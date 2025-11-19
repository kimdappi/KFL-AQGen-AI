# -------------------------------------
# TOPIK 단어 Retriever (BGE Reranker 적용)
# -------------------------------------
import time, random, hashlib
from collections import deque
from typing import List, Dict
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever

class BGEReranker:
    """BGE-reranker-v2-m3 기반 reranker"""
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def rerank(self, query: str, docs: List[Document], top_k: int = 10) -> List[Document]:
        if not docs:
            return []
        
        pairs = [[query, d.page_content] for d in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu()
        
        ranked_idx = scores.argsort(descending=True)[:top_k]
        return [docs[i] for i in ranked_idx]


class TOPIKVocabularyRetriever:
    """TOPIK 단어 CSV 파일 기반 Retriever (BGE Reranker + 난이도 준수)"""
    
    LEVEL_ORDER = ["basic", "intermediate", "advanced"]
    NEAR_LEVELS = {
        "basic":        ["basic", "intermediate"],
        "intermediate": ["intermediate", "basic", "advanced"],
        "advanced":     ["advanced", "intermediate"]
    }

    def __init__(self, csv_paths: Dict[str, List[str]]):
        self.csv_paths = csv_paths
        self.vocabulary_data = {}
        self.retrievers = {}
        self.level_docs_flat = {}
        self.recent_words = deque(maxlen=200)  # 전역 최근 단어 (모든 쿼리 공통)
        self.query_recent_words = {}  # 쿼리별 최근 단어 캐시 (쿼리별 중복 방지)
        self.query_call_count = {}  # 쿼리별 실행 횟수 (매번 다른 결과 보장)
        self.reranker = BGEReranker()  # Reranker 초기화
        self._load_vocabulary()
        self._create_retrievers()
    
    def _load_vocabulary(self):
        """CSV 파일들을 레벨별로 로드"""
        for level, paths in self.csv_paths.items():
            level_documents = []
            for path in paths:
                try:
                    df = pd.read_csv(path, encoding='utf-8')
                    for _, row in df.iterrows():
                        vocabulary = row.get('Vocabulary', '')
                        wordclass = row.get('Wordclass', '')
                        guide = row.get('Guide', '')
                        topik_level = str(row.get('Level', '')).strip()
                        
                        doc_content = (
                            f"단어: {vocabulary}\n"
                            f"품사: {wordclass}\n"
                            f"설명: {guide}\n"
                            f"TOPIK 레벨: {topik_level}"
                        )
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                'difficulty_level': level,
                                'source': path,
                                'word': vocabulary,
                                'wordclass': wordclass,
                                'guide': guide,
                                'topik_level': topik_level,
                                'file_name': path.split('/')[-1].replace('.csv', '')
                            }
                        )
                        level_documents.append(doc)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            self.vocabulary_data[level] = level_documents
    
    def _create_retrievers(self):
        """레벨별 retriever 생성: 일반 similarity search (reranker가 다양성 처리)"""
        embeddings = OpenAIEmbeddings()
        for level, documents in self.vocabulary_data.items():
            if not documents:
                continue

            self.level_docs_flat[level] = documents

            vectorstore = FAISS.from_documents(documents, embeddings)
            # MMR 제거, 일반 검색으로 변경 (reranker가 정렬)
            vector_retriever = vectorstore.as_retriever(
                search_kwargs={"k": 80}  # 넓게 가져와서 reranker에게 위임
            )

            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 80

            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            self.retrievers[level] = ensemble_retriever


    def _dedup_by_word(self, docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for d in docs:
            w = d.metadata.get('word', '').strip()
            if w and w not in seen:
                seen.add(w); unique.append(d)
        return unique

    def _filter_recent(self, docs: List[Document]) -> List[Document]:
        """전역 최근 단어 필터링"""
        return [d for d in docs if d.metadata.get('word', '').strip() not in self.recent_words]
    
    def _filter_recent_by_query(self, docs: List[Document], query: str) -> List[Document]:
        """
        쿼리별 최근 단어 필터링
        같은 쿼리를 여러 번 실행해도 이전에 나온 단어 제외
        """
        if query not in self.query_recent_words:
            self.query_recent_words[query] = deque(maxlen=50)  # 쿼리별 최근 50개
        
        recent = self.query_recent_words[query]
        return [d for d in docs if d.metadata.get('word', '').strip() not in recent]

    def _seed_from_query(self, query: str):
        """
        쿼리 + 실행 횟수 기반 시드 생성
        같은 쿼리라도 실행 횟수가 다르면 다른 결과 보장
        """
        # 쿼리별 실행 횟수 추적
        if query not in self.query_call_count:
            self.query_call_count[query] = 0
        self.query_call_count[query] += 1
        
        # 쿼리 + 실행 횟수로 시드 생성 (매번 다른 결과)
        key = f"{query}:{self.query_call_count[query]}"
        seed = int(hashlib.md5(key.encode()).hexdigest(), 16) & 0xFFFFFFFF
        random.seed(seed)
    
    def _query_hash_based_sample(self, candidates: List[Document], query: str, k: int) -> List[Document]:
        """
        쿼리 해시 기반 가중 랜덤 샘플링
        - 같은 쿼리면 같은 단어 선택 (일관성)
        - 다른 쿼리면 다른 단어 선택 (다양성 보장)
        - "블랙핑크 관련 중급" vs "스트레이키즈 관련 중급" → 다른 단어 선택
        """
        if not candidates:
            return []
        
        # 쿼리 해시로 시드 생성 (쿼리별로 고유한 시드)
        query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16) & 0xFFFFFFFF
        random.seed(query_hash)
        
        # 순위 기반 가중치 (상위 단어가 더 높은 확률, 하지만 랜덤성도 포함)
        weights = [1/(i+1) for i in range(len(candidates))]
        
        picked = []
        used_indices = set()
        
        while len(picked) < min(k, len(candidates)):
            idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
            if idx not in used_indices:
                used_indices.add(idx)
                picked.append(candidates[idx])
        
        return picked

    def _level_match(self, target_level: str, doc_level: str) -> bool:
        return target_level == doc_level

    def _within_near_levels(self, target_level: str, doc_level: str) -> bool:
        return doc_level in self.NEAR_LEVELS.get(target_level, [target_level])


    def invoke(self, query: str, level: str) -> List[Document]:
        """
        BGE Reranker + 쿼리 해시 기반 다양성 보장 검색
        1. 앙상블로 80개 후보 수집
        2. Reranker로 재정렬 (쿼리 관련성 고려)
        3. 난이도 필터링
        4. 쿼리 해시 기반 가중 랜덤 샘플링 (쿼리별 다른 단어 보장)
        
        예: "블랙핑크 관련 중급" vs "스트레이키즈 관련 중급" → 다른 단어 선택
        """
        # 쿼리 해시로 시드 고정 (같은 쿼리면 같은 결과, 다른 쿼리면 다른 결과)
        self._seed_from_query(query)
        
        retriever = self.retrievers.get(level)
        if not retriever:
            return []

        # 1단계: 넓게 후보 수집
        docs = retriever.get_relevant_documents(query)
        docs = self._dedup_by_word(docs)[:80]
        docs = self._filter_recent(docs)  # 전역 최근 단어 제외
        docs = self._filter_recent_by_query(docs, query)  # 쿼리별 최근 단어 제외 (중복 방지)

        if not docs:
            return []

        # 2단계: BGE Reranker로 재정렬 (쿼리-단어 관련성 고려)
        reranked = self.reranker.rerank(query, docs, top_k=30)

        # 3단계: 난이도 필터링
        exact = [d for d in reranked if self._level_match(level, d.metadata.get('difficulty_level', ''))]
        near = [d for d in reranked if d not in exact and self._within_near_levels(level, d.metadata.get('difficulty_level', ''))]

        # 4단계: 실행 횟수 기반 가중 랜덤 샘플링 (매번 다른 결과 보장)
        picked = []
        if exact:
            # 실행 횟수 기반 샘플링 (같은 쿼리라도 매번 다른 단어)
            picked += self._query_hash_based_sample(exact, query, k=5 - len(picked))
        if len(picked) < 5 and near:
            picked += self._query_hash_based_sample(near, query, k=5 - len(picked))

        picked = picked[:5]

        # 최근 캐시 업데이트 (전역 + 쿼리별)
        for d in picked:
            w = d.metadata.get('word', '').strip()
            if w:
                self.recent_words.append(w)  # 전역 캐시
                if query not in self.query_recent_words:
                    self.query_recent_words[query] = deque(maxlen=50)
                self.query_recent_words[query].append(w)  # 쿼리별 캐시

        return picked