# -------------------------------------
# TOPIK 단어 Retriever (난이도 준수 + 랜덤성 강화 + MMR)
# -------------------------------------
import time, random, hashlib
from collections import deque
from typing import List, Dict
import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever

class TOPIKVocabularyRetriever:
    """TOPIK 단어 CSV 파일 기반 Retriever (난이도 준수 + 랜덤성 + MMR)"""
    
    # 난이도 정규화
    LEVEL_ORDER = ["basic", "intermediate", "advanced"]
    NEAR_LEVELS = {
        "basic":        ["basic", "intermediate"],          # 부족할 때 인접 난이도 보충 순서
        "intermediate": ["intermediate", "basic", "advanced"],
        "advanced":     ["advanced", "intermediate"]
    }

    def __init__(self, csv_paths: Dict[str, List[str]]):
        self.csv_paths = csv_paths
        self.vocabulary_data = {}
        self.retrievers = {}
        self.level_docs_flat = {}     # ε-greedy용 전체 풀
        self.recent_words = deque(maxlen=200)  # 최근 노출 억제
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
                        topik_level = str(row.get('Level', '')).strip()  # CSV의 Level 그대로 보관(숫자/문자)
                        
                        doc_content = (
                            f"단어: {vocabulary}\n"
                            f"품사: {wordclass}\n"
                            f"설명: {guide}\n"
                            f"TOPIK 레벨: {topik_level}"
                        )
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                'difficulty_level': level,    # 우리 시스템 난이도
                                'source': path,
                                'word': vocabulary,
                                'wordclass': wordclass,
                                'guide': guide,
                                'topik_level': topik_level,   # 원본 TOPIK Level
                                'file_name': path.split('/')[-1].replace('.csv', '')
                            }
                        )
                        level_documents.append(doc)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            self.vocabulary_data[level] = level_documents
    
    def _create_retrievers(self):
        """레벨별 retriever 생성: MMR + fetch_k로 후보 다양화"""
        embeddings = OpenAIEmbeddings()
        for level, documents in self.vocabulary_data.items():
            if not documents:
                continue

            self.level_docs_flat[level] = documents  # ε-greedy 탐험용 전체 풀 저장

            vectorstore = FAISS.from_documents(documents, embeddings)
            # 다양성 강제: MMR + 넓은 fetch_k
            vector_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 30, "fetch_k": 300, "lambda_mult": 0.5}
            )

            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 60  # 텍스트 키워드 기반 후보폭 확대

            # 앙상블: 의미(임베딩) 0.6, 키워드 0.4
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            self.retrievers[level] = ensemble_retriever


    # 중복체크, 랜덤 및 난이도 유지

    def _dedup_by_word(self, docs: List[Document]) -> List[Document]:
        seen, unique = set(), []
        for d in docs:
            w = d.metadata.get('word', '').strip()
            if w and w not in seen:
                seen.add(w); unique.append(d)
        return unique

    def _filter_recent(self, docs: List[Document]) -> List[Document]:
        return [d for d in docs if d.metadata.get('word', '').strip() not in self.recent_words]

    def _seed_from_query(self, query: str):
        # 쿼리 + 분 단위 타임슬라이스로 시드 변화
        key = f"{query}:{int(time.time()//60)}"
        seed = int(hashlib.md5(key.encode()).hexdigest(), 16) & 0xFFFFFFFF
        random.seed(seed)

    def _level_match(self, target_level: str, doc_level: str) -> bool:
        # 시스템 난이도 정확 일치
        return target_level == doc_level

    def _within_near_levels(self, target_level: str, doc_level: str) -> bool:
        # 인접 난이도 허용(부족 시 보충)
        return doc_level in self.NEAR_LEVELS.get(target_level, [target_level])

    def _weighted_sample(self, candidates: List[Document], k: int) -> List[Document]:
        # 순위 기반 가중치(1/rank) 비복원 샘플링 
        if not candidates:
            return []
        weights = [1/(i+1) for i in range(len(candidates))]
        picked, used = [], set()
        while len(picked) < min(k, len(candidates)):
            idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
            if idx in used: 
                continue
            used.add(idx)
            picked.append(candidates[idx])
        return picked

    def _epsilon_explore(self, level: str, m: int) -> List[Document]:
        # 완전 랜덤 탐험(ε-greedy) 최근 단어 제외 후 랜덤 
        pool = self._filter_recent(self.level_docs_flat.get(level, []))
        if len(pool) <= m:
            random.shuffle(pool); return pool
        return random.sample(pool, m)


    def invoke(self, query: str, level: str, epsilon: float = 0.2) -> List[Document]:
        """
        난이도 준수 + 랜덤성 + 다양성(MMR)로 5개 반환
        - 1순위: 요청 난이도와 정확히 일치하는 후보
        - 2순위: 인접 난이도(±1)에서 보충
        - 3순위: 그래도 부족하면 최소량만 임시 보충
        """
        self._seed_from_query(query)

        retriever = self.retrievers.get(level)
        if not retriever:
            return []

        # 풍부한 후보 수집(MMR+BM25 앙상블 결과)
        docs = retriever.get_relevant_documents(query)
        docs = self._dedup_by_word(docs)[:60]   # 다양성 유지
        docs = self._filter_recent(docs)        # 최근 노출 회피

        # 난이도 우선 필터
        exact = [d for d in docs if self._level_match(level, d.metadata.get('difficulty_level', ''))]
        near  = [d for d in docs if d not in exact and self._within_near_levels(level, d.metadata.get('difficulty_level', ''))]
        far   = [d for d in docs if d not in exact and d not in near]  # 그 외(되도록 지양)

        picked: List[Document] = []

        # ε-greedy: 소량 탐험(레벨 내)
        take_explore = 1 if random.random() < epsilon else 0

        # 우선순위대로 채우기 (가중 랜덤 → 탐험 → 보충) - 5개로 증가
        if exact:
            picked += self._weighted_sample(exact, k=5 - len(picked))
        if len(picked) < 5 and take_explore:
            picked += self._epsilon_explore(level, m=min(1, 5 - len(picked)))
        if len(picked) < 5 and near:
            picked += self._weighted_sample(near, k=5 - len(picked))
        if len(picked) < 5 and far:
            # 정말 부족할 때만 아주 소량 보충 (난이도 비약 방지)
            picked += self._weighted_sample(far, k=5 - len(picked))

        picked = picked[:5]

        # 최근 캐시에 등록(중복 회피)
        for d in picked:
            w = d.metadata.get('word', '').strip()
            if w:
                self.recent_words.append(w)

        return picked

