"""
문법 Retriever (BM25 + Reranker 개선)
"""
import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import random

# BGE Reranker 공유 (vocabulary_retriever와 동일)
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
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
    
    _BGE_RERANKER_AVAILABLE = True
except ImportError:
    _BGE_RERANKER_AVAILABLE = False
    BGEReranker = None

class GrammarRetriever:
    """문법 JSON 파일 기반 Retriever (BM25 + Reranker 개선)"""
    
    def __init__(self, json_paths: Dict[str, str], use_reranker: bool = True):
        self.json_paths = json_paths
        self.grammar_data = {}
        self.retrievers = {}
        self.use_reranker = use_reranker and _BGE_RERANKER_AVAILABLE
        self.reranker = None
        from collections import deque
        self.query_recent_grammar = {}  # 쿼리별 최근 문법 캐시 (쿼리별 중복 방지)
        self.query_call_count = {}  # 쿼리별 실행 횟수 (매번 다른 결과 보장)
        if self.use_reranker:
            try:
                self.reranker = BGEReranker()
                print("   ✅ Grammar Retriever: BGE Reranker 초기화 완료")
            except Exception as e:
                print(f"   ⚠️ Grammar Retriever: Reranker 초기화 실패: {e}")
                self.use_reranker = False
        self._load_grammar()
        self._create_retrievers()
    
    def _load_grammar(self):
        """JSON 파일들을 레벨별로 로드"""
        for level, path in self.json_paths.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    grammar_list = json.load(f)
                
                # grade 순으로 정렬
                grammar_list.sort(key=lambda x: x.get('grade', 999))
                
                level_documents = []
                for item in grammar_list:
                    doc_content = f"문법: {item.get('grammar', '')}\n"
                    doc_content += f"등급: {item.get('grade', '')}\n"
                    doc_content += f"레벨: {item.get('level', '')}"
                    
                    if 'description' in item:
                        doc_content += f"\n설명: {item.get('description', '')}"
                    if 'example' in item:
                        doc_content += f"\n예문: {item.get('example', '')}"
                    
                    doc = Document(
                        page_content=doc_content,
                        metadata={
                            'level': level,
                            'grade': item.get('grade', 0),
                            'grammar': item.get('grammar', ''),
                            'level_code': item.get('level', ''),
                            'source': path
                        }
                    )
                    level_documents.append(doc)
            except Exception as e:
                print(f"Error loading {path}: {e}")
            
            self.grammar_data[level] = level_documents
    
    def _create_retrievers(self):
        """레벨별 retriever 생성: BM25 + Vector 앙상블"""
        embeddings = OpenAIEmbeddings()
        
        for level, documents in self.grammar_data.items():
            if documents:
                # Vector search
                vectorstore = FAISS.from_documents(documents, embeddings)
                vector_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 50}  # 넓게 가져와서 reranker에게 위임
                )
                
                # BM25 추가 (키워드 검색 강화)
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 50
                
                # 앙상블: 의미(임베딩) 0.6, 키워드 0.4
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.6, 0.4]
                )
                self.retrievers[level] = ensemble_retriever
    
    
    def invoke(self, query: str, level: str, k: int = 10) -> List[Document]:
        """
        개선된 문법 검색 파이프라인 (쿼리별 중복 방지)
        1. BM25 + Vector 앙상블로 넓게 후보 수집
        2. Reranker로 재정렬 (선택적)
        3. 쿼리별 최근 문법 제외
        4. grade 정렬 후 실행 횟수 기반 랜덤 샘플링 (매번 다른 결과)
        """
        if level not in self.retrievers:
            return []
        
        # 쿼리별 실행 횟수 추적
        if query not in self.query_call_count:
            self.query_call_count[query] = 0
        self.query_call_count[query] += 1
        
        # 실행 횟수 기반 시드 생성 (매번 다른 결과)
        import hashlib
        seed_key = f"{query}:{self.query_call_count[query]}"
        seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) & 0xFFFFFFFF
        random.seed(seed)
        
        # 1단계: 앙상블로 넓게 후보 수집
        docs = self.retrievers[level].get_relevant_documents(query)
        
        if not docs:
            return [] 
        
        # 2단계: Reranker로 재정렬 (선택적, 쿼리-문법 관련성 향상)
        if self.use_reranker and self.reranker and len(docs) > 20:
            docs = self.reranker.rerank(query, docs, top_k=30)
        
        # 3단계: 쿼리별 최근 문법 제외 (중복 방지)
        from collections import deque
        if query not in self.query_recent_grammar:
            self.query_recent_grammar[query] = deque(maxlen=50)  # 쿼리별 최근 50개
        
        recent_grammar = set(self.query_recent_grammar[query])  # 빠른 검색을 위해 set 변환
        docs = [d for d in docs if d.metadata.get('grammar', '') not in recent_grammar]
        
        if not docs:
            # 최근 문법이 너무 많으면 캐시 초기화
            self.query_recent_grammar[query] = deque(maxlen=50)
            docs = self.retrievers[level].get_relevant_documents(query)
            if self.use_reranker and self.reranker and len(docs) > 20:
                docs = self.reranker.rerank(query, docs, top_k=30)
        
        # 4단계: grade로 정렬
        docs.sort(key=lambda x: x.metadata.get('grade', 999))
        
        # 5단계: 상위 후보 중 실행 횟수 기반 랜덤 샘플링 (매번 다른 결과)
        top_candidates = docs[:50]
        sample_size = min(k, len(top_candidates))
        picked = random.sample(top_candidates, sample_size) if len(top_candidates) >= sample_size else top_candidates
        
        # 쿼리별 최근 문법 캐시 업데이트
        for d in picked:
            grammar = d.metadata.get('grammar', '')
            if grammar:
                self.query_recent_grammar[query].append(grammar)  # deque는 자동으로 maxlen 처리
        
        return picked