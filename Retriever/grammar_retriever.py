"""
문법 Retriever
"""
import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import random

class GrammarRetriever:
    """문법 JSON 파일 기반 Retriever"""
    
    def __init__(self, json_paths: Dict[str, str]):
        self.json_paths = json_paths
        self.grammar_data = {}
        self.retrievers = {}
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
        """레벨별 retriever 생성"""
        embeddings = OpenAIEmbeddings()
        
        for level, documents in self.grammar_data.items():
            if documents:
                vectorstore = FAISS.from_documents(documents, embeddings)
                vector_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                )
                self.retrievers[level] = vector_retriever
    
    
    def invoke(self, query: str, level: str, k: int = 10) -> List[Document]:
        """난이도에 따른 문법 검색 (랜덤성 추가)"""
        if level in self.retrievers:
            # 우선 관련 문법 후보 50개 가져오기
            docs = self.retrievers[level].get_relevant_documents(query)

            if docs:
                # grade로 정렬
                docs.sort(key=lambda x: x.metadata.get('grade', 999))
                
                # 상위 50개 중에서 랜덤으로 10개 추출
                top_candidates = docs[:50]
                sample_size = min(k, len(top_candidates))
                return random.sample(top_candidates, sample_size)
        
        return []