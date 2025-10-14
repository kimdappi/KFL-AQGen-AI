# =====================================
# grammar_retriever.py
# =====================================
"""
문법 Retriever
"""
import json
from typing import List, Dict
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document  # <-- 올바른 경로로 수정


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
    
    # def invoke(self, query: str, level: str) -> List[Document]:
    #     """난이도에 따른 문법 검색 (grade 낮은 순으로)"""
    #     if level in self.retrievers:
    #         docs = self.retrievers[level].get_relevant_documents(query)
    #         # grade 순으로 재정렬
    #         docs.sort(key=lambda x: x.metadata.get('grade', 999))
    #         return docs[:1]
    #     return []

    # def invoke(self, query: str, level: str, keyword: str = None) -> List[Document]:
    #     if level not in self.retrievers:
    #         return []
            
    #     docs = self.retrievers[level].get_relevant_documents(query)
    
    # # 1. 키워드 필터링 단계 (추가!)
    #     if keyword:
    #         filtered_docs = [
    #             doc for doc in docs if keyword in doc.metadata.get('grammar', '')
    #         ]
    #     # 키워드와 일치하는 문법이 하나라도 있으면, 그것만 반환
    #     if filtered_docs:
    #         filtered_docs.sort(key=lambda x: x.metadata.get('grade', 999))
    #         return filtered_docs[:5]

    # # 2. 키워드가 없거나 일치하는게 없으면, 기존 로직 수행
    #     docs.sort(key=lambda x: x.metadata.get('grade', 999))
    #     return docs[:5]

    def invoke(self, query: str, level: str, keyword: str = None) -> List[Document]:
        """난이도에 따른 문법 검색. 특정 keyword가 주어지면 우선적으로 필터링합니다."""
        if level not in self.retrievers:
            return []
            
        docs = self.retrievers[level].get_relevant_documents(query)
        
        # 1. 키워드 필터링 단계
        if keyword:
            normalized_keyword = keyword.strip()
            
            # 'filtered_docs'는 이 if 블록 안에서만 정의되고 사용됩니다.
            filtered_docs = [
                doc for doc in docs if normalized_keyword == doc.metadata.get('grammar', '').strip()
            ]
            
            # 키워드와 일치하는 문법이 하나라도 있으면, 그것만 반환하고 함수를 즉시 종료합니다.
            if filtered_docs:
                filtered_docs.sort(key=lambda x: x.metadata.get('grade', 999))
                print(f"   - 키워드 필터링 성공: '{keyword}'에 해당하는 문법을 찾았습니다.")
                return filtered_docs[:5]

        # 2. 키워드가 없거나(keyword is None) 키워드 필터링에 실패했을 경우,
        #    이곳으로 와서 기본 유사도 검색 결과를 반환합니다.
        docs.sort(key=lambda x: x.metadata.get('grade', 999))
        return docs[:5]