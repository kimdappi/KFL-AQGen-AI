# =====================================
# vocabulary_retriever.py
# =====================================
"""
TOPIK 단어 Retriever
"""
import pandas as pd
from typing import List, Dict
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class TOPIKVocabularyRetriever:
    """TOPIK 단어 CSV 파일 기반 Retriever"""
    
    def __init__(self, csv_paths: Dict[str, List[str]]):
        self.csv_paths = csv_paths
        self.vocabulary_data = {}
        self.retrievers = {}
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
                        # CSV 컬럼: Number, Level, Vocabulary, Wordclass, Guide
                        vocabulary = row.get('Vocabulary', '')
                        wordclass = row.get('Wordclass', '')
                        guide = row.get('Guide', '')
                        topik_level = row.get('Level', '')
                        
                        doc_content = f"단어: {vocabulary}\n"
                        doc_content += f"품사: {wordclass}\n"
                        doc_content += f"설명: {guide}\n"
                        doc_content += f"TOPIK 레벨: {topik_level}"
                        
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
        """레벨별 retriever 생성"""
        embeddings = OpenAIEmbeddings()
        
        for level, documents in self.vocabulary_data.items():
            if documents:
                # Vector Store Retriever
                vectorstore = FAISS.from_documents(documents, embeddings)
                vector_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 10}
                )
                
                # BM25 Retriever
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 10
                
                # Ensemble Retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.5, 0.5]
                )
                
                self.retrievers[level] = ensemble_retriever
    
    def invoke(self, query: str, level: str) -> List[Document]:
        """난이도에 따른 단어 검색"""
        if level in self.retrievers:
            return self.retrievers[level].get_relevant_documents(query)
        return []


