# =====================================
# kpop_retriever.py
# =====================================
"""
K-pop 문장 Retriever
"""
import pandas as pd
from typing import List, Dict
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever


class KpopSentenceRetriever:
    """K-pop 문장 기반 Retriever"""

    def __init__(self, csv_paths: Dict[str, List[str]]):
        self.csv_paths = csv_paths
        self.kpop_data = {}
        self.retrievers = {}
        self._load_data()
        self._create_retrievers()

    def _load_data(self):
        """CSV 파일들을 레벨별로 로드"""
        for level, paths in self.csv_paths.items():
            level_documents = []
            for path in paths:
                try:
                    df = pd.read_csv(path, encoding="utf-8")
                    for _, row in df.iterrows():
                        # CSV 컬럼 예시: Sentence, Song, Group, Context
                        sentence = row.get("Sentence", "")
                        song = row.get("Song", "")
                        group = row.get("Group", "")
                        context = row.get("Context", "")

                        doc_content = f"K-pop 문장: {sentence}\n"
                        if song:
                            doc_content += f"노래: {song}\n"
                        if group:
                            doc_content += f"그룹: {group}\n"
                        if context:
                            doc_content += f"설명: {context}"

                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                "difficulty_level": level,
                                "source": path,
                                "sentence": sentence,
                                "song": song,
                                "group": group,
                                "context": context,
                                "file_name": path.split("/")[-1].replace(".csv", ""),
                            },
                        )
                        level_documents.append(doc)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            self.kpop_data[level] = level_documents

    def _create_retrievers(self):
        """레벨별 retriever 생성"""
        embeddings = OpenAIEmbeddings()

        for level, documents in self.kpop_data.items():
            if documents:
                # 벡터 검색
                vectorstore = FAISS.from_documents(documents, embeddings)
                vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

                # BM25 검색
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 10

                # 단순 앙상블 (우선은 벡터 기반만 리턴해도 OK)
                self.retrievers[level] = vector_retriever

    def invoke(self, query: str, level: str) -> List[Document]:
        """난이도에 따른 K-pop 문장 검색"""
        if level in self.retrievers:
            return self.retrievers[level].get_relevant_documents(query)
        return []
