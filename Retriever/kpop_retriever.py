# =====================================
# kpop_retriever.py (영어 메타데이터 지원)
# =====================================
"""
K-pop 그룹 정보 Retriever (영어 메타데이터 → 한국어 학습 자료)
"""
import json
from typing import List
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


class KpopSentenceRetriever:
    """K-pop 그룹 메타데이터 기반 Retriever (영어 → 한국어 자동 처리)"""

    def __init__(self, json_path: str):
        """
        Args:
            json_path: K-pop 그룹 메타데이터가 담긴 JSON 파일 경로
        """
        self.json_path = json_path
        self.kpop_data = []
        self.retriever = None
        self._load_data()
        self._create_retriever()

    def _load_data(self):
        """JSON 파일에서 K-pop 그룹 메타데이터 로드"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 데이터가 리스트인 경우와 딕셔너리인 경우 모두 처리
            if isinstance(data, dict):
                data = [data]
            
            for item in data:
                # 그룹 기본 정보 추출
                group = item.get("group", "")
                agency = item.get("agency", "")
                fandom = item.get("fandom", "")
                concepts = item.get("concepts", [])
                debut = item.get("debut", "")
                
                # 멤버 정보 추출
                members = item.get("members", [])
                member_info_list = []
                for member in members:
                    member_name = member.get("name", "")
                    role = member.get("role", "")
                    member_debut = member.get("debut", "")
                    member_info_list.append({
                        "name": member_name,
                        "role": role,
                        "debut": member_debut
                    })
                
                if not group:  # 그룹명이 없으면 스킵
                    continue
                
                # Document 내용 구성 (검색용 - 영어 포함)
                doc_content = f"K-pop Group: {group}\n"
                doc_content += f"그룹: {group}\n"
                doc_content += f"Agency: {agency}\n"
                doc_content += f"소속사: {agency}\n"
                doc_content += f"Fandom: {fandom}\n"
                doc_content += f"팬덤: {fandom}\n"
                doc_content += f"Concepts: {', '.join(concepts)}\n"
                doc_content += f"컨셉: {', '.join(concepts)}\n"
                
                # 멤버 정보
                member_names = [m["name"] for m in member_info_list]
                doc_content += f"Members: {', '.join(member_names)}\n"
                doc_content += f"멤버: {', '.join(member_names)}"
                
                # Document 생성 (메타데이터에 모든 정보 저장)
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        "source": self.json_path,
                        "group": group,
                        "agency": agency,
                        "fandom": fandom,
                        "concepts": concepts,
                        "debut": debut,
                        "members": member_info_list,  # 전체 멤버 정보
                        "member_names": member_names   # 멤버 이름 리스트
                    }
                )
                self.kpop_data.append(doc)
                
            print(f"   ✅ K-pop 데이터 로드 완료: {len(self.kpop_data)}개 그룹")
            
        except Exception as e:
            print(f"   ❌ K-pop 데이터 로드 실패 ({self.json_path}): {e}")
            self.kpop_data = []

    def _create_retriever(self):
        """벡터 기반 retriever 생성"""
        if not self.kpop_data:
            print("   ⚠️ K-pop 데이터가 없어 retriever를 생성할 수 없습니다.")
            return
        
        try:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(self.kpop_data, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
            print(f"   ✅ K-pop retriever 생성 완료")
        except Exception as e:
            print(f"   ❌ K-pop retriever 생성 실패: {e}")
            self.retriever = None

    def invoke(self, query: str, level: str = None) -> List[Document]:
        """
        쿼리 기반 K-pop 그룹 정보 검색
        
        Args:
            query: 검색 쿼리 (그룹명, 멤버명, 컨셉 등)
            level: 하위 호환성을 위한 파라미터 (사용하지 않음)
            
        Returns:
            검색된 Document 리스트 (메타데이터에 영어 정보 포함)
        """
        if not self.retriever:
            print("   ⚠️ Retriever가 초기화되지 않았습니다.")
            return []
        
        try:
            results = self.retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            print(f"   ❌ K-pop 검색 실패: {e}")
            return []