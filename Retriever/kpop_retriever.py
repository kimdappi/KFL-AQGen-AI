from typing import List, Dict, Tuple, Optional
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

class KpopSentenceRetriever:
    """
    - 멀티링구얼 임베딩 사용
    - 그룹명 전용 인덱스(group_name_index)로 타깃 그룹 선별
    """
    def __init__(self, json_path: str, embedding_model: str = "text-embedding-3-large",
                 group_match_topk: int = 1, group_match_threshold: float = 0.75):
        self.json_path = json_path
        self.kpop_data: List[Document] = []
        self.retriever = None

        # 임베딩 객체
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self._load_data()
        self._create_retriever()

        # 그룹명 전용 임베딩 인덱스 
        self.group_name_index: Dict[str, np.ndarray] = {}
        self.group_match_topk = group_match_topk
        self.group_match_threshold = group_match_threshold
        self._build_group_name_index()

    def _load_data(self):
        import json
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]

            for item in data:
                group = item.get("group", "")
                if not group:
                    continue
                agency  = item.get("agency", "")
                fandom  = item.get("fandom", "")
                concepts = item.get("concepts", [])
                debut   = item.get("debut", "")
                members = item.get("members", [])

                member_info_list = [{
                    "name": m.get("name",""),
                    "role": m.get("role",""),
                    "debut": m.get("debut","")
                } for m in members]
                member_names = [m["name"] for m in member_info_list]


                content = []
                content.append(f"K-pop Group: {group}")
                content.append(f"Agency: {agency}")
                content.append(f"Fandom: {fandom}")
                content.append(f"Concepts: {', '.join(concepts)}")
                content.append(f"Members: {', '.join(member_names)}")
                if debut:
                    content.append(f"Debut: {debut}")

                # 한국어 태그 라벨 
                content.append(f"그룹: {group}")
                content.append(f"소속사: {agency}")
                content.append(f"팬덤: {fandom}")
                content.append(f"컨셉: {', '.join(concepts)}")
                if debut:
                    content.append(f"데뷔: {debut}")

                doc = Document(
                    page_content="\n".join(content),
                    metadata={
                        "source": self.json_path,
                        "group": group,
                        "agency": agency,
                        "fandom": fandom,
                        "concepts": concepts,
                        "debut": debut,
                        "members": member_info_list,
                        "member_names": member_names,
                    }
                )
                self.kpop_data.append(doc)
            print(f"   ✅ K-pop 데이터 로드 완료: {len(self.kpop_data)}개 그룹")
        except Exception as e:
            print(f"   ❌ K-pop 데이터 로드 실패 ({self.json_path}): {e}")
            self.kpop_data = []

    def _create_retriever(self):
        if not self.kpop_data:
            print("   ⚠️ K-pop 데이터가 없어 retriever를 생성할 수 없습니다.")
            return
        try:
            vectorstore = FAISS.from_documents(self.kpop_data, self.embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
            print("   ✅ K-pop retriever 생성 완료")
        except Exception as e:
            print(f"   ❌ K-pop retriever 생성 실패: {e}")
            self.retriever = None

    def _build_group_name_index(self):
        """
        각 그룹명(영문)만 따로 임베딩해두는 소형 인덱스.
        한국어 질의도 같은 임베딩 공간에서 유사도 비교가 가능하므로
        별칭/하드코딩 없이 매칭된다.
        """
        try:
            names = [d.metadata["group"] for d in self.kpop_data]
            vecs = self.embeddings.embed_documents(names)  # List[List[float]]
            self.group_name_index = {
                name: np.array(vec, dtype=np.float32) for name, vec in zip(names, vecs)
            }
            print(f"   ✅ 그룹명 인덱스 구축 완료: {len(self.group_name_index)}개")
        except Exception as e:
            print(f"   ❌ 그룹명 인덱스 구축 실패: {e}")
            self.group_name_index = {}

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _match_groups_by_query(self, query: str) -> List[Tuple[str, float]]:
        """
        질의 임베딩과 그룹명 임베딩을 비교해 유사도 순으로 정렬 반환
        """
        if not self.group_name_index:
            return []
        try:
            qv = np.array(self.embeddings.embed_query(query), dtype=np.float32)
            scored = []
            for name, vec in self.group_name_index.items():
                sim = self._cosine_sim(qv, vec)
                scored.append((name, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        except Exception:
            return []

    def invoke(self, query: str, level: str = None) -> List[Document]:
        """
        질의 -> 그그룹명 임베딩 매칭으로 타깃 그룹 선별
        임계치 이상이면 해당 그룹 문서만 반환
        실패시 일반 FAISS 검색 + 상위 20 랜덤 10 
        """
        import random

        if not self.retriever:
            print("   ⚠️ Retriever가 초기화되지 않았습니다.")
            return []

        try:
            # 그룹명 매칭 시도
            ranked = self._match_groups_by_query(query)
            selected_groups = []
            if ranked:
                # 상위 k개 중 임계치 이상인 것만 채택
                for name, score in ranked[:self.group_match_topk]:
                    if score >= self.group_match_threshold:
                        selected_groups.append(name)

            # 타깃 그룹이 있으면 그 문서만 반환
            if selected_groups:
                filtered = [d for d in self.kpop_data if d.metadata.get("group") in selected_groups]
                random.shuffle(filtered)
                return filtered[:10]

            # 매칭 실패 → 일반 벡터 검색 폴백(상위 20 중 랜덤 10)
            results = self.retriever.get_relevant_documents(query)
            if len(results) > 10:
                return random.sample(results[:20], 10)
            return results

        except Exception as e:
            print(f"   ❌ K-pop 검색 실패: {e}")
            return []