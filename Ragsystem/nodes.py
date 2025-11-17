"""
LangGraph 노드 정의 (개선된 재생성 로직 - 간결 버전)
"""
from typing import List, Dict, Any

from langchain.chat_models import ChatOpenAI
from Ragsystem.schema import GraphState
from utils import (
    extract_words_from_docs,
    extract_grammar_with_grade,
    get_group_type,
)
from config import LLM_CONFIG
from agents import QueryAnalysisAgent, QualityCheckAgent


#기본 RAG 노드
class KoreanLearningNodes:
    """한국어 학습 노드 클래스"""

    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.kpop_retriever = kpop_retriever
        self.llm = llm or ChatOpenAI(
            model="gpt-5",
            temperature=LLM_CONFIG.get("temperature", 0.7),
            max_completion_tokens=LLM_CONFIG.get("max_completion_tokens", 1000),
        )

    def retrieve_vocabulary(self, state: GraphState) -> GraphState:
        """단어 검색 노드"""
        level = state["difficulty_level"]
        query = state["input_text"]
        vocab_docs = self.vocabulary_retriever.invoke(query, level)
        return {"vocabulary_docs": vocab_docs}

    def retrieve_grammar(self, state: GraphState) -> GraphState:
        """문법 검색 노드"""
        level = state["difficulty_level"]
        query = state["input_text"]
        grammar_docs = self.grammar_retriever.invoke(query, level)
        return {"grammar_docs": grammar_docs}



#Agentic RAG 노드 - 쿼리 분석
class AgenticKoreanLearningNodes(KoreanLearningNodes):
    """Agentic RAG 노드 - 리트리버 정보 기반 자연스러운 3문장 생성 (간결 버전)"""

    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
        # kpop_retriever를 QueryAnalysisAgent에 전달하여 임베딩 기반 매칭 활성화
        self.query_agent = QueryAnalysisAgent(llm, kpop_retriever=kpop_retriever)
        self.quality_agent = QualityCheckAgent(llm)

    #Agents Nodes 

    def analyze_query_agent(self, state: GraphState) -> GraphState:
        """쿼리 분석 에이전트 노드"""
        print("\n🔍 [Agent] Query Analysis")
        analysis = self.query_agent.analyze(state["input_text"])

        print(f"   Difficulty: {analysis['difficulty']}")
        print(f"   Topic: {analysis['topic']}")
        print(f"   Needs K-pop: {analysis.get('needs_kpop', False)}")
        kpop_filters = analysis.get('kpop_filters', {})
        if kpop_filters:
            filter_info = []
            if kpop_filters.get('groups'):
                filter_info.append(f"그룹: {kpop_filters['groups']}")
            if kpop_filters.get('members'):
                filter_info.append(f"멤버: {kpop_filters['members']}")
            if kpop_filters.get('agencies'):
                filter_info.append(f"소속사: {kpop_filters['agencies']}")
            if kpop_filters.get('fandoms'):
                filter_info.append(f"팬덤: {kpop_filters['fandoms']}")
            if kpop_filters.get('concepts'):
                filter_info.append(f"컨셉: {kpop_filters['concepts']}")
            if kpop_filters.get('debut_year'):
                filter_info.append(f"데뷔: {kpop_filters['debut_year']}년")
            if kpop_filters.get('group_type'):
                filter_info.append(f"타입: {kpop_filters['group_type']}")
            if filter_info:
                print(f"   K-pop 필터: {', '.join(filter_info)}")

        return {
            "difficulty_level": analysis["difficulty"],
            "query_analysis": analysis,
        }

    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드 (리소스 충분한지만 확인)"""
        print("\n✅ [Agent] 품질 검증")

        query_analysis = state.get("query_analysis", {})
        needs_kpop = query_analysis.get("needs_kpop", False)

        result = self.quality_agent.check(
            vocab_count=len(state.get("vocabulary_docs", [])),
            grammar_count=len(state.get("grammar_docs", [])),
            kpop_db_count=len(state.get("kpop_docs", [])),
            needs_kpop=needs_kpop,
        )

        print(f"   어휘: {result['vocab_count']}개")
        print(f"   문법: {result['grammar_count']}개")
        print(f"   K-pop: {result['kpop_db_count']}개")
        print(f"   상태: {result['message']}")

        return {"quality_check": result}

    def _process_kpop_docs_enhanced(
        self,
        kpop_docs,
    ):
        """
        K-pop 문서에서 메타데이터 추출
        주의: 필터링은 이미 retrieve_kpop_routed에서 벡터 기반 검색과 메타데이터 필터링으로 완료됨
        이 함수는 단순히 메타데이터를 추출하여 구조화된 형태로 변환하는 역할만 수행
        """
        kpop_metadata: List[Dict[str, Any]] = []

        if not kpop_docs:
            return kpop_metadata

        # 이미 필터링된 문서들에서 메타데이터만 추출 (최대 5개)
        for doc in kpop_docs[:5]:
            meta = doc.metadata
            group = meta.get("group", "")
            if not group:
                continue

            # 전체 그룹 정보를 하나의 메타데이터로 저장 (모든 정보 포함)
            full_meta = {
                "group": group,
                "agency": meta.get("agency", ""),
                "fandom": meta.get("fandom", ""),
                "concepts": meta.get("concepts", []),
                "members": [
                    {
                        "name": m.get("name", ""),
                        "role": m.get("role", ""),
                    }
                    for m in meta.get("members", [])  # 모든 멤버 포함
                ],
            }
            kpop_metadata.append(full_meta)

        return kpop_metadata

    def generate_question_directly(self, state: GraphState) -> GraphState:
        """
        문장 생성 없이 추출된 정보로 바로 문제 생성용 payload 구성
        - 단어 5개 추출 (난이도에 맞는 것) - 자연스러운 문제 생성을 위해 증가
        - 문법 1개 추출 (난이도에 맞는 것)
        - K-pop 정보 최대 5개 추출 (쿼리에 K-pop 관련이 있을 때만) - 더 풍부한 컨텍스트 제공
        """
        print("\n🎯 [Agent] 정보 추출 및 문제 생성용 payload 구성")

        # 1) 단어 추출 (난이도에 맞는 것, 최대 5개)
        words_info = extract_words_from_docs(state.get("vocabulary_docs", []))
        vocab_list = [word for word, _ in words_info][:5]
        vocab_details = []
        for word, wordclass in words_info[:5]:
            vocab_details.append({
                "word": word,
                "wordclass": wordclass
            })
        
        print(f"   ✅ 단어 추출: {len(vocab_list)}개 - {vocab_list}")

        # 2) 문법 추출 (난이도에 맞는 것, 1개)
        grammar_info = extract_grammar_with_grade(state.get("grammar_docs", []))
        target_grammar = grammar_info[0]["grammar"] if grammar_info else "기본 문법"
        target_grade = grammar_info[0]["grade"] if grammar_info else 1
        
        print(f"   ✅ 문법 추출: {target_grammar} (Grade {target_grade})")

        # 3) K-pop 정보 추출 (최대 5개) - 동적 필터링
        query_analysis = state.get("query_analysis", {})
        needs_kpop = query_analysis.get("needs_kpop", False)
        kpop_metadata = []
        
        if needs_kpop and state.get("kpop_docs"):
            # 필터링은 이미 retrieve_kpop_routed에서 완료되었으므로 메타데이터만 추출
            kpop_metadata = self._process_kpop_docs_enhanced(
                state.get("kpop_docs", []),
            )
            kpop_metadata = kpop_metadata[:5]  # 최대 5개로 증가
            
            # 실제 추출된 정보 확인
            extracted_groups = set([m.get("group", "") for m in kpop_metadata])
            if extracted_groups:
                print(f"   ✅ K-pop 정보 추출: {len(kpop_metadata)}개")
                print(f"   📋 추출된 그룹: {list(extracted_groups)}")
            else:
                print(f"   ✅ K-pop 정보 추출: {len(kpop_metadata)}개")
        else:
            print(f"   ⏭️  K-pop 정보 없음 (쿼리에 K-pop 관련 키워드 없음)")

        # 난이도 매핑 (TOPIK 레벨 → 시스템 난이도)
        difficulty = state["difficulty_level"]
        level_mapping = {
            "basic": "grade1-2",
            "intermediate": "grade3-4",
            "advanced": "grade5-6"
        }
        level = level_mapping.get(difficulty, f"grade{target_grade}")

        # 문제 생성용 넘겨줄 정보 구성
        question_payload = {
            "level": level,
            "target_grammar": target_grammar,
            "vocabulary": vocab_list,
            "vocabulary_details": vocab_details,
            "difficulty": difficulty,
            "grade": target_grade,
        }

        # K-pop 정보가 있으면 추가
        if kpop_metadata:
            question_payload["kpop_references"] = kpop_metadata



        print(f"   ✅ Payload 구성 완료")
        print(f"      - Level: {level}")
        print(f"      - Grammar: {target_grammar}")
        print(f"      - Vocabulary: {len(vocab_list)}개")
        print(f"      - K-pop: {len(kpop_metadata)}개")

        return {
            "question_payload": question_payload,
            "target_grade": target_grade,
        }

    def format_output_agentic(self, state: GraphState) -> GraphState:
        """Agentic RAG 출력 포맷팅"""
        print("\n📄 [Agent] 최종 출력")

        output = "=" * 80 + "\n"
        output += "🎓 한국어 학습 문제 생성 (Agentic RAG)\n"
        output += "=" * 80 + "\n\n"

        # 추출된 정보 출력
        if "question_payload" in state:
            question_payload = state.get("question_payload", {})
            output += "【추출된 학습 정보】\n"
            output += f"   목표 문법: {question_payload.get('target_grammar', 'N/A')}\n"
            vocab_list = question_payload.get("vocabulary", [])
            if vocab_list:
                output += f"   학습 단어: {', '.join(vocab_list)}\n"
            kpop_refs = question_payload.get("kpop_references", [])
            if kpop_refs:
                output += f"   K-pop 참조: {len(kpop_refs)}개\n"

        output += "\n" + "=" * 80 + "\n"

        return {"final_output": output}