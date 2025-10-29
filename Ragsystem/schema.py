"""
KFL-AQGen-AI 데이터 스키마 정의
LangGraph 상태 구조 정의
수정 완료
"""

from typing import Annotated, TypedDict, List, Optional, Dict
from langchain.schema import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    LangGraph 상태 정의
    
    Agentic RAG 파이프라인을 통해 흐르는 모든 데이터 포함
    """
    
    # 핵심 입력/출력
    input_text: Annotated[str, "사용자 입력 쿼리"]
    difficulty_level: Annotated[str, "난이도 수준 (basic/intermediate/advanced)"]
    final_output: Annotated[str, "최종 포맷팅된 출력"]
    
    # 검색 결과
    vocabulary_docs: Annotated[List[Document], "검색된 어휘 문서"]
    grammar_docs: Annotated[List[Document], "검색된 문법 문서"]
    kpop_docs: Annotated[List[Document], "DB에서 검색된 K-pop 문장"]
    
    # 생성 결과
    generated_sentences: Annotated[List[str], "생성된 예문"]
    sentence_data: Optional[Dict]  # JSON 파일 저장용 데이터
    target_grade: Optional[int]  # 타겟 문법 등급 (1-6)
    
    # 메시지 히스토리
    messages: Annotated[list, add_messages]
    
    # Agentic RAG 필드
    query_analysis: Optional[Dict]  # 쿼리 분석 결과
    quality_check: Optional[Dict]  # 품질 체크 결과
    routing_decision: Optional[object]  # 라우팅 결정 객체
    search_params: Optional[Dict]  # 검색 파라미터 (vocabulary, grammar, kpop)
    search_strategies: Optional[List[Dict]]  # 검색 전략 (딕셔너리 리스트)
    rerank_count: Optional[int]  # 재검색 시도 카운터
    rerank_decision: Optional[object]  # 재검색 결정 객체