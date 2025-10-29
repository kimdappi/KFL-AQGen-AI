# =====================================
# schema.py (Updated) - target_grade 필드 추가
# =====================================
"""
데이터 스키마 정의 (업데이트)
"""
from typing import Annotated, TypedDict, List, Optional, Dict
from langchain.schema import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph State 정의"""
    input_text: Annotated[str, "사용자 입력 텍스트"]
    difficulty_level: Annotated[str, "난이도 (basic/intermediate/advanced)"]
    vocabulary_docs: Annotated[List[Document], "검색된 단어 문서"]
    grammar_docs: Annotated[List[Document], "검색된 문법 문서"]
    generated_sentences: Annotated[List[str], "생성된 예문"]
    final_output: Annotated[str, "최종 출력"]
    messages: Annotated[list, add_messages]  # 메시지 히스토리
    sentence_data: Optional[Dict]  # JSON 저장용 데이터
    target_grade: Optional[int]  # 타겟 문법의 grade