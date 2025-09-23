# =====================================
# schemas.py - 데이터 스키마
# =====================================
"""
데이터 스키마 정의
"""
from typing import Annotated, TypedDict, List
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