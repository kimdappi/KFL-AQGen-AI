# =====================================
# schema.py (Updated) - target_grade 필드 추가
# =====================================

"""
데이터 스키마 정의 (에이전트 상태 추가)
"""
from typing import TypedDict, List, Optional, Dict, Annotated
from langgraph.graph.message import add_messages



class AgentState(TypedDict):
    """Agentic RAG 워크플로우의 상태를 정의합니다."""
    
    # 사용자의 초기 입력
    input_text: str
    
    # 에이전트가 수립한 단계별 계획
    plan: List[Dict]
    
    # 실행된 도구의 결과물들을 저장하는 리스트
    tool_outputs: List[Dict]
    
    # 현재 실행해야 할 계획의 순번 (인덱스)
    current_step: int
    
    # 최종적으로 생성된 문장 (JSON 문자열)
    generated_sentences_json: str
    
    # 최종 사용자 출력
    final_output: str
    
    # 메시지 히스토리 (채팅 기능용)
    messages: Annotated[list, add_messages]