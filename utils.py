# =====================================
# utils.py (Updated) - grade 추출 함수 추가
# =====================================

from typing import List, Dict, Optional
from langchain.schema import Document

def get_group_type(group_name: str, kpop_retriever=None) -> Optional[str]:
    """
    그룹명으로 그룹 타입 판단 (girl_group, boy_group, None)
    K-pop 데이터에서 동적으로 추출 (하드코딩 제거)
    
    Args:
        group_name: 그룹명 (대소문자 무관)
        kpop_retriever: K-pop 리트리버 (선택적, 없으면 None 반환)
        
    Returns:
        'girl_group', 'boy_group', 또는 None
    """
    if not kpop_retriever or not hasattr(kpop_retriever, 'kpop_data'):
        return None
    
    group_upper = (group_name or '').upper()
    
    # K-pop 데이터에서 해당 그룹 찾기
    for doc in kpop_retriever.kpop_data:
        doc_group = doc.metadata.get('group', '')
        if doc_group and doc_group.upper() == group_upper:
            # 멤버 정보에서 role로 판단 (보통 걸그룹은 여성, 보이그룹은 남성)
            # 하지만 더 정확한 방법은 DB에 group_type 필드 추가
            # 현재는 임시로 None 반환 (필터링은 다른 방식으로 처리)
            return None
    
    return None


def detect_difficulty_from_text(text: str) -> str:
    """
    텍스트에서 난이도 감지 (하드코딩 방식 - 효율적)
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        'basic', 'intermediate', 또는 'advanced'
    """
    text_lower = text.lower()
    
    if 'basic' in text_lower or '초급' in text_lower or '기초' in text_lower:
        return 'basic'
    elif 'intermediate' in text_lower or '중급' in text_lower:
        return 'intermediate'
    elif 'advanced' in text_lower or '고급' in text_lower or '상급' in text_lower:
        return 'advanced'
    else:
        return 'basic'  # 기본값


def extract_words_from_docs(docs: List[Document], limit: int = 10) -> List[tuple]:
    """문서에서 단어와 품사 추출"""
    words_info = []
    for doc in docs[:limit]:
        word = doc.metadata.get('word', '')
        wordclass = doc.metadata.get('wordclass', '')
        if word:
            words_info.append((word, wordclass))
    return words_info


def extract_grammar_with_grade(docs: List[Document], limit: int = 10) -> List[Dict]:
    """
    문서에서 문법과 grade 함께 추출
    Returns: [{'grammar': '-(으)면서', 'grade': 2}, ...]
    """
    grammar_info = []
    for doc in docs[:limit]:
        grammar = doc.metadata.get('grammar', '')
        grade = doc.metadata.get('grade', 1)  # 기본값 1
        if grammar:
            grammar_info.append({
                'grammar': grammar,
                'grade': grade
            })
    return grammar_info