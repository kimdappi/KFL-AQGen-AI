# =====================================
# utils.py (Updated) - grade 추출 함수 추가
# =====================================

from typing import List, Dict, Optional
from langchain.schema import Document

# K-pop 그룹 타입 판단을 위한 하드코딩 (DB에 group_type 필드가 없으므로 필요)
GIRL_GROUPS = {'BLACKPINK', 'TWICE', 'RED VELVET', 'IVE', 'NEWJEANS', 'LE SSERAFIM', 'AESPA'}
BOY_GROUPS = {'BTS', 'EXO', 'SEVENTEEN', 'STRAY KIDS', 'NCT', 'SHINEE', 'SUPER JUNIOR'}


def get_group_type(group_name: str) -> Optional[str]:
    """
    그룹명으로 그룹 타입 판단 (girl_group, boy_group, None)
    
    Args:
        group_name: 그룹명 (대소문자 무관)
        
    Returns:
        'girl_group', 'boy_group', 또는 None
    """
    group_upper = (group_name or '').upper()
    if group_upper in GIRL_GROUPS:
        return 'girl_group'
    elif group_upper in BOY_GROUPS:
        return 'boy_group'
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