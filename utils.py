# =====================================
# utils.py (Updated) - grade 추출 함수 추가
# =====================================

from typing import List, Dict
from langchain.schema import Document


def format_docs(docs: List[Document]) -> str:
    """문서를 포맷팅하여 문자열로 변환"""
    formatted = []
    for doc in docs:
        formatted.append(doc.page_content)
    return "\n\n---\n\n".join(formatted)


def detect_difficulty_from_text(text: str) -> str:
    """텍스트에서 난이도 감지"""
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


def extract_grammars_from_docs(docs: List[Document], limit: int = 10) -> List[str]:
    """문서에서 문법 추출 (기존 함수 유지)"""
    grammars = []
    for doc in docs[:limit]:
        grammar = doc.metadata.get('grammar', '')
        if grammar:
            grammars.append(grammar)
    return grammars


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