# =====================================
# config.py - 설정 파일
# =====================================
"""
프로젝트 설정 파일
"""

# 파일 경로 설정
TOPIK_PATHS = {
    'basic': ['data/words/TOPIK1.csv', 'data/words/TOPIK2.csv'],
    'intermediate': ['data/words/TOPIK3.csv', 'data/words/TOPIK4.csv'],
    'advanced': ['data/words/TOPIK5.csv', 'data/words/TOPIK6.csv']
}

GRAMMAR_PATHS = {
    'basic': 'data/grammar/grammar_list_A.json',
    'intermediate': 'data/grammar/grammar_list_B.json',
    'advanced': 'data/grammar/grammar_list_C.json'
}

# Retriever 설정
RETRIEVER_CONFIG = {
    'top_k': 10,
    'ensemble_weights': [0.5, 0.5],
    'vector_search_type': 'similarity',
}

# LLM 설정
LLM_CONFIG = {
    'temperature': 0.7,
    'max_tokens': 1000,
}