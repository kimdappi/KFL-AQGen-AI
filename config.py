# =====================================
# config.py - 설정 파일
# =====================================
"""
프로젝트 설정 파일
"""

# 파일 경로 설정
TOPIK_PATHS = {
    'basic': [r'data\words\TOPIK1.csv', r'data\words\TOPIK2.csv'],
    'intermediate': [r'data\words\TOPIK3.csv', r'data\words\TOPIK4.csv'],
    'advanced': [r'data\words\TOPIK5.csv', r'data\words\TOPIK6.csv']
}

GRAMMAR_PATHS = {
    'basic': r'data\grammar\grammar_list_A.json',
    'intermediate': r'data\grammar\grammar_list_B.json',
    'advanced': r'data\grammar\grammar_list_C.json'
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

# Kpop 데이터 설정
# 변경됨
KPOP_JSON_PATH = r'data\kpop\kpop_db.json'

# 생성문장 위치(nodes.py에서 사용)
SENTENCE_SAVE_DIR = "output/sentence"
