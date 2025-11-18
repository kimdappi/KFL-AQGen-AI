"""
프로젝트 설정 파일
"""
MODEL_NAME = "gpt-5"

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

# LLM 설정
LLM_CONFIG = {
    'temperature': 1.0,
    'max_completion_tokens': 1000,
}

# Kpop 데이터 설정
KPOP_JSON_PATH = r'data\kpop\kpop_db.json'