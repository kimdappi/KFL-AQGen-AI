# =====================================
# tools.py (Pydantic V2 최종 버전)
# =====================================
"""
Pydantic V2 스키마와 에이전트가 사용할 순수 파이썬 함수 로직을 정의합니다.
"""
from typing import List, Optional
from langchain_core.documents import Document
# ==========================================================
#        ↓↓↓ Pydantic V1 대신 V2를 임포트합니다 ↓↓↓
# ==========================================================
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict  # pydantic v2


# 외부 클래스들은 타입 힌팅 용도로만 사용됩니다.
from vocabulary_retriever import TOPIKVocabularyRetriever
from grammar_retriever import GrammarRetriever
from langchain_openai import ChatOpenAI
from utils import extract_words_from_docs, extract_grammar_with_grade, detect_difficulty_from_text

# --- Pydantic V2 입력 스키마 정의 ---

class DifficultyInput(BaseModel):
    user_query: str = Field(description="난이도를 감지할 사용자의 원본 요청 텍스트")
    model_config = ConfigDict(extra="forbid")

class VocabularyInput(BaseModel):
    query: str = Field(description="검색의 중심이 되는 일반적인 사용자 질의. 예: '식당에서 쓸만한 표현'")
    level: str = Field(description="학습자의 난이도 수준. 'basic', 'intermediate', 'advanced' 중 하나.")
    model_config = ConfigDict(extra="forbid")

class GrammarInput(BaseModel):
    query: str = Field(description="검색의 중심이 되는 일반적인 사용자 질의. 예: '후회와 관련된 문법'")
    level: str = Field(description="학습자의 난이도 수준. 'basic', 'intermediate', 'advanced' 중 하나.")
    keyword: Optional[str] = Field(default=None, description="사용자가 명시적으로 사용하길 원하는 특정 문법 패턴. 예: '-(으)면서'")
    model_config = ConfigDict(extra="forbid")

class GeneratorInput(BaseModel):
    difficulty_level: str = Field(description="생성할 문장의 목표 난이도. 'basic', 'intermediate', 'advanced' 중 하나.")
    vocabulary_docs: List[Document] = Field(description="retrieve_vocabulary 툴을 통해 검색된 단어 문서 리스트")
    grammar_docs: List[Document] = Field(description="retrieve_grammar 툴을 통해 검색된 문법 문서 리스트")

    # Pydantic V2에서는 Config 클래스 대신 model_config를 사용합니다.
    model_config = ConfigDict(extra="forbid")

# --- 순수 함수 로직 정의 ---
# (함수 내용은 변경할 필요 없습니다)

def detect_difficulty_func(user_query: str) -> str:
    return detect_difficulty_from_text(user_query)

def retrieve_vocabulary_func(retriever: TOPIKVocabularyRetriever, query: str, level: str) -> List[Document]:
    return retriever.invoke(query, level)

def retrieve_grammar_func(retriever: GrammarRetriever, query: str, level: str, keyword: Optional[str] = None) -> List[Document]:
    return retriever.invoke(query, level, keyword=keyword)

def korean_sentence_generator_func(llm: ChatOpenAI, difficulty_level: str, vocabulary_docs: List[Document], grammar_docs: List[Document]) -> str:
    words_info = extract_words_from_docs(vocabulary_docs)
    grammar_info = extract_grammar_with_grade(grammar_docs)
    words_formatted = [f"{word}({wordclass})" for word, wordclass in words_info[:5]]
    
    if grammar_info:
        target_grammar = grammar_info[0]['grammar']
        target_grade = grammar_info[0]['grade']
    else:
        target_grammar = "기본 문법"
        target_grade = 1

    prompt = f"""
# 페르소나
너는 20년 경력의 베테랑 한국어 교육 교수다. 특히 일본인 학습자를 가르치는 데 독보적인 전문가이며, 한일 대조언어학과 오류 분석에 대한 깊은 지식을 갖추고 있다. 너의 역할은 단순한 문장 생성이 아니라, 학습자의 잠재적 오류를 예측하고 이를 근본적으로 예방할 수 있는 최적의 교육 예문을 직접 만드는 것이다.

# 컨텍스트
- 학습자 레벨: {difficulty_level} (Grade {target_grade})
- 학습 목표 문법: {target_grammar}
- 활용 가능 어휘 (품사): {', '.join(words_formatted)}
- 핵심 지식 베이스: [일본인 학습자 핵심 오류 유형 총정리 (생략)]

# 목표
주어진 '학습 목표 문법'과 '활용 가능 어휘'를 사용하여, 아래 3가지 역할에 맞는 가장 교육적인 예문 3개를 **직접 생성**하고, 각 문장의 교육적 의도를 명확하게 서술한다.

# 지침 (단계별 사고 과정)
1.  **3가지 역할 구상**: '기본 모델', '확장 모델', '오류 예방 모델'이라는 3가지 교육적 목표를 명확히 이해한다.
2.  **문장 생성**: 각 역할에 맞춰 문장을 1개씩 생성한다.
    - 모든 문장은 '학습 목표 문법'을 반드시 포함해야 한다.
    - '활용 가능 어휘' 목록의 단어를 3개 이상 자연스럽게 사용한다.
    - 문장의 전체적인 난이도는 '학습자 레벨'에 맞춘다.
3.  **역할별 문장 설계**: (생략)
4.  **교육적 의도 작성**: (생략)

# 출력 형식 (반드시 아래 JSON 형식만 출력할 것)
{{
  "sentences": [
    {{
      "sentence": "생성된 첫 번째 문장 (기본 모델)",
      "role": "기본 모델",
      "pedagogical_rationale": "..."
    }},
    {{
      "sentence": "생성된 두 번째 문장 (확장 모델)",
      "role": "확장 모델",
      "pedagogical_rationale": "..."
    }},
    {{
      "sentence": "생성된 세 번째 문장 (오류 예방 모델)",
      "role": "오류 예방 모델",
      "pedagogical_rationale": "..."
    }}
  ]
}}
"""
    # 프롬프트의 일부 내용을 생략하여 간결하게 표시했습니다. 실제 코드에는 전체 프롬프트를 넣어주세요.
    
    response = llm.predict(prompt)
    return response