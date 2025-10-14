# =====================================
# nodes.py (Updated) - grade를 level로 사용
# =====================================
"""
LangGraph 노드 정의 (문장 저장 기능 및 grade 사용)
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from schema import GraphState
from utils import (
    detect_difficulty_from_text,
    extract_words_from_docs,
    extract_grammars_from_docs,
    extract_grammar_with_grade,  # 새로운 유틸 함수
    format_docs
)
from config import LLM_CONFIG

INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1F]'

def sanitize_filename(name: str, replacement: str = "_") -> str:
# 금지문자 -> _
    safe = re.sub(INVALID_CHARS, replacement, name)
    # 마지막의 점/공백 제거
    safe = safe.strip().strip(".")
    # Windows 예약어 회피
    RESERVED = {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}
    if safe.upper() in RESERVED:
        safe = f"_{safe}"
        # 너무 긴 파일명 방지 (경로 전체 길이 여유 주기)
    return safe[:120] if len(safe) > 120 else safe

class KoreanLearningNodes:
    """한국어 학습 노드 클래스"""
    
    def __init__(self, vocabulary_retriever, grammar_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini", #범준이 api 대신 임시
            temperature=LLM_CONFIG.get('temperature', 0.7),
            max_tokens=LLM_CONFIG.get('max_tokens', 1000)
        )
        
        # sentence 폴더 생성
        self.output_dir = "sentence"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def detect_difficulty(self, state: GraphState) -> GraphState:
        """입력 텍스트에서 난이도 감지"""
        difficulty = detect_difficulty_from_text(state['input_text'])
        return {"difficulty_level": difficulty}
    
    def retrieve_vocabulary(self, state: GraphState) -> GraphState:
        """단어 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        
        vocab_docs = self.vocabulary_retriever.invoke(query, level)
        return {"vocabulary_docs": vocab_docs}
    
    def retrieve_grammar(self, state: GraphState) -> GraphState:
        """문법 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        
        grammar_docs = self.grammar_retriever.invoke(query, level)
        return {"grammar_docs": grammar_docs}
    
    def generate_sentences(self, state: GraphState) -> GraphState:
        """검색된 단어와 문법을 활용한 문장 생성"""
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        
        # 문법과 grade 정보 함께 추출
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 단어와 품사 정보 포맷팅
        words_formatted = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
        
        # 첫 번째 문법과 그 grade를 메인으로 사용 (임시) -> 단점 : 계속 1,3,5 만 나옴
        if grammar_info:
            target_grammar = grammar_info[0]['grammar']
            target_grade = grammar_info[0]['grade']  # 실제 grade 값 사용
        else:
            target_grammar = "기본 문법"
            target_grade = 1
        
        
        prompt = f"""
# 페르소나
너는 20년 경력의 베테랑 한국어 교육 교수다. 특히 일본인 학습자를 가르치는 데 독보적인 전문가이며, 한일 대조언어학과 오류 분석에 대한 깊은 지식을 갖추고 있다. 너의 역할은 단순한 문장 생성이 아니라, 학습자의 잠재적 오류를 예측하고 이를 근본적으로 예방할 수 있는 최적의 교육 예문을 직접 만드는 것이다.

# 컨텍스트
- 학습자 레벨: {state['difficulty_level']} (Grade {target_grade})
- 학습 목표 문법: {target_grammar}
- 활용 가능 어휘 (품사): {', '.join(words_formatted)}
- 핵심 지식 베이스: 아래의 `[일본인 학습자 핵심 오류 유형 총정리]`

    [일본인 학습자 핵심 오류 유형 총정리 (논문 기반)]

    1. 조사 오류 (가장 우선적으로 고려할 항목)
        1-1. 격조사 선택 오류: `좋아하다`, `잘하다` 등 특정 서술어 앞에서 목적격 `을/를` 대신 주격 `이/가`를 쓰는 오류 방지. 정적인 장소(`에`)와 동적인 장소(`에서`) 구분. 대상이 무정물(`에`)인지 유정물(`에게/한테`)인지 구분.
        1-2. 보조사 선택 오류: 신정보 제시에 `은/는` 대신 `이/가`를 사용하도록 유도. `은/는`의 '대조/화제' 기능 명시.
        1-3. 관형격 조사 `의`의 남용: 일본어 번역투 방지.

    2. 용언 활용 및 문법 범주 오류
        2-1. 품사 구분: 형용사를 동사처럼 활용하는(`-는다`) 오류 방지.
        2-2. 불규칙 활용: `덥다`, `춥다`, `듣다` 등 불규칙 용언의 정확한 활용 제시.

    3. 시제(Tense) 및 상(Aspect) 오류
        3-1. 상태 vs. 진행: `-아/어 있다`(상태)와 `-고 있다`(진행)의 구분. (예: `의자에 앉아 있다`)
        3-2. 회상 `던`: 단순 과거 사실에 불필요한 `던` 사용 방지.
        3-3. `-겠-`: '의지'나 '추측'의 뉘앙스를 명확히 전달.

    4. 연결어미 및 문장 구조 오류
        4-1. 원인/이유: `-어서/아서` 뒤에 명령문/청유문이 오지 않도록 예문 구성.
        4-2. 일본어식 표현: `~에 대해서` 보다 자연스러운 `~에 대해` 등으로 표현.

# 목표
주어진 '학습 목표 문법'과 '활용 가능 어휘'를 사용하여, 아래 3가지 역할에 맞는 가장 교육적인 예문 3개를 **직접 생성**하고, 각 문장의 교육적 의도를 명확하게 서술한다.

# 지침 (단계별 사고 과정)
1.  **3가지 역할 구상**: '기본 모델', '확장 모델', '오류 예방 모델'이라는 3가지 교육적 목표를 명확히 이해한다.
2.  **문장 생성**: 각 역할에 맞춰 문장을 1개씩 생성한다.
    - 모든 문장은 '학습 목표 문법'을 반드시 포함해야 한다.
    - '활용 가능 어휘' 목록의 단어를 3개 이상 자연스럽게 사용한다.
    - 문장의 전체적인 난이도는 '학습자 레벨'에 맞춘다.
3.  **역할별 문장 설계**:
    - **역할 1 (기본 모델)**: 목표 문법의 핵심 의미와 형태를 가장 단순하고 명료하게 보여주는 문장을 생성한다.
    - **역할 2 (확장 모델)**: 목표 문법을 다른 문법 요소와 결합하거나, 실제 대화에서 쓰일 법한 실용적인 문장을 생성한다.
    - **역할 3 (오류 예방 모델)**: `[핵심 오류 유형 총정리]`의 항목 중 하나를 예방하는 것을 명시적인 목표로 삼는 문장을 생성한다. 예를 들어, 목표 문법과 함께 '1-1. 서술어 제약'에 해당하는 '음악을 좋아하다' 같은 표현을 의도적으로 포함시킨다.
4.  **교육적 의도 작성**: 생성한 각 문장에 대해 `pedagogical_rationale`을 작성한다. 특히 '오류 예방 모델'의 경우, 총정리 목록의 어떤 항목(예: 1-1. 격조사 선택 오류)을 예방하기 위한 것인지 반드시 명시한다.

# 출력 형식 (반드시 아래 JSON 형식만 출력할 것)
{{
  "sentences": [
    {{
      "sentence": "생성된 첫 번째 문장 (기본 모델)",
      "role": "기본 모델",
      "pedagogical_rationale": "목표 문법의 핵심 의미를 가장 직관적으로 보여줍니다. 학습자가 문법의 기본 형태와 의미를 처음 접할 때 가장 효과적입니다."
    }},
    {{
      "sentence": "생성된 두 번째 문장 (확장 모델)",
      "role": "확장 모델",
      "pedagogical_rationale": "실제 대화처럼 다른 문법 요소와 결합된 예시입니다. 이를 통해 문법의 활용 범위를 넓히고 응용력을 기를 수 있습니다."
    }},
    {{
      "sentence": "생성된 세 번째 문장 (오류 예방 모델)",
      "role": "오류 예방 모델",
      "pedagogical_rationale": "이 문장은 일본인 학습자가 자주 범하는 '1-1. 격조사 선택 오류'를 예방하기 위해 설계되었습니다. 'OO을/를 좋아하다' 형태를 명확히 제시하여 올바른 사용법을 각인시킵니다."
    }}
  ]
}}
"""
        
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # JSON 형식으로 저장할 데이터 생성 (grade를 level로 사용)
        save_data = {
            "level": target_grade,  # grade 값을 level로 사용 1-6
            "target_grammar": target_grammar,
            "critique_summary": [{"sentence": s} for s in sentences]
        }
        
        # 메시지 히스토리 추가
        messages = [
            ("user", state['input_text']),
            ("assistant", "\n".join(sentences))
        ]
        
        return {
            "generated_sentences": sentences,
            "messages": messages,
            "sentence_data": save_data,
            "target_grade": target_grade  # state에 grade 정보 추가
        }
    
    def format_output(self, state: GraphState) -> GraphState:
        """최종 출력 포맷팅 및 JSON 저장"""
        output = f"=== 한국어 학습 문제 생성 결과 ===\n"
        output += f"난이도: {state['difficulty_level']}\n"
        
        # target_grade가 있으면 표시
        if 'target_grade' in state:
            output += f"문법 Grade: {state['target_grade']}\n"
        
        output += "\n선택된 단어 (상위 10개):\n"
        for i, doc in enumerate(state['vocabulary_docs'][:10], 1):
            word = doc.metadata.get('word', 'N/A')
            wordclass = doc.metadata.get('wordclass', 'N/A')
            guide = doc.metadata.get('guide', 'N/A')
            topik_level = doc.metadata.get('topik_level', 'N/A')
            output += f"{i}. {word} ({wordclass}) - {guide[:30]}... [TOPIK{topik_level}]\n"
        
        output += "\n선택된 문법 (상위 10개, grade 낮은 순):\n"
        for i, doc in enumerate(state['grammar_docs'][:10], 1):
            grammar = doc.metadata.get('grammar', 'N/A')
            grade = doc.metadata.get('grade', 'N/A')
            output += f"{i}. {grammar} (Grade: {grade})\n"
        
        output += "\n생성된 예문:\n"
        for i, sentence in enumerate(state['generated_sentences'], 1):
            output += f"{i}. {sentence}\n"
        
        # JSON 파일로 저장 (sentence_data가 있을 때만)
        if 'sentence_data' in state and state['sentence_data']:
            saved_file = self._save_to_json(state['sentence_data'])
            output += f"\n 예문이 저장되었습니다: {saved_file}\n"
        
        return {"final_output": output}
    
    def _save_to_json(self, sentence_data: dict) -> str:
        out_dir = Path("sentence")
        out_dir.mkdir(parents=True, exist_ok=True)

        level = sentence_data.get("level", "grade1")
        title = sentence_data.get("title", "untitled")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base = f"sentences_{level}_{title}_{timestamp}"
        safe_base = sanitize_filename(base)
        filepath = out_dir / f"{safe_base}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            import json
            json.dump(sentence_data, f, ensure_ascii=False, indent=2)

        return str(filepath)