# test_maker.py

import os
import json
import textwrap
import random
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()
# ==============================================================================
# 0. 문제 유형별 프롬프트 템플릿 딕셔너리 (TMPLS)
# 문제는 총 6유형이다.
# (내용은 이전과 동일)
# ==============================================================================
FILL_IN_BLANK_TMPL = """\
[ROLE] 너는 '외국어로서의 한국어' 교재 편집자다. 반드시 JSON만 출력한다.
[GOAL] [INPUT_SENTENCES]에 주어진 문장 중 하나를 활용하여 **주관식 빈칸 채우기** 문제 1개를 만든다.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `instruction`: "<보기>와 같이 괄호 안의 단어를 사용하여 문장을 완성하십시오." 와 같은 명확한 지시문을 작성한다.
- `stem_with_blank`: [INPUT_SENTENCES]의 문장 중 하나를 선택하여, 타깃 문법 부분을 빈칸( ___ )으로 바꾼다.
- `hint`: 빈칸에 들어갈 동사/형용사의 기본형을 힌트로 제시한다.
- `example`: 문제에 사용하지 않은 다른 입력 문장 하나를 골라 동일한 형식의 보기 문항을 생성한다.
- **절대로 `options` 필드를 만들지 않는다.**
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{ "schema_id": "{schema_id}", "format": "fill_in_blank", "input": {{"instruction": "<보기>와 같이 괄호 안의 단어를 사용하여 문장을 완성하십시오.", "example": {{"stem": "나는 점심을 ___ TV를 봤어요. (먹다)", "answer": "먹으면서"}}, "stem_with_blank": "저는 음악을 ___ 공부합니다. ({{hint}})", "hint": "듣다"}}, "answer": {{"completed_sentence": "저는 음악을 들으면서 공부합니다."}}, "rationale": "두 가지 행동을 동시에 함을 나타내는 '-으면서'가 자연스럽습니다. '듣다'는 불규칙 동사이므로 '들으면서'로 활용됩니다."}}"""

MATCH_AND_CONNECT_TMPL = """\
[ROLE] 너는 '외국어로서의 한국어' 교재 편집자다. 반드시 JSON만 출력한다.
[GOAL] 입력된 문장들을 분해하고 재조합하여, 문장 연결하기 문제를 생성한다. 반드시 입력된 문장들만 사용한다.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `instruction`: "다음 문장을 연결하여 <보기>와 같이 하나의 문장을 만드십시오." 와 같은 지시문을 작성한다.
- `clause_set_A`, `clause_set_B`: 입력된 문장들에서 3~4개를 골라 각각 앞부분과 뒷부분으로 분해하고, 순서를 섞어서 배치한다.
- `example`: **문제에 사용되지 않은 다른 입력 문장 하나를 골라** 분해하여 <보기>를 만든다.
- `answer`: `clause_set_A`의 각 항목에 맞는 `clause_set_B`의 항목을 연결하여 만든 완성 문장들의 배열.
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{ "schema_id": "{schema_id}", "format": "match_and_connect", "input": {{"instruction": "다음 문장을 연결하여 <보기>와 같이 하나의 문장을 만드십시오.", "example": {{ "clause_A": "아버지는 운동을 하다", "clause_B": "건강을 챙기다", "connected": "아버지는 운동을 하면서 건강을 챙깁니다." }}, "clause_set_A": ["저는 음악을 듣다", "그녀는 친구와 이야기를 나누다", "우리는 여행 계획을 세우다"], "clause_set_B": ["즐거운 시간을 보내다", "공부하다", "웃고 있다"]}}, "answer": {{ "connected_sentences": ["저는 음악을 들으면서 공부합니다.", "그녀는 친구와 이야기를 나누면서 웃고 있습니다.", "우리는 여행 계획을 세우면서 즐거운 시간을 보냈습니다."] }}, "rationale": "조건을 나타내는 '-으면'을 사용하여 앞선 절과 뒷선 절을 자연스럽게 연결할 수 있습니다."}}"""


SENTENCE_CONNECTION_TMPL = """\
[ROLE] 한국어 문장 연결 문제 출제자. JSON만 출력.

[GOAL]
- 입력된 문장 하나를 두 개의 절로 분해하여 문장 연결 문제를 생성한다.
{kpop_info}[INPUT_SENTENCES]
- 아래 정확히 3개의 예문 중에서만 사용한다. 
{sentences_bullets}

[INSTRUCTIONS]
- `instruction`: "다음 두 문장을 <보기>와 같이 목표 문법을 사용하여 한 문장으로 만드십시오." 와 같이 지시문을 작성한다.
- `input`: [INPUT_SENTENCES]의 문장 중 하나를 선택하여 두 개의 독립된 문장(`clause_A`, `clause_B`)으로 분해한다.
- `answer`: 분해되기 전의 원본 문장을 `connected_sentence` 값으로 설정한다.
- `example`: **문제에 사용되지 않은 다른 입력 문장 하나를 골라** 분해하여 <보기>를 만든다.

[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
  "format": "sentence_connection",
  "input": {{
    "instruction": "다음 두 문장을 <보기>와 같이 '-(으)면서'를 사용하여 한 문장으로 만드십시오.",
    "example": {{ "clause_A": "나는 점심을 먹습니다.", "clause_B": "TV를 봅니다.", "connected": "나는 점심을 먹으면서 TV를 봅니다." }},
    "clause_A": "저는 음악을 듣습니다.",
    "clause_B": "저는 공부를 합니다."
  }},
  "answer": {{ "connected_sentence": "저는 음악을 들으면서 공부합니다." }},
  "rationale": "두 가지 행동이 동시에 일어남을 나타낼 때 동사 어간에 '-으면서'를 붙여 연결할 수 있습니다."
}}
"""

SENTENCE_CREATION_TMPL = """\
[ROLE] 한국어 문장 생성 문제 출제자. JSON만 출력.

[GOAL]
- 입력된 문장에서 핵심 표현을 추출하여 문장 생성 문제를 만든다. 
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}

[INSTRUCTIONS]
- `cues`: [INPUT_SENTENCES]의 문장 중 하나에서 핵심이 되는 표현 2~4개를 추출하여 조합할 요소로 제시한다.
- `created_sentence`: `cues`가 추출된 원본 문장을 정답으로 설정한다.
- `example`: **문제에 사용되지 않은 다른 입력 문장 하나에서** 핵심 표현을 추출하여 <보기>를 만든다.
- `instruction`: "<보기>와 같이 주어진 표현을 사용하여 문장을 완성하십시오." 같은 지시문을 작성한다.

[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
  "format": "sentence_creation",
  "input": {{
    "instruction": "<보기>와 같이 주어진 표현을 사용하여 '-(으)면서' 문법으로 문장을 완성하십시오.",
    "example": {{ "cues": ["점심을 먹다", "TV를 보다"], "answer": "점심을 먹으면서 TV를 봅니다." }},
    "cues": ["음악을 듣다", "공부하다"]
  }},
  "answer": {{ "created_sentence": "음악을 들으면서 공부합니다." }},
  "rationale": "핵심 표현들을 '-으면서' 문법을 사용하여 자연스러운 문장으로 만들 수 있습니다."
}}
"""

CHOICE_COMPLETION_TMPL = """\
[ROLE] 한국어 문제 출제자. JSON만 출력한다.

[GOAL]
- 목표 문법: {target_grammar}
- 레벨: {level}
- 제시문(prompt)에 맞는 선택지로 문장을 완성하는 문제 1개 생성.

[INSTRUCTIONS]
- `prompt`: 간단한 상황이나 질문을 제시한다.
- `options`: 4개(정답 1, 오답 3)의 선택지를 만든다. 선택지들은 목표 문법의 사용 여부나 정확성으로 정답과 오답이 갈리도록 설계한다.
- `completed_sentence`: `prompt`와 정답 `option`을 자연스럽게 연결한 완성 문장을 만든다.
- `rationale`: 왜 그것이 정답인지 문법적, 문맥적 근거를 설명한다.

---
[COMPLETE_EXAMPLE]
아래는 이 작업을 어떻게 수행해야 하는지에 대한 완벽한 예시다.

## Input Sentences For Example:
- 저는 학교에 가는 길에 친구를 만났어요.
- 퇴근하는 길에 빵을 좀 샀어요.

## Corresponding Output JSON:
{{
    "schema_id": "{{schema_id}}",
    "format": "choice_completion",
    "input": {{
        "prompt": "어제는 정말 바빴어요. 아침 일찍 일어나서 운동을 하고...",
        "options": [
            "회사에 가는 길에 세탁소에 들렀어요.",
            "회사에 가고 세탁소에 들렀어요.",
            "회사에 가려고 세탁소에 들렀어요.",
            "회사에 가지만 세탁소에 들렀어요."
        ]
    }},
    "answer": {{
        "completed_sentence": "어제는 정말 바빴어요. 아침 일찍 일어나서 운동을 하고 회사에 가는 길에 세탁소에 들렀어요."
    }},
    "rationale": "'-는 길에'는 어떤 목적지로 이동하는 도중에 다른 행동을 할 때 사용하는 문법으로, 바쁜 하루의 일과를 설명하는 문맥에 가장 자연스럽습니다."
}}
---

{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}

[OUTPUT_JSON]
"""


DIALOGUE_COMPLETION_TMPL = """\
[ROLE] 한국어 대화 완성 문제 출제자. JSON만 출력.

[GOAL]
- 목표 문법: {target_grammar}
- 레벨: {level}
- 대화(turn 3개 내외)에서 1곳을 빈칸으로 두고 자연스럽게 채우게 한다.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}

[INSTRUCTIONS]
- dialogue_with_missing_turns: A/B 대화 배열. 한 턴은 "___" 로 빈칸 표기.
- completed_dialogue: 빈칸을 채운 최종 대화 배열.
- 타깃 문법은 최소 1회 자연스럽게 등장.

[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
  "format": "dialogue_completion",
  "input": {{
    "dialogue_with_missing_turns": [
      {{"speaker":"A","text":"..."}},
      {{"speaker":"B","text":"___"}},
      {{"speaker":"A","text":"..."}}
    ]
  }},
  "answer": {{
    "completed_dialogue": [
      {{"speaker":"A","text":"..."}},
      {{"speaker":"B","text":"채워진 문장"}},
      {{"speaker":"A","text":"..."}}
    ]
  }},
  "rationale": "문맥 상의 연결 근거"
}}
"""

TMPLS = {
    "fill_in_blank": FILL_IN_BLANK_TMPL,
    "match_and_connect": MATCH_AND_CONNECT_TMPL,
    "sentence_connection": SENTENCE_CONNECTION_TMPL,
    "sentence_creation": SENTENCE_CREATION_TMPL,
    "choice_completion": CHOICE_COMPLETION_TMPL,
    "dialogue_completion": DIALOGUE_COMPLETION_TMPL,
}

# ==============================================================================
# 1. [1단계] 선택 AI 에이전트
# ==============================================================================

AGENT_PROMPT_TEMPLATE = """\
[ROLE]
당신은 10년차 한국어 교육 과정 설계 전문가입니다. 당신의 임무는 주어진 학습 정보에 기반하여 가장 교육적으로 효과적인 문제 유형을 단 하나만 추천하고, 그 이유를 논리적으로 설명하는 것입니다. 반드시 JSON 형식으로만 출력해야 합니다.
[CONTEXT]
- Target Grammar: {target_grammar}
- Learner Level: {level}
- Available Example Sentences:
{sentences_bullets}
{kpop_context}
- Available Question Formats:
{formats_bullets}
[INSTRUCTIONS]
다음 3단계의 사고 과정에 따라 최적의 문제 유형을 결정하십시오.
1.  **문법-유형 적합도 분석**: '{target_grammar}' 문법의 핵심 기능은 어떤 문제 유형으로 평가할 때 가장 효과적입니까?
2.  **입력 문장 구조 분석**: 제공된 예문들의 구조적 특징이 어떤 문제 유형을 만들기에 가장 유리합니까?
3.  **학습 목표 및 난이도 고려**: 학습자 레벨({level})을 고려할 때, 어떤 유형이 적절한 학습 효과를 유발할 수 있습니까?
위 분석을 종합하여, 가장 추천하는 문제 유형 하나를 `chosen_format`으로, 그리고 결정 이유를 `rationale`에 서술하여 아래 JSON 스키마에 맞춰 출력하십시오.
[OUTPUT_JSON_SCHEMA]
{{
  "chosen_format": "하나의 문제 유형(string)",
  "rationale": "위 3단계 분석에 기반한 구체적인 선택 이유(string)"
}}"""
#config.py에서 변경된 MODEL_NAME과 temperature를 반영합니다.
def call_llm(prompt: str, model: str = "gpt-4o") -> str:
    """OpenAI 모델을 호출하는 범용 함수"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})

def bullets(items: list) -> str:
    """리스트를 불렛 포인트 문자열로 변환합니다."""
    return "\n".join(f"- {s}" for s in items)

def select_best_schema(payload: dict) -> dict:
    """최적의 문제 유형을 '결정'하는 AI 에이전트"""
    print("🤖 [1단계] 최적 문제 유형 선택 AI 에이전트를 시작합니다...")
    
    # payload에 "critique_summary" 키가 없을 경우를 대비하여 .get() 사용
    valid_sentences = [item["sentence"] for item in payload.get("critique_summary", [])]
    available_formats = list(TMPLS.keys())
    
    # ✅ K-pop 정보 포맷팅
    kpop_context = ""
    if "kpop_references" in payload and payload["kpop_references"]:
        kpop_list = []
        for ref in payload["kpop_references"]:
            sentence = ref.get('sentence', '')
            song = ref.get('song', '')
            group = ref.get('group', '')
            kpop_list.append(f"  - \"{sentence}\" ({song} - {group})")
        kpop_context = "\n- K-pop References (학습자의 관심사):\n" + "\n".join(kpop_list)
        print(f"✨ K-pop 참조 {len(payload['kpop_references'])}개 발견")
    else:
        print("ℹ️ K-pop 참조 없음")

    prompt = AGENT_PROMPT_TEMPLATE.format(
        target_grammar=payload.get("target_grammar", "N/A"),
        level=payload.get("level", "N/A"),
        sentences_bullets=bullets(valid_sentences),
        kpop_context=kpop_context,  # ✅ 필수!
        formats_bullets=bullets(available_formats)
    )
    
    print("🧠 선택 LLM을 호출하여 분석 중입니다...")
    raw_json_output = call_llm(prompt)
    
    try:
        decision = json.loads(raw_json_output)
        print("✅ 분석 완료!")
        return decision
    except json.JSONDecodeError:
        return {
            "chosen_format": random.choice(available_formats), 
            "rationale": "AI 에이전트 응답 오류로 랜덤 선택."
        }
# ==============================================================================
# 2. [2단계] 문제 생성기
# ==============================================================================

def generate_question_item(agent_decision: dict, payload: dict) -> dict:
    """AI 에이전트의 결정을 바탕으로 실제 문제를 '생성'하는 함수"""
    chosen_format = agent_decision.get("chosen_format")
    print(f"\n🚀 [2단계] 선택된 유형 '{chosen_format}'으로 문제 생성을 시작합니다...")

    template = TMPLS.get(chosen_format)
    
    if not template:
        return {"error": f"'{chosen_format}'에 해당하는 프롬프트 템플릿이 정의되지 않았습니다."}
        
    valid_sentences = [item["sentence"] for item in payload.get("critique_summary", [])]

    # ✅ K-pop 정보 처리 추가
    kpop_info = ""
    if "kpop_references" in payload and payload["kpop_references"]:
        kpop_list = []
        for ref in payload["kpop_references"]:
            sentence = ref.get('sentence', '')
            song = ref.get('song', '')
            group = ref.get('group', '')
            kpop_list.append(f"- \"{sentence}\" ({song} - {group})")
        kpop_info = "\n[K-POP REFERENCES]\n" + "\n".join(kpop_list) + "\n"
    else:
        kpop_info = ""  # 없어도 포맷팅에 문제없게

    try:
        # ✅ 'kpop_info' 추가
        format_args = {
            "sentences_bullets": bullets(valid_sentences),
            "target_grammar": payload.get("target_grammar", "N/A"),
            "level": payload.get("level", "N/A"),
            "schema_id": "Q_generated_1",
            "kpop_info": kpop_info,
        }

        print(f"DEBUG: Formatting template '{chosen_format}' with keys: {list(format_args.keys())}")

        prompt = template.format(**format_args)

    except KeyError as e:
        print(f"❌ CRITICAL ERROR: Formatting failed with KeyError.")
        print(f"   템플릿 '{chosen_format}'에 필요한 키가 format_args에 없는지 확인하세요.")
        print(f"   오류 메시지: {e}")
        return {"error": "Template formatting failed.", "details": str(e)}
    
    print("✍️ 생성 LLM을 호출하여 문제 구성 중입니다...")
    raw_json_output = call_llm(prompt)

    try:
        generated_question = json.loads(raw_json_output)
        print("✅ 문제 생성 완료!")
        return generated_question
    except json.JSONDecodeError:
        return {"error": "문제 생성 LLM의 응답이 유효한 JSON이 아닙니다."}



# ==============================================================================
# 3. 전체 파이프라인 실행 함수 (main.py에서 호출할 함수)
# ==============================================================================

def create_korean_test_from_payload(payload: dict) -> dict:
    """
    입력받은 payload로 한국어 연습 문제를 생성하는 전체 파이프라인을 실행합니다.
    payload는 'level', 'target_grammar', 'critique_summary' 키를 포함해야 합니다.
    """
    if not all(k in payload for k in ['level', 'target_grammar', 'critique_summary']):
        return {"error": "Payload must contain 'level', 'target_grammar', and 'critique_summary' keys."}
        
    # --- 1단계 실행 ---
    agent_decision = select_best_schema(payload)
    
    # --- 2단계 실행 ---
    if "error" not in agent_decision:
        final_question = generate_question_item(agent_decision, payload)
        return final_question
    else:
        print("에이전트 결정 단계에서 오류가 발생하여 문제 생성을 진행하지 않습니다.")
        return {"error": "Agent decision failed.", "details": agent_decision}


def create_korean_test_set(payload: dict, num_questions: int = 5) -> list:
    """
    동일한 payload를 기반으로 서로 다른 유형(format)의 문제를 여러 개 생성합니다.
    - num_questions: 생성할 문항 개수
    - 문제 유형은 TMPLS의 key를 순환하며 중복되지 않게 선택
    """
    available_formats = list(TMPLS.keys())
    questions = []

    print(f"\n🧩 [확장 모드] 서로 다른 유형으로 {num_questions}개 문제를 생성합니다.")
    print(f"   사용 가능한 문제 유형: {available_formats}")

    # 문제 유형 순환 (필요 시 랜덤 셔플)
    random.shuffle(available_formats)

    for i in range(num_questions):
        fmt = available_formats[i % len(available_formats)]  # 순환 방식
        agent_decision = {
            "chosen_format": fmt,
            "rationale": "자동 생성 세트 모드에서 유형 다양화를 위해 선택됨."
        }

        print(f"\n{'='*80}")
        print(f"🧠 [{i+1}/{num_questions}] '{fmt}' 유형 문제 생성 중...")
        print('='*80)

        question = generate_question_item(agent_decision, payload)
        if "error" not in question:
            questions.append(question)
        else:
            print(f"⚠️ {fmt} 유형 문제 생성 실패: {question}")

    print(f"\n✅ 총 {len(questions)}개의 문제 생성 완료.")
    return questions