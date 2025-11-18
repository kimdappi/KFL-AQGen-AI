# test_maker.py

import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from config import MODEL_NAME

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
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 문제에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.
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
- **중요: 절을 분해할 때 반드시 자연스러운 한국어 문법을 유지해야 한다.**
  - 주어 뒤에는 조사(이/가, 은/는, 을/를 등)가 필요하면 반드시 포함
  - 장소나 시간 표현 뒤에는 조사(에서, 에, 로 등)가 필요하면 반드시 포함
  - 예: "로제와 리사를 체험함으로써" (O), "로제·리사 체험함으로써" (X)
  - 예: "YG 로비에서 머무름으로써" (O), "YG 로비 머무름으로써" (X)
- `example`: **문제에 사용되지 않은 다른 입력 문장 하나를 골라** 분해하여 <보기>를 만든다.
- `answer`: `clause_set_A`의 각 항목에 맞는 `clause_set_B`의 항목을 연결하여 만든 완성 문장들의 배열.
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 문제에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.
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
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 문제에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

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
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 문제에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

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
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 문제에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

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
- **K-pop 정보가 제공되면 모든 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 대화에 자연스럽게 활용하세요.**
- **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
  - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
  - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
  - 모든 대화는 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

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
def call_llm(prompt: str, model: str =MODEL_NAME, temperature: float = 1.0, require_json: bool = True) -> str:
    """OpenAI 모델을 호출하는 범용 함수"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant." + (" You must output JSON only." if require_json else "")},
            {"role": "user", "content": prompt}
        ]
        
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,  # 기본값 1.0 (모델이 지원하는 기본값)
        }
        
        # JSON 형식이 필요한 경우만 추가
        if require_json:
            request_params["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**request_params)
        return response.choices[0].message.content
    except Exception as e:
        if require_json:
            return json.dumps({"error": str(e)})
        else:
            return f"Error: {str(e)}"


def bullets(items: list) -> str:
    """리스트를 불렛 포인트 문자열로 변환합니다."""
    return "\n".join(f"- {s}" for s in items)

def select_best_schema(payload: dict) -> dict:
    """최적의 문제 유형을 '결정'하는 AI 에이전트"""
    print("🤖 [1단계] 최적 문제 유형 선택 AI 에이전트를 시작합니다...")
    
    available_formats = list(TMPLS.keys())
    
    # 추출된 정보 구성 (문장 대신 정보 사용)
    # 1) 문장이 있으면 사용 (기존 호환성)
    valid_sentences = [item["sentence"] for item in payload.get("critique_summary", [])]
    
    # 2) 문장이 없으면 추출된 정보로 설명 구성
    if not valid_sentences:
        vocab_list = payload.get("vocabulary", [])
        vocab_details = payload.get("vocabulary_details", [])
        grammar = payload.get("target_grammar", "N/A")
        
        info_parts = [f"목표 문법: {grammar}"]
        if vocab_list:
            vocab_str = ", ".join([f"{v['word']}({v['wordclass']})" if isinstance(v, dict) else v 
                                   for v in (vocab_details if vocab_details else vocab_list)])
            info_parts.append(f"학습 단어: {vocab_str}")
        
        valid_sentences = [" | ".join(info_parts)]  # 정보를 하나의 "문장"처럼 처리
    
    # ✅ K-pop 정보 포맷팅
    kpop_context = ""
    if "kpop_references" in payload and payload["kpop_references"]:
        kpop_list = []
        for ref in payload["kpop_references"]:
            # 새로운 형식 (sentence 없이 group, members 등만)
            group = ref.get('group', '')
            agency = ref.get('agency', '')
            fandom = ref.get('fandom', '')
            members = ref.get('members', [])
            concepts = ref.get('concepts', [])
            
            parts = [f"그룹: {group}"]
            if agency:
                parts.append(f"소속사: {agency}")
            if fandom:
                parts.append(f"팬덤: {fandom}")
            if members:
                member_names = [m.get('name', '') if isinstance(m, dict) else m for m in members[:3]]
                parts.append(f"멤버: {', '.join([n for n in member_names if n])}")
            if concepts:
                parts.append(f"컨셉: {', '.join(concepts[:2])}")
            
            kpop_list.append(" | ".join(parts))
        
        kpop_context = "\n- K-pop References (학습자의 관심사):\n" + "\n".join([f"  - {p}" for p in kpop_list])
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
    
    # 문장 추출 (기존 호환성)
    valid_sentences = [item["sentence"] for item in payload.get("critique_summary", [])]
    
    # 문장이 없으면 추출된 정보로 문장 생성
    # 각 문제 유형에 맞는 형식으로 문장 생성 필요
    sentences_required_formats = [
        "match_and_connect",      # 여러 문장 (분해/재조합)
        "sentence_connection",    # 문장 쌍 (분해 가능)
        "fill_in_blank",          # 단일 문장들
        "sentence_creation",       # 문장들 (핵심 표현 추출)
        "dialogue_completion"     # 대화 형식
    ]
    needs_actual_sentences = chosen_format in sentences_required_formats
    
    if not valid_sentences:
        vocab_list = payload.get("vocabulary", [])
        vocab_details = payload.get("vocabulary_details", [])
        grammar = payload.get("target_grammar", "N/A")
        difficulty = payload.get("difficulty", "intermediate")
        level = payload.get("level", "grade3-4")
        
        # 문장이 필요한 유형이면 먼저 문장 생성
        if needs_actual_sentences:
            print(f"   📝 '{chosen_format}' 유형은 실제 문장이 필요합니다. 유형에 맞게 문장을 생성합니다...")
            
            vocab_info = ""
            if vocab_details:
                vocab_info = ", ".join([f"{v['word']}({v['wordclass']})" for v in vocab_details])
            elif vocab_list:
                vocab_info = ", ".join(vocab_list)
            
            kpop_context = ""
            if "kpop_references" in payload and payload["kpop_references"]:
                kpop_refs = payload["kpop_references"]
                kpop_parts = []
                # 모든 K-pop 정보를 포함 (요구사항: 모든 추출 정보 사용)
                for ref in kpop_refs:
                    group = ref.get('group', '')
                    agency = ref.get('agency', '')
                    fandom = ref.get('fandom', '')
                    members = ref.get('members', [])
                    concepts = ref.get('concepts', [])
                    # 모든 멤버 정보 포함
                    member_names = [m.get('name', '') if isinstance(m, dict) else m for m in members]
                    
                    parts = []
                    if group:
                        parts.append(f"그룹: {group}")
                    if agency:
                        parts.append(f"소속사: {agency}")
                    if fandom:
                        parts.append(f"팬덤: {fandom}")
                    if member_names:
                        parts.append(f"멤버: {', '.join([n for n in member_names if n])}")
                    if concepts:
                        parts.append(f"컨셉: {', '.join(concepts)}")
                    
                    if parts:
                        kpop_parts.append(" | ".join(parts))
                
                if kpop_parts:
                    kpop_context = f"\n- K-pop 컨텍스트 (모든 정보 포함):\n  " + "\n  ".join(kpop_parts)
            
            # 문제 유형별 맞춤 프롬프트 생성
            if chosen_format == "dialogue_completion":
                # 대화 형식 생성
                sentence_gen_prompt = f"""너는 한국어를 가르치는 교사야. 다음 정보를 바탕으로 자연스럽고 일상적인 한국어 대화를 만들어라.

[학습 정보]
- 목표 문법: {grammar}
- 학습 단어: {vocab_info}
- 난이도: {difficulty} ({level}){kpop_context}

[요구사항]
1. 두 사람(A와 B)이 주고받는 자연스러운 일상 대화 (친구, 동료, 가족 등)
2. **목표 문법 '{grammar}'가 대화 안에서 반드시 사용되어야 함** (학습자가 명확히 학습할 수 있도록)
3. **최소 2개 이상의 학습 단어를 대화에 자연스럽게 포함** (맥락에 맞는 단어를 선택하여 사용)
4. {difficulty} 수준에 맞는 일상적인 대화 톤 (과도하게 격식적이지 않게)
5. **K-pop 컨텍스트가 있으면 제공된 모든 K-pop 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 활용한 자연스러운 주제로 대화**
6. 대화는 4~6턴 정도 (A와 B가 번갈아 말함, 자연스러운 흐름)
7. 구체적인 상황 설정 (예: 카페에서, 학교에서, 집에서 등)
8. **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
   - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
   - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
   - 모든 대화는 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

[출력 형식]
대화만 출력하세요. 각 턴은 "A: ..." 또는 "B: ..." 형식으로:
A: 첫 번째 말
B: 두 번째 말
A: 세 번째 말
B: 네 번째 말
"""
            elif chosen_format == "match_and_connect":
                # 분해/재조합 가능한 여러 문장 생성
                sentence_gen_prompt = f"""너는 한국어를 가르치는 교사야. 다음 정보를 바탕으로 자연스럽고 다양한 상황의 한국어 예문 6~8개를 만들어라.

[학습 정보]
- 목표 문법: {grammar}
- 학습 단어: {vocab_info}
- 난이도: {difficulty} ({level}){kpop_context}

[요구사항]
1. **각 문장은 목표 문법 '{grammar}'를 반드시 사용** (학습자가 명확히 학습할 수 있도록)
2. **최소 2개 이상의 학습 단어를 문장에 자연스럽게 포함** (맥락에 맞는 단어를 선택하여 사용)
3. {difficulty} 수준에 맞는 자연스러운 한국어 문장 (과도하게 격식적이지 않게)
4. **K-pop 컨텍스트가 있으면 제공된 K-pop 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 활용한 다양한 상황의 문장**
5. 문장은 12~30자 정도의 완성된 문장 (너무 짧거나 길지 않게)
6. 각 문장은 앞부분과 뒷부분으로 분해 가능한 구조로 작성
7. 다양한 상황과 맥락 (일상, 취미, 학교, 직장, 여행 등)을 포함하여 다양성 확보
8. **중요: 문장을 분해할 때 자연스러운 절이 되도록 작성**
   - 주어 뒤에는 필요한 조사(이/가, 은/는, 을/를 등)를 반드시 포함
   - 장소나 시간 표현 뒤에는 필요한 조사(에서, 에, 로 등)를 반드시 포함
   - 예: "제니가 비공식 토크를 진행함으로써" (O), "제니 비공식 토크 진행함으로써" (X)
9. **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
   - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
   - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
   - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

[출력 형식]
문장만 한 줄에 하나씩 출력하세요. 번호나 설명 없이 문장만:
문장1
문장2
문장3
문장4
문장5
문장6
"""
            elif chosen_format == "sentence_connection":
                # 분해 가능한 문장 쌍 생성
                sentence_gen_prompt = f"""너는 한국어를 가르치는 교사야. 다음 정보를 바탕으로 자연스러운 한국어 예문 4~5개를 만들어라.

[학습 정보]
- 목표 문법: {grammar}
- 학습 단어: {vocab_info}
- 난이도: {difficulty} ({level}){kpop_context}

[요구사항]
1. **각 문장은 목표 문법 '{grammar}'를 반드시 사용**하여 두 개의 독립된 절로 명확하게 분해 가능한 구조 (학습자가 명확히 학습할 수 있도록)
2. **최소 2개 이상의 학습 단어를 문장에 자연스럽게 포함** (맥락에 맞는 단어를 선택하여 사용)
3. {difficulty} 수준에 맞는 자연스러운 한국어 문장 (과도하게 격식적이지 않게)
4. **K-pop 컨텍스트가 있으면 제공된 모든 K-pop 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 활용한 다양한 상황의 문장**
5. 문장은 18~35자 정도 (두 절을 연결한 자연스러운 구조)
6. 다양한 상황과 맥락을 포함하여 다양성 확보
7. **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
   - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
   - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
   - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

[출력 형식]
문장만 한 줄에 하나씩 출력하세요. 번호나 설명 없이 문장만:
문장1
문장2
문장3
문장4
문장5
"""
            else:
                # fill_in_blank, sentence_creation 등 일반 문장들
                sentence_gen_prompt = f"""너는 한국어를 가르치는 교사야. 다음 정보를 바탕으로 자연스럽고 다양한 상황의 한국어 예문 6~8개를 만들어라.

[학습 정보]
- 목표 문법: {grammar}
- 학습 단어: {vocab_info}
- 난이도: {difficulty} ({level}){kpop_context}

[요구사항]
1. **각 문장은 목표 문법 '{grammar}'를 반드시 사용** (학습자가 명확히 학습할 수 있도록)
2. **최소 2개 이상의 학습 단어를 문장에 자연스럽게 포함** (맥락에 맞는 단어를 선택하여 사용)
3. {difficulty} 수준에 맞는 자연스러운 한국어 문장 (과도하게 격식적이지 않게)
4. **K-pop 컨텍스트가 있으면 제공된 모든 K-pop 정보(그룹, 멤버, 소속사, 팬덤, 컨셉)를 활용한 다양한 상황의 문장**
5. 문장은 12~28자 정도의 완성된 문장 (너무 짧거나 길지 않게)
6. 다양한 상황과 맥락 (일상, 취미, 학교, 직장, 여행, 감정 표현 등)을 포함하여 다양성 확보
7. **중요: 영어 단어나 영어 컨셉트는 반드시 한국어로 번역하여 사용하세요.**
   - 예: "self-love" → "자기 사랑" 또는 "자기애", "youth" → "젊음" 또는 "청춘", "storytelling" → "스토리텔링"
   - 예: "girl crush" → "걸크러시" (한국어로 통용되는 경우는 그대로 사용 가능)
   - 모든 문장은 자연스러운 한국어로 작성되어야 하며, 번역투를 피하세요.

[출력 형식]
문장만 한 줄에 하나씩 출력하세요. 번호나 설명 없이 문장만:
문장1
문장2
문장3
문장4
문장5
문장6
"""
            
            try:
                raw_sentences = call_llm(sentence_gen_prompt, temperature=1.0, require_json=False)
                
                # 유형별 파싱
                if chosen_format == "dialogue_completion":
                    # 대화 형식 파싱
                    dialogue_lines = [s.strip() for s in raw_sentences.strip().split('\n') if s.strip()]
                    generated_sentences = []
                    for line in dialogue_lines:
                        # "A: ..." 또는 "B: ..." 형식 파싱
                        if ':' in line:
                            speaker_text = line.split(':', 1)[1].strip()
                            if len(speaker_text) > 5:
                                generated_sentences.append(speaker_text)
                    if not generated_sentences:
                        # 형식이 다르면 그냥 줄 단위로 처리
                        generated_sentences = [s.lstrip('0123456789.-) ').strip(' "“"') 
                                             for s in dialogue_lines if len(s.strip()) > 5]
                else:
                    # 일반 문장 파싱
                    generated_sentences = [s.strip() for s in raw_sentences.strip().split('\n') if s.strip()]
                    # 번호나 불릿 제거
                    generated_sentences = [s.lstrip('0123456789.-) ').strip(' "“"') 
                                         for s in generated_sentences if len(s.strip()) > 5]
                
                if generated_sentences:
                    max_sentences = 6 if chosen_format != "sentence_connection" else 4
                    valid_sentences = generated_sentences[:max_sentences]
                    print(f"   ✅ {len(valid_sentences)}개의 문장 생성 완료 ({chosen_format} 유형에 맞게)")
                else:
                    print(f"   ⚠️ 문장 생성 실패, 지시사항으로 대체")
                    instruction_text = f"목표 문법: {grammar} | 학습 단어: {vocab_info} | 난이도: {difficulty}"
                    valid_sentences = [instruction_text]
            except Exception as e:
                print(f"   ⚠️ 문장 생성 중 오류: {e}, 지시사항으로 대체")
                instruction_text = f"목표 문법: {grammar} | 학습 단어: {vocab_info} | 난이도: {difficulty}"
                valid_sentences = [instruction_text]
        else:
            # 문장이 필수 아닌 유형은 지시사항으로 처리
            vocab_info = ""
            if vocab_details:
                vocab_info = ", ".join([f"{v['word']}({v['wordclass']})" for v in vocab_details])
            elif vocab_list:
                vocab_info = ", ".join(vocab_list)
            
            instruction_text = f"""
[학습 정보]
- 목표 문법: {grammar}
- 학습 단어: {vocab_info}
- 난이도: {difficulty}

위 정보를 바탕으로 자연스러운 한국어 예문을 생성하여 문제를 만들어주세요.
"""
            valid_sentences = [instruction_text]  # 지시사항을 "문장"처럼 처리

    # ✅ K-pop 정보 처리 추가
    kpop_info = ""
    if "kpop_references" in payload and payload["kpop_references"]:
        kpop_list = []
        for ref in payload["kpop_references"]:
            # 새로운 형식 지원
            group = ref.get('group', '')
            agency = ref.get('agency', '')
            fandom = ref.get('fandom', '')
            members = ref.get('members', [])
            concepts = ref.get('concepts', [])
            song = ref.get('song', '')  # 기존 형식도 지원
            
            if song:  # 기존 형식 (sentence 포함)
                sentence = ref.get('sentence', '')
                kpop_list.append(f"- \"{sentence}\" ({song} - {group})")
            else:  # 새로운 형식 (정보만)
                parts = [f"그룹: {group}"]
                if agency:
                    parts.append(f"소속사: {agency}")
                if fandom:
                    parts.append(f"팬덤: {fandom}")
                if members:
                    # 모든 멤버 정보 포함 (요구사항: 모든 추출 정보 사용)
                    member_names = [m.get('name', '') if isinstance(m, dict) else m for m in members]
                    parts.append(f"멤버: {', '.join([n for n in member_names if n])}")
                if concepts:
                    # 모든 컨셉 정보 포함
                    parts.append(f"컨셉: {', '.join(concepts)}")
                kpop_list.append(f"- {' | '.join(parts)}")
        
        kpop_info = "\n[K-POP REFERENCES - 모든 정보를 문제에 활용하세요]\n" + "\n".join(kpop_list) + "\n"
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
    try:
        raw_json_output = call_llm(prompt)
        
        # 에러 체크
        if not raw_json_output:
            return {"error": "LLM 응답이 비어있습니다."}
        
        # JSON 파싱 시도
        try:
            generated_question = json.loads(raw_json_output)
            
            # 에러 필드가 있는지 확인
            if "error" in generated_question:
                print(f"   ⚠️ LLM이 에러를 반환했습니다: {generated_question.get('error')}")
                return generated_question
            
            print("✅ 문제 생성 완료!")
            return generated_question
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON 파싱 실패")
            print(f"      응답 일부: {raw_json_output[:200]}...")
            return {"error": "문제 생성 LLM의 응답이 유효한 JSON이 아닙니다.", "details": str(e), "raw_response": raw_json_output[:500]}
    except Exception as e:
        print(f"   ❌ LLM 호출 중 예외 발생: {e}")
        return {"error": f"LLM 호출 실패: {str(e)}"}



# 3. 전체 파이프라인 실행 함수 (main.py에서 호출할 함수)

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
            print(f"   ✅ '{fmt}' 유형 문제 생성 성공")
        else:
            print(f"   ❌ {fmt} 유형 문제 생성 실패")
            print(f"      에러 내용: {question.get('error', 'Unknown error')}")
            if 'details' in question:
                print(f"      상세: {question.get('details')}")

    print(f"\n✅ 총 {len(questions)}개의 문제 생성 완료.")
    return questions