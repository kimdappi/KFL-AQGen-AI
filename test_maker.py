import json
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
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] [INPUT_SENTENCES]에서 문장 하나를 선택하여 빈칸 채우기 문제 1개 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `stem_with_blank`: 타깃 문법 부분만 빈칸( ___)으로 변경. 빈칸 부분에만 목표 문법 필요.
- `hint`: 빈칸에 들어갈 동사/형용사 기본형.
- `example`: 다른 입력 문장으로 보기 생성.
- 자연스러운 일상 문장만 사용. 학습 단어 억지 사용 금지.
- K-pop 정보는 자연스럽게만 활용 (나열 금지).
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{ "schema_id": "{schema_id}", "format": "fill_in_blank", "input": {{"instruction": "<보기>와 같이 괄호 안의 단어를 사용하여 문장을 완성하십시오.", "example": {{"stem": "나는 점심을 ___ TV를 봤어요. (먹다)", "answer": "먹으면서"}}, "stem_with_blank": "저는 음악을 ___ 공부합니다. ({{hint}})", "hint": "듣다"}}, "answer": {{"completed_sentence": "저는 음악을 들으면서 공부합니다."}}, "rationale": "두 가지 행동을 동시에 함을 나타내는 '-으면서'가 자연스럽습니다."}}"""

MATCH_AND_CONNECT_TMPL = """\
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] 입력 문장들을 분해·재조합하여 문장 연결 문제 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `clause_set_A`, `clause_set_B`: 입력 문장 3~4개를 앞부분/뒷부분으로 분해, 순서 섞기.
- 연결 부분에만 목표 문법 사용. 각 절은 자연스러운 일상 문장.
- 절 분해 시 조사 포함 (주어 뒤: 이/가/은/는, 장소 뒤: 에/에서/로).
- `example`: 다른 입력 문장으로 보기 생성.
- `answer`: 목표 문법으로 연결한 완성 문장 배열.
- 자연스러운 일상 문장만. 학습 단어 억지 사용 금지.
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{ "schema_id": "{schema_id}", "format": "match_and_connect", "input": {{"instruction": "다음 문장을 연결하여 <보기>와 같이 하나의 문장을 만드십시오.", "example": {{ "clause_A": "아버지는 운동을 하다", "clause_B": "건강을 챙기다", "connected": "아버지는 운동을 하면서 건강을 챙깁니다." }}, "clause_set_A": ["저는 음악을 듣다", "그녀는 친구와 이야기를 나누다"], "clause_set_B": ["공부하다", "웃고 있다"]}}, "answer": {{ "connected_sentences": ["저는 음악을 들으면서 공부합니다.", "그녀는 친구와 이야기를 나누면서 웃고 있습니다."] }}, "rationale": "두 절을 자연스럽게 연결합니다."}}"""


SENTENCE_CONNECTION_TMPL = """\
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] 입력 문장 하나를 두 절로 분해하여 문장 연결 문제 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `input`: 문장 하나를 `clause_A`, `clause_B`로 분해. 각 절은 자연스러운 일상 문장.
- `answer`: 원본 문장을 `connected_sentence`로 설정. 연결 부분에만 목표 문법 사용.
- `example`: 다른 입력 문장으로 보기 생성.
- 자연스러운 일상 문장만. 학습 단어 억지 사용 금지.
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
  "format": "sentence_connection",
  "input": {{
    "instruction": "다음 두 문장을 <보기>와 같이 목표 문법을 사용하여 한 문장으로 만드십시오.",
    "example": {{ "clause_A": "나는 점심을 먹습니다.", "clause_B": "TV를 봅니다.", "connected": "나는 점심을 먹으면서 TV를 봅니다." }},
    "clause_A": "저는 음악을 듣습니다.",
    "clause_B": "저는 공부를 합니다."
  }},
  "answer": {{ "connected_sentence": "저는 음악을 들으면서 공부합니다." }},
  "rationale": "두 절을 목표 문법으로 자연스럽게 연결합니다."
}}
"""

SENTENCE_CREATION_TMPL = """\
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] 입력 문장에서 핵심 표현 추출하여 문장 생성 문제 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `cues`: 핵심 표현 2~4개 추출.
- `created_sentence`: 원본 문장을 정답으로 설정. 전체 문장에 목표 문법 포함.
- `example`: 다른 입력 문장으로 보기 생성.
- 자연스러운 일상 문장만. 학습 단어 억지 사용 금지.
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
  "format": "sentence_creation",
  "input": {{
    "instruction": "<보기>와 같이 주어진 표현을 사용하여 문장을 완성하십시오.",
    "example": {{ "cues": ["점심을 먹다", "TV를 보다"], "answer": "점심을 먹으면서 TV를 봅니다." }},
    "cues": ["음악을 듣다", "공부하다"]
  }},
  "answer": {{ "created_sentence": "음악을 들으면서 공부합니다." }},
  "rationale": "핵심 표현을 목표 문법으로 자연스럽게 연결합니다."
}}
"""

CHOICE_COMPLETION_TMPL = """\
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] 목표 문법: {target_grammar}, 레벨: {level}. 선택형 문제 1개 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- `prompt`: 자연스러운 상황 설명. 목표 문법 불필요.
- `options`: 4개 선택지 (정답 1, 오답 3). 정답에만 목표 문법 사용.
- `completed_sentence`: prompt + 정답 연결.
- `rationale`: 정답 근거 설명.
- 자연스러운 일상 문장만. 학습 단어 억지 사용 금지.
[OUTPUT_JSON_SCHEMA_EXAMPLE]
{{
  "schema_id": "{schema_id}",
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
  "rationale": "목표 문법이 정확히 사용된 선택지가 정답입니다."
}}
"""


DIALOGUE_COMPLETION_TMPL = """\
[ROLE] 너는 외국인들을 위한 한국어 교재 편집자다. 반드시 JSON만 출력.
[GOAL] 목표 문법: {target_grammar}, 레벨: {level}. 대화 완성 문제 생성.
{kpop_info}[INPUT_SENTENCES]
{sentences_bullets}
[INSTRUCTIONS]
- dialogue_with_missing_turns: A/B 대화 배열, 한 턴은 "___"로 빈칸.
- completed_dialogue: 빈칸 채운 최종 대화.
- 빈칸 부분에만 목표 문법 사용. 다른 턴은 자연스러운 일상 대화.
- 자연스러운 일상 대화만. 학습 단어 억지 사용 금지.
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
  "rationale": "문맥 연결 근거"
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
def call_llm(prompt: str, model: str = "gpt-5", temperature: float = 1.0, require_json: bool = True) -> str:
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
                sentence_gen_prompt = f"""한국어 교사. 자연스러운 일상 대화 생성.

목표 문법: {grammar}
학습 단어: {vocab_info}
난이도: {difficulty}{kpop_context}

요구사항:
- A와 B가 주고받는 자연스러운 대화 (4~6턴)
- 목표 문법은 학습자가 학습할 수 있게 한두 부분에만 사용 (모든 턴에 불필요)
- 학습 단어는 맥락에 맞는 것만 선택적 사용
- K-pop 정보는 자연스럽게만 활용 (나열 금지)
- 학습 단어 억지 사용 금지

출력: "A: ..." 또는 "B: ..." 형식으로만
"""
            elif chosen_format == "match_and_connect":
                sentence_gen_prompt = f"""한국어 교사. 자연스러운 예문 6~8개 생성.

목표 문법: {grammar}
학습 단어: {vocab_info}
난이도: {difficulty}{kpop_context}

요구사항:
- 연결 부분에만 목표 문법 사용. 각 절은 자연스러운 문장으로 만들어야 함.
- 학습 단어는 맥락에 맞는 것만 선택적 사용.
- 문장 12~30자, 앞부분/뒷부분 분해 가능 구조.
- 절 분해 시 조사 포함 (주어 뒤: 이/가/은/는, 장소 뒤: 에/에서/로).
- 학습 단어 억지 사용 금지.

출력: 문장만 한 줄에 하나씩
"""
            elif chosen_format == "sentence_connection":
                sentence_gen_prompt = f"""한국어 교사. 자연스러운 예문 4~5개 생성.

목표 문법: {grammar}
학습 단어: {vocab_info}
난이도: {difficulty}{kpop_context}

요구사항:
- 두 절을 목표 문법으로 연결. 각 절은 자연스러운 일상 문장.
- 학습 단어는 맥락에 맞는 것만 선택적 사용.
- 문장 18~35자.
- 학습 단어 억지 사용 금지.

출력: 문장만 한 줄에 하나씩
"""
            else:
                sentence_gen_prompt = f"""한국어 교사. 자연스러운 예문 6~8개 생성.

목표 문법: {grammar}
학습 단어: {vocab_info}
난이도: {difficulty}{kpop_context}

요구사항:
- 전체 문장에 목표 문법 포함.
- 학습 단어는 맥락에 맞는 것만 선택적 사용.
- 문장 12~28자.
- 학습 단어 억지 사용 금지.

출력: 문장만 한 줄에 하나씩
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