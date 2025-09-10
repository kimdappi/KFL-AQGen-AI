
## Korean as a Foreign Language — Adaptive Question Generation (문제 생성) 


### 개요

KFL-AQG의 문제 생성 모듈은 다음을 담당합니다.

* 입력(프로필: 국적/레벨/관심사 등)을 받아 학습자 맞춤 문제지를 생성
* RAG(검색된 실제 예문/대화)를 근거로 authentic한 문항 생성
* 생성된 문항을 Critic(검수) 모듈로 품질 보정
* 난이도·문제유형(Reading/Listening/Writing)·오답지(선지) 설계까지 자동화

핵심 설계 원칙: **일관성, 근거(예문)/해설 포함, 국적별 오류 보정(일본인 패턴)**

---

### 목표(정량적)

* 초기 MVP: `Reading`과 `Listening` 중심 문제 3문항 세트 생성(각 세트에 해설 포함)
* 난이도 매칭 정확도(사전 레벨 vs 후속 검증) ≥ 70%
* Critic 통과율(기본 검수) ≥ 90%

---

### 문제 정의 (구체화)

* **타깃:** 일본인, 정착자·초중급(대략 TOPIK A1\~B1)
* **포커스:** 생활대화(식당/교통/쇼핑/병원/학교 등)
* **입력(프로필):** `{ nationality, age, level, interest, goal }`
* **산출(문제세트):** JSON 배열 — 각 항목은 `{id, type, question, choices?, answer, explain, source_contexts, difficulty_score}`

예시 출력(요약):

```json
{
  "questions": [
    {
      "id":1,
      "type":"multiple_choice",
      "question":"식당에서 '이거 하나 주세요'와 가장 유사한 표현은?",
      "choices":["이거 하나 주세요","이거 하나 빼주세요","이거 많이 주세요","이건 안 됩니다"],
      "answer":"이거 하나 주세요",
      "explain":"~",
      "source_contexts":["이거 한 그릇 주세요.", "주문하실게요?"],
      "difficulty_score":0.35
    }, ...
  ]
}
```

---

### 핵심 구성 요소 (요약)

1. **Profile Parser** — 입력 프로필을 표준화 (level → 난이도 스칼라)
2. **RAG Retriever** — 관심사/주제 기반 검색(벡터 또는 키워드)로 contexts 반환
3. **Generator (LLM)** — RAG 결과 + 프롬프트 템플릿으로 문제 초안 생성
4. **Critic (검수)** — 문법·난이도·문화적 민감성 평가 및 수정 제안
5. **Postprocessor** — 포맷, 난이도 수치화, 오답지 검증(상식/중복 제거)

---

### 난이도 설계(예시)

* 레벨 → 난이도 스케일(0.0 \~ 1.0)

  * A1: 0.0 \~ 0.25
  * A2: 0.25 \~ 0.45
  * B1: 0.45 \~ 0.7
* 난이도 산출 요소: 어휘레벨(주빈도 사전 기준), 문장 길이(토큰수), 문법 구조 복잡도, 추론 요구도
* 생성 후 `difficulty_score` 계산: 가중합(예: 0.5*vocab + 0.3*length + 0.2\*grammar\_complexity)

---

### 데이터 & RAG

* **기본 코퍼스(초기):** 교재 예문, 생활회화 문장 CSV(간단 텍스트 라인)
* **필수 메타:** 문장별 난이도 태그, 주제 태그(식당/교통 등), 출처
* **RAG 방식(초기 MVP):** 키워드 매칭 → 상위 N 문장 반환(벡터 인덱스는 차후 확장)
* **국적별 오류 DB:** 일본인 학습자가 자주 틀리는 패턴(띄어쓰기, 조사 혼동 등)을 수집해 distractor/해설에 반영

---

### 프롬프트 템플릿(실전 예시)

**1) 질문 생성 프롬프트 (Generator)**
```
당신은 한국어 교육 전문가이며, 일본인 초중급 학습자에게 적합한 문제를 생성해야 합니다.
입력:
- 레벨: A2
- 주제: 식당 주문
- 목표: Reading 3문제 (객관식 2, 단답형 1)
- 근거 문장(문맥): 
  1) "이거 한 그릇 주세요."
  2) "영수증 부탁합니다."
요구사항:
1) 각 문항은 JSON 객체로 출력할 것.
2) 객관식은 4지선다, 정답은 하나, 오답(선지)은 학습자가 흔히 틀리는 패턴을 반영할 것(일본인 오류 패턴 고려).
3) 각 문항에 'explain'을 포함해 정답 근거와 오해될 만한 점을 서술할 것.
4) 난이도는 0~1 사이 숫자(difficulty_score)로 표기할 것.
출력예시: JSON 배열
```

**2) Critic(검수) 프롬프트**
```
다음은 생성된 문제입니다. 문법, 의미적 정확성, 문화적 민감성, 난이도 타당성을 검사하시오.
- 각 문항에 대해 score(0-100), 수정된 문항(있다면), 수정 이유를 JSON으로 반환.
- 특히 일본인 학습자가 혼동할만한 조사/띄어쓰기 표현을 지적하시오.
```

> 팁: Generator에게는 `few-shot` 예시 1\~3개를 함께 주어 출력 포맷을 확실히 고정하세요.

---

**LLM 하이퍼파라미터 권장값 (일관성 유지)**

* Temperature: `0.0 ~ 0.3` (일관성/정답성 우선)
* Top-p: `0.8 ~ 0.95`
* Max tokens: 생성 포맷에 따라 150\~512
* System prompt(역할 고정) + user prompt(템플릿) 조합 사용
* 프롬프트 내부에 `Output must be valid JSON` 명시

---

## 생성 파이프라인 (의사코드)

```python
def generate_pipeline(profile, focus, n):
    # 1. parse profile -> level_score
    level_score = map_level(profile['level'])
    # 2. retrieve contexts
    contexts = rag_retrieve(profile['interest'], top_k=5)
    # 3. build prompt (few-shot + contexts + constraints)
    prompt = build_generator_prompt(profile, contexts, focus, n)
    # 4. call LLM -> draft_questions
    draft = call_llm(prompt, temp=0.2)
    # 5. call Critic -> reviewed_questions
    reviewed = call_critic(draft, profile)
    # 6. postprocess: parse, difficulty_score 계산, validate choices
    final = postprocess(reviewed)
    return final
```

---

## 오답지(선지) 설계 가이드라인

* 1개는 문법형 오답(일본인 자주 실수하는 조사/띄어쓰기)
* 1개는 의미 유사 오답(동사/명사 유사)
* 1개는 문장 불완전/비문(명백한 오답 — 안전장치)
* Shuffle: 정답 위치 고정하지 않음
* 난이도에 따라 distractor의 '유혹도' 제어(난이도 높을수록 더 유사한 오답)

---

## 평가 지표

* **정답률(Accuracy):** 사용자 시험 데이터에서의 정답률
* **난이도 일치율:** 생성 난이도와 실제 사용자 성취도(후속 테스트) 일치도
* **Critic 수정률:** 생성물 대비 Critic이 수정한 비율(낮을수록 안정적)
* **사용자 만족도(UX):** 간단 설문(이해도/난이도 적정성)
* **오답 유형 분포:** 어떤 유형 distractor가 자주 맞는지 분석

---

## 데이터 보강(데이터 부족 대응)

* **역번역(Back-translation)**: 문장 다양성 확보
* **패턴 시뮬레이션:** 일본인 오답 패턴 규칙을 만들어 synthetic wrong-choices 생성
* **데이터 증강:** 의도 변형(질문문→대화→문장) 자동 생성 스크립트

---

## 안전·윤리·문화적 고려

* 문화적으로 민감한 표현 제거 (음식/종교/성 관련 민감어 체크)
* 학습자 신상정보 수집 최소화(프로필은 익명화 가능 항목만 사용)
* 출처 표기: RAG에서 인용한 문장은 `source_contexts`에 출처 표기

---

## 실전 적용 팁

* **LLM 변수 고정**: `TEMPERATURE`, `TOP_P`를 환경변수로 고정해 실험간 재현성 확보
* **템플릿 버전 관리**: 프롬프트 템플릿을 버전별로 저장하고 A/B 테스트 수행
* **Critic을 '검수자'로 쓰되 자동수정 루프를 1회만 허용**: 무한 루프 방지
* **모니터링 로그**: 생성 프롬프트/응답 원본을 로그로 저장(디버깅용)

---

## 파일/스크립트(권장 최소셋)

프로젝트에서 문제생성 모듈만 따로 관리할 경우 권장 파일들:

```
/generator
├─ generate.py            # 엔트리 포인트 (generate_pipeline)
├─ prompt_templates.md    # 프롬프트 및 few-shot 예시
├─ rag/
│  └─ retriever.py        # contexts 불러오기
├─ critic/
│  └─ critic.py           # 검수 로직
├─ data/
│  ├─ textbook_sentences.csv
│  └─ jp_error_patterns.json
├─ utils/
│  └─ difficulty.py
└─ tests/
   └─ sample_requests.json
```

---

## 예시: CLI 간단 사용 (로컬)

`generate.py`가 있다고 가정할 때:

```bash
python generate.py --profile '{"nationality":"Japan","level":"A2","interest":"식당"}' --focus reading --n 3
```

출력: JSON 문자열(문제세트)

---

## 향후 로드맵 (우선순위)

1. RAG: 키워드 → 벡터(FAISS) 전환
2. Generator: 프롬프트 개선 + few-shot 표준화
3. Critic: GPT 기반 점수/교정 → 자동 교정 루프 1회 도입
4. 사용자 데이터(실제 일본인 학습자)로 난이도 리프팅/보정
5. Listening: TTS 연동 및 리스닝 문제 자동 생성

---

