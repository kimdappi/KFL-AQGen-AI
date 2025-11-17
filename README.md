# KFL-AQGen-AI

> **한국어 학습용 문항 자동 생성 시스템**  
> 지능형 라우터 기반 Agentic RAG 시스템으로 TOPIK 어휘, 문법, K-pop 문맥을 결합해 난이도별 한국어 학습 문제를 자동 생성합니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-green.svg)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-orange.svg)](https://openai.com)

## 🚀 핵심 특징

- **🧠 지능형 라우터**: 쿼리 분석 기반으로 필요한 리트리버만 선택적 실행
- **🔄 Agentic RAG**: 질의 분석 → 지식 검색 → 품질 점검 → 정보 추출의 자동화된 흐름
- **📊 난이도 인식**: 입력에서 난이도를 자동 감지하여 TOPIK 1-6급과 연동
- **🎵 K-pop 통합**: K-pop 문맥을 활용한 재미있는 한국어 학습 자료 생성
- **📚 멀티 소스**: TOPIK 어휘, 문법 패턴, K-pop 정보의 통합 검색
- **🎯 직접 문제 생성**: 문장 생성 단계 없이 추출된 정보로 바로 문제 생성
- **🔍 동적 필터링**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입으로 자동 필터링
- **🌐 임베딩 기반 매칭**: 하드코딩 없이 한글/영어 그룹명 자동 변환 (임계치 0.75)
- **🌍 자연스러운 한국어**: 번역투 방지, 영어 단어를 한국어로 자동 번역

## 📁 프로젝트 구조

```
KFL-AQGen-AI/
│
├── 📂 data/                          # 학습 데이터
│   ├── 📂 words/                     # TOPIK 어휘
│   │   ├── TOPIK1.csv               # 기초 (basic)
│   │   ├── TOPIK2.csv               # 기초 (basic)
│   │   ├── TOPIK3.csv               # 중급 (intermediate)
│   │   ├── TOPIK4.csv               # 중급 (intermediate)
│   │   ├── TOPIK5.csv               # 고급 (advanced)
│   │   └── TOPIK6.csv               # 고급 (advanced)
│   ├── 📂 grammar/                   # 문법 패턴
│   │   ├── grammar_list_A.json      # 기초 (basic)
│   │   ├── grammar_list_B.json      # 중급 (intermediate)
│   │   └── grammar_list_C.json      # 고급 (advanced)
│   └── 📂 kpop/                      # K-pop 학습 자료
│       └── kpop_db.json             # K-pop 그룹/멤버 정보
│
├── 📂 Retriever/                     # 검색 모듈
│   ├── vocabulary_retriever.py      # TOPIK 어휘 검색 (난이도별)
│   ├── grammar_retriever.py         # 문법 패턴 검색 (난이도별)
│   └── kpop_retriever.py            # K-pop 정보 검색
│
├── 📂 Ragsystem/                     # RAG 시스템 핵심
│   ├── schema.py                     # GraphState 스키마 정의
│   ├── router.py                     # 지능형 라우터 (검색 전략 결정)
│   ├── nodes.py                      # 기본 노드 (정보 추출, 문제 생성용 payload 구성)
│   ├── nodes_router_intergration.py  # 라우터 통합 노드
│   ├── graph.py                      # 기본 그래프 (레거시)
│   └── graph_agentic_router.py       # 라우터 통합 그래프 (메인)
│
├── 📂 output/                        # 출력 결과
│   └── final_v.1.json                # 최종 생성된 문제들
│
├── 🐍 agents.py                      # AI 에이전트 (쿼리 분석, 품질 검증)
├── ⚙️ config.py                      # 설정 파일 (경로, LLM 설정)
├── 🛠️ utils.py                       # 유틸리티 함수
├── 🎯 test_maker.py                  # 문제 생성기 (6가지 유형)
├── 🚀 main_router.py                 # 메인 실행 파일 (권장)
├── 📋 requirements.txt               # 의존성
└── 📖 README.md                      # 문서
```

## 🔄 전체 워크플로우 (Input → Output)

### 📥 입력 단계

**파일: `main_router.py`**
```
사용자 쿼리 입력
예: "Create intermediate level Korean grammar practice questions about BLACKPINK"
```

### 🔧 초기화 단계

**파일: `main_router.py` → `config.py`**

1. **리트리버 초기화**
   - `TOPIKVocabularyRetriever`: `data/words/TOPIK*.csv` 로드
   - `GrammarRetriever`: `data/grammar/grammar_list_*.json` 로드
   - `KpopSentenceRetriever`: `data/kpop/kpop_db.json` 로드

2. **그래프 구축**
   - `RouterAgenticGraph` 생성 (LangGraph 워크플로우)

### 🧠 LangGraph 워크플로우 실행

**파일: `Ragsystem/graph_agentic_router.py`**

#### 노드 1: `analyze_query`
**파일: `agents.py` → `QueryAnalysisAgent.analyze()`**
- 쿼리 분석 (난이도, 주제, K-pop 필요성)
- **동적 필터 조건 추출**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입
- **임베딩 기반 그룹명 자동 매칭**: 하드코딩 없이 "아이브" → "IVE" 자동 변환
- 출력: `query_analysis` (difficulty, topic, needs_kpop, kpop_filters)

#### 노드 2: `routing`
**파일: `Ragsystem/nodes_router_intergration.py` → `routing_node()`**
- **파일: `Ragsystem/router.py` → `IntelligentRouter.route()`**
- 검색 전략 결정
  - **Vocabulary 리트리버**: 무조건 활성화
  - **Grammar 리트리버**: 문법 관련 키워드("문법", "패턴", "grammar", "pattern", "표현", "구조")가 있을 때만 활성화
  - **K-pop 리트리버**: K-pop 관련 내용이 쿼리에 있을 때만 활성화
- 출력: `routing_decision` (strategies)

#### 노드 3-5: 리트리버 실행 (순차)
**파일: `Ragsystem/nodes_router_intergration.py`**

**3-1. `retrieve_vocabulary`**
- **파일: `Retriever/vocabulary_retriever.py` → `TOPIKVocabularyRetriever.invoke()`**
- 난이도별 TOPIK 단어 검색
  - `basic` (TOPIK 1-2): `data/words/TOPIK1.csv`, `TOPIK2.csv`
  - `intermediate` (TOPIK 3-4): `data/words/TOPIK3.csv`, `TOPIK4.csv`
  - `advanced` (TOPIK 5-6): `data/words/TOPIK5.csv`, `TOPIK6.csv`
- 최대 5개 단어 추출 (자연스러운 문제 생성을 위해 증가)
- 출력: `vocabulary_docs`

**3-2. `retrieve_grammar`**
- **파일: `Retriever/grammar_retriever.py` → `GrammarRetriever.invoke()`**
- 난이도별 문법 검색
  - `basic`: `data/grammar/grammar_list_A.json`
  - `intermediate`: `data/grammar/grammar_list_B.json`
  - `advanced`: `data/grammar/grammar_list_C.json`
- 1개 문법 추출
- 출력: `grammar_docs`

**3-3. `retrieve_kpop`** (조건부)
- **파일: `Retriever/kpop_retriever.py` → `KpopSentenceRetriever.invoke()`**
- K-pop DB에서 그룹/멤버 정보 검색
- 쿼리에 K-pop 키워드가 있을 때만 활성화
- **동적 필터링**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입으로 필터링
- **임베딩 기반 그룹명 매칭**: 임계치 0.75로 정확한 매칭
- 최대 5개 정보 추출 (더 풍부한 컨텍스트 제공)
- 출력: `kpop_docs`

#### 노드 6: `check_quality`
**파일: `Ragsystem/nodes_router_intergration.py` → `check_quality_agent()`**
- 검색 결과 품질 검증
- 기준: 어휘 5개 이상, 문법 1개 이상, K-pop 3개 이상(필요시)
- 출력: `quality_check` (sufficient 여부)

#### 노드 7: `rerank` (조건부)
**파일: `Ragsystem/nodes_router_intergration.py` → `rerank_node()`**
- 품질 부족 시 재검색 (최대 1회)
- 다시 `check_quality`로 이동

#### 노드 8: `generate`
**파일: `Ragsystem/nodes.py` → `generate_question_directly()`**
- **문장 생성 없이 정보만 추출**
- 단어 5개 추출 (난이도에 맞는 것, 자연스러운 문제 생성을 위해 증가)
- 문법 1개 추출 (난이도에 맞는 것)
- K-pop 정보 최대 5개 추출 (쿼리에 K-pop 관련이 있을 때만, 더 풍부한 컨텍스트 제공)
- **동적 필터링 적용**: kpop_filters 기반으로 필터링된 정보만 사용
- 문제 생성용 payload 구성
- 출력: `question_payload` (그래프 결과에서 직접 반환)

#### 노드 9: `format_output`
**파일: `Ragsystem/nodes.py` → `format_output_agentic()`**
- 최종 출력 포맷팅
- 출력: `final_output` (문자열)

### 🎯 문제 생성 파이프라인

**파일: `main_router.py`**

```
그래프 실행 결과에서 question_payload 직접 추출
    ↓
test_maker.py 호출 (question_payload 전달)
```

**파일: `test_maker.py`**

#### 3-1. 문제 유형 선택
**`select_best_schema(payload)`**
- LLM으로 최적 문제 유형 선택
- 6가지 유형:
  - `fill_in_blank`: 빈칸 채우기
  - `match_and_connect`: 문장 연결하기
  - `sentence_connection`: 문장 연결
  - `sentence_creation`: 문장 생성
  - `choice_completion`: 선택지 완성
  - `dialogue_completion`: 대화 완성
- 출력: `chosen_format`

#### 3-2. 문장 생성 (필요시)
**`generate_question_item()` 내부**
- 문장이 필요한 유형 (`match_and_connect`, `sentence_connection`, `fill_in_blank`, `dialogue_completion`, `sentence_creation`)
- 추출된 정보로 유형별 맞춤 문장 생성:
  - `dialogue_completion`: 대화 형식 (A/B 턴)
  - `match_and_connect`: 분해/재조합 가능한 문장들 (자연스러운 조사 포함)
  - `sentence_connection`: 두 절로 분해 가능한 문장들
  - 기타: 일반 예문들
- **번역투 방지**: 영어 단어/컨셉트는 한국어로 번역하여 사용
  - 예: "self-love" → "자기 사랑", "youth" → "청춘", "storytelling" → "스토리텔링"
- **요구사항**:
  - 목표 문법 반드시 사용
  - 최소 2개 이상의 학습 단어 자연스럽게 포함
  - 모든 K-pop 컨텍스트 정보 활용

#### 3-3. 문제 생성
**`generate_question_item(agent_decision, payload)`**
- 선택된 유형의 템플릿 사용
- LLM으로 실제 문제 생성
- 출력: 문제 JSON

#### 3-4. 문제 세트 생성
**`create_korean_test_set(payload, num_questions=6)`**
- 위 과정을 6번 반복하여 다양한 유형의 문제 생성
- 출력: 문제 리스트

### 📤 최종 저장

**파일: `main_router.py`**

```
생성된 문제들을
    ↓
output/final_v.1.json에 저장
```

## 📊 상세 워크플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT: 사용자 쿼리 입력                                  │
│    "Create intermediate Korean questions about BLACKPINK"   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. INITIALIZATION: main_router.py                           │
│    ├─ 리트리버 초기화 (vocabulary, grammar, kpop)           │
│    └─ RouterAgenticGraph 생성                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LANGGRAPH WORKFLOW: graph_agentic_router.py              │
│                                                              │
│    analyze_query (agents.py)                                │
│         │                                                    │
│         ▼                                                    │
│    routing (router.py)                                       │
│         │                                                    │
│         ├─► retrieve_vocabulary (vocabulary_retriever.py)   │
│         │   └─► TOPIK CSV에서 단어 5개 추출                 │
│         │                                                    │
│         ├─► retrieve_grammar (grammar_retriever.py)         │
│         │   └─► JSON에서 문법 1개 추출                       │
│         │                                                    │
│         └─► retrieve_kpop (kpop_retriever.py) [조건부]      │
│             └─► 임베딩 기반 매칭 + 동적 필터링              │
│                 └─► JSON에서 K-pop 정보 최대 5개 추출        │
│                                                              │
│         ▼                                                    │
│    check_quality                                             │
│         │                                                    │
│         ├─► 충족 ──► generate_question_directly             │
│         │   └─► 정보 추출 + payload 구성                     │
│         │                                                    │
│         └─► 부족 ──► rerank ──► check_quality               │
│                                                              │
│         ▼                                                    │
│    format_output                                             │
│         └─► 최종 출력 포맷팅                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. QUESTION GENERATION: test_maker.py                       │
│                                                              │
│    select_best_schema()                                      │
│         └─► 최적 문제 유형 선택                              │
│                                                              │
│    generate_question_item()                                 │
│         ├─► 문장 생성 (필요시, 유형별 맞춤)                  │
│         └─► 문제 생성                                        │
│                                                              │
│    create_korean_test_set()                                 │
│         └─► 6개 문제 생성                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. OUTPUT: output/final_v.1.json                            │
│    생성된 한국어 학습 문제들                                 │
└─────────────────────────────────────────────────────────────┘
```

## 📂 파일별 역할 상세

### 🚀 실행 파일

#### `main_router.py` (메인 진입점)
- **역할**: 전체 파이프라인 실행
- **기능**:
  1. 리트리버 초기화
  2. 그래프 구축
  3. 사용자 쿼리 입력 받기
  4. 그래프 실행 (question_payload 직접 반환)
  5. 문제 생성 호출
  6. 최종 결과 저장

### 🕸️ 그래프 워크플로우

#### `Ragsystem/graph_agentic_router.py`
- **역할**: LangGraph 워크플로우 정의
- **노드 순서**:
  1. `analyze_query` → 쿼리 분석
  2. `routing` → 검색 전략 결정
  3. `retrieve_vocabulary` → 어휘 검색
  4. `retrieve_grammar` → 문법 검색
  5. `retrieve_kpop` → K-pop 검색 (조건부)
  6. `check_quality` → 품질 검증
  7. `rerank` → 재검색 (조건부)
  8. `generate` → 정보 추출 및 payload 구성
  9. `format_output` → 출력 포맷팅

### 🔗 노드 구현

#### `Ragsystem/nodes.py`
- **`generate_question_directly()`**: 문장 생성 없이 정보만 추출
  - 단어 5개 추출 (자연스러운 문제 생성을 위해 증가)
  - 문법 1개 추출
  - K-pop 정보 최대 5개 추출 (더 풍부한 컨텍스트 제공)
  - **동적 필터링**: kpop_filters 기반으로 필터링된 정보만 사용
  - 문제 생성용 payload 구성

#### `Ragsystem/nodes_router_intergration.py`
- **`routing_node()`**: 라우팅 결정
- **`retrieve_vocabulary_routed()`**: 라우터 기반 어휘 검색
- **`retrieve_grammar_routed()`**: 라우터 기반 문법 검색
- **`retrieve_kpop_routed()`**: 라우터 기반 K-pop 검색
  - **동적 필터링**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입으로 필터링
  - 여러 조건이 있으면 모두 만족하는 문서만 선택
- **`check_quality_agent()`**: 품질 검증
- **`rerank_node()`**: 재검색

#### `Ragsystem/router.py`
- **`IntelligentRouter`**: 검색 전략 결정
  - 쿼리 분석 기반 리트리버 활성화
  - 각 리트리버별 최적화된 쿼리 생성
  - 재검색 전략 결정

### 🔍 데이터 처리

#### `Retriever/vocabulary_retriever.py`
- **`TOPIKVocabularyRetriever`**: TOPIK 어휘 검색
- 난이도별 CSV 파일에서 단어 검색
- MMR + BM25 앙상블 검색
- 난이도 매핑:
  - `basic` → TOPIK 1-2급
  - `intermediate` → TOPIK 3-4급
  - `advanced` → TOPIK 5-6급

#### `Retriever/grammar_retriever.py`
- **`GrammarRetriever`**: 문법 패턴 검색
- 난이도별 JSON 파일에서 문법 검색
- 벡터 검색 + BM25 앙상블

#### `Retriever/kpop_retriever.py`
- **`KpopSentenceRetriever`**: K-pop 정보 검색
- **임베딩 기반 그룹명 매칭**: 멀티링구얼 임베딩으로 한글/영어 자동 매칭
- 그룹명 전용 임베딩 인덱스 (`group_name_index`) 구축
- 임계치: 0.75 (정확한 매칭 보장)
- 그룹, 멤버, 소속사, 팬덤, 컨셉 정보 추출

### 🤖 AI 에이전트

#### `agents.py`
- **`QueryAnalysisAgent`**: 쿼리 분석
  - 난이도 감지 (basic/intermediate/advanced)
    - 한국어 키워드: "초급"/"기초" → `basic`, "중급" → `intermediate`, "고급"/"상급" → `advanced`
    - 영어 키워드: "basic", "beginner", "intermediate", "middle", "advanced" 인식
    - 난이도 미지정 시 기본값: `basic`
  - 주제 추출
  - K-pop 필요성 판단
  - **동적 필터 조건 추출**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입
  - **임베딩 기반 그룹명 자동 매칭**: 하드코딩 없이 한글/영어 그룹명 자동 변환
    - 예: "아이브" → "IVE", "블랙핑크" → "BLACKPINK"
    - 임계치: 0.75 (정확한 매칭 보장)

- **`QualityCheckAgent`**: 품질 검증
  - 검색 결과 충분성 확인
  - 재검색 필요성 판단

### 🎯 문제 생성

#### `test_maker.py`
- **`select_best_schema()`**: 최적 문제 유형 선택
- **`generate_question_item()`**: 문제 생성
  - 문장이 필요한 유형: 유형별 맞춤 문장 생성
  - 문제 유형별 템플릿 사용
- **`create_korean_test_set()`**: 문제 세트 생성 (6개)

**6가지 문제 유형**:
1. `fill_in_blank`: 빈칸 채우기
2. `match_and_connect`: 문장 연결하기
3. `sentence_connection`: 문장 연결
4. `sentence_creation`: 문장 생성
5. `choice_completion`: 선택지 완성
6. `dialogue_completion`: 대화 완성

### ⚙️ 설정 및 유틸리티

#### `config.py`
- 파일 경로 설정
- LLM 설정 (temperature, max_completion_tokens)
- 리트리버 설정

#### `utils.py`
- `get_group_type()`: 그룹명으로 그룹 타입 판단 (girl_group, boy_group)
- `detect_difficulty_from_text()`: 텍스트에서 난이도 감지 (하드코딩 방식 - 효율적)
- `extract_words_from_docs()`: 문서에서 단어 추출
- `extract_grammar_with_grade()`: 문서에서 문법 추출

## 🛠️ 설치 및 실행

### 1. 환경 준비

#### Python 버전
```bash
python --version  # Python 3.8+ 필요
```

#### 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 데이터 확인

프로젝트에 다음 데이터 파일들이 있는지 확인하세요:

```
data/
├── words/          # TOPIK1.csv ~ TOPIK6.csv
├── grammar/        # grammar_list_A.json, B.json, C.json
└── kpop/          # kpop_db.json
```

### 4. 실행

```bash
python main_router.py
```

### 5. 실행 결과

**콘솔 출력:**
- 에이전트 진행상황과 생성 결과가 실시간으로 출력됩니다
- 라우터 결정 과정과 검색 전략이 표시됩니다
- 품질 검증 및 재검색 과정을 확인할 수 있습니다
- 추출된 정보 (단어, 문법, K-pop)가 표시됩니다

**파일 생성:**
- `output/final_v.1.json`에 생성된 문제들이 저장됩니다

## 📝 입출력 형식

### 입력 예시
```
"Create intermediate level Korean grammar practice questions about BLACKPINK"
"Generate advanced Korean word exercises"
"Create basic level Korean grammar practice questions"
```

### 중간 출력 (그래프 결과의 `question_payload`)
```json
{
  "level": "grade3-4",
  "target_grammar": "-(으)면서",
  "vocabulary": ["회복", "클래식", "아울러", "흔적", "희생"],
  "vocabulary_details": [
    {"word": "회복", "wordclass": "명사"},
    {"word": "클래식", "wordclass": "명사"},
    {"word": "아울러", "wordclass": "부사"},
    {"word": "흔적", "wordclass": "명사"},
    {"word": "희생", "wordclass": "명사"}
  ],
  "difficulty": "intermediate",
  "grade": 3,
  "kpop_references": [
    {
      "group": "BLACKPINK",
      "agency": "YG Entertainment",
      "fandom": "BLINK",
      "members": [
        {"name": "Jisoo", "role": "vocal"},
        {"name": "Jennie", "role": "rapper"},
        {"name": "Rosé", "role": "vocal"},
        {"name": "Lisa", "role": "rapper"}
      ],
      "concepts": ["girl crush", "hip-hop", "confidence"]
    }
  ]
}
```

### 최종 출력 (`output/final_v.1.json`)
```json
[
  {
    "schema_id": "Q_generated_1",
    "format": "fill_in_blank",
    "input": {
      "instruction": "...",
      "stem_with_blank": "...",
      "hint": "..."
    },
    "answer": {
      "completed_sentence": "..."
    },
    "rationale": "..."
  },
  ...
]
```

## 🔧 설정 (`config.py`)

### 파일 경로 설정
```python
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

KPOP_JSON_PATH = 'data/kpop/kpop_db.json'
```

### LLM 설정
```python
LLM_CONFIG = {
    'temperature': 1.0,  # gpt-5 모델은 기본값 1.0만 지원
    'max_completion_tokens': 1000,
}
```

## 🧭 지능형 라우터 기능

### 핵심 특징
- **🎯 검색 전략 자동 결정**: 쿼리 분석 결과를 바탕으로 필요한 리트리버만 선택적 실행
- **🔧 쿼리 최적화**: 각 리트리버별로 최적화된 검색 쿼리 자동 생성
- **🔄 재검색 기능**: 품질 검증 후 부족한 결과에 대해 개선된 전략으로 재검색
- **⚡ 효율성 향상**: 불필요한 검색을 줄여 실행 시간과 비용 절약
- **🔍 동적 필터링**: K-pop 정보를 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입으로 자동 필터링
- **🌐 임베딩 기반 매칭**: 하드코딩 없이 한글/영어 그룹명 자동 변환 (임계치 0.75)

### 라우터 동작 방식

#### 1. 쿼리 분석 기반 리트리버 활성화
- **Vocabulary 리트리버**: 무조건 활성화 (항상 실행)
- **Grammar 리트리버**: 문법 관련 키워드가 있을 때만 활성화
  - 키워드: "문법", "패턴", "grammar", "pattern", "표현", "구조"
- **K-pop 리트리버**: K-pop 관련 내용이 쿼리에 있을 때만 활성화
  - 키워드: "케이팝", "kpop", "k-pop", "가사", "lyrics", "노래", "song", "아이돌", "idol", "음악", "music", 그룹명 등

#### 2. 검색 전략 수립
- **우선순위 설정**: 어휘(1) → 문법(2) → K-pop(3)
- **검색 파라미터**: 난이도별 검색 개수 및 방식 조정
- **쿼리 최적화**: 각 리트리버 특성에 맞는 검색어 생성

#### 3. 품질 기반 재검색
- **품질 기준**: 어휘 5개 이상, 문법 1개 이상, K-pop 3개 이상
- **재검색 전략**: 부족한 리트리버에 대해 확장된 쿼리로 재검색
- **재시도 제한**: 최대 1회 재검색으로 무한 루프 방지

#### 4. 동적 K-pop 필터링
- **지원 필터**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입
- **자동 인식**: 쿼리에서 필터 조건 자동 추출
  - 예: "아이브 관련" → 그룹: IVE
  - 예: "걸크러시 컨셉" → 컨셉: girl crush
  - 예: "2013년 데뷔" → 데뷔 연도: 2013
  - 예: "걸그룹" → 그룹 타입: girl_group
- **임베딩 매칭**: 한글 그룹명을 영어로 자동 변환 (임계치 0.75)

## 📊 난이도 매핑

| 시스템 난이도 | TOPIK 등급 | 어휘 파일 | 문법 파일 | 한국어 키워드 |
|------------|----------|---------|---------|------------|
| `basic` | 1-2급 | TOPIK1.csv, TOPIK2.csv | grammar_list_A.json | "초급", "기초" |
| `intermediate` | 3-4급 | TOPIK3.csv, TOPIK4.csv | grammar_list_B.json | "중급" |
| `advanced` | 5-6급 | TOPIK5.csv, TOPIK6.csv | grammar_list_C.json | "고급", "상급" |

### 난이도 처리 규칙
- **난이도 미지정 시**: 기본값 `basic` 사용
- **한국어 키워드 변환**: 
  - "초급" 또는 "기초" → `basic`
  - "중급" → `intermediate`
  - "고급" 또는 "상급" → `advanced`
- **영어 키워드**: "basic", "beginner", "intermediate", "middle", "advanced"도 인식
- **문법과 단어 모두 동일한 난이도에서 추출**: 난이도가 명시되면 문법과 단어 모두 해당 난이도 그룹에서 가져옴

## 🔧 의존성

### 핵심 라이브러리
- **LangChain**: RAG 파이프라인 구축
- **LangGraph**: 워크플로우 관리
- **OpenAI**: GPT 모델 사용
- **Pandas**: 데이터 처리
- **FAISS**: 벡터 검색

### 설치된 패키지
```
langchain>=0.3,<0.4
langchain-core>=0.3,<0.4
langchain-community>=0.3,<0.4
langchain-openai>=0.2
numpy>=2.0
langgraph>=0.6
python-dotenv
pandas
openai
pydantic<3
rank_bm25
faiss-cpu>=1.8.0  # 또는 faiss-gpu
```

## 💡 주요 개선사항

### 새로운 워크플로우
- ✅ **문장 생성 단계 제거**: 정보 추출 후 바로 문제 생성
- ✅ **유형별 맞춤 문장 생성**: 각 문제 유형에 맞는 형식으로 문장 생성
- ✅ **효율성 향상**: 불필요한 문장 생성 단계 제거로 속도 향상
- ✅ **동적 필터링**: 하드코딩 없이 쿼리에서 필터 조건 자동 추출
- ✅ **임베딩 기반 매칭**: 한글/영어 그룹명 자동 변환 (임계치 0.75)
- ✅ **번역투 방지**: 영어 단어를 한국어로 자동 번역하여 자연스러운 문장 생성

### 문제 유형별 문장 생성
- `dialogue_completion`: 대화 형식 (A/B 턴)
- `match_and_connect`: 분해/재조합 가능한 문장들 (자연스러운 조사 포함)
- `sentence_connection`: 두 절로 분해 가능한 문장들
- 기타: 일반 예문들

### 정보 추출 개선
- **단어**: 3개 → 5개 (자연스러운 문제 생성을 위해)
- **K-pop**: 3개 → 최대 5개 (더 풍부한 컨텍스트 제공)
- **동적 필터링**: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입 지원

## 🐛 문제 해결

### 일반적인 오류
- **FAISS 설치 오류**: `pip install faiss-cpu` 또는 `pip install faiss-gpu` 실행
- **API 키 오류**: `.env` 파일에 올바른 OpenAI API 키가 설정되었는지 확인
- **데이터 파일 누락**: `data/` 폴더에 필요한 CSV/JSON 파일들이 있는지 확인
- **Temperature 오류**: `gpt-5` 모델은 `temperature=1.0`만 지원 (기본값)

### 디버깅
- 콘솔 출력에서 각 단계별 진행상황 확인
- 그래프 실행 결과의 `question_payload`로 추출된 정보 확인
- 에러 메시지에서 상세한 원인 확인

## 📄 라이선스

내부 프로젝트 용도로 사용되는 예시입니다. 외부 배포 시 데이터셋 저작권(K-pop 가사/문장 등)을 확인하세요.

## 🤝 기여

프로젝트 개선을 위한 제안이나 버그 리포트는 언제든 환영합니다!

---

**KFL-AQGen-AI** - 지능형 한국어 학습 문제 자동 생성 시스템 🚀
