# KFL-AQGen-AI

> **한국어 학습용 문항/예문 자동 생성 시스템**  
> 단순화된 라우터 기반 Agentic RAG 시스템으로 TOPIK 어휘, 문법, K-pop 문맥을 결합해 난이도별 한국어 학습 자료를 생성합니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-green.svg)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com)

## 🚀 핵심 특징

- **🧠 단순화된 라우터**: 쿼리 전체 맥락 이해로 필요한 리트리버만 선택적 실행
- **🔄 Agentic RAG**: 질의 분석 → 지식 검색 → 품질 점검 → 생성의 자동화된 흐름
- **📚 어휘 필수 포함**: 문법 문제여도 어휘 최소 3개 이상 보장으로 의미 있는 문장 생성
- **📊 난이도 인식**: 입력에서 난이도를 자동 감지하여 TOPIK 1-6급과 연동
- **🎵 K-pop 통합**: K-pop 문맥을 활용한 재미있는 한국어 학습 자료 생성
- **📚 멀티 소스**: TOPIK 어휘, 문법 패턴, K-pop 문장의 통합 검색
- **💾 버전 관리**: 생성된 예문을 JSON으로 저장하여 추적 가능

## 📁 프로젝트 구조

```
KFL-AQGen-AI/
├── 📂 data/                           # 학습 데이터
│   ├── 📂 words/                       # TOPIK 어휘 데이터
│   │   ├── TOPIK1.csv                  # TOPIK 1급 어휘
│   │   ├── TOPIK2.csv                  # TOPIK 2급 어휘
│   │   ├── TOPIK3.csv                  # TOPIK 3급 어휘
│   │   ├── TOPIK4.csv                  # TOPIK 4급 어휘
│   │   ├── TOPIK5.csv                  # TOPIK 5급 어휘
│   │   └── TOPIK6.csv                  # TOPIK 6급 어휘
│   ├── 📂 grammar/                     # 문법 패턴 데이터
│   │   ├── grammar_list_A.json         # 기초 문법 (A급)
│   │   ├── grammar_list_B.json         # 중급 문법 (B급)
│   │   └── grammar_list_C.json         # 고급 문법 (C급)
│   └── 📂 kpop/                        # K-pop 학습 자료
│       ├── kpop_basic.csv              # 초급 K-pop 문장
│       ├── kpop_intermediate.csv       # 중급 K-pop 문장
│       └── kpop_advanced.csv           # 고급 K-pop 문장
├── 📂 sentence/                        # 생성된 예문 저장 (64개 JSON 파일)
├── 📂 __pycache__/                     # Python 캐시
├── 🤖 agents.py                        # 쿼리 분석/문제 품질 검증 에이전트
├── ⚙️ config.py                         # 프로젝트 설정
├── 📄 final_output_agentic.json        # 최종 생성된 문제
├── 📄 final_output.json                # 이전 결과
├── 🔍 grammar_retriever.py             # 문법 검색 리트리버
├── 🕸️ graph_agentic_router.py           # 🆕 단순화된 라우터 통합 그래프 (권장)
├── 🕸️ graph.py                          # 기본 Agentic RAG 그래프
├── 🎵 kpop_retriever.py                 # K-pop 검색 리트리버
├── 🚀 main_router.py                    # 🆕 라우터 통합 메인 실행 파일 (권장)
├── 🔗 nodes_router_intergration.py      # 🆕 단순화된 라우터 통합 노드 (권장)
├── 🔗 nodes.py                          # 기본 Agentic RAG 노드들
├── 📖 README.md                         # 프로젝트 문서
├── 📋 requirements.txt                  # Python 의존성
├── 🧭 router.py                         # 🆕 단순화된 라우터 (쿼리 기반 판단)
├── 📊 schema.py                         # LangGraph 상태 스키마
├── 🎯 test_maker.py                     # 문제 생성기
├── 🛠️ utils.py                          # 유틸리티 함수들
└── 📚 vocabulary_retriever.py           # TOPIK 어휘 검색 리트리버
```

## 🏗️ 아키텍처 개요

### 핵심 컴포넌트

#### 1. 🚀 실행 파일
- **`main_router.py`**: 단순화된 라우터 통합 메인 실행 파일 (권장)
- **`main.py`**: 기본 Agentic RAG 실행 파일

#### 2. 🕸️ 그래프 워크플로우
- **`graph_agentic_router.py`**: 단순화된 라우터 통합 LangGraph 워크플로우 (최신)
- **`graph.py`**: 기본 Agentic RAG 워크플로우

#### 3. 🤖 에이전트 시스템
- **`agents.py`**: 
  - `QueryAnalysisAgent`: 쿼리 분석 및 난이도/주제/K-pop 필요성 추출
  - `ProblemImprovementAgent`: 문제 품질 검증 및 개선 제안
- **`router.py`**: 단순화된 라우터 - 쿼리 기반 검색 전략 결정

#### 4. 🔗 노드 구현
- **`nodes_router_intergration.py`**: 단순화된 라우터 통합 노드 (최신, 권장)
- **`nodes.py`**: 기본 Agentic RAG 노드들

#### 5. 🔍 데이터 처리
- **`vocabulary_retriever.py`**: TOPIK 어휘 검색
- **`grammar_retriever.py`**: 문법 패턴 검색  
- **`kpop_retriever.py`**: K-pop 문장 검색

#### 6. 🎯 문제 생성 및 설정
- **`test_maker.py`**: 생성된 예문으로 연습문제 생성
- **`config.py`**: 경로 및 LLM/리트리버 설정
- **`utils.py`**: 유틸리티 함수들

### 데이터 소스
- **어휘**: `data/words/TOPIK{1..6}.csv` (TOPIK 1-6급 어휘)
- **문법**: `data/grammar/grammar_list_{A|B|C}.json` (기초/중급/고급)
- **K-pop**: `data/kpop/kpop_{basic|intermediate|advanced}.csv`
## 🔄 처리 흐름 (단순화된 라우터 통합 Agentic RAG)

### 1. 📝 쿼리 분석 단계
- **QueryAnalysisAgent**: 입력 질의에서 난이도, 주제, K-pop 필요성 추출
- 난이도: `basic`/`intermediate`/`advanced` → TOPIK 1-6급 매핑
- K-pop 연관성: K-pop 키워드, 아이돌 그룹/멤버 이름, 노래, 기획사 등으로 판단

### 2. 🧭 단순화된 라우팅 단계
- **SimplifiedRouter**: 쿼리 전체 맥락을 이해하여 검색 전략 결정
- **핵심 원칙**: 
  - 어휘는 항상 필수 (문법 문제여도 최소 3개 이상)
  - 문법은 문법 연습이 언급된 경우에만
  - K-pop은 K-pop 관련 키워드가 언급된 경우에만

### 3. 🔍 순차적 검색 단계
- **어휘 검색**: 항상 실행 (최소 3개 보장)
- **문법 검색**: 라우팅 결정에 따라 조건부 실행
- **K-pop 검색**: K-pop 언급시에만 실행
- 각 리트리버는 TOPIK/문법/K-pop 데이터베이스에서 검색

### 4. ✅ 품질 검증 및 재검색
- **품질 체크**: 어휘 3개 이상, 문법 1개 이상, K-pop 3개 이상 확인
- **재검색**: 어휘 부족시 추가 검색 (최대 1회)
- **무한 루프 방지**: 재검색 횟수 제한

### 5. 🎨 문장 생성 및 출력
- **LLM 생성**: 검색된 자료를 바탕으로 3개 예문 생성
- **JSON 저장**: `sentence/` 폴더에 메타데이터와 함께 저장
- **문제 생성**: `test_maker.py`로 최종 연습문제 생성

### 워크플로우 다이어그램
```
User Query
   │
   ▼
QueryAnalysisAgent ──► difficulty/topic/needs_kpop
   │
   ▼
SimplifiedRouter ──► 검색 전략 결정
   │
   ├─► VocabularyRetriever (항상 실행, 최소 3개)
   ├─► GrammarRetriever (조건부)  
   └─► KpopRetriever (K-pop 언급시에만)
                │
                ▼
         QualityCheck ──► 품질 검증 (어휘 3개 이상)
                │
                ├─► 충족 ──► 문장 생성
                └─► 부족 ──► 재검색 (최대 1회)
                            │
                            └─► 문장 생성
                │
                ▼
         LLM Generation ──► 3개 예문 생성
                │
                ├─► sentence/*.json 저장
                └─► test_maker → final_output_agentic.json
```

## 🛠️ 설치 및 실행

### 1. 환경 준비

#### Python 버전
```bash
python --version  # Python 3.8+ 필요
```

#### 가상환경 생성 및 활성화
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

#### 의존성 설치
```bash
pip install -r requirements.txt
```

#### 추가 설치 (FAISS)
```bash
pip install faiss-cpu  # CPU 버전
# 또는
pip install faiss-gpu  # GPU 버전 (CUDA 지원 시)
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
└── kpop/          # kpop_basic.csv, intermediate.csv, advanced.csv
```

경로는 `config.py`의 `TOPIK_PATHS`, `GRAMMAR_PATHS`, `KPOP_PATHS`에서 관리됩니다.

### 4. 실행

#### 🚀 권장 실행 방법 (단순화된 라우터 통합)
```bash
python main_router.py
```

#### 기본 실행 방법
```bash
python main.py
```

### 5. 실행 결과

**콘솔 출력:**
- 에이전트 진행상황과 생성 결과가 실시간으로 출력됩니다
- 단순화된 라우터 결정 과정과 검색 전략이 표시됩니다
- 품질 검증 및 재검색 과정을 확인할 수 있습니다

**파일 생성:**
- `sentence/` 폴더에 최신 예문 JSON이 저장됩니다
- `final_output_agentic.json`에 최종 연습문제가 저장됩니다
- 단순화된 라우터를 통해 검색 효율성이 향상됩니다

## 🧭 단순화된 라우터 기능

### 핵심 특징
- **🎯 쿼리 기반 판단**: 쿼리 전체 맥락을 이해하여 검색 전략 결정
- **📚 어휘 필수 포함**: 문법 문제여도 어휘 최소 3개 이상 보장
- **🔄 단순한 재검색**: 어휘 부족시에만 추가 검색 (최대 1회)
- **⚡ 효율성 향상**: 복잡한 전략 제거로 실행 시간 단축

### 라우터 동작 방식

#### 1. 쿼리 분석 기반 리트리버 활성화
```python
# 쿼리 전체 맥락 이해
- 어휘: 항상 활성화 (최소 3개 보장)
- 문법: 문법 연습이 언급된 경우에만
- K-pop: K-pop 관련 키워드가 언급된 경우에만
```

#### 2. 검색 전략 수립
- **어휘 우선**: 모든 문제에 어휘 필수 포함
- **조건부 검색**: 문법과 K-pop은 필요시에만
- **순차적 실행**: 어휘 → 문법 → K-pop 순서로 검색

#### 3. 품질 기반 재검색
- **품질 기준**: 어휘 3개 이상, 문법 1개 이상, K-pop 3개 이상
- **재검색 전략**: 어휘 부족시에만 추가 검색
- **재시도 제한**: 최대 1회 재검색으로 무한 루프 방지

### 사용 예시

#### 기본 쿼리
```
"Create basic level Korean practice questions"
```
→ 어휘 (필수) + 문법 리트리버 활성화 (K-pop 제외)

#### K-pop 관련 쿼리  
```
"Create intermediate Korean questions about BTS"
```
→ 어휘 (필수) + 문법 + K-pop 리트리버 모두 활성화

#### 문법 중심 쿼리
```
"Generate advanced Korean grammar exercises with -는 patterns"
```
→ 어휘 (필수) + 문법 리트리버 활성화 (K-pop 제외)

### 성능 개선 효과
- **⏱️ 검색 시간 단축**: 단순화된 로직으로 빠른 실행
- **🎯 결과 품질 향상**: 어휘 필수 포함으로 의미 있는 문장 생성
- **💰 비용 절약**: 최소한의 재검색으로 API 비용 감소
- **🛡️ 안정성 향상**: 단순한 재검색 로직으로 무한 루프 방지

## 📊 구성요소 상세

### 상태 스키마 (`schema.py`)
- `GraphState`에 다음 필드 포함: `input_text`, `difficulty_level`, `vocabulary_docs`, `grammar_docs`, `kpop_docs`, `generated_sentences`, `final_output`, `messages`, `sentence_data`, `target_grade`, `query_analysis`, `quality_check`, `routing_decision`, `search_strategies`, `rerank_count`, `rerank_decision`

### 에이전트 (`agents.py`)
- **QueryAnalysisAgent**: 쿼리 분석 및 난이도/주제/K-pop 필요성 추출
- **ProblemImprovementAgent**: 문제 품질 검증 및 개선 제안
- **품질 평가**: 5가지 기준으로 문제 적합성 평가 (0-100점)

### 노드 (`nodes_router_intergration.py`)
- **단순화된 라우터 통합 노드**: `routing_node` → `retrieve_*_routed` → `check_quality_agent` → `rerank_simple` → `generate_sentences_with_kpop` → `format_output_agentic`
- **재검색 기능**: 어휘 부족시에만 추가 검색 (최대 1회)
- **어휘 필수 포함**: 문법 문제여도 어휘 최소 3개 이상 보장

### 그래프 (`graph_agentic_router.py`)
- LangGraph `StateGraph`로 노드 연결, `MemorySaver` 체크포인트 사용
- `invoke()`는 입력을 초기 `GraphState`로 만들어 실행하고 최종 문자열을 반환
- 단순화된 조건부 분기로 재검색 필요성 판단

### 메인 파이프라인 (`main_router.py`)
- 리트리버 초기화 → 단순화된 라우터 통합 Agentic 그래프 실행 → 최신 `sentence/*.json` 로드 → `test_maker.create_korean_test_from_payload()`로 문제 생성/저장

## ⚙️ 설정 (`config.py`)

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

KPOP_PATHS = {
    'basic': ['data/kpop/kpop_basic.csv'],
    'intermediate': ['data/kpop/kpop_intermediate.csv'],
    'advanced': ['data/kpop/kpop_advanced.csv']
}
```

### 리트리버 설정
```python
RETRIEVER_CONFIG = {
    'top_k': 10,
    'ensemble_weights': [0.5, 0.5],
    'vector_search_type': 'similarity',
}
```

### LLM 설정
```python
LLM_CONFIG = {
    'temperature': 0.7,
    'max_tokens': 1000,
}
```

## 📝 입출력 형식

### 입력 예시
```
"Create **middle** level Korean practice questions about K-pop"
"Generate **advanced** Korean word exercises about k-pop"
"Create **basic** level Korean grammar practice questions about blackpink"
```

### `sentence/*.json` 예시 필드
```json
{
  "level": 3,
  "target_grammar": "~(으)면서",
  "kpop_references": [
    { 
      "sentence": "...", 
      "song": "...", 
      "group": "...", 
      "source": "database" 
    }
  ],
  "critique_summary": [
    { "sentence": "생성된 예문 1" }, 
    { "sentence": "생성된 예문 2" }, 
    { "sentence": "생성된 예문 3" }
  ]
}
```

### 최종 문제 (`final_output_agentic.json`)
- `test_maker.py`가 `sentence` 페이로드를 받아 생성한 문제 리스트 저장

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
faiss-cpu  # 또는 faiss-gpu
```

## 💡 팁

### 성능 최적화
- **Windows 경로 제한**: 파일명은 `sanitize_filename`으로 안전 처리됩니다
- **LLM 비용/속도**: `config.py`의 `LLM_CONFIG`와 모델을 상황에 맞게 조정하세요
- **재검색 제한**: 최대 2회로 설정되어 무한 루프를 방지합니다

### 문제 해결
- **FAISS 설치 오류**: `pip install faiss-cpu` 또는 `pip install faiss-gpu` 실행
- **API 키 오류**: `.env` 파일에 올바른 OpenAI API 키가 설정되었는지 확인
- **데이터 파일 누락**: `data/` 폴더에 필요한 CSV/JSON 파일들이 있는지 확인

## 📄 라이선스

내부 프로젝트 용도로 사용되는 예시입니다. 외부 배포 시 데이터셋 저작권(K-pop 가사/문장 등)을 확인하세요.

## 🤝 기여

프로젝트 개선을 위한 제안이나 버그 리포트는 언제든 환영합니다!

---

**KFL-AQGen-AI** - 단순화된 라우터 기반 한국어 학습 자료 생성 시스템 🚀