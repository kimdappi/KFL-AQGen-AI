
# 비타민 외국어→한국어 워크시트 생성기

K-POP 콘텐츠 기반 한국어 학습 워크시트 자동 생성 시스템입니다. 세대별·난이도별 맞춤 문장 생성, 문제지 구성, 품질 검증을 에이전트 구조로 제공합니다.

## 폴더/파일 구조

```
.
├── main.py                # 메인 실행 스크립트
├── config.py              # 환경설정 및 경로 관리
├── requirements.txt       # 의존성 패키지 목록
├── agents/                # 에이전트 모듈
│   ├── base_agent.py      # 에이전트 공통 베이스
│   ├── kpop_agent.py      # K-POP 문장 생성
│   ├── worksheet_agent.py # 워크시트/문제지 생성
│   ├── critic_agent.py    # 품질 검증 및 피드백
│   └── __init__.py
├── data/
│   └── schemas/
│       ├── worksheet_schema.json      # 워크시트 JSON 스키마
│       └── difficulty_levels.json    # 난이도별 세부 설정
├── output/
│   └── worksheet_intermediate_kpop_*.json # 생성된 워크시트 결과
└── README.md
```

## 주요 기능

- **KpopAgent**: 세대별 K-POP 아티스트/토픽 기반 문장 생성
- **WorksheetAgent**: 난이도·관심사별 문제지 자동 생성(JSON)
- **CriticAgent**: 문제/문장 품질 평가 및 개선 피드백
- **Schema 기반**: 모든 워크시트는 `worksheet_schema.json` 구조를 따름
- **난이도/세대별 맞춤**: beginner/intermediate/advanced, 10대~40대+ 지원

## 실행 방법

1. **패키지 설치**
   ```powershell
   pip install -r requirements.txt
   ```

2. **워크시트 생성**
   ```powershell
   python main.py
   ```

3. **결과 확인**
   - `output/worksheet_*.json` 파일에서 생성 결과 및 정답지 확인

## 설정 및 커스터마이즈

- `config.py`에서 데이터 경로, 난이도, MCP 옵션 등 조정 가능
- 난이도/세대별 상세 설정은 `data/schemas/difficulty_levels.json` 참고
- 워크시트 구조는 `data/schemas/worksheet_schema.json`에서 확인

## 출력물

- 워크시트/문제지: JSON 파일(`output/worksheet_*.json`)
- 정답지 및 메타데이터 포함
- 품질 평가 결과 및 개선 제안(에이전트별)

## 스키마 예시

`data/schemas/worksheet_schema.json` 참고

```json
{
  "worksheet": {
    "id": "string",
    "title": "string",
    "difficulty": "easy|medium|hard",
    "target_audience": { ... },
    "sections": [ ... ],
    "metadata": { ... }
  }
}
```

## 에이전트 설명

- **KpopAgent**: 세대/난이도별 K-POP 문장 10~12개 생성
- **WorksheetAgent**: 문장 기반 문제지 자동 생성(객관식/빈칸/참거짓)
- **CriticAgent**: 문제/문장 품질 평가 및 개선 피드백 제공

## 향후 개선 방향

- 실제 MCP 연동 및 K-POP 실시간 정보 활용
- PDF 템플릿 자동 생성/파싱
- 웹 기반 UI 및 API 제공
- 다국어/다문화 지원

---
문의: ezzyoung