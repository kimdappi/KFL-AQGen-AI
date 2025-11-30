"""
Agentic RAG 메인 파일 - FastAPI 버전 (안전한 초기화 버전)

- 전역에서 바로 Retriever / Graph를 만들지 않는다.
- init_resources() 안에서 한 번만 lazy 초기화하고,
  에러가 나면 서버가 죽지 않고 /generate 요청에서 500으로만 응답한다.
- /generate 엔드포인트는 query를 받아서 _run_pipeline_once를 실행하고,
  생성된 문제를 JSON으로 반환한다.
"""

import json
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse ## respone 모델 사용하지 않고 바로 호출 시도
from pydantic import BaseModel

# 프로젝트 내부 모듈
from Retriever.vocabulary_retriever import TOPIKVocabularyRetriever
from Retriever.grammar_retriever import GrammarRetriever
from Retriever.kpop_retriever import KpopSentenceRetriever

from Ragsystem.graph_agentic_router import RouterAgenticGraph
from config import TOPIK_PATHS, GRAMMAR_PATHS, KPOP_JSON_PATH
from test_maker import create_korean_test_set

load_dotenv()

# -------------------------------------------------------------------
# FastAPI 기본 세팅
# -------------------------------------------------------------------
app = FastAPI(
    title="KFL-AQGen-AI API",
    description="외국인을 위한 한국어 학습 문제 자동 생성 시스템 (FastAPI)",
    version="0.1.0",
)

# CORS: 프론트(index.html)에서 호출할 수 있게 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요하면 특정 도메인으로 제한 가능
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# 입력/출력 모델
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class GenerateResponse(BaseModel):
    query: str
    num_questions: int
    questions: List[dict]


# -------------------------------------------------------------------
# 전역 상태 (lazy 초기화)
# -------------------------------------------------------------------
topik_retriever: Optional[TOPIKVocabularyRetriever] = None
grammar_retriever: Optional[GrammarRetriever] = None
kpop_retriever: Optional[KpopSentenceRetriever] = None
graph: Optional[RouterAgenticGraph] = None

# 누적 문제 리스트 + 출력 경로
all_generated_questions: list = []
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "final_v.1.json"


def init_resources():
    """
    Retriever와 Agentic RAG Graph를 '필요할 때 한 번만' 초기화하는 함수.
    - 전역에서 바로 초기화하지 않고, 요청 시에 호출해서 에러가 나도 서버가 죽지 않도록 한다.
    """
    global topik_retriever, grammar_retriever, kpop_retriever, graph

    if graph is not None:
        # 이미 초기화되어 있으면 바로 리턴
        return

    print("\n" + "=" * 80)
    print("🚀 외국인을 위한 한국어 학습 문제 자동 생성 시스템 (FastAPI)")
    print("   KFL-AQGen-AI with Intelligent Router")
    print("=" * 80)

    print("\n📚 데이터베이스 초기화 중...")

    try:
        print("   ├─ TOPIK 어휘 데이터베이스")
        topik_retriever = TOPIKVocabularyRetriever(TOPIK_PATHS)

        print("   ├─ 문법 패턴 데이터베이스")
        grammar_retriever = GrammarRetriever(GRAMMAR_PATHS)

        print("   └─ K-pop 학습 자료 데이터베이스")
        kpop_retriever = KpopSentenceRetriever(KPOP_JSON_PATH)

        print("   ✅ 모든 데이터베이스 초기화 완료")
    except Exception as e:
        # 여기서 예외를 그대로 터뜨리면 앱이 죽으니, RuntimeError로 감싸서 위에서 500으로 처리
        print(f"❌ 리트리버 초기화 중 오류 발생: {e}")
        raise RuntimeError(f"리소스 초기화 실패(리트리버): {e}")

    print("\n🔧 지능형 라우터 기반 Agentic RAG 그래프 구축 중...")
    try:
        graph = RouterAgenticGraph(
            topik_retriever,
            grammar_retriever,
            kpop_retriever,
        )
        print("   ✅ 그래프 구축 완료")
    except Exception as e:
        print(f"❌ 그래프 구축 중 오류 발생: {e}")
        raise RuntimeError(f"리소스 초기화 실패(그래프): {e}")

    print("\n" + "=" * 80)
    print("🎯 Agentic RAG 시스템 (지능형 라우터, FastAPI 모드)")
    print("=" * 80)


def _run_pipeline_once(query: str):
    """
    한 번의 쿼리 처리 로직.

    - 리소스 초기화 (lazy)
    - 그래프 실행
    - question_payload로부터 문제 생성
    - 전역 all_generated_questions에 누적
    - output/final_v.1.json에 저장
    - 이번에 새로 생성된 문제 리스트 반환
    """
    global all_generated_questions

    query = query.strip()
    if not query:
        raise ValueError("쿼리가 비어 있습니다.")

    # 0. 리소스 초기화 (필요시)
    init_resources()

    # 설정 (요청마다 thread_id 새로 부여)
    config = RunnableConfig(
        recursion_limit=25,
        configurable={"thread_id": str(uuid.uuid4())},
    )

    print(f"\n{'=' * 80}")
    print(f"🔹 처리 중...")
    print(f"   입력: {query}")
    print("=" * 80)

    # 1. Agentic RAG 실행
    try:
        graph_result = graph.invoke(query, config)  # type: ignore[arg-type]
        rag_output_string = graph_result.get("final_output", "")
        question_payload = graph_result.get("question_payload")
        print("\n" + "=" * 80)
        print("📤 RAG 최종 출력:")
        print(rag_output_string)
        print("=" * 80)
    except Exception as e:
        print(f"❌ 그래프 실행 중 오류 발생: {e}")
        raise RuntimeError(f"그래프 실행 실패: {e}")

    # 2. question_payload 확인 및 정보 출력
    if not question_payload:
        print("❌ question_payload를 찾을 수 없습니다.")
        raise RuntimeError("question_payload가 없습니다.")

    print("\n" + "=" * 70)
    print("📋 추출된 학습 자료 정보")
    print("=" * 70)
    print(f"   학습자 수준 (등급): {question_payload.get('level')}")
    print(f"   목표 문법: {question_payload.get('target_grammar')}")

    # critique_summary
    if question_payload.get("critique_summary"):
        print(f"   생성된 예문: {len(question_payload.get('critique_summary', []))}개")
        for i, item in enumerate(question_payload.get("critique_summary", []), 1):
            print(f"      {i}. {item.get('sentence', 'N/A')}")

    # vocabulary
    if question_payload.get("vocabulary"):
        vocab_list = question_payload.get("vocabulary", [])
        vocab_details = question_payload.get("vocabulary_details", [])
        print(f"   추출된 단어: {len(vocab_list)}개")
        if vocab_details:
            for i, v in enumerate(vocab_details, 1):
                print(f"      {i}. {v.get('word', 'N/A')} ({v.get('wordclass', 'N/A')})")
        else:
            for i, v in enumerate(vocab_list, 1):
                print(f"      {i}. {v}")

    # K-pop 정보 확인
    if "kpop_references" in question_payload:
        kpop_refs = question_payload["kpop_references"] or []
        if kpop_refs:
            print(f"\n   ✨ K-pop 참조 자료: 총 {len(kpop_refs)}개")
            for i, ref in enumerate(kpop_refs, 1):
                group = ref.get("group", "N/A")
                song = ref.get("song", "")
                if song:
                    print(f"      {i}. [DB] {group} - {song}")
                else:
                    agency = ref.get("agency", "")
                    fandom = ref.get("fandom", "")
                    members = ref.get("members", [])
                    concepts = ref.get("concepts", [])

                    member_names = [
                        m.get("name", "") if isinstance(m, dict) else m
                        for m in members
                    ]
                    member_names = [n for n in member_names if n]

                    info_parts = []
                    if agency:
                        info_parts.append(f"소속사: {agency}")
                    if fandom:
                        info_parts.append(f"팬덤: {fandom}")
                    if member_names:
                        info_parts.append(f"멤버: {', '.join(member_names)}")
                    if concepts:
                        info_parts.append(f"컨셉: {', '.join(concepts)}")

                    info_str = " | ".join(info_parts) if info_parts else ""
                    print(f"      {i}. [DB] {group}" + (f" ({info_str})" if info_str else ""))

    print("=" * 70)

    # 3. 문제 생성
    print("\n🎯 한국어 학습 문제 생성 파이프라인 시작...")
    print("   Payload 확인:")
    print(f"      - level: {question_payload.get('level')}")
    print(f"      - target_grammar: {question_payload.get('target_grammar')}")
    print(f"      - vocabulary: {len(question_payload.get('vocabulary', []))}개")

    generated_questions = create_korean_test_set(
        question_payload,
        num_questions=6,
    )

    if not generated_questions:
        print("\n❌ 문제 생성 실패 - 생성된 문제가 없습니다.")
        print("   가능한 원인:")
        print("   1. LLM 호출 실패")
        print("   2. JSON 파싱 실패")
        print("   3. 모든 문제 유형에서 에러 발생")
        raise RuntimeError("문제 생성 실패")

    print("\n" + "=" * 70)
    print("✅ 생성된 한국어 학습 문제 세트 (이번 요청)")
    print("=" * 70)
    print(json.dumps(generated_questions, indent=2, ensure_ascii=False))
    print("=" * 70)

    # 4. 누적 + 저장
    all_generated_questions.extend(generated_questions)
    print(f"\n   📊 현재까지 누적된 문제 수: {len(all_generated_questions)}개")

    print("\n" + "=" * 80)
    print("💾 최종 결과 저장 중...")
    print("=" * 80)
    print(f"   생성된 문제 수(누적): {len(all_generated_questions)}개")
    print(f"   저장 파일명: {OUTPUT_PATH}")

    try:
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(all_generated_questions, f, ensure_ascii=False, indent=2)
        print(f"   ✅ '{OUTPUT_PATH}' 저장 완료")
    except Exception as e:
        print(f"   ❌ 파일 저장 실패: {e}")
        # 저장 실패해도 문제 생성은 됐으니, 에러만 찍고 진행

    print("\n" + "=" * 80)
    print("🎉 이번 쿼리에 대한 문제 생성 완료!")
    print("=" * 80 + "\n")

    return generated_questions

# -------------------------------------------------------------------
# FastAPI 엔드포인트
# -------------------------------------------------------------------
@app.post("/generate")
async def generate_questions(payload: QueryRequest):
    """
    HTML에서 쿼리를 받아서:
      1) 내부 파이프라인 실행
      2) output/final_v.1.json에 누적 저장
      3) 생성된 문제 리스트를 그대로 JSON으로 반환
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query가 비어 있습니다.")

    try:
        questions = _run_pipeline_once(query)
    except Exception as e:
        # 터미널에서도 확인하고 싶으면 여기서 print(e) 한 번 더 가능
        raise HTTPException(status_code=500, detail=str(e))

    # ★ Pydantic 모델 안 거치고, 그냥 순수 JSON으로 응답
    return JSONResponse(
        content={
            "query": query,
            "num_questions": len(questions),
            "questions": questions,
        },
    )



@app.get("/health")
async def health_check():
    """
    간단한 헬스체크 엔드포인트.
    - /docs가 떠 있는지 확인 + 서버 살아있는지 빠르게 확인용.
    """
    return {"status": "ok"}

"""
python -m uvicorn main_router_api:app --reload
"""