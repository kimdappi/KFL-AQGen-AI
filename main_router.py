"""
Agentic RAG 메인 파일
"""

import json
import uuid
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
# 
from Retriever.vocabulary_retriever import TOPIKVocabularyRetriever
from Retriever.grammar_retriever import GrammarRetriever
from Retriever.kpop_retriever import KpopSentenceRetriever

from Ragsystem.graph_agentic_router import RouterAgenticGraph
from config import TOPIK_PATHS, GRAMMAR_PATHS, KPOP_JSON_PATH
from test_maker import create_korean_test_set

load_dotenv()


def main():
    """메인 실행 함수 (라우터 통합 버전)"""
    
    print("\n" + "="*80)
    print("🚀 외국인을 위한 한국어 학습 문제 자동 생성 시스템")
    print("   KFL-AQGen-AI with Intelligent Router")
    print("="*80)
    
    # 리트리버 초기화
    print("\n📚 데이터베이스 초기화 중...")
    print("   ├─ TOPIK 어휘 데이터베이스")
    topik_retriever = TOPIKVocabularyRetriever(TOPIK_PATHS)
    print("   ├─ 문법 패턴 데이터베이스")
    grammar_retriever = GrammarRetriever(GRAMMAR_PATHS)
    print("   └─ K-pop 학습 자료 데이터베이스")
    kpop_retriever = KpopSentenceRetriever(KPOP_JSON_PATH)
    print("   ✅ 모든 데이터베이스 초기화 완료")
    
    # 라우터 통합 Agentic RAG 그래프 구축
    print("\n🔧 지능형 라우터 기반 Agentic RAG 그래프 구축 중...")
    graph = RouterAgenticGraph(
        topik_retriever,
        grammar_retriever,
        kpop_retriever
    )
    print("   ✅ 그래프 구축 완료")
    
    # 설정
    config = RunnableConfig(
        recursion_limit=25,  # 재검색을 위해 약간 증가
        configurable={"thread_id": str(uuid.uuid4())}
    )
    
    print("\n" + "="*80)
    print("🎯 Agentic RAG 시스템 시작 (지능형 라우터)")
    print("="*80)
    
    all_generated_questions = []
    
    
    # 사용자 입력 받기
    print("\n💬 한국어 학습 문제를 생성할 쿼리를 입력하세요.")
    print("   예시: 'Create intermediate level Korean grammar practice questions about BLACKPINK'")
    print("   종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        query = input("📝 쿼리 입력: ").strip()
        
        if query.lower() in ['quit', 'exit', '종료', '그만']:
            print("\n👋 프로그램을 종료합니다.")
            break
        
        if not query:
            print("   ⚠️ 쿼리를 입력해주세요.\n")
            continue
        
        print(f"\n{'='*80}")
        print(f"🔹 처리 중...")
        print(f"   입력: {query}")
        print('='*80)

        # 1. 라우터 기반 Agentic RAG 실행
        try:
            graph_result = graph.invoke(query, config)
            rag_output_string = graph_result.get('final_output', '')
            question_payload = graph_result.get('question_payload')
            print("\n" + "="*80)
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
            continue

        # 2. question_payload 확인 및 정보 출력
        if not question_payload:
            print("❌ question_payload를 찾을 수 없습니다.")
            continue

        print("\n" + "="*70)
        print("📋 추출된 학습 자료 정보")
        print("="*70)
        print(f"   학습자 수준 (등급): {question_payload.get('level')}")
        print(f"   목표 문법: {question_payload.get('target_grammar')}")
        
        # 기존 형식 호환성 체크 (critique_summary가 있으면 표시)
        if 'critique_summary' in question_payload and question_payload.get('critique_summary'):
            print(f"   생성된 예문: {len(question_payload.get('critique_summary', []))}개")
            for i, item in enumerate(question_payload.get('critique_summary', []), 1):
                print(f"      {i}. {item.get('sentence', 'N/A')}")
        # 새로운 형식 (정보만 추출)
        if 'vocabulary' in question_payload and question_payload.get('vocabulary'):
            vocab_list = question_payload.get('vocabulary', [])
            vocab_details = question_payload.get('vocabulary_details', [])
            print(f"   추출된 단어: {len(vocab_list)}개")
            if vocab_details:
                # 모든 단어 출력 (5개)
                for i, v in enumerate(vocab_details, 1):
                    print(f"      {i}. {v.get('word', 'N/A')} ({v.get('wordclass', 'N/A')})")
            elif vocab_list:
                # 모든 단어 출력 (5개)
                for i, v in enumerate(vocab_list, 1):
                    print(f"      {i}. {v}")
        
        # K-pop 정보 확인
        if 'kpop_references' in question_payload:
            kpop_refs = question_payload['kpop_references']
            
            if kpop_refs:
                print(f"\n   ✨ K-pop 참조 자료: 총 {len(kpop_refs)}개")
                
                for i, ref in enumerate(kpop_refs, 1):
                    group = ref.get('group', 'N/A')
                    song = ref.get('song', '')
                    if song:
                        print(f"      {i}. [DB] {group} - {song}")
                    else:
                        # 새로운 형식 - 모든 정보 표시
                        agency = ref.get('agency', '')
                        fandom = ref.get('fandom', '')
                        members = ref.get('members', [])
                        concepts = ref.get('concepts', [])
                        
                        # 모든 멤버 이름 추출
                        member_names = [m.get('name', '') if isinstance(m, dict) else m for m in members]
                        member_names = [n for n in member_names if n]  # 빈 문자열 제거
                        
                        # 정보 구성
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
        
        print("="*70)
        
        # 3. 문제 생성
        print("\n🎯 한국어 학습 문제 생성 파이프라인 시작...")
        print(f"   Payload 확인:")
        print(f"      - level: {question_payload.get('level')}")
        print(f"      - target_grammar: {question_payload.get('target_grammar')}")
        print(f"      - vocabulary: {len(question_payload.get('vocabulary', []))}개")
        
        generated_questions = create_korean_test_set(question_payload, num_questions=6)

        if generated_questions:
            print("\n" + "="*70)
            print("✅ 생성된 한국어 학습 문제 세트")
            print("="*70)
            print(json.dumps(generated_questions, indent=2, ensure_ascii=False))
            print("="*70)

            all_generated_questions.extend(generated_questions)
            print(f"\n   📊 현재까지 누적된 문제 수: {len(all_generated_questions)}개")
        else:
            print("\n❌ 문제 생성 실패 - 생성된 문제가 없습니다.")
            print("   가능한 원인:")
            print("   1. LLM 호출 실패")
            print("   2. JSON 파싱 실패")
            print("   3. 모든 문제 유형에서 에러 발생")
            print("   위의 에러 메시지를 확인해주세요.")


        print("\n" + "="*80)
    
    # 최종 출력 저장
    output_filename = "output/final_v.1.json"
    print(f"\n{'='*80}")
    print(f"💾 최종 결과 저장 중...")
    print(f"{'='*80}")
    print(f"   생성된 문제 수: {len(all_generated_questions)}개")
    print(f"   저장 파일명: {output_filename}")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_generated_questions, f, ensure_ascii=False, indent=2)
        print(f"   ✅ '{output_filename}' 저장 완료")
    except Exception as e:
        print(f"   ❌ 파일 저장 실패: {e}")

    print("\n" + "="*80)
    print("🎉 모든 작업 완료!")
    print("   외국인을 위한 한국어 학습 문제가 성공적으로 생성되었습니다.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()