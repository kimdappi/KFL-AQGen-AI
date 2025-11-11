"""
라우터 통합 Agentic RAG 메인 실행 파일
기존 main.py와 동일한 출력 형식 유지
수정 완료
"""

import json
import uuid
import glob
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
# 
from Retriever.vocabulary_retriever import TOPIKVocabularyRetriever
from Retriever.grammar_retriever import GrammarRetriever
from Retriever.kpop_retriever import KpopSentenceRetriever

from Ragsystem.graph_agentic_router import RouterAgenticGraph
from config import TOPIK_PATHS, GRAMMAR_PATHS, KPOP_JSON_PATH , SENTENCE_SAVE_DIR
from test_maker import create_korean_test_set
load_dotenv()


def find_latest_sentence_file(directory=SENTENCE_SAVE_DIR):
    """지정된 디렉토리에서 가장 최근에 생성된 JSON 파일 찾기"""
    try:
        list_of_files = glob.glob(os.path.join(directory, '*.json'))
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"파일 검색 중 오류 발생: {e}")
        return None


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
            rag_output_string = graph.invoke(query, config)
            print("\n" + "="*80)
        except Exception as e:
            print(f"❌ 오류가 발생했습니다: {e}")
            continue

        # 2. 최신 JSON 파일 찾기
        latest_payload_file = find_latest_sentence_file()

        if latest_payload_file:
            print(f"\n📄 생성된 예문 파일: {latest_payload_file}")
            
            with open(latest_payload_file, 'r', encoding='utf-8') as f:
                sentence_payload = json.load(f)
            
            # Payload 검증
            print("\n" + "="*70)
            print("📋 생성된 학습 자료 정보")
            print("="*70)
            print(f"   학습자 수준 (등급): {sentence_payload.get('level')}")
            print(f"   목표 문법: {sentence_payload.get('target_grammar')}")
            print(f"   생성된 예문: {len(sentence_payload.get('critique_summary', []))}개")
            
            # 생성된 문장 출력
            for i, item in enumerate(sentence_payload.get('critique_summary', []), 1):
                print(f"      {i}. {item.get('sentence', 'N/A')}")
            
            # K-pop 정보 확인
            if 'kpop_references' in sentence_payload:
                kpop_refs = sentence_payload['kpop_references']
                db_count = len(kpop_refs)  # 모두 DB에서
                
                print(f"\n   ✨ K-pop 참조 자료: 총 {len(kpop_refs)}개")
                print(f"      - 데이터베이스: {db_count}개")
                
                for i, ref in enumerate(kpop_refs[:5], 1):
                    print(f"      {i}. [DB] {ref.get('group', 'N/A')} - {ref.get('song', 'N/A')}")
            
            print("="*70)
            
            # 3. 문제 생성
            print("\n🎯 한국어 학습 문제 생성 파이프라인 시작...")
            generated_questions = create_korean_test_set(sentence_payload, num_questions=6)

            if generated_questions:
                    print("\n" + "="*70)
                    print("✅ 생성된 한국어 학습 문제 세트")
                    print("="*70)
                    print(json.dumps(generated_questions, indent=2, ensure_ascii=False))
                    print("="*70)

                    all_generated_questions.extend(generated_questions)
            else:
                print("\n❌ 문제 생성 실패")
                
        else:
            print("\n⚠️ 'sentence' 폴더에서 JSON 파일을 찾을 수 없습니다.")


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
