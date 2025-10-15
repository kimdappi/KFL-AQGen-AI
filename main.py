# =====================================
# main.py - 메인 실행 파일
# =====================================
"""
한국어 학습 RAG 시스템 메인 실행 파일
"""
from langchain_core.runnables import RunnableConfig
from vocabulary_retriever import TOPIKVocabularyRetriever
from grammar_retriever import GrammarRetriever
from kpop_retriever import KpopContextProvider   # ✅ 변경됨
from config import TOPIK_PATHS, GRAMMAR_PATHS
from graph import KoreanLearningGraph
import uuid
from dotenv import load_dotenv
import os

# .env 파일의 내용을 로드
load_dotenv()

def main():
    """메인 실행 함수"""
    
    # Retriever / Context Provider 초기화
    print("Initializing retrievers...")
    topik_retriever = TOPIKVocabularyRetriever(TOPIK_PATHS)
    grammar_retriever = GrammarRetriever(GRAMMAR_PATHS)
    kpop_context = KpopContextProvider("/Users/jiho/Bitamin/KFL-AQGen-AI/data/kpop/kpop_db.json")   
    
    # Graph 생성
    print("Building graph...")
    graph = KoreanLearningGraph(topik_retriever, grammar_retriever, kpop_context)  
    
    # Config 설정
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": str(uuid.uuid4())}
    )
    
    # 예제 실행
    print("\n=== 한국어 학습 시스템 시작 ===\n")

    # 컨텍스트 프롬프트 테스트 (선택)
    print("🧩 예시 K-pop 문맥 프롬프트:")
    print(kpop_context.format_context_as_prompt(grammar="현재완료", vocab="춤추다"))
    print("\n" + "="*70 + "\n")

    # 테스트 질문들
    test_queries = [
        "Create basic level Korean practice questions about K-pop songs and idols",
        "중급 수준의 한국어 연습 문제를 만들어주세요, 반드시 K-pop 그룹과 관련된 문장을 포함하세요",
        "Generate advanced Korean grammar exercises using contexts from K-pop lyrics, wiki, and news"
    ]
    
    for query in test_queries:
        print(f"\n입력: {query}")
        print("-" * 50)
        result = graph.invoke(query, config)
        print(result)
        print("=" * 70)


if __name__ == "__main__":
    main()
