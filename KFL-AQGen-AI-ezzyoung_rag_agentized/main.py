# =====================================
# main.py (Updated for Agentic RAG)
# =====================================
"""
한국어 학습 Agentic RAG 시스템 메인 실행 파일
"""
from langchain_core.runnables import RunnableConfig
from vocabulary_retriever import TOPIKVocabularyRetriever
from grammar_retriever import GrammarRetriever
# 변경: KoreanLearningGraph 대신 AgenticKoreanLearningGraph 임포트
from graph import AgenticKoreanLearningGraph
from config import TOPIK_PATHS, GRAMMAR_PATHS
import uuid
from dotenv import load_dotenv


# .env 파일의 내용을 로드
load_dotenv()

def main():
    """메인 실행 함수"""
    print("Initializing retrievers...")
    topik_retriever = TOPIKVocabularyRetriever(TOPIK_PATHS)
    grammar_retriever = GrammarRetriever(GRAMMAR_PATHS)
    
    # 변경: AgenticKoreanLearningGraph 사용
    print("Building agentic graph...")
    graph_builder = AgenticKoreanLearningGraph(topik_retriever, grammar_retriever)
    
    config = RunnableConfig(
        recursion_limit=20,
        configurable={"thread_id": str(uuid.uuid4())}
    )
    
    print("\n=== 한국어 학습 에이전트 시스템 시작 ===\n")
    
    test_queries = [
        "식당에서 쓸 만한 쉬운 한국어 문장을 공부하고 싶어."
    ]
    
    for query in test_queries:
        print(f"\n입력: {query}")
        print("-" * 50)
        # 변경: graph_builder.invoke 호출
        result = graph_builder.invoke(query, config)
        print(result)
        print("=" * 70)

if __name__ == "__main__":
    main()