# =====================================
# graph.py - LangGraph 워크플로우
# =====================================
"""
LangGraph 워크플로우 정의 (K-pop Context 반영)
"""
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from schema import GraphState
from nodes import KoreanLearningNodes


class KoreanLearningGraph:
    """한국어 학습 LangGraph"""
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_context, llm=None):
        """
        Args:
            vocabulary_retriever: TOPIKVocabularyRetriever 인스턴스
            grammar_retriever: GrammarRetriever 인스턴스
            kpop_context: KpopContextProvider 인스턴스
            llm: (선택) LLM 객체
        """
        self.nodes = KoreanLearningNodes(vocabulary_retriever, grammar_retriever, kpop_context, llm)
        self.workflow = None
        self.app = None
        self._build_graph()
    
    def _build_graph(self):
        """그래프 구축"""
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("detect_difficulty", self.nodes.detect_difficulty)
        workflow.add_node("retrieve_vocabulary", self.nodes.retrieve_vocabulary)
        workflow.add_node("retrieve_grammar", self.nodes.retrieve_grammar)
        workflow.add_node("generate_sentences", self.nodes.generate_sentences)
        workflow.add_node("format_output", self.nodes.format_output)
        
        # 엣지 연결
        workflow.set_entry_point("detect_difficulty")
        workflow.add_edge("detect_difficulty", "retrieve_vocabulary")
        workflow.add_edge("retrieve_vocabulary", "retrieve_grammar")
        workflow.add_edge("retrieve_grammar", "generate_sentences")
        workflow.add_edge("generate_sentences", "format_output")
        workflow.add_edge("format_output", END)
        
        # 체크포인터 설정
        memory = MemorySaver()
        
        # 컴파일
        self.workflow = workflow
        self.app = workflow.compile(checkpointer=memory)
    
    def invoke(self, input_text: str, config=None):
        """그래프 실행"""
        inputs = GraphState(
            input_text=input_text,
            difficulty_level="",
            vocabulary_docs=[],
            grammar_docs=[],
            generated_sentences=[],
            final_output="",
            messages=[]
        )
        
        result = self.app.invoke(inputs, config)
        return result['final_output']
    
    def stream(self, input_text: str, config=None):
        """그래프 스트리밍 실행"""
        inputs = GraphState(
            input_text=input_text,
            difficulty_level="",
            vocabulary_docs=[],
            grammar_docs=[],
            generated_sentences=[],
            final_output="",
            messages=[]
        )
        
        for output in self.app.stream(inputs, config):
            yield output
