"""
라우터 통합 Agentic RAG 그래프
지능형 라우팅 기능이 포함된 LangGraph 워크플로우
수정 완료
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from Ragsystem.schema import GraphState
from Ragsystem.nodes_router_intergration import RouterIntegratedNodes  


class RouterAgenticGraph:
    """
    라우터 통합 Agentic RAG 그래프
    
    워크플로우:
    1. analyze_query: 쿼리 분석 (난이도, 주제, K-pop 필요성)
    2. routing: 검색 전략 수립 (어떤 리트리버를 어떻게 사용할지)
    3. retrieve_*: 전략에 따른 선택적 검색
    4. check_quality: 검색 결과 품질 검증
    5. rerank (조건부): 품질 부족 시 재검색
    6. generate: 예문 생성
    7. format_output: 최종 출력
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.nodes = RouterIntegratedNodes(
            vocabulary_retriever,
            grammar_retriever,
            kpop_retriever,
            llm
        )
        self.workflow = None
        self.app = None
        self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # ====================================================================
        # Add Nodes
        # ====================================================================
        
        # Stage 1: Query Analysis
        workflow.add_node("analyze_query", self.nodes.analyze_query_agent)
        
        # Stage 2: Routing (NEW)
        workflow.add_node("routing", self.nodes.routing_node)
        
        # Stage 3: Retrieval (Router-based)
        workflow.add_node("retrieve_vocabulary", self.nodes.retrieve_vocabulary_routed)
        workflow.add_node("retrieve_grammar", self.nodes.retrieve_grammar_routed)
        workflow.add_node("retrieve_kpop", self.nodes.retrieve_kpop_routed)
        
        # Stage 4: Quality Check
        workflow.add_node("check_quality", self.nodes.check_quality_agent)
        
        # Stage 5: Reranking (NEW)
        workflow.add_node("rerank", self.nodes.rerank_node)
        
        # Stage 6: Generation
        workflow.add_node("generate", self.nodes.generate_sentences_with_kpop)
        
        # Stage 7: Output Formatting
        workflow.add_node("format_output", self.nodes.format_output_agentic)
        
        # ====================================================================
        # Connect Edges
        # ====================================================================
        
        # Entry point
        workflow.set_entry_point("analyze_query")
        
        # 1. Query Analysis → Routing
        workflow.add_edge("analyze_query", "routing")
        
        # 2. Routing → Sequential Retrieval
        workflow.add_edge("routing", "retrieve_vocabulary")
        workflow.add_edge("retrieve_vocabulary", "retrieve_grammar")
        workflow.add_edge("retrieve_grammar", "retrieve_kpop")
        
        # 3. K-pop Retrieval → Quality Check
        workflow.add_edge("retrieve_kpop", "check_quality")
        
        # 4. 품질 체크 → 조건부 분기
        def should_rerank(state: GraphState) -> str:
            """
            재검색 필요 여부 판단
            """
            quality = state.get('quality_check', {})
            sufficient = quality.get('sufficient', True)
            
            # 재검색 횟수 제한 (무한 루프 방지)
            rerank_count = state.get('rerank_count', 0)
            
            if not sufficient and rerank_count < 2:
                print(f"   [결정] 재검색 필요 (시도: {rerank_count + 1}/2회)")
                return "rerank"
            else:
                if sufficient:
                    print("   [결정] 품질 충족 → 문장 생성 단계로")
                else:
                    print("   [결정] 재검색 횟수 초과 → 문장 생성 단계로")
                return "generate"
        
        workflow.add_conditional_edges(
            "check_quality",
            should_rerank,
            {
                "rerank": "rerank",
                "generate": "generate"
            }
        )
        
        # 5. 재검색 → 다시 품질 체크 (카운터 증가)
        def update_rerank_count(state: GraphState) -> GraphState:
            """재검색 후 카운터 증가"""
            current_count = state.get('rerank_count', 0)
            state['rerank_count'] = current_count + 1
            return state
        
        workflow.add_edge("rerank", "check_quality")
        
        # 6. Generation → Output Formatting
        workflow.add_edge("generate", "format_output")
        
        # 7. Output Formatting → End
        workflow.add_edge("format_output", END)
        
        # ====================================================================
        # Compile
        # ====================================================================
        
        memory = MemorySaver()
        self.workflow = workflow
        self.app = workflow.compile(checkpointer=memory)
    
    def invoke(self, input_text: str, config=None):
        """그래프 워크플로우 실행"""
        inputs = GraphState(
            input_text=input_text,
            difficulty_level="",
            vocabulary_docs=[],
            grammar_docs=[],
            kpop_docs=[],
            generated_sentences=[],
            final_output="",
            messages=[],
            query_analysis={},
            quality_check={},
            routing_decision=None,
            search_strategies=[],
            rerank_count=0,  # 재검색 카운터 초기화
            rerank_decision=None
        )
        
        result = self.app.invoke(inputs, config)
        return result['final_output']
    
    def stream(self, input_text: str, config=None):
        """그래프 워크플로우 스트리밍 실행"""
        inputs = GraphState(
            input_text=input_text,
            difficulty_level="",
            vocabulary_docs=[],
            grammar_docs=[],
            kpop_docs=[],
            generated_sentences=[],
            final_output="",
            messages=[],
            query_analysis={},
            quality_check={},
            routing_decision=None,
            search_strategies=[],
            rerank_count=0,
            rerank_decision=None
        )
        
        for output in self.app.stream(inputs, config):
            yield output


def create_router_graph(vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
    """라우터 통합 그래프 생성 (편의 함수)"""
    return RouterAgenticGraph(
        vocabulary_retriever,
        grammar_retriever,
        kpop_retriever,
        llm
    )