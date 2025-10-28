"""
Simplified Router Agentic Graph
단순화된 라우팅과 필수 어휘 포함 그래프
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from Ragsystem.schema import GraphState
from .nodes_router_intergration import SimplifiedRouterNodes


class SimplifiedRouterGraph:
    """
    단순화된 라우터 통합 그래프
    
    주요 특징:
    1. 쿼리 기반 라우팅 (키워드 매칭 X)
    2. 문법 문제여도 어휘 필수 포함 (최소 3개)
    3. K-pop은 언급시에만 검색
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.nodes = SimplifiedRouterNodes(
            vocabulary_retriever,
            grammar_retriever,
            kpop_retriever,
            llm
        )
        self.workflow = None
        self.app = None
        self._build_graph()
    
    def _build_graph(self):
        """단순화된 워크플로우 구축"""
        workflow = StateGraph(GraphState)
        
        # ====================================================================
        # 노드 추가
        # ====================================================================
        
        # Stage 1: 쿼리 분석
        workflow.add_node("analyze_query", self.nodes.analyze_query_agent)
        
        # Stage 2: 라우팅 결정 (단순화)
        workflow.add_node("routing", self.nodes.routing_node)
        
        # Stage 3: 검색 (어휘는 필수)
        workflow.add_node("retrieve_vocabulary", self.nodes.retrieve_vocabulary_routed)
        workflow.add_node("retrieve_grammar", self.nodes.retrieve_grammar_routed)
        workflow.add_node("retrieve_kpop", self.nodes.retrieve_kpop_routed)
        
        # Stage 4: 품질 체크
        workflow.add_node("check_quality", self.nodes.check_quality_agent)
        
        # Stage 5: 재검색 (필요시)
        workflow.add_node("rerank", self.nodes.rerank_simple)
        
        # Stage 6: 문장 생성
        workflow.add_node("generate", self.nodes.generate_sentences_with_kpop)
        
        # Stage 7: 출력
        workflow.add_node("format_output", self.nodes.format_output_agentic)
        
        # ====================================================================
        # 엣지 연결
        # ====================================================================
        
        # Entry point
        workflow.set_entry_point("analyze_query")
        
        # 1. 분석 → 라우팅
        workflow.add_edge("analyze_query", "routing")
        
        # 2. 라우팅 → 검색 (순차적)
        workflow.add_edge("routing", "retrieve_vocabulary")  # 어휘는 항상
        workflow.add_edge("retrieve_vocabulary", "retrieve_grammar")
        workflow.add_edge("retrieve_grammar", "retrieve_kpop")
        
        # 3. 검색 → 품질 체크
        workflow.add_edge("retrieve_kpop", "check_quality")
        
        # 4. 품질 체크 → 조건부 분기
        def should_rerank(state: GraphState) -> str:
            """재검색 필요 여부 (단순 판단)"""
            quality = state.get('quality_check', {})
            sufficient = quality.get('sufficient', True)
            rerank_count = state.get('rerank_count', 0)
            
            # 어휘가 3개 미만이면 무조건 재검색
            vocab_count = quality.get('vocab_count', 0)
            if vocab_count < 3 and rerank_count < 1:
                print(f"   [결정] 어휘 부족 ({vocab_count}개) → 재검색")
                return "rerank"
            
            if not sufficient and rerank_count < 1:
                print(f"   [결정] 품질 부족 → 재검색")
                return "rerank"
            else:
                print(f"   [결정] 품질 충족 → 문장 생성")
                return "generate"
        
        workflow.add_conditional_edges(
            "check_quality",
            should_rerank,
            {
                "rerank": "rerank",
                "generate": "generate"
            }
        )
        
        # 5. 재검색 → 품질 체크 (루프)
        workflow.add_edge("rerank", "check_quality")
        
        # 6. 생성 → 출력
        workflow.add_edge("generate", "format_output")
        
        # 7. 출력 → 종료
        workflow.add_edge("format_output", END)
        
        # ====================================================================
        # 컴파일
        # ====================================================================
        
        memory = MemorySaver()
        self.workflow = workflow
        self.app = workflow.compile(checkpointer=memory)
        
        print("✅ Simplified Router Graph 구축 완료")
        print("   - 쿼리 기반 라우팅")
        print("   - 어휘 필수 포함 (최소 3개)")
        print("   - 단순화된 품질 체크")
    
    def invoke(self, input_text: str, config=None):
        """그래프 실행"""
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
            search_params={},
            rerank_count=0
        )
        
        result = self.app.invoke(inputs, config)
        return result['final_output']