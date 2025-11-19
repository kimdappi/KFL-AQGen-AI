"""
라우터 통합 Agentic RAG 그래프
지능형 라우팅 기능이 포함된 LangGraph 워크플로우 (graph 구현)
수정 완료
"""

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import List
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
        self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        #노드 추가
        
        # 쿼리 분석
        workflow.add_node("analyze_query", self.nodes.analyze_query_agent)
        
        # 라우팅
        workflow.add_node("routing", self.nodes.routing_node)
        
        # 라우팅 결과로 어떤 리트리버를 활성화할 것인가
        workflow.add_node("retrieve_vocabulary", self.nodes.retrieve_vocabulary_routed)
        workflow.add_node("retrieve_grammar", self.nodes.retrieve_grammar_routed)
        workflow.add_node("retrieve_kpop", self.nodes.retrieve_kpop_routed)
        
        # 품질 체크 에이전트
        workflow.add_node("check_quality", self.nodes.check_quality_agent)
        
        # 재검색 
        workflow.add_node("rerank", self.nodes.rerank_node)
        
        # 생성 (문장 생성 없이 정보 추출 후 문제 생성)
        workflow.add_node("generate", self.nodes.generate_question_directly)
        
        # output 포맷
        workflow.add_node("format_output", self.nodes.format_output_agentic)
        

        # 엣지 연결
        # 입력
        workflow.set_entry_point("analyze_query")
        
        # 분석 -> 라우팅
        workflow.add_edge("analyze_query", "routing")
        
        # 라우팅 결과에 따라 조건부로 리트리버 실행
        def route_to_retrievers(state: GraphState) -> List[str]:
            """
            라우팅 결정에 따라 활성화된 리트리버만 반환
            병렬 실행 가능하도록 리스트 반환
            """
            decision = state.get("routing_decision")
            if not decision:
                # 기본값: vocabulary만 실행
                return ["retrieve_vocabulary"]
            
            active_retrievers = []
            strategies = decision.strategies if hasattr(decision, 'strategies') else []
            
            for strategy in strategies:
                if strategy.retriever_type.value == "vocabulary":
                    active_retrievers.append("retrieve_vocabulary")
                elif strategy.retriever_type.value == "grammar":
                    active_retrievers.append("retrieve_grammar")
                elif strategy.retriever_type.value == "kpop":
                    active_retrievers.append("retrieve_kpop")
            
            # vocabulary는 항상 포함 (라우터에서 항상 활성화)
            if "retrieve_vocabulary" not in active_retrievers:
                active_retrievers.insert(0, "retrieve_vocabulary")
            
            print(f"   🎯 활성화된 리트리버: {', '.join(active_retrievers)}")
            return active_retrievers
        
        # 조건부 병렬 실행: 라우팅 결정에 따라 필요한 리트리버만 실행
        # LangGraph 0.6+에서는 리스트 반환 시 자동으로 병렬 실행됨
        workflow.add_conditional_edges(
            "routing",
            route_to_retrievers,  # 리스트 반환 → 병렬 실행
            {
                "retrieve_vocabulary": "retrieve_vocabulary",
                "retrieve_grammar": "retrieve_grammar",
                "retrieve_kpop": "retrieve_kpop"
            }
        )
        
        # 리트리버 실행 후 모두 check_quality로 수렴
        # LangGraph는 자동으로 병렬 실행된 노드들을 수렴시킴
        # 모든 리트리버 노드가 check_quality로 연결되어 있어야 병렬 실행 후 수렴 가능
        workflow.add_edge("retrieve_vocabulary", "check_quality")
        workflow.add_edge("retrieve_grammar", "check_quality")
        workflow.add_edge("retrieve_kpop", "check_quality")
        
        # 품질 체크 → 조건부 분기
        def should_rerank(state: GraphState) -> str:
            """
            재검색 필요 여부 판단 (1회만)
            """
            quality = state.get('quality_check', {})
            sufficient = quality.get('sufficient', True)
            
            # 재검색 1회만
            rerank_count = state.get('rerank_count', 0)
            
            if not sufficient and rerank_count < 1:  # 2회에서 1회로 변경
                print(f"   [결정] 재검색 실행 (1회만)")
                return "rerank"
            else:
                if sufficient:
                    print("   [결정] 품질 충족 → 문장 생성")
                else:
                    print("   [결정] 재검색 완료 → 문장 생성")
                return "generate"
        
        workflow.add_conditional_edges(
            "check_quality",
            should_rerank,
            {
                "rerank": "rerank",
                "generate": "generate"
            }
        )
        
        # 5. 재검색 → 다시 품질 체크
        workflow.add_edge("rerank", "check_quality")
        
        # 생성
        workflow.add_edge("generate", "format_output")
        
        # 결과, 끝
        workflow.add_edge("format_output", END)
        

        # 컴파일링
        memory = MemorySaver()
        self.workflow = workflow.compile(checkpointer=memory)
    
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
            rerank_count=0,
            question_payload=None,
            sentence_data=None,
            target_grade=None
        )
        
        result = self.workflow.invoke(inputs, config)
        # final_output과 question_payload 모두 반환
        return {
            'final_output': result.get('final_output', ''),
            'question_payload': result.get('question_payload')
        }
    
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
            question_payload=None,
            sentence_data=None,
            target_grade=None
        )
        
        for output in self.workflow.stream(inputs, config):
            yield output
    
    def print_graph_structure(self):
        """그래프 구조를 텍스트로 출력"""
        print("\n" + "="*80)
        print("📊 Agentic RAG 그래프 구조")
        print("="*80)
        print("\n🔄 워크플로우:")
        print("\n1️⃣  [Entry] analyze_query")
        print("   └─> 쿼리 분석 (난이도, 주제, K-pop 필요성, 필터 조건)")
        print("   └─> QueryAnalysisAgent 사용")
        print("   └─> 출력: query_analysis, difficulty_level")
        
        print("\n2️⃣  [Node] routing")
        print("   └─> 검색 전략 수립 (어떤 리트리버를 어떻게 사용할지)")
        print("   └─> IntelligentRouter 사용")
        print("   └─> 출력: routing_decision, search_strategies")
        
        print("\n3️⃣  [Node] retrieve_vocabulary")
        print("   └─> 어휘 검색 (항상 활성화)")
        print("   └─> TOPIKVocabularyRetriever 사용")
        print("   └─> 출력: vocabulary_docs (5개)")
        
        print("\n4️⃣  [Node] retrieve_grammar")
        print("   └─> 문법 검색 (문법 관련 키워드 있을 때만 활성화)")
        print("   └─> GrammarRetriever 사용")
        print("   └─> 출력: grammar_docs (10개)")
        
        print("\n5️⃣  [Node] retrieve_kpop")
        print("   └─> K-pop 검색 (K-pop 관련 키워드 있을 때만 활성화)")
        print("   └─> KpopSentenceRetriever 사용 (DB 전용)")
        print("   └─> 동적 필터링: 그룹, 멤버, 소속사, 팬덤, 컨셉, 데뷔 연도, 그룹 타입")
        print("   └─> 출력: kpop_docs (5개)")
        
        print("\n6️⃣  [Node] check_quality")
        print("   └─> 검색 결과 품질 검증")
        print("   └─> QualityCheckAgent 사용")
        print("   └─> 기준: 어휘 3개, 문법 1개, K-pop 3개 (필요시)")
        print("   └─> 출력: quality_check")
        
        print("\n7️⃣  [Conditional] should_rerank")
        print("   ├─> 품질 충족 → generate")
        print("   └─> 품질 부족 + rerank_count < 1 → rerank")
        
        print("\n8️⃣  [Node] rerank (조건부)")
        print("   └─> 재검색 실행 (최대 1회)")
        print("   └─> 부족한 리트리버만 재검색")
        print("   └─> 출력: vocabulary_docs, grammar_docs, kpop_docs 업데이트")
        print("   └─> rerank_count 증가")
        print("   └─> → check_quality로 돌아감")
        
        print("\n9️⃣  [Node] generate")
        print("   └─> 문제 생성용 payload 구성")
        print("   └─> 문장 생성 없이 정보만 추출")
        print("   └─> 출력: question_payload, sentence_data")
        
        print("\n🔟 [Node] format_output")
        print("   └─> 최종 출력 포맷팅")
        print("   └─> 출력: final_output")
        
        print("\n⏹️  [End] END")
        print("   └─> 최종 결과 반환")
        
        print("\n" + "="*80)
        print("📋 노드 연결 구조:")
        print("="*80)
        print("""
Entry
  │
  ▼
analyze_query
  │
  ▼
routing
  │
  ├─[활성화된 리트리버만 병렬 실행]─┐
  │                                  │
  ├─> retrieve_vocabulary ──────────┤
  ├─> retrieve_grammar ──────────────┤ (병렬)
  └─> retrieve_kpop ─────────────────┘
  │
  ▼ (모든 리트리버 완료 후 수렴)
check_quality
  │
  ├─[sufficient]──────────┐
  │                        │
  └─[insufficient]        │
     │                     │
     ▼                     │
  rerank ──────────────────┘
     │                     │
     └─> check_quality ────┘
              │
              ▼
         generate
              │
              ▼
      format_output
              │
              ▼
            END
        """)
        print("="*80)
        print("\n💡 주요 특징:")
        print("   • Vocabulary 리트리버: 항상 활성화")
        print("   • Grammar 리트리버: 문법 관련 키워드 있을 때만 활성화")
        print("   • K-pop 리트리버: K-pop 관련 키워드 있을 때만 활성화")
        print("   • 병렬 실행: 활성화된 리트리버는 동시에 실행 (성능 향상)")
        print("   • 재검색: 최대 1회만 실행 (무한 루프 방지)")
        print("   • 쿼리별 중복 방지: 단어/문법 캐시로 이전 결과 제외")
        print("\n⚡ 성능:")
        print("   • 선형 실행: 3개 리트리버 = 3초")
        print("   • 병렬 실행: 3개 리트리버 = 1초 (가장 긴 것 기준)")
        print("="*80 + "\n")

