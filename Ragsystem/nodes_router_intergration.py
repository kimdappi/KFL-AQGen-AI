"""
Router-Integrated Nodes for KFL-AQGen-AI
Extends AgenticKoreanLearningNodes with intelligent routing capabilities
"""

from typing import Any
from Ragsystem.schema import GraphState
from Ragsystem.nodes import AgenticKoreanLearningNodes
from Ragsystem.router import IntelligentRouter, format_routing_summary, RetrieverType


class RouterIntegratedNodes(AgenticKoreanLearningNodes):
    """
    Router-Integrated Nodes
    Combines all features from AgenticKoreanLearningNodes + Intelligent Routing
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        # Initialize parent class (all existing features)
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
        
        # Add intelligent router
        self.router = IntelligentRouter(llm=llm)
        print("✅ [Router] IntelligentRouter initialized (DB only mode)")
    
    def routing_node(self, state: GraphState) -> GraphState:
        """
        라우팅 노드: 쿼리 분석 후 검색 전략 결정
        analyze_query_agent 노드 다음에 실행됨
        """
        print("\n" + "="*70)
        print("🔀 [라우터] 한국어 학습 자료 검색 전략 수립")
        print("="*70)
        
        # 쿼리 분석 결과 추출
        query = state.get("input_text", "")
        difficulty = state.get("difficulty_level", "intermediate")
        query_analysis = state.get("query_analysis", {})
        
        topic = query_analysis.get("topic", "")
        
        # 라우팅 결정
        decision = self.router.route(
            query=query,
            difficulty=difficulty,
            topic=topic,
            query_analysis=query_analysis
        )
        
        # 결과 출력
        print(format_routing_summary(decision))
        print("="*70)
        
        # 상태 업데이트
        return {
            "routing_decision": decision,
            "search_strategies": [s.to_dict() for s in decision.strategies]
        }
    
    def retrieve_vocabulary_routed(self, state: GraphState) -> GraphState:
        """라우터 기반 어휘 검색"""
        decision = state.get("routing_decision")
        
        # 라우팅 결정이 없으면 기본 방식 사용
        if not decision:
            print("   ⚠️ 라우팅 정보 없음, 기본 검색 실행")
            return super().retrieve_vocabulary(state)
        
        # Vocabulary 전략 찾기
        strategy = decision.get_strategy(RetrieverType.VOCABULARY)
        if not strategy:
            print("   ⏭️  어휘 검색 스킵됨 (라우터 결정)")
            return {"vocabulary_docs": []}
        
        # 전략에 따른 검색 실행
        print(f"\n📚 [어휘 검색] TOPIK 어휘 데이터베이스")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        vocab_docs = self.vocabulary_retriever.invoke(strategy.query, level)
        
        # limit 적용
        limit = strategy.params.get("limit", 10)
        vocab_docs = vocab_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(vocab_docs)}개 어휘")
        
        return {"vocabulary_docs": vocab_docs}
    
    def retrieve_grammar_routed(self, state: GraphState) -> GraphState:
        """라우터 기반 문법 검색"""
        decision = state.get("routing_decision")
        
        if not decision:
            print("   ⚠️ 라우팅 정보 없음, 기본 검색 실행")
            return super().retrieve_grammar(state)
        
        strategy = decision.get_strategy(RetrieverType.GRAMMAR)
        if not strategy:
            print("   ⏭️  문법 검색 스킵됨 (라우터 결정)")
            return {"grammar_docs": []}
        
        print(f"\n📖 [문법 검색] 한국어 문법 패턴 데이터베이스")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        grammar_docs = self.grammar_retriever.invoke(strategy.query, level)
        
        limit = strategy.params.get("limit", 5)
        grammar_docs = grammar_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(grammar_docs)}개 문법 패턴")
        
        return {"grammar_docs": grammar_docs}
    
    def retrieve_kpop_routed(self, state: GraphState) -> GraphState:
        """
        라우터 기반 K-pop 검색 (조건부 - 쿼리에 K-pop 키워드 있을 때만)
        웹 검색 없음 - 데이터베이스만 사용
        """
        decision = state.get("routing_decision")
        
        if not decision:
            print("   ⚠️ 라우팅 정보 없음")
            return {"kpop_docs": []}
        
        strategy = decision.get_strategy(RetrieverType.KPOP)
        if not strategy:
            print("   ⏭️  K-pop 검색 스킵 (쿼리에 K-pop 키워드 없음)")
            return {"kpop_docs": []}
        
        print(f"\n🎵 [K-pop 검색] 한국어 학습용 K-pop 문장 (DB 전용)")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        
        # DB 검색만 수행
        db_limit = strategy.params.get("db_limit", 5)
        kpop_db_docs = self.kpop_retriever.invoke(strategy.query, level)
        kpop_db_docs = kpop_db_docs[:db_limit]
        print(f"   ✅ DB 검색 완료: {len(kpop_db_docs)}개 K-pop 문장")
        
        return {
            "kpop_docs": kpop_db_docs
        }

    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드 - K-pop 전용 쿼리 지원"""
        print("\n✅ [Agent] 품질 검증")
        
        # ✅ 1. needs_kpop 먼저 정의!
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        # ✅ 2. kpop_only 판단
        routing_decision = state.get('routing_decision')
        kpop_only = False
        
        if routing_decision:
            active_retrievers = routing_decision.get_active_retrievers()
            # K-pop만 활성화되고 어휘/문법이 없으면 K-pop 전용
            if RetrieverType.KPOP in active_retrievers and \
            RetrieverType.VOCABULARY not in active_retrievers and \
            RetrieverType.GRAMMAR not in active_retrievers:
                kpop_only = True
        
        # ✅ 3. 품질 체크 (needs_kpop과 kpop_only 모두 전달)
        result = self.quality_agent.check(
            vocab_count=len(state.get('vocabulary_docs', [])),
            grammar_count=len(state.get('grammar_docs', [])),
            kpop_db_count=len(state.get('kpop_docs', [])),
            needs_kpop=needs_kpop,
            kpop_only=kpop_only
        )
        
        # ✅ 4. 결과 출력
        print(f"   어휘: {result['vocab_count']}개")
        print(f"   문법: {result['grammar_count']}개")
        if kpop_only:
            print(f"   K-pop DB: {result['kpop_db_count']}개 (K-pop 전용 쿼리)")
        elif needs_kpop:
            print(f"   K-pop DB: {result['kpop_db_count']}개 (필요)")
        else:
            print(f"   K-pop DB: {result['kpop_db_count']}개 (불필요)")
        print(f"   상태: {result['message']}")
        
        # ✅ 5. 결과 반환
        return {"quality_check": result}
    
    def rerank_node(self, state: GraphState) -> GraphState:
        print("\n" + "="*70)
        print("🔄 [재검색] 검색 결과 품질 분석 및 개선")
        print("="*70)
        
        quality_check = state.get("quality_check", {})
        decision = state.get("routing_decision")
        current_count = state.get("rerank_count", 0)  # ✅ 1. 현재 카운터 가져오기
        
        if not decision:
            print("   ⚠️ 라우팅 정보 없음, 재검색 스킵")
            return {}
        
        # 재검색 결정
        rerank_decision = self.router.decide_rerank(
            quality_check=quality_check,
            current_strategies=decision.strategies,
            difficulty=state.get("difficulty_level", "intermediate")
        )
        
        if not rerank_decision.should_rerank:
            print("   ✅ 품질 기준 충족, 재검색 불필요")
            return {"rerank_decision": rerank_decision}
        
        print(f"   ⚠️ 재검색 필요: {rerank_decision.reasoning}")
        print("="*70)
        
        # ✅ 2. 재검색 카운터 증가
        new_count = current_count + 1
        print(f"   🔢 재검색 카운터: {current_count} → {new_count}")
        
        # 재검색 실행
        for improved_strategy in rerank_decision.improved_strategies:
            retriever_type = improved_strategy.retriever_type
            
            if retriever_type == RetrieverType.VOCABULARY:
                print(f"\n🔁 [어휘 재검색] 개선된 검색 실행")
                print(f"   개선된 검색어: '{improved_strategy.query}'")
                
                vocab_docs = self.vocabulary_retriever.invoke(
                    improved_strategy.query,
                    improved_strategy.params.get("level", "intermediate")
                )
                limit = improved_strategy.params.get("limit", 15)
                vocab_docs = vocab_docs[:limit]
                
                print(f"   ✅ 재검색 완료: {len(vocab_docs)}개 어휘")
                state["vocabulary_docs"] = vocab_docs
            
            elif retriever_type == RetrieverType.GRAMMAR:
                print(f"\n🔁 [문법 재검색] 개선된 검색 실행")
                print(f"   개선된 검색어: '{improved_strategy.query}'")
                
                grammar_docs = self.grammar_retriever.invoke(
                    improved_strategy.query,
                    improved_strategy.params.get("level", "intermediate")
                )
                limit = improved_strategy.params.get("limit", 10)
                grammar_docs = grammar_docs[:limit]
                
                print(f"   ✅ 재검색 완료: {len(grammar_docs)}개 문법 패턴")
                state["grammar_docs"] = grammar_docs
            
            elif retriever_type == RetrieverType.KPOP:
                print(f"\n🔁 [K-pop 재검색] DB에서 추가 검색")
                
                level = improved_strategy.params.get("level", "intermediate")
                db_limit = improved_strategy.params.get("db_limit", 8)
                
                # DB 전용 검색
                kpop_db_docs = self.kpop_retriever.invoke(improved_strategy.query, level)
                kpop_db_docs = kpop_db_docs[:db_limit]
                
                print(f"   ✅ DB 재검색 완료: {len(kpop_db_docs)}개 K-pop 문장")
                state["kpop_docs"] = kpop_db_docs
        
        print("="*70)
        
        # ✅ 3. 업데이트된 카운터를 상태에 저장
        return {
            "rerank_decision": rerank_decision,
            "rerank_count": new_count  # ✅ 이 줄 추가!
        }
        
    def llm_query_rewrite_node(self, state: GraphState) -> GraphState:
        """
        LLM 기반 쿼리 재작성 노드 (고급 기능)
        재검색으로도 결과가 부족할 때 사용
        """
        print("\n🤖 [LLM 재작성] 지능형 검색어 개선")
        
        decision = state.get("routing_decision")
        quality_check = state.get("quality_check", {})
        
        if not decision:
            return {}
        
        # 2회 이상 재시도한 리트리버에 대해 LLM 재작성 시도
        for strategy in decision.strategies:
            if strategy.retry_count >= 2:  # 2회 이상 재시도 후 LLM 사용
                retriever_type = strategy.retriever_type
                
                failure_reason = f"검색 결과 부족 ({quality_check.get(f'{retriever_type.value}_count', 0)}개)"
                
                improved_query = self.router.rewrite_query_with_llm(
                    original_query=strategy.query,
                    retriever_type=retriever_type,
                    difficulty=state.get("difficulty_level", "intermediate"),
                    failure_reason=failure_reason
                )
                
                # 쿼리 업데이트
                strategy.query = improved_query
        
        return {"routing_decision": decision}