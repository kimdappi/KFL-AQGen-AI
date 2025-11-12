"""
Router-Integrated Nodes for KFL-AQGen-AI
Extends AgenticKoreanLearningNodes with intelligent routing capabilities
"""

from typing import Any
import re
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
        # 하드 필터: 쿼리에 특정 그룹/멤버/컨셉/곡이 언급되면 해당되는 문서만 선택
        raw_query = state.get('input_text', '')
        q_tokens = set([t.strip().lower() for t in re.split(r"[^\w가-힣]+", raw_query) if len(t.strip()) >= 2])
        specified_groups = []
        qa = state.get('query_analysis', {})
        if qa:
            specified_groups = [g.strip() for g in qa.get('kpop_groups', []) if g.strip()]

        filtered = []
        if specified_groups:
            sg_set = {g.lower() for g in specified_groups}
            for d in kpop_db_docs:
                g = (d.metadata.get('group', '') or '').lower()
                if g in sg_set:
                    filtered.append(d)
        else:
            # 멤버/컨셉/곡 토큰 일치 시 포함
            for d in kpop_db_docs:
                group = (d.metadata.get('group', '') or '').lower()
                song = (d.metadata.get('song', '') or '').lower()
                member_names = [m.lower() for m in (d.metadata.get('member_names', []) or [])]
                concepts = [c.lower() for c in (d.metadata.get('concepts', []) or []) if isinstance(c, str)]
                fields = set()
                if group:
                    fields.add(group)
                if song:
                    fields.add(song)
                fields.update(member_names)
                fields.update(concepts)
                if any(tok in fields for tok in q_tokens):
                    filtered.append(d)

        if filtered:
            kpop_db_docs = filtered

        kpop_db_docs = kpop_db_docs[:db_limit]
        print(f"   ✅ DB 검색 완료: {len(kpop_db_docs)}개 K-pop 문장")
        
        return {
            "kpop_docs": kpop_db_docs
        }

    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드 - 간소화"""
        print("\n✅ [Agent] 품질 검증")
        
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        # 간소화된 기준: 어휘 3개, 문법 1개, K-pop 3개
        vocab_count = len(state.get('vocabulary_docs', []))
        grammar_count = len(state.get('grammar_docs', []))
        kpop_count = len(state.get('kpop_docs', []))
        
        sufficient = (vocab_count >= 3 and grammar_count >= 1)
        if needs_kpop:
            sufficient = sufficient and (kpop_count >= 3)
        
        result = {
            "sufficient": sufficient,
            "vocab_count": vocab_count,
            "grammar_count": grammar_count,
            "kpop_db_count": kpop_count,
            "needs_kpop": needs_kpop,
            "message": "충분함" if sufficient else "추가 검색 필요"
        }
        
        print(f"   어휘: {vocab_count}개 (목표 3개)")
        print(f"   문법: {grammar_count}개 (목표 1개)")
        if needs_kpop:
            print(f"   K-pop: {kpop_count}개 (목표 3개)")
        
        return {"quality_check": result}
    
    def rerank_node(self, state: GraphState) -> GraphState:
        """재검색 노드 - 간소화"""
        print("\n🔄 [재검색] 품질 개선을 위한 재검색 (1회만)")
        
        quality_check = state.get("quality_check", {})
        current_count = state.get("rerank_count", 0)
        new_count = current_count + 1
        
        # 간단한 재검색: 어휘 5개, 문법 3개, K-pop 5개 추가 검색
        level = state.get("difficulty_level", "intermediate")
        query = state.get("input_text", "")
        
        # 어휘 재검색
        if quality_check.get("vocab_count", 0) < 3:
            print(f"   📚 어휘 재검색 (현재 {quality_check.get('vocab_count')}개)")
            vocab_docs = self.vocabulary_retriever.invoke(query, level)[:5]
            state["vocabulary_docs"] = vocab_docs
        
        # 문법 재검색
        if quality_check.get("grammar_count", 0) < 1:
            print(f"   📖 문법 재검색 (현재 {quality_check.get('grammar_count')}개)")
            grammar_docs = self.grammar_retriever.invoke(query, level)[:3]
            state["grammar_docs"] = grammar_docs
        
        # K-pop 재검색 (필요시)
        if quality_check.get("needs_kpop") and quality_check.get("kpop_db_count", 0) < 3:
            print(f"   🎵 K-pop 재검색 (현재 {quality_check.get('kpop_db_count')}개)")
            kpop_docs = self.kpop_retriever.invoke(query, level)[:5]
            state["kpop_docs"] = kpop_docs
        
        print(f"   ✅ 재검색 완료 (카운터: {new_count})")
        
        return {
            "rerank_count": new_count
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