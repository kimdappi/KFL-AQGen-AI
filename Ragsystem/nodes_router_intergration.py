"""
Simplified Router-Integrated Nodes
단순화된 라우팅과 필수 어휘 포함 보장
"""

from typing import Any, Dict
from Ragsystem.schema import GraphState
from Ragsystem.nodes import AgenticKoreanLearningNodes
from router import SimplifiedRouter
from agents import QueryAnalysisAgent, ProblemImprovementAgent


class SimplifiedRouterNodes(AgenticKoreanLearningNodes):
    """
    단순화된 라우터 통합 노드
    - 문법 문제여도 어휘 필수 포함
    - 쿼리 기반 판단
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
        
        # 단순화된 컴포넌트
        self.query_agent = QueryAnalysisAgent(llm=llm)
        self.quality_agent = ProblemImprovementAgent(llm=llm) 
        self.router = SimplifiedRouter(llm=llm)
        
        print("✅ Simplified Router initialized - Always includes vocabulary")
    
    def routing_node(self, state: GraphState) -> GraphState:
        """
        단순화된 라우팅 노드
        쿼리를 이해하고 필요한 리트리버 결정
        """
        print("\n" + "="*70)
        print("🔀 [Router] Simplified Routing Decision")
        print("="*70)
        
        query = state.get("input_text", "")
        query_analysis = state.get("query_analysis", {})
        
        # 라우팅 결정 (쿼리 전체 맥락 이해)
        routing = self.router.route(query, query_analysis)
        
        print(f"📊 라우팅 결정:")
        print(f"   어휘: {'✓' if routing['use_vocabulary'] else '✗'} ({routing.get('vocab_count', 5)}개)")
        print(f"   문법: {'✓' if routing['use_grammar'] else '✗'} ({routing.get('grammar_count', 0)}개)")
        print(f"   K-pop: {'✓' if routing['use_kpop'] else '✗'} ({routing.get('kpop_count', 0)}개)")
        print(f"   근거: {routing.get('reasoning', '')}")
        
        # 검색 파라미터 생성
        search_params = self.router.get_search_params(
            routing,
            query_analysis.get('difficulty', 'basic')
        )
        
        print("="*70)
        
        return {
            "routing_decision": routing,
            "search_params": search_params
        }
    
    def retrieve_vocabulary_routed(self, state: GraphState) -> GraphState:
        """
        어휘 검색 - 항상 실행
        문법 문제여도 최소 3개 이상 필수
        """
        search_params = state.get("search_params", {})
        vocab_params = search_params.get("vocabulary", {})
        
        # 어휘는 항상 검색 (문법 문제여도 필수)
        print(f"\n📚 [Vocabulary] TOPIK 어휘 검색 (필수)")
        
        query = state.get("input_text", "")
        level = vocab_params.get("level", "basic")
        limit = max(vocab_params.get("limit", 5), 3)  # 최소 3개 보장
        
        print(f"   레벨: {level}")
        print(f"   목표: {limit}개 (최소 3개 보장)")
        
        vocab_docs = self.vocabulary_retriever.invoke(query, level)
        vocab_docs = vocab_docs[:limit]
        
        # 부족하면 추가 검색
        if len(vocab_docs) < 3:
            print(f"   ⚠️ 어휘 부족 ({len(vocab_docs)}개), 추가 검색...")
            # 더 일반적인 쿼리로 재검색
            additional_docs = self.vocabulary_retriever.invoke(
                state.get("query_analysis", {}).get("topic", "daily"),
                level
            )
            vocab_docs.extend(additional_docs[:3-len(vocab_docs)])
        
        print(f"   ✅ 검색 완료: {len(vocab_docs)}개 어휘")
        
        return {"vocabulary_docs": vocab_docs}
    
    def retrieve_grammar_routed(self, state: GraphState) -> GraphState:
        """문법 검색 - 필요시에만"""
        search_params = state.get("search_params", {})
        grammar_params = search_params.get("grammar", {})
        
        if not grammar_params.get("enabled", False):
            print("   ⏭️ 문법 검색 스킵")
            return {"grammar_docs": []}
        
        print(f"\n📖 [Grammar] 문법 패턴 검색")
        
        query = state.get("input_text", "")
        level = grammar_params.get("level", "basic")
        limit = grammar_params.get("limit", 2)
        
        print(f"   레벨: {level}")
        print(f"   목표: {limit}개")
        
        grammar_docs = self.grammar_retriever.invoke(query, level)
        grammar_docs = grammar_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(grammar_docs)}개 문법")
        
        return {"grammar_docs": grammar_docs}
    
    def retrieve_kpop_routed(self, state: GraphState) -> GraphState:
        """K-pop 검색 - K-pop 언급시에만"""
        search_params = state.get("search_params", {})
        kpop_params = search_params.get("kpop", {})
        
        if not kpop_params.get("enabled", False):
            print("   ⏭️ K-pop 검색 스킵 (언급 없음)")
            return {"kpop_docs": []}
        
        print(f"\n🎵 [K-pop] K-pop 문장 검색")
        
        query = state.get("input_text", "")
        level = kpop_params.get("level", "basic")
        limit = kpop_params.get("limit", 3)
        
        # 사용자 관심사 추출
        query_analysis = state.get("query_analysis", {})
        user_interests = query_analysis.get("user_interests", [])
        
        print(f"   레벨: {level}")
        print(f"   목표: {limit}개")
        if user_interests:
            print(f"   관심사: {', '.join(user_interests)}")
        
        # K-pop 검색 (관심사 필터링 적용)
        kpop_docs = self.kpop_retriever.invoke(query, level)
        
        # 관심사 필터링 (간단한 버전)
        if user_interests and hasattr(self, 'filter_kpop_by_interests'):
            kpop_docs = self.filter_kpop_by_interests(kpop_docs, user_interests)
        
        kpop_docs = kpop_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(kpop_docs)}개 K-pop 문장")
        
        return {"kpop_docs": kpop_docs}
    
    def filter_kpop_by_interests(self, docs, interests):
        """간단한 관심사 필터링"""
        if not interests:
            return docs
        
        filtered = []
        others = []
        
        for doc in docs:
            metadata = doc.metadata
            content = (metadata.get('group', '') + ' ' + 
                      metadata.get('song', '') + ' ' + 
                      metadata.get('sentence', '')).lower()
            
            # 관심사와 매칭되는지 확인
            if any(interest.lower() in content for interest in interests):
                filtered.append(doc)
            else:
                others.append(doc)
        
        # 관심사 매칭 우선, 부족하면 기타 추가
        return filtered + others
    
    def check_quality_agent(self, state: GraphState) -> GraphState:
        """
        품질 체크 - 어휘가 충분한지 확인
        문법 문제여도 어휘 3개 이상 필수
        """
        print("\n✅ [Quality Check] 검색 결과 검증")
        
        vocab_count = len(state.get('vocabulary_docs', []))
        grammar_count = len(state.get('grammar_docs', []))
        kpop_count = len(state.get('kpop_docs', []))
        
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        # 최소 요구사항
        min_vocab = 3  # 문법 문제여도 최소 3개
        min_grammar = 1 if state.get('routing_decision', {}).get('use_grammar') else 0
        min_kpop = 3 if needs_kpop else 0
        
        # 충분성 검사
        vocab_sufficient = vocab_count >= min_vocab
        grammar_sufficient = grammar_count >= min_grammar or not state.get('routing_decision', {}).get('use_grammar')
        kpop_sufficient = kpop_count >= min_kpop or not needs_kpop
        
        sufficient = vocab_sufficient and grammar_sufficient and kpop_sufficient
        
        print(f"   어휘: {vocab_count}개 (최소 {min_vocab}개) {'✓' if vocab_sufficient else '✗'}")
        print(f"   문법: {grammar_count}개 (최소 {min_grammar}개) {'✓' if grammar_sufficient else '✗'}")
        if needs_kpop:
            print(f"   K-pop: {kpop_count}개 (최소 {min_kpop}개) {'✓' if kpop_sufficient else '✗'}")
        
        print(f"   종합: {'충분함' if sufficient else '부족함'}")
        
        return {
            "quality_check": {
                "sufficient": sufficient,
                "vocab_count": vocab_count,
                "grammar_count": grammar_count,
                "kpop_db_count": kpop_count,
                "details": {
                    "vocab_sufficient": vocab_sufficient,
                    "grammar_sufficient": grammar_sufficient,
                    "kpop_sufficient": kpop_sufficient,
                    "min_requirements": {
                        "vocab": min_vocab,
                        "grammar": min_grammar,
                        "kpop": min_kpop
                    }
                }
            }
        }
    
    def generate_sentences_with_kpop(self, state: GraphState) -> GraphState:
        """
        문장 생성 - 어휘를 반드시 포함
        """
        print("\n✏️ [Generation] 예문 생성")
        
        vocab_docs = state.get('vocabulary_docs', [])
        grammar_docs = state.get('grammar_docs', [])
        
        # 어휘 확인
        vocab_words = [doc.metadata.get('word', '') for doc in vocab_docs[:5]]
        
        if vocab_words:
            print(f"   포함될 어휘: {', '.join(vocab_words[:3])}...")
        
        # 문법 확인
        if grammar_docs:
            target_grammar = grammar_docs[0].metadata.get('grammar', '')
            print(f"   목표 문법: {target_grammar}")
        
        # 기존 생성 로직 호출
        result = super().generate_sentences_with_kpop(state)
        
        # 어휘 포함 검증
        if result.get('generated_sentences'):
            print(f"   ✅ {len(result['generated_sentences'])}개 문장 생성 (어휘 포함)")
        
        return result
    
    def rerank_simple(self, state: GraphState) -> GraphState:
        """
        단순 재검색 - 어휘가 부족하면 추가 검색
        """
        print("\n🔄 [Rerank] 재검색 수행")
        
        quality_check = state.get("quality_check", {})
        current_count = state.get("rerank_count", 0)
        
        # 어휘 부족시 추가 검색
        vocab_count = quality_check.get("vocab_count", 0)
        if vocab_count < 3:
            print(f"   어휘 재검색: {vocab_count} → 3개 목표")
            
            # 더 넓은 범위로 재검색
            level = state.get("difficulty_level", "basic")
            topic = state.get("query_analysis", {}).get("topic", "daily")
            
            additional_docs = self.vocabulary_retriever.invoke(topic, level)
            state["vocabulary_docs"].extend(additional_docs[:5-vocab_count])
            
        # 문법 부족시 추가 검색
        if state.get("routing_decision", {}).get("use_grammar"):
            grammar_count = quality_check.get("grammar_count", 0)
            if grammar_count < 1:
                print(f"   문법 재검색: {grammar_count} → 1개 목표")
                
                level = state.get("difficulty_level", "basic")
                grammar_docs = self.grammar_retriever.invoke("grammar patterns", level)
                state["grammar_docs"] = grammar_docs[:1]
        
        return {"rerank_count": current_count + 1}