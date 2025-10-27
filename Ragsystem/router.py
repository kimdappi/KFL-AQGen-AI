"""
KFL-AQGen-AI용 지능형 라우터
Agentic RAG 시스템을 위한 검색 전략 결정 및 쿼리 최적화
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI


class RetrieverType(Enum):
    """사용 가능한 리트리버 타입"""
    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    KPOP = "kpop"


@dataclass
class SearchStrategy:
    """개별 리트리버의 검색 전략"""
    retriever_type: RetrieverType
    query: str  # 검색에 사용할 최적화된 쿼리
    priority: int  # 실행 우선순위 (낮을수록 먼저 실행)
    params: Dict[str, Any]  # 추가 검색 파라미터
    retry_count: int = 0  # 재검색 재시도 횟수
    
    def to_dict(self) -> Dict:
        return {
            "retriever": self.retriever_type.value,
            "query": self.query,
            "priority": self.priority,
            "params": self.params,
            "retry_count": self.retry_count
        }


@dataclass
class RoutingDecision:
    """라우팅 결정 결과"""
    strategies: List[SearchStrategy]
    reasoning: str
    confidence: float
    needs_quality_check: bool = True
    
    def get_active_retrievers(self) -> Set[RetrieverType]:
        """활성화된 리트리버 타입 반환"""
        return {s.retriever_type for s in self.strategies}
    
    def get_strategy(self, retriever_type: RetrieverType) -> Optional[SearchStrategy]:
        """특정 리트리버의 전략 반환"""
        for strategy in self.strategies:
            if strategy.retriever_type == retriever_type:
                return strategy
        return None


@dataclass
class RerankDecision:
    """재검색 결정"""
    should_rerank: bool
    target_retrievers: List[RetrieverType]
    improved_strategies: List[SearchStrategy]
    reasoning: str


class IntelligentRouter:
    """
    지능형 라우터: 검색 전략 결정, 재검색, 쿼리 최적화
    
    핵심 역할:
    - 쿼리 분석 및 활성화할 리트리버 결정
    - 각 리트리버에 최적화된 검색 쿼리 생성
    - 품질 체크 기반 재검색 필요성 판단
    - 필요 시 LLM 사용한 쿼리 개선
    """
    
    # 리트리버 활성화 판단용 트리거 키워드 (ON/OFF 스위치)
    VOCABULARY_TRIGGERS = {
        "단어", "어휘", "vocabulary", "word", "voca", "TOPIK",
        "명사", "동사", "형용사", "부사"
    }
    
    GRAMMAR_TRIGGERS = {
        "문법", "패턴", "grammar", "pattern", "표현", "구조"
    }
    
    KPOP_TRIGGERS = {
        "케이팝", "kpop", "k-pop", "가사", "lyrics", "노래", "song",
        "아이돌", "idol", "음악", "music",
        "bts", "blackpink", "뉴진스", "newjeans", "아이브", "ive"
    }
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def route(
        self,
        query: str,
        difficulty: str,
        topic: str = "",
        query_analysis: Optional[Dict] = None
    ) -> RoutingDecision:
        """
        초기 라우팅 결정
        
        Args:
            query: 사용자 쿼리
            difficulty: QueryAnalysisAgent가 결정한 난이도
            topic: 추출된 주제
            query_analysis: QueryAnalysisAgent의 전체 분석 결과
        
        Returns:
            검색 전략이 포함된 RoutingDecision
        """
        query_lower = query.lower()
        topic_lower = topic.lower()
        
        strategies = []
        reasons = []
        
        # ✅ 문제 생성 쿼리 감지: "questions", "practice", "문제", "연습" 등
        is_question_generation = any(keyword in query_lower for keyword in [
            "question", "practice", "exercise", "문제", "연습", "생성", "만들"
        ])
        
        # 1. Vocabulary Retriever Activation
        # 문제 생성 쿼리면 항상 활성화, 그 외에는 키워드 체크
        if is_question_generation or self._should_activate_vocabulary(query_lower, topic_lower):
            vocab_query = self._extract_vocab_query(query, topic, difficulty)
            strategies.append(SearchStrategy(
                retriever_type=RetrieverType.VOCABULARY,
                query=vocab_query,
                priority=1,
                params={"level": difficulty, "limit": 5}
            ))
            reasons.append(f"Vocabulary search (query: '{vocab_query}', level: {difficulty})")
        
        # 2. Grammar Retriever Activation
        # 문제 생성 쿼리면 항상 활성화, 그 외에는 키워드 체크
        if is_question_generation or self._should_activate_grammar(query_lower, topic_lower):
            grammar_query = self._extract_grammar_query(query, topic, difficulty)
            strategies.append(SearchStrategy(
                retriever_type=RetrieverType.GRAMMAR,
                query=grammar_query,
                priority=2,
                params={"level": difficulty, "limit": 5}
            ))
            reasons.append(f"Grammar search (query: '{grammar_query}', level: {difficulty})")
        
        # 3. K-pop Retriever Activation (DB only)
        if self._should_activate_kpop(query_lower, topic_lower):
            kpop_query = self._extract_kpop_query(query, topic)
            strategies.append(SearchStrategy(
                retriever_type=RetrieverType.KPOP,
                query=kpop_query,
                priority=3,
                params={"level": difficulty, "db_limit": 5}
            ))
            reasons.append(f"K-pop search (DB only, query: '{kpop_query}')")
        
        # Default strategy: At least grammar + vocabulary
        if not strategies:
            strategies.extend([
                SearchStrategy(
                    RetrieverType.VOCABULARY,
                    query=query,
                    priority=1,
                    params={"level": difficulty, "limit": 5}
                ),
                SearchStrategy(
                    RetrieverType.GRAMMAR,
                    query=query,
                    priority=2,
                    params={"level": difficulty, "limit": 5}
                )
            ])
            reasons.append("Default strategy: vocabulary + grammar")
        
        # Sort by priority
        strategies.sort(key=lambda x: x.priority)
        
        reasoning = " | ".join(reasons)
        confidence = self._calculate_confidence(strategies, query, topic)
        
        return RoutingDecision(
            strategies=strategies,
            reasoning=reasoning,
            confidence=confidence,
            needs_quality_check=True
        )
    
    def decide_rerank(
        self,
        quality_check: Dict[str, Any],
        current_strategies: List[SearchStrategy],
        difficulty: str
    ) -> RerankDecision:
        """
        품질 체크 결과 기반 재검색 필요성 판단
        
        Args:
            quality_check: QualityCheckAgent의 결과
            current_strategies: 현재 사용된 검색 전략들
            difficulty: 난이도
        
        Returns:
            RerankDecision
        """
        print("\n🔄 [Router] Analyzing rerank necessity...")
        
        sufficient = quality_check.get('sufficient', True)
        vocab_count = quality_check.get('vocab_count', 0)
        grammar_count = quality_check.get('grammar_count', 0)
        total_kpop = quality_check.get('total_kpop', 0)
        
        should_rerank = False
        target_retrievers = []
        improved_strategies = []
        reasons = []
        
        if not sufficient:
            # 1. Vocabulary insufficient (< 5)
            if vocab_count < 5:
                should_rerank = True
                target_retrievers.append(RetrieverType.VOCABULARY)
                
                vocab_strategy = self._find_strategy(current_strategies, RetrieverType.VOCABULARY)
                if vocab_strategy:
                    expanded_query = self._expand_query(vocab_strategy.query, difficulty)
                    improved_strategies.append(SearchStrategy(
                        retriever_type=RetrieverType.VOCABULARY,
                        query=expanded_query,
                        priority=1,
                        params={"level": difficulty, "limit": 15},
                        retry_count=vocab_strategy.retry_count + 1
                    ))
                    reasons.append(f"Vocabulary insufficient ({vocab_count}/5) → '{expanded_query}'")
            
            # 2. Grammar insufficient (< 1)
            if grammar_count < 1:
                should_rerank = True
                target_retrievers.append(RetrieverType.GRAMMAR)
                
                grammar_strategy = self._find_strategy(current_strategies, RetrieverType.GRAMMAR)
                if grammar_strategy:
                    improved_query = self._improve_grammar_query(grammar_strategy.query, difficulty)
                    improved_strategies.append(SearchStrategy(
                        retriever_type=RetrieverType.GRAMMAR,
                        query=improved_query,
                        priority=2,
                        params={"level": difficulty, "limit": 5},
                        retry_count=grammar_strategy.retry_count + 1
                    ))
                    reasons.append(f"Grammar insufficient ({grammar_count}/1) → '{improved_query}'")
            
            # 3. K-pop insufficient (< 5) - DB only
            if total_kpop < 5:
                should_rerank = True
                target_retrievers.append(RetrieverType.KPOP)
                
                kpop_strategy = self._find_strategy(current_strategies, RetrieverType.KPOP)
                if kpop_strategy:
                    improved_strategies.append(SearchStrategy(
                        retriever_type=RetrieverType.KPOP,
                        query=kpop_strategy.query,
                        priority=3,
                        params={"level": difficulty, "db_limit": 8},
                        retry_count=kpop_strategy.retry_count + 1
                    ))
                    reasons.append(f"K-pop insufficient ({total_kpop}/5) → More from DB")
        
        reasoning = " | ".join(reasons) if reasons else "Quality criteria met"
        
        print(f"   Rerank needed: {'Yes' if should_rerank else 'No'}")
        if should_rerank:
            print(f"   Targets: {[r.value for r in target_retrievers]}")
            print(f"   Reasoning: {reasoning}")
        
        return RerankDecision(
            should_rerank=should_rerank,
            target_retrievers=target_retrievers,
            improved_strategies=improved_strategies,
            reasoning=reasoning
        )
    
    def rewrite_query_with_llm(
        self, 
        original_query: str, 
        retriever_type: RetrieverType,
        difficulty: str, 
        failure_reason: str = ""
    ) -> str:
        """LLM을 사용하여 쿼리를 지능적으로 재작성"""
        print(f"\n🤖 [Router] LLM 기반 쿼리 개선 중 ({retriever_type.value})...")
        
        retriever_purpose = {
            "vocabulary": "외국인 학습자에게 적합한 한국어 어휘",
            "grammar": "해당 난이도에 맞는 한국어 문법 패턴",
            "kpop": "한국어 학습에 활용 가능한 K-pop 관련 문장"
        }
        
        difficulty_desc = {
            "basic": "초급 (TOPIK 1-2급): 기본 어휘와 간단한 문법",
            "intermediate": "중급 (TOPIK 3-4급): 다양한 표현과 일상 대화",
            "advanced": "고급 (TOPIK 5-6급): 복잡한 문법과 추상적 개념"
        }
        
        prompt = f"""당신은 외국인을 위한 한국어 교육 자료 검색 전문가입니다.
다음 검색 쿼리를 개선하여 더 나은 한국어 학습 자료를 찾을 수 있도록 도와주세요.

**현재 상황:**
- 원본 쿼리: "{original_query}"
- 검색 대상: {retriever_purpose.get(retriever_type.value, "한국어 학습 자료")}
- 학습자 수준: {difficulty_desc.get(difficulty, "일반")}
{f"- 검색 실패 이유: {failure_reason}" if failure_reason else ""}

**개선 목표:**
1. {retriever_type.value} 데이터베이스 검색에 최적화
2. {difficulty} 수준에 적합한 키워드 포함
3. 외국인 학습자 관점에서 유용한 내용
4. 실용적이고 자연스러운 한국어 표현 중심

**형식:**
- 1-6단어의 간결한 검색어
- 한국어와 영어 키워드 적절히 혼합
- 구체적이고 명확한 표현 사용

개선된 검색 쿼리만 출력하세요 (설명 없이):
"""
        
        try:
            response = self.llm.predict(prompt)
            improved_query = response.strip().strip('"').strip("'")
            print(f"   원본: '{original_query}'")
            print(f"   개선: '{improved_query}'")
            return improved_query
        except Exception as e:
            print(f"   ⚠️ LLM 호출 실패: {e}")
            return original_query
    
    # 헬퍼 메서드들
    def _should_activate_vocabulary(self, query: str, topic: str) -> bool:
        """Vocabulary 리트리버 활성화 여부 확인"""
        if any(kw in query for kw in self.VOCABULARY_TRIGGERS):
            return True
        if any(kw in topic for kw in self.VOCABULARY_TRIGGERS):
            return True
        if "예문" in query or "문장" in query or "sentence" in query:
            return True
        return False
    
    def _should_activate_grammar(self, query: str, topic: str) -> bool:
        """Grammar 리트리버 활성화 여부 확인"""
        if any(kw in query for kw in self.GRAMMAR_TRIGGERS):
            return True
        if any(kw in topic for kw in self.GRAMMAR_TRIGGERS):
            return True
        if "예문" in query or "문장" in query or "sentence" in query:
            return True
        return False
    
    def _should_activate_kpop(self, query: str, topic: str) -> bool:
        """K-pop 리트리버 활성화 여부 확인"""
        if any(kw in query for kw in self.KPOP_TRIGGERS):
            return True
        if any(kw in topic for kw in self.KPOP_TRIGGERS):
            return True
        return False
    
    def _extract_vocab_query(self, query: str, topic: str, difficulty: str) -> str:
        """어휘 검색용 최적화된 쿼리 생성"""
        if topic and len(topic.split()) <= 3:
            return f"{topic} {difficulty}"
        
        for kw in self.VOCABULARY_TRIGGERS:
            if kw in query.lower():
                clean_query = query.lower().replace(kw, "").strip()
                return f"{clean_query[:20]} {difficulty}"
        
        return f"{query[:20]} {difficulty}"
    
    def _extract_grammar_query(self, query: str, topic: str, difficulty: str) -> str:
        """문법 검색용 최적화된 쿼리 생성"""
        import re
        
        pattern = r'-[가-힣ㄱ-ㅎㅏ-ㅣ/()]+\s*'
        matches = re.findall(pattern, query)
        if matches:
            return f"{matches[0].strip()} {difficulty}"
        
        if topic:
            return f"{topic[:20]} {difficulty}"
        
        return f"{query[:20]} {difficulty}"
    
    def _extract_kpop_query(self, query: str, topic: str) -> str:
        """K-pop 검색용 최적화된 쿼리 생성"""
        for kw in self.KPOP_TRIGGERS:
            if kw in query.lower():
                return f"{kw} 한국어 학습"
        
        if topic:
            return f"{topic} K-pop"
        
        return "K-pop 한국어"
    
    def _calculate_confidence(
        self, 
        strategies: List[SearchStrategy],
        query: str, 
        topic: str
    ) -> float:
        """라우팅 결정의 신뢰도 계산"""
        confidence = 1.0
        
        if not strategies:
            confidence = 0.5
        
        if len(topic.split()) >= 2 or len(query.split()) >= 3:
            confidence = min(1.0, confidence + 0.2)
        
        keyword_matches = sum([
            any(kw in query.lower() for kw in self.VOCABULARY_TRIGGERS),
            any(kw in query.lower() for kw in self.GRAMMAR_TRIGGERS),
            any(kw in query.lower() for kw in self.KPOP_TRIGGERS)
        ])
        confidence = min(1.0, confidence + keyword_matches * 0.1)
        
        return confidence
    
    def _find_strategy(
        self, 
        strategies: List[SearchStrategy],
        retriever_type: RetrieverType
    ) -> Optional[SearchStrategy]:
        """전략 리스트에서 특정 리트리버의 전략 찾기"""
        for strategy in strategies:
            if strategy.retriever_type == retriever_type:
                return strategy
        return None
    
    def _expand_query(self, original_query: str, difficulty: str) -> str:
        """더 많은 결과를 위한 쿼리 확장"""
        difficulty_keywords = {
            "basic": "기초 초급",
            "intermediate": "중급",
            "advanced": "고급 상급"
        }
        expanded = f"{original_query} {difficulty_keywords.get(difficulty, '')}"
        return expanded.strip()
    
    def _improve_grammar_query(self, original_query: str, difficulty: str) -> str:
        """문법 쿼리 개선"""
        if "문법" not in original_query and "grammar" not in original_query.lower():
            return f"{original_query} 문법"
        return original_query


def format_routing_summary(decision: RoutingDecision) -> str:
    """라우팅 결정 요약 (한국어)"""
    
    retriever_names = {
        "vocabulary": "어휘",
        "grammar": "문법",
        "kpop": "K-pop"
    }
    
    lines = [
        "🔀 라우팅 결정 결과",
        f"   신뢰도: {decision.confidence:.0%}",
        f"   결정 근거: {decision.reasoning}",
        "\n   📋 검색 전략:"
    ]
    
    for strategy in decision.strategies:
        retriever_kr = retriever_names.get(strategy.retriever_type.value, strategy.retriever_type.value)
        lines.append(
            f"      {strategy.priority}. [{retriever_kr}] "
            f"쿼리: '{strategy.query}' (재시도: {strategy.retry_count}회)"
        )
    
    return "\n".join(lines)