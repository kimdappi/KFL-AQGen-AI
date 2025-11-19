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


class IntelligentRouter:
    """
    지능형 라우터: 검색 전략 결정, 재검색, 쿼리 최적화
    
    핵심 역할:
    - 쿼리 분석 및 활성화할 리트리버 결정
    - 각 리트리버에 최적화된 검색 쿼리 생성
    - 품질 체크 기반 재검색 필요성 판단
    - 필요 시 LLM 사용한 쿼리 개선
    """
    
    # 리트리버 활성화 판단용 트리거 키워드 기반
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
        "bts", "blackpink", "뉴진스", "newjeans", "아이브", "ive",
        # 한글 그룹명 추가
        "방탄소년단", "블랙핑크", "트와이스", "르세라핌", "에스파",
        "세븐틴", "스트레이키즈", "엑소", "레드벨벳", "걸그룹", "보이그룹"
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
        
        # 1. 단어 리트리버 활성화

        vocab_query = self._extract_vocab_query(query, topic, difficulty)
        strategies.append(SearchStrategy(
            retriever_type=RetrieverType.VOCABULARY,
            query=vocab_query,
            priority=1,
            params={"level": difficulty, "limit": 5}
        ))
        reasons.append(f"Vocabulary search (query: '{vocab_query}', level: {difficulty})")
        
       
        # 문법 관련 키워드가 있을 때만 활성화
        if self._should_activate_grammar(query_lower, topic_lower):
            grammar_query = self._extract_grammar_query(query, topic, difficulty)
            strategies.append(SearchStrategy(
                retriever_type=RetrieverType.GRAMMAR,
                query=grammar_query,
                priority=2,
                params={"level": difficulty, "limit": 5}
            ))
            reasons.append(f"Grammar search (query: '{grammar_query}', level: {difficulty})")
        
       
        # K-pop 관련 내용이 쿼리에 있을 때만 활성화
        # query_analysis의 needs_kpop도 확인
        if self._should_activate_kpop(query_lower, topic_lower, query_analysis):
            kpop_query = self._extract_kpop_query(query, topic, query_analysis)
            strategies.append(SearchStrategy(
                retriever_type=RetrieverType.KPOP,
                query=kpop_query,
                priority=3,
                params={"level": difficulty, "db_limit": 5}
            ))
            reasons.append(f"K-pop search (DB only, query: '{kpop_query}')")
        
        # 중요도 순 정렬화
        strategies.sort(key=lambda x: x.priority)
        
        reasoning = " | ".join(reasons)
        confidence = self._calculate_confidence(strategies, query, topic)
        
        return RoutingDecision(
            strategies=strategies,
            reasoning=reasoning,
            confidence=confidence,
            needs_quality_check=True
        )
    
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
        """Grammar 리트리버 활성화 여부 확인 - 문법 관련 키워드가 있을 때만"""
        if any(kw in query for kw in self.GRAMMAR_TRIGGERS):
            return True
        if any(kw in topic for kw in self.GRAMMAR_TRIGGERS):
            return True
        return False
    
    def _should_activate_kpop(self, query: str, topic: str, query_analysis: Optional[Dict] = None) -> bool:
        """
        K-pop 리트리버 활성화 여부 확인
        - query_analysis의 needs_kpop 우선 확인
        - 키워드 매칭도 확인
        """
        # 1. query_analysis의 needs_kpop 우선 확인 (가장 정확)
        if query_analysis:
            needs_kpop = query_analysis.get('needs_kpop', False)
            if needs_kpop:
                return True
            
            # kpop_filters에 그룹이 있으면 활성화
            kpop_filters = query_analysis.get('kpop_filters', {})
            if kpop_filters.get('groups'):
                return True
        
        # 2. 키워드 매칭 확인
        query_lower = query.lower()
        topic_lower = topic.lower()
        
        if any(kw in query_lower for kw in self.KPOP_TRIGGERS):
            return True
        if any(kw in topic_lower for kw in self.KPOP_TRIGGERS):
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
    
    def _extract_kpop_query(self, query: str, topic: str, query_analysis: Optional[Dict] = None) -> str:
        """
        K-pop 검색용 최적화된 쿼리 생성
        query_analysis의 kpop_filters에서 그룹명 우선 추출
        """
        # 1. query_analysis에서 그룹명 추출 (가장 정확)
        if query_analysis:
            kpop_filters = query_analysis.get('kpop_filters', {})
            groups = kpop_filters.get('groups', [])
            if groups:
                # 첫 번째 그룹명 사용
                return f"{groups[0]} 한국어 학습"
        
        # 2. 키워드 매칭
        query_lower = query.lower()
        for kw in self.KPOP_TRIGGERS:
            if kw in query_lower:
                return f"{kw} 한국어 학습"
        
        # 3. topic 사용
        if topic:
            return f"{topic} K-pop"
        
        # 4. 기본값
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