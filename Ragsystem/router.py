"""
Simplified Router for KFL-AQGen-AI
쿼리 기반 판단으로 단순화
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
import json


class SimplifiedRouter:
    """
    단순화된 라우터
    - 복잡한 전략 제거
    - 쿼리 전체 맥락 이해
    - 문법 문제여도 어휘 포함 보장
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def route(
        self, 
        query: str,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        쿼리를 이해하고 필요한 리트리버 결정
        
        Args:
            query: 사용자 쿼리
            query_analysis: 쿼리 분석 결과
            
        Returns:
            라우팅 결정
        """
        
        prompt = f"""한국어 학습 쿼리를 분석하고 사용할 리트리버를 결정해주세요.

【쿼리 정보】
- 사용자 쿼리: "{query}"
- 난이도: {query_analysis.get('difficulty', 'basic')}
- 주제: {query_analysis.get('topic', 'general')}

【리트리버 선택 규칙】
1. 어휘 리트리버는 항상 사용 (모든 문제에 최소 3개 단어 필요)
2. 문법 리트리버는 문법 연습이 언급되거나 필요한 경우에만 사용
3. K-pop 리트리버는 K-pop/한국 음악/아티스트가 언급된 경우에만 사용

⚠️ 중요: 문법 중심 문제라도 의미 있는 문장 생성을 위해 어휘가 반드시 필요합니다.

【응답 형식】
다음 JSON 형식으로 응답해주세요:
{{
  "use_vocabulary": true/false,    // 어휘 리트리버 사용 여부
  "use_grammar": true/false,       // 문법 리트리버 사용 여부  
  "use_kpop": true/false,          // K-pop 리트리버 사용 여부
  "vocab_count": 5-10,             // 필요한 어휘 개수
  "grammar_count": 1,              // 필요한 문법 패턴 개수
  "kpop_count": 0-5,               // 필요한 K-pop 콘텐츠 개수
  "reasoning": "간단한 설명"        // 선택 이유
}}
"""
        
        response = self.llm.predict(prompt)
        
        try:
            decision = json.loads(response)
            # 어휘는 항상 필요 (최소 3개)
            decision['use_vocabulary'] = True
            if decision.get('vocab_count', 0) < 3:
                decision['vocab_count'] = 5
                
            return decision
            
        except json.JSONDecodeError:
            # 기본값: 어휘는 필수, 문법은 쿼리에 따라
            return {
                "use_vocabulary": True,
                "use_grammar": "grammar" in query.lower() or "문법" in query,
                "use_kpop": any(kw in query.lower() for kw in ['kpop', 'k-pop', 'blackpink', 'bts']),
                "vocab_count": 5,
                "grammar_count": 2,
                "kpop_count": 3,
                "reasoning": "쿼리 키워드 기반 기본 라우팅"
            }
    
    def get_search_params(
        self,
        routing_decision: Dict[str, Any],
        difficulty: str
    ) -> Dict[str, Any]:
        """
        라우팅 결정을 검색 파라미터로 변환
        
        Args:
            routing_decision: 라우팅 결정
            difficulty: 난이도
            
        Returns:
            검색 파라미터
        """
        
        params = {
            "vocabulary": {
                "enabled": routing_decision.get('use_vocabulary', True),
                "level": difficulty,
                "limit": routing_decision.get('vocab_count', 3)
            },
            "grammar": {
                "enabled": routing_decision.get('use_grammar', False),
                "level": difficulty,
                "limit": routing_decision.get('grammar_count', 1)
            },
            "kpop": {
                "enabled": routing_decision.get('use_kpop', False),
                "level": difficulty,
                "limit": routing_decision.get('kpop_count', 3)
            }
        }
        
        return params