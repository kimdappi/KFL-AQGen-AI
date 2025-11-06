"""
KFL-AQGen-AI Agentic RAG 에이전트
쿼리 분석 및 품질 검증 기능 제공
K-pop 그룹 필터링 지원 추가
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json


class QueryAnalysisAgent:
    """
    쿼리 분석 에이전트
    사용자 쿼리를 분석하여 난이도, 주제, 검색 필요성, K-pop 그룹 파악
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return structured analysis
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with:
            - difficulty: basic/intermediate/advanced
            - topic: Main topic
            - needs_kpop: Whether K-pop content is relevant (true/false)
            - kpop_groups: List of specific K-pop groups mentioned (NEW!)
        """
        prompt = f"""Analyze the following Korean language learning query and respond in JSON format:

Query: "{query}"

Analysis items:
1. difficulty: basic/intermediate/advanced (check for keywords: basic, middle, intermediate, advanced, beginner, 초급, 중급, 고급)
2. topic: Main topic (e.g., restaurant, travel, school, K-pop, daily life)
3. needs_kpop: true if query mentions K-pop, Korean music, or specific K-pop artists, false otherwise
4. kpop_groups: List of specific K-pop group names mentioned in the query (e.g., ["BLACKPINK"], ["BTS", "TWICE"], or empty list [])

**IMPORTANT for needs_kpop**:
- Set to true if query contains: k-pop, kpop, K-pop, BTS, BLACKPINK, blackpink, EXO, TWICE, NewJeans, or any K-pop artist name
- Set to true if query mentions Korean music, Korean songs, or Korean idols
- Set to false for general Korean learning queries without K-pop references

**IMPORTANT for kpop_groups**:
- Extract ONLY the specific group names mentioned in the query
- Use standardized names: "BLACKPINK", "BTS", "TWICE", "NewJeans", "EXO", "Stray Kids", "aespa", "SEVENTEEN"
- If query says "about blackpink" → ["BLACKPINK"]
- If query says "about BTS and TWICE" → ["BTS", "TWICE"]
- If query says "K-pop" without specific groups → []
- Case-insensitive matching (blackpink = BLACKPINK)

JSON format:
{{
  "difficulty": "basic/intermediate/advanced",
  "topic": "topic name",
  "needs_kpop": true/false,
  "kpop_groups": ["GROUP1", "GROUP2"] or []
}}

Respond ONLY with valid JSON, no additional text.
"""
        
        response = self.llm.predict(prompt)

        # 표준화 매핑 (영어 표기)
        normalization_map = {
            "블랙핑크": "BLACKPINK",
            "방탄소년단": "BTS",
            "트와이스": "TWICE",
            "뉴진스": "NewJeans",
            "엑소": "EXO",
            "스트레이키즈": "Stray Kids",
            "에스파": "aespa",
            "세븐틴": "SEVENTEEN",
            " seventeen ": "SEVENTEEN",
            " blackpink ": "BLACKPINK",
            " bts ": "BTS",
            " twice ": "TWICE",
            " newjeans ": "NewJeans",
            " exo ": "EXO",
            " stray kids ": "Stray Kids",
            " aespa ": "aespa"
        }

        try:
            result = json.loads(response)
            # Ensure keys
            if 'kpop_groups' not in result:
                result['kpop_groups'] = []

            # 후처리 표준화: 쿼리 원문과 LLM 결과를 함께 사용
            q_lower = f" {query.lower()} "
            normalized = set()
            # 1) LLM 결과 표준화
            for g in result.get('kpop_groups', []):
                name = g.strip()
                if not name:
                    continue
                # 역매핑 시도
                key = f" {name.lower()} "
                std = normalization_map.get(key) or normalization_map.get(name) or name
                normalized.add(std)
            # 2) 쿼리에서 직접 감지
            for key, std in normalization_map.items():
                if key.strip() and key in q_lower:
                    normalized.add(std)
            result['kpop_groups'] = list(normalized)
            return result
        except json.JSONDecodeError:
            # Default fallback
            return {
                "difficulty": "basic",
                "topic": "general",
                "needs_kpop": False,
                "kpop_groups": []
            }


class QualityCheckAgent:
    """
    품질 검증 에이전트
    검색 결과의 충분성 검증
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def check(
        self, 
        vocab_count: int,
        grammar_count: int,
        kpop_db_count: int,
        needs_kpop: bool = False,
        kpop_only: bool = False
    ) -> Dict[str, Any]:
        """
        검색 결과가 충분한지 확인
        
        Args:
            vocab_count: 검색된 어휘 항목 수
            grammar_count: 검색된 문법 항목 수
            kpop_db_count: 데이터베이스에서 검색된 K-pop 항목 수
            needs_kpop: K-pop 검색이 필요한지 여부
            kpop_only: K-pop 전용 쿼리인지 여부
            
        Returns:
            Dictionary with:
            - sufficient: 결과가 충분한지 여부 (Boolean)
            - vocab_count, grammar_count, kpop_db_count: 각 항목 수
            - total_kpop: 총 K-pop 항목 수
            - message: 상태 메시지
        """
        
        # 기본 최소 요구사항
        basic_sufficient = (
            vocab_count >= 5 and
            grammar_count >= 1
        )
        
        # K-pop이 필요한 경우 추가 체크
        if needs_kpop:
            sufficient = basic_sufficient and kpop_db_count >= 3
        else:
            sufficient = basic_sufficient
        
        return {
            "sufficient": sufficient,
            "vocab_count": vocab_count,
            "grammar_count": grammar_count,
            "kpop_db_count": kpop_db_count,
            "total_kpop": kpop_db_count,
            "needs_kpop": needs_kpop,
            "message": "충분함" if sufficient else "추가 검색 필요"
        }