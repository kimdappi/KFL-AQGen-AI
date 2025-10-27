"""
KFL-AQGen-AI Agentic RAG 에이전트
쿼리 분석 및 품질 검증 기능 제공
(수정 완료!)
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json


class QueryAnalysisAgent:
    """
    쿼리 분석 에이전트
    사용자 쿼리를 분석하여 난이도, 주제, 검색 필요성 파악
    일단은 gpt 기 때문에 영어로 하라고 명시해놓음 (수정해도됨됨)
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
            - topic: Main topic (e.g., restaurant, travel, school, K-pop, daily life)
            - needs_kpop: Whether K-pop content is relevant (true/false)
        """
        prompt = f"""Analyze the following Korean language learning query and respond in JSON format:

Query: "{query}"

Analysis items:
1. difficulty: basic/intermediate/advanced (check for keywords: basic, middle, intermediate, advanced, beginner, 초급, 중급, 고급)
2. topic: Main topic (e.g., restaurant, travel, school, K-pop, daily life)
3. needs_kpop: true if query mentions K-pop, Korean music, or specific K-pop artists (e.g., BTS, BLACKPINK, blackpink, EXO, TWICE, NewJeans, etc.), false otherwise

**IMPORTANT for needs_kpop**:
- Set to true if query contains: k-pop, kpop, K-pop, BTS, BLACKPINK, blackpink, EXO, TWICE, NewJeans, or any K-pop artist name
- Set to true if query mentions Korean music, Korean songs, or Korean idols
- Set to false for general Korean learning queries without K-pop references

JSON format:
{{
  "difficulty": "basic/intermediate/advanced",
  "topic": "topic name",
  "needs_kpop": true/false
}}

Respond ONLY with valid JSON, no additional text.
"""
        
        response = self.llm.predict(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Default fallback
            return {
                "difficulty": "basic",
                "topic": "general",
                "needs_kpop": False
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
        needs_kpop: bool = False,  # K-pop 필요 여부 추가
        kpop_only: bool = False  # ✅ NEW 파라미터
    ) -> Dict[str, Any]:
        """
        검색 결과가 충분한지 확인
        
        Args:
            vocab_count: 검색된 어휘 항목 수
            grammar_count: 검색된 문법 항목 수
            kpop_db_count: 데이터베이스에서 검색된 K-pop 항목 수
            needs_kpop: K-pop 검색이 필요한지 여부 (쿼리에 K-pop 키워드 있을 때)
            
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
            sufficient = basic_sufficient and kpop_db_count >= 3  # K-pop 필요 시 최소 3개
        else:
            sufficient = basic_sufficient  # K-pop 불필요 시 체크 안 함
        
        return {
            "sufficient": sufficient,
            "vocab_count": vocab_count,
            "grammar_count": grammar_count,
            "kpop_db_count": kpop_db_count,
            "total_kpop": kpop_db_count,
            "needs_kpop": needs_kpop,
            "message": "충분함" if sufficient else "추가 검색 필요"
        }