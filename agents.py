"""
KFL-AQGen-AI Agentic RAG 에이전트
쿼리 분석 및 품질 검증 기능 제공
(개선 버전 - 문제 품질 검증 추가)
"""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import re


class QueryAnalysisAgent:
    """
    쿼리 분석 에이전트
    프롬프트 기반 분석으로 개선
    """
    
    def __init__(self, llm=None):
        # 기본 LLM 설정 추가
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query using prompt-based approach for better understanding
        
        Args:
            query: User input query
        
        """
        prompt = f"""한국어 교육 목적의 문제지 생성을 위한 쿼리를 아래기준으로 분석하시오:

쿼리: "{query}"

분석 기준:
1. 난이도: 초급/중급/고급 (명시 없으면 문맥으로 사용자 필요 난이도 추론)
2. 주제: (예: 일상/여행/음식/학교/비즈니스/K-pop 등)
3. K-pop 연관 여부: kpop 관련 키워드, 아이돌 그룹 혹은 멤버 이름으로 판단, 노래, 기획사, 콘셉트 등 언급 여부로 판단
4. 관심 요소: K-pop 관련 그룹/가수/세계관/콘셉트 등 구체 관심사 확인 후 삽입
5. 학습 목표: 어휘/문법/표현 학습 의도

**중요: 반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 포함하지 마세요.**

JSON 형식:
{{
  "difficulty": "초급/중급/고급 중 하나",
  "topic": "주제",
  "needs_kpop": true,
  "user_interests": ["관심사1", "관심사2"],
  "learning_goals": ["목표1", "목표2"],
  "query_intent": "사용자 의도 요약"
}}
"""
        
        # invoke() 메서드 사용 (predict() deprecated)
        response = self.llm.invoke(prompt)
        
        # AIMessage 객체에서 content 추출
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # JSON 파싱 시도
        try:
            # 응답에서 JSON 부분만 추출 (마크다운 코드블록 제거)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            # 중괄호로 시작하는 부분 찾기
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패. 원본 응답: {response_text}")
            # 기본값 반환
            return {
                "difficulty": "중급",
                "topic": "일반",
                "needs_kpop": False,
                "user_interests": [],
                "learning_goals": ["한국어 학습"],
                "query_intent": query
            }

    
    def check_problem_quality(
        self,
        generated_problem: Dict[str, Any],
        target_level: str,
        learning_goals: List[str] = None,
        target_grammar: str = None,
        target_vocab: List[str] = None
    ) -> Dict[str, Any]:
        """
        생성된 문제의 교육적 품질을 검증
        한국어 학습 목적에 적합한지 평가
        
        Args:
            generated_problem: 생성된 문제 데이터
            target_level: 목표 난이도
            learning_goals: 학습 목표
            target_grammar: 목표 문법
            target_vocab: 목표 어휘 리스트
            
        Returns:
            문제 품질 평가 결과
        """
        
        prompt = f"""외국인의 한국어 교육을 위해 생성된 아래 문제가 한국어 교육 문제로서 적합한지 아래 기준에 의거하여 판단하시오.:

Problem Data: {json.dumps(generated_problem, ensure_ascii=False)}
Target Level: {target_level}
Learning Goals: {learning_goals}
Target Grammar: {target_grammar}
Target Vocabulary: {target_vocab}

Evaluate based on these criteria:

1. **Appropriateness for Foreign Learners** (외국인 학습자 적합성)
2. **Educational Value** (교육적 가치)
3. **Problem Design Quality** (문제 설계 품질)
4. **Learning Goal Alignment** (학습 목표 정렬)
5. **Engagement and Relevance** (참여도와 관련성)

**중요: 반드시 아래 JSON 형식으로만 응답하세요.**

{{{{
  "overall_quality_score": 80,
  "is_appropriate": true,
  "criteria_scores": {{{{
    "appropriateness_for_learners": 80,
    "educational_value": 80,
    "problem_design": 80,
    "goal_alignment": 80,
    "engagement": 80
  }}}},
  "strengths": ["강점1", "강점2"],
  "weaknesses": ["약점1"],
  "improvement_suggestions": ["제안1"],
  "recommendation": "accept",
  "detailed_feedback": "종합 피드백"
}}}}
"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # JSON 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            evaluation = json.loads(response_text)
            
            # 종합 판단
            if evaluation.get("overall_quality_score", 0) >= 70:
                evaluation["final_verdict"] = "한국어 학습에 적합한 문제"
            else:
                evaluation["final_verdict"] = "문제 재생성 필요"
                
            return evaluation
            
        except json.JSONDecodeError:
            # Fallback evaluation
            return {
                "overall_quality_score": 60,
                "is_appropriate": True,
                "criteria_scores": {
                    "appropriateness_for_learners": 60,
                    "educational_value": 60,
                    "problem_design": 60,
                    "goal_alignment": 60,
                    "engagement": 60
                },
                "strengths": ["기본 구조 갖춤"],
                "weaknesses": ["평가 불가"],
                "improvement_suggestions": ["재검토 필요"],
                "recommendation": "revise",
                "detailed_feedback": "자동 평가 실패 - 수동 검토 필요",
                "final_verdict": "조건부 합격 - 검토 필요"
            }


class ProblemImprovementAgent:
    """
    문제 개선 에이전트 (추가)
    품질 검증 후 문제점 발견 시 개선 제안
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    def improve_problem(
        self,
        original_problem: Dict[str, Any],
        evaluation: Dict[str, Any],
        target_level: str
    ) -> Dict[str, Any]:
        """
        평가 결과를 바탕으로 문제 개선
        
        Args:
            original_problem: 원본 문제
            evaluation: 품질 평가 결과
            target_level: 목표 수준
            
        Returns:
            개선된 문제 또는 개선 제안
        """
        
        if evaluation.get("recommendation") == "accept":
            return original_problem
        
        prompt = f"""아래 원문 문제를 분석하고, 평가 내용을 반영하여 한국어 교육에 더 적합한 문제로 재구성하세요.

원문 문제: {json.dumps(original_problem, ensure_ascii=False)}
평가 결과: {json.dumps(evaluation, ensure_ascii=False)}
목표 난이도: {target_level}

**중요: 반드시 아래 JSON 형식으로만 응답하세요.**

{{{{
  "improved_problem": {{}},
  "changes_made": ["변경사항1", "변경사항2"],
  "rationale": "개선 이유"
}}}}
"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        try:
            # JSON 추출
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            
            return json.loads(response_text)
            
        except json.JSONDecodeError:
            return {
                "improved_problem": original_problem,
                "changes_made": ["개선 실패"],
                "rationale": "자동 개선 실패 - 수동 수정 필요"
            }