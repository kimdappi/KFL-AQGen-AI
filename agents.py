"""
KFL-AQGen-AI Agentic RAG 에이전트
쿼리 분석 및 품질 검증 기능 제공
(개선 버전 - 문제 품질 검증 추가)
"""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
import json


class QueryAnalysisAgent:
    """
    쿼리 분석 에이전트
    프롬프트 기반 분석으로 개선
    """
    
    def __init__(self, llm=None):
        self.llm = llm 
    
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

JSON 형식으로 반환:
{
  "difficulty": "",
  "topic": "",
  "needs_kpop": true/false,
  "user_interests": [],
  "learning_goals": [],
  "query_intent": ""
}
"""
        
        response = self.llm.predict(prompt)
        
        return json.loads(response)

    
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

1. **Appropriateness for Foreign Learners** (외국인 학습자 적합성):
- 난이도가 사용자가 원하는 난이도인가?
- 외국인이 문제를 풀기에 가독성이 좋은 문제인가?
- 문제, 정답, 예문 모두 오로지 **한국어 교육** 에 적합한가?

2. **Educational Value** (교육적 가치):
- 효과적으로 타겟 문법을 배울 수 있는 문제로 생성되었나?
- 데이터베이스에서 가져온 해당 난이도에 알맞는 단어를 사용해서 문제를 만들었나?
- 한국어를 배우는 목표에 맞는 문제인가?

3. **Problem Design Quality** (문제 설계 품질):
- 해당 문제가 주어진 문제 형식에 적합하게 만들어졌나?

4. **Learning Goal Alignment** (학습 목표 정렬):
- 학습해야 하는 목표에 알맞는 문제가 생성이 되었는가?

5. **Engagement and Relevance** (참여도와 관련성):
- 생성된 문제와 예상 답안이 모두 번역체가 아닌 자연스러운 한국어로 구성되어 있나?

Json 포멧으로 각 항목에 대해 점수를 반환하세요:
{{
  "overall_quality_score": 0-100,
  "is_appropriate": true/false,
  "criteria_scores": {{
    "appropriateness_for_learners": 0-100,
    "educational_value": 0-100,
    "problem_design": 0-100,
    "goal_alignment": 0-100,
    "engagement": 0-100
  }},
  "strengths": ["list of strengths"],
  "weaknesses": ["list of weaknesses"],
  "improvement_suggestions": ["specific suggestions"],
  "recommendation": "accept/revise/reject",
  "detailed_feedback": "comprehensive feedback message"
}}
"""
        
        response = self.llm.predict(prompt)
        
        try:
            evaluation = json.loads(response)
            
            # 종합 판단
            if evaluation["overall_quality_score"] >= 70:
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
        
        if evaluation["recommendation"] == "accept":
            return original_problem
        
        prompt = f""" 아래 원문 문제를 분석하고, 평가 내용을 반영하여 한국어 교육에 더 적합한 문제로 재구성하세요.

원문 문제: {json.dumps(original_problem, ensure_ascii=False)}
평가 결과: {json.dumps(evaluation, ensure_ascii=False)}
목표 난이도: {target_level}

개선 시 중점 사항:
1. 평가에서 드러난 약점 보완
2. 제시된 개선 의견 반영
3. 한국어 학습자를 위한 문제 및 학습 목표에 맞춘 문제
4. {target_level} 수준에 맞는 난이도 조정

출력 형식(JSON):
{
  "improved_problem": { ... },  
  "changes_made": ["구체적으로 어떤 부분을 어떻게 바꿨는지"],
  "rationale": "왜 그렇게 수정했는지 (교육적 이유)"
}

반드시 JSON만 출력하고, 설명은 JSON 내부 'changes_made'와 'rationale'에 포함하세요.

"""
        
        response = self.llm.predict(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "improved_problem": original_problem,
                "changes_made": ["개선 실패"],
                "rationale": "자동 개선 실패 - 수동 수정 필요"
            }