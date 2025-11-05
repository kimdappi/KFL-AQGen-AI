# =====================================
# kpop_evaluator.py (개선 버전)
# =====================================
"""
K-pop 문장 생성 결과 평가 모듈 (개선)
- 문법, 어휘 포함 여부 평가
- 어휘 중복 없이 사용되었는지 체크
- 상세한 피드백 제공
"""
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI


class KpopSentenceEvaluator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def evaluate(self, sentence: str, grammar: str = None, vocab: List[str] = None):
        """
        LLM을 이용한 문법/어휘 평가
        - 문법 포함 여부
        - 어휘 포함 여부
        - 사용된 어휘 반환
        """
        vocab_str = ", ".join(vocab) if vocab else "없음"
        
        prompt = f"""아래 문장을 평가해 주세요.

문장: "{sentence}"

평가 기준:
1. 문법 '{grammar}' 포함 여부
2. 어휘 목록 [{vocab_str}] 중 하나 이상 포함 여부
3. 사용된 어휘 목록

JSON 형식으로 답하세요:
{{
    "grammar_ok": true/false,
    "vocab_ok": true/false,
    "used_vocab": ["사용된어휘1", "사용된어휘2"]
}}
"""
        
        try:
            result = self.llm.invoke(prompt).content
            
            # JSON 파싱
            if "```" in result:
                result = result.split("```")[1].replace("json", "").strip()
            
            parsed = json.loads(result)
            
            # 기본값 설정
            return {
                "grammar_ok": parsed.get("grammar_ok", False),
                "vocab_ok": parsed.get("vocab_ok", False),
                "used_vocab": parsed.get("used_vocab", [])
            }
            
        except Exception as e:
            print(f"   평가 오류: {e}")
            return {
                "grammar_ok": False,
                "vocab_ok": False,
                "used_vocab": []
            }
    
    def evaluate_batch(self, sentences: List[str], grammar: str = None, vocab: List[str] = None):
        """
        여러 문장 평가 (개선)
        - 각 문장 평가
        - 어휘 중복 체크
        - 전체 통계 제공
        """
        evaluation_results = []
        all_used_vocab = []
        
        print(f"\n   📊 문장 평가 시작 (목표 어휘: {vocab})")
        
        for i, sentence in enumerate(sentences, 1):
            eval_result = self.evaluate(sentence, grammar, vocab)
            
            # 평가 결과
            grammar_status = "✅" if eval_result.get("grammar_ok") else "❌"
            vocab_status = "✅" if eval_result.get("vocab_ok") else "❌"
            used = eval_result.get("used_vocab", [])
            
            print(f"      문장{i}: 문법{grammar_status} 어휘{vocab_status}")
            print(f"         사용 어휘: {used if used else '없음'}")
            print(f"         내용: {sentence[:60]}...")
            
            evaluation_results.append({
                "sentence": sentence,
                "grammar_ok": eval_result.get("grammar_ok", False),
                "vocab_ok": eval_result.get("vocab_ok", False),
                "used_vocab": used
            })
            
            all_used_vocab.extend(used)
        
        # 어휘 중복 체크
        vocab_duplicates = []
        vocab_counts = {}
        for v in all_used_vocab:
            vocab_counts[v] = vocab_counts.get(v, 0) + 1
            if vocab_counts[v] > 1 and v not in vocab_duplicates:
                vocab_duplicates.append(v)
        
        # 전체 평가 요약
        grammar_pass = sum(1 for r in evaluation_results if r["grammar_ok"])
        vocab_pass = sum(1 for r in evaluation_results if r["vocab_ok"])
        
        print(f"\n   📈 평가 요약:")
        print(f"      문법 충족: {grammar_pass}/3")
        print(f"      어휘 충족: {vocab_pass}/3")
        
        if vocab_duplicates:
            print(f"      ⚠️ 중복 어휘: {vocab_duplicates}")
        
        # 미사용 어휘
        unused_vocab = [v for v in vocab if v not in all_used_vocab]
        if unused_vocab:
            print(f"      ℹ️ 미사용 어휘: {unused_vocab}")
        
        return evaluation_results
    
    def get_feedback(self, evaluation_results: List[Dict], grammar: str, vocab: List[str]) -> str:
        """
        평가 결과 기반 상세 피드백 생성
        """
        feedback = []
        
        # 문법 피드백
        grammar_fail = [i+1 for i, r in enumerate(evaluation_results) if not r["grammar_ok"]]
        if grammar_fail:
            feedback.append(f"문법 '{grammar}' 미포함: 문장 {grammar_fail}")
        
        # 어휘 피드백
        vocab_fail = [i+1 for i, r in enumerate(evaluation_results) if not r["vocab_ok"]]
        if vocab_fail:
            feedback.append(f"어휘 미포함: 문장 {vocab_fail}")
        
        # 어휘 중복 피드백
        all_used = []
        for r in evaluation_results:
            all_used.extend(r.get("used_vocab", []))
        
        duplicates = [v for v in set(all_used) if all_used.count(v) > 1]
        if duplicates:
            feedback.append(f"어휘 중복 사용: {duplicates}")
        
        # 미사용 어휘
        unused = [v for v in vocab if v not in all_used]
        if unused:
            feedback.append(f"미사용 어휘: {unused}")
        
        return " | ".join(feedback) if feedback else "모든 조건 충족"