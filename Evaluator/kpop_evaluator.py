# =====================================
# kpop_evaluator.py
# =====================================
"""
K-pop 문장 생성 결과 평가 모듈
- 문법, 어휘, 자연스러움 등 규칙 기반 평가
"""
import json
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI


class KpopSentenceEvaluator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def evaluate(self, sentence: str, grammar: str = None, vocab: List[str] = None):
        """LLM을 이용한 문법/어휘 평가"""
        vocab_str = ", ".join(vocab) if vocab else "없음"
        
        prompt = f"""
        아래 문장이 주어진 조건을 충족하는지 평가해 주세요.
        문장: "{sentence}"
        문법 조건: {grammar}
        어휘 조건: {vocab_str}
        '문법과 어휘를 모두 포함하면 True, 아니면 False'로 JSON 형태로 답하세요.
        예시: {{"grammar_ok": true, "vocab_ok": false}}
        """
        
        try:
            result = self.llm.invoke(prompt).content
            # JSON 파싱
            if "```" in result:
                result = result.split("```")[1].replace("json", "").strip()
            return json.loads(result)
        except Exception as e:
            print(f"   평가 오류: {e}")
            return {"grammar_ok": False, "vocab_ok": False}
    
    def evaluate_batch(self, sentences: List[str], grammar: str = None, vocab: List[str] = None):
        """여러 문장 평가 - 3개 문장 그대로 평가"""
        evaluation_results = []
        
        for i, sentence in enumerate(sentences, 1):
            eval_result = self.evaluate(sentence, grammar, vocab)
            
            # 평가 결과 출력
            grammar_status = "✅" if eval_result.get("grammar_ok") else "❌"
            vocab_status = "✅" if eval_result.get("vocab_ok") else "❌"
            
            print(f"      {i}. 문법{grammar_status} 어휘{vocab_status}: {sentence[:50]}...")
            
            evaluation_results.append({
                "sentence": sentence,
                "grammar_ok": eval_result.get("grammar_ok", False),
                "vocab_ok": eval_result.get("vocab_ok", False)
            })
        
        # 전체 평가 요약
        grammar_pass = sum(1 for r in evaluation_results if r["grammar_ok"])
        vocab_pass = sum(1 for r in evaluation_results if r["vocab_ok"])
        
        print(f"\n   📊 평가 결과: 문법 충족 {grammar_pass}/3, 어휘 충족 {vocab_pass}/3")
        
        return evaluation_results