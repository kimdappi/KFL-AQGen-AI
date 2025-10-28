# =====================================
# kpop_evaluator.py
# =====================================
"""
K-pop 문장 생성 결과 평가 모듈
- 문법, 어휘, 자연스러움 등 규칙 기반 평가
"""
import json
from langchain_openai import ChatOpenAI

class KpopSentenceEvaluator:
    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def evaluate(self, sentence: str, grammar: str = None, vocab: str = None):
        """LLM을 이용한 문법/어휘 평가"""
        prompt = f"""
        아래 문장이 주어진 조건을 충족하는지 평가해 주세요.

        문장: "{sentence}"
        문법 조건: {grammar}
        어휘 조건: {vocab}

        '문법과 어휘를 모두 포함하면 True, 아니면 False'로 JSON 형태로 답하세요.
        예시: {{"grammar_ok": true, "vocab_ok": false}}
        """

        result = self.llm.invoke(prompt).content
        return json.loads(result)
