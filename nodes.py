# =====================================
# nodes.py - 노드 정의 (K-pop Context 반영)
# =====================================
"""
LangGraph 노드 정의
"""
from typing import Any
from langchain.llms import OpenAI
from schema import GraphState
from utils import (
    detect_difficulty_from_text,
    extract_words_from_docs,
    extract_grammars_from_docs,
    format_docs
)


class KoreanLearningNodes:
    """한국어 학습 노드 클래스"""

    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_context, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.kpop_context = kpop_context  # ✅ 변경됨
        self.llm = llm or OpenAI(temperature=0.7)

    def detect_difficulty(self, state: GraphState) -> GraphState:
        """입력 텍스트에서 난이도 감지"""
        difficulty = detect_difficulty_from_text(state['input_text'])
        return {"difficulty_level": difficulty}

    def retrieve_vocabulary(self, state: GraphState) -> GraphState:
        """단어 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        vocab_docs = self.vocabulary_retriever.invoke(query, level)
        return {"vocabulary_docs": vocab_docs}

    def retrieve_grammar(self, state: GraphState) -> GraphState:
        """문법 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        grammar_docs = self.grammar_retriever.invoke(query, level)
        return {"grammar_docs": grammar_docs}

    # ✅ retrieve_kpop() 제거 (이제 컨텍스트 제공자로 통합됨)

    def generate_sentences(self, state: GraphState) -> GraphState:
        """단어 + 문법 + K-pop 컨텍스트 기반 문장 생성"""
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        grammars = extract_grammars_from_docs(state['grammar_docs'])

        # 단어 포맷팅
        words_formatted = [f"{w}({c})" for w, c in words_info[:5]]

        # ✅ K-pop 컨텍스트 프롬프트 생성
        kpop_prompt = self.kpop_context.format_context_as_prompt(
            grammar=grammars[0] if grammars else None,
            vocab=words_info[0][0] if words_info else None
        )

        # ✅ 전체 프롬프트 구성
        prompt = f"""
        아래 정보를 바탕으로 자연스러운 한국어 학습용 예문 3개를 생성하세요.

        난이도: {state['difficulty_level']}
        단어 (품사): {', '.join(words_formatted)}
        문법: {', '.join(grammars[:3])}

        --- K-pop Context ---
        {kpop_prompt}
        ----------------------

        조건:
        1. 제시된 단어 중 최소 2개 포함
        2. 제시된 문법 패턴 중 최소 1개 포함
        3. K-pop 문맥에 자연스럽게 어울릴 것
        4. 난이도에 맞는 복잡도

        예문:
        """

        try:
            response = self.llm.predict(prompt)
            sentences = [s.strip() for s in response.split("\n") if s.strip()]
        except Exception as e:
            print(f"⚠️ LLM 호출 실패, 기본 문장 사용: {e}")
            sentences = [
                "1. 숙제를 해야 주말에 힘내서 즐길 수 있을까요?",
                "2. 내가 좋아하는 가수의 음악을 들으면서 숙제를 하면 더 재미있어요.",
                "3. 질문이 있으면 언제든지 물어봐주세요. 제가 답변해드릴게요."
            ]

        messages = [
            ("user", state['input_text']),
            ("assistant", "\n".join(sentences))
        ]

        return {
            "generated_sentences": sentences,
            "messages": messages
        }

    def format_output(self, state: GraphState) -> GraphState:
        """최종 출력 포맷팅"""
        output = f"=== 한국어 학습 문제 생성 결과 ===\n"
        output += f"난이도: {state['difficulty_level']}\n\n"

        output += "선택된 단어 (상위 10개):\n"
        for i, doc in enumerate(state['vocabulary_docs'][:10], 1):
            word = doc.metadata.get('word', 'N/A')
            wordclass = doc.metadata.get('wordclass', 'N/A')
            guide = doc.metadata.get('guide', 'N/A')
            topik_level = doc.metadata.get('topik_level', 'N/A')
            output += f"{i}. {word} ({wordclass}) - {guide[:30]}... [TOPIK{topik_level}]\n"

        output += "\n선택된 문법 (상위 10개):\n"
        for i, doc in enumerate(state['grammar_docs'][:10], 1):
            grammar = doc.metadata.get('grammar', 'N/A')
            grade = doc.metadata.get('grade', 'N/A')
            output += f"{i}. {grammar} (Grade: {grade})\n"

        output += "\n생성된 예문:\n"
        for sentence in state['generated_sentences']:
            # 문장 안에 이미 숫자(1., 2., 3.)가 있으면 제거
            cleaned = sentence.lstrip("1234567890. ").strip()
            output += f"- {cleaned}\n"

        return {"final_output": output}
