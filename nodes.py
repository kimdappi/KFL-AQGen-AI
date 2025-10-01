# =====================================
# nodes.py - 노드 정의
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
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.kpop_retriever = kpop_retriever   # ✅ 추가
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
    
    def retrieve_kpop(self, state: GraphState) -> GraphState:
        """K-pop 문장 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']

        kpop_docs = self.kpop_retriever.invoke(query, level)
        return {"kpop_docs": kpop_docs}
    
    def generate_sentences(self, state: GraphState) -> GraphState:
        """검색된 단어와 문법을 활용한 문장 생성"""
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        grammars = extract_grammars_from_docs(state['grammar_docs'])
        
        # 단어와 품사 정보 포맷팅
        words_formatted = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
        
        prompt = f"""
        다음 단어와 문법, K-pop 문장을 참고하여 한국어 학습용 예문을 3개 생성해주세요.
        
        난이도: {state['difficulty_level']}
        단어 (품사): {', '.join(words_formatted)}
        문법: {', '.join(grammars[:3])}
        
        각 문장은:
        1. 제시된 단어를 최소 2개 이상 포함
        2. 제시된 문법 패턴을 최소 1개 포함
        3. 난이도에 적합한 복잡도
        4. 자연스러운 한국어 문장
        
        예문:
        """
        
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 메시지 히스토리 추가
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
        
        output += "\n선택된 문법 (상위 10개, grade 낮은 순):\n"
        for i, doc in enumerate(state['grammar_docs'][:10], 1):
            grammar = doc.metadata.get('grammar', 'N/A')
            grade = doc.metadata.get('grade', 'N/A')
            output += f"{i}. {grammar} (Grade: {grade})\n"
        
        output += "\n생성된 예문:\n"
        for i, sentence in enumerate(state['generated_sentences'], 1):
            output += f"{i}. {sentence}\n"
        
        return {"final_output": output}