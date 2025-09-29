# =====================================
# nodes.py (Updated) - grade를 level로 사용
# =====================================
"""
LangGraph 노드 정의 (문장 저장 기능 및 grade 사용)
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from schema import GraphState
from utils import (
    detect_difficulty_from_text,
    extract_words_from_docs,
    extract_grammars_from_docs,
    extract_grammar_with_grade,  # 새로운 유틸 함수
    format_docs
)
from config import LLM_CONFIG

INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1F]'

def sanitize_filename(name: str, replacement: str = "_") -> str:
# 금지문자 -> _
    safe = re.sub(INVALID_CHARS, replacement, name)
    # 마지막의 점/공백 제거
    safe = safe.strip().strip(".")
    # Windows 예약어 회피
    RESERVED = {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}
    if safe.upper() in RESERVED:
        safe = f"_{safe}"
        # 너무 긴 파일명 방지 (경로 전체 길이 여유 주기)
    return safe[:120] if len(safe) > 120 else safe

class KoreanLearningNodes:
    """한국어 학습 노드 클래스"""
    
    def __init__(self, vocabulary_retriever, grammar_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini", #범준이 api 대신 임시
            temperature=LLM_CONFIG.get('temperature', 0.7),
            max_tokens=LLM_CONFIG.get('max_tokens', 1000)
        )
        
        # sentence 폴더 생성
        self.output_dir = "sentence"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

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
    
    def generate_sentences(self, state: GraphState) -> GraphState:
        """검색된 단어와 문법을 활용한 문장 생성"""
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        
        # 문법과 grade 정보 함께 추출
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 단어와 품사 정보 포맷팅
        words_formatted = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
        
        # 첫 번째 문법과 그 grade를 메인으로 사용 (임시) -> 단점 : 계속 1,3,5 만 나옴
        if grammar_info:
            target_grammar = grammar_info[7]['grammar']
            target_grade = grammar_info[7]['grade']  # 실제 grade 값 사용
        else:
            target_grammar = "기본 문법"
            target_grade = 1
        
        
        prompt = f"""
        다음 단어와 문법을 사용하여 한국어 학습용 예문을 3개 생성해주세요.
        
        난이도: {state['difficulty_level']} (Grade {target_grade})
        단어 (품사): {', '.join(words_formatted)}
        학습 목표 문법: {target_grammar} (Grade {target_grade})
        
        
        각 문장은:
        1. 제시된 단어를 최소 5개 이상 포함
        2. 주요 문법 패턴을 반드시 포함
        3. Grade {target_grade} 수준에 적합한 복잡도
        4. 외국인이 한국어를 배울 때 유용한 문장
        
        예문 (번호 없이 문장만):
        """
        
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # JSON 형식으로 저장할 데이터 생성 (grade를 level로 사용)
        save_data = {
            "level": target_grade,  # grade 값을 level로 사용 1-6
            "target_grammar": target_grammar,
            "critique_summary": [{"sentence": s} for s in sentences]
        }
        
        # 메시지 히스토리 추가
        messages = [
            ("user", state['input_text']),
            ("assistant", "\n".join(sentences))
        ]
        
        return {
            "generated_sentences": sentences,
            "messages": messages,
            "sentence_data": save_data,
            "target_grade": target_grade  # state에 grade 정보 추가
        }
    
    def format_output(self, state: GraphState) -> GraphState:
        """최종 출력 포맷팅 및 JSON 저장"""
        output = f"=== 한국어 학습 문제 생성 결과 ===\n"
        output += f"난이도: {state['difficulty_level']}\n"
        
        # target_grade가 있으면 표시
        if 'target_grade' in state:
            output += f"문법 Grade: {state['target_grade']}\n"
        
        output += "\n선택된 단어 (상위 10개):\n"
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
        
        # JSON 파일로 저장 (sentence_data가 있을 때만)
        if 'sentence_data' in state and state['sentence_data']:
            saved_file = self._save_to_json(state['sentence_data'])
            output += f"\n 예문이 저장되었습니다: {saved_file}\n"
        
        return {"final_output": output}
    
    def _save_to_json(self, sentence_data: dict) -> str:
        out_dir = Path("sentence")
        out_dir.mkdir(parents=True, exist_ok=True)

        level = sentence_data.get("level", "grade1")
        title = sentence_data.get("title", "untitled")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base = f"sentences_{level}_{title}_{timestamp}"
        safe_base = sanitize_filename(base)
        filepath = out_dir / f"{safe_base}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            import json
            json.dump(sentence_data, f, ensure_ascii=False, indent=2)

        return str(filepath)