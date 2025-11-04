# =====================================
# nodes.py (Updated) - Evaluator 통합 버전
# 수정 완료
# =====================================
"""
LangGraph 노드 정의 (문장 저장 기능 및 평가 기능 포함)
"""
import json
import os
import re
import random
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from Ragsystem.schema import GraphState
from utils import (
    detect_difficulty_from_text,
    extract_words_from_docs,
    extract_grammar_with_grade  
)
from config import LLM_CONFIG, SENTENCE_SAVE_DIR
from agents import QueryAnalysisAgent, QualityCheckAgent

# Evaluator 임포트 (optional)
try:
    from Evaluator.kpop_evaluator import KpopSentenceEvaluator
    EVALUATOR_ENABLED = True
except ImportError:
    EVALUATOR_ENABLED = False
    print("⚠️ Evaluator 모듈 없음 - 기본 모드로 실행")

INVALID_CHARS = r'[<>:"/\\|?*\x00-\x1F]'

def sanitize_filename(name: str, replacement: str = "_") -> str:
    """Windows 파일명 안전 처리"""
    safe = re.sub(INVALID_CHARS, replacement, name)
    safe = safe.strip().strip(".")
    RESERVED = {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}
    if safe.upper() in RESERVED:
        safe = f"_{safe}"
    return safe[:120] if len(safe) > 120 else safe

class KoreanLearningNodes:
    """한국어 학습 노드 클래스"""
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.kpop_retriever = kpop_retriever
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=LLM_CONFIG.get('temperature', 0.7),
            max_tokens=LLM_CONFIG.get('max_tokens', 1000)
        )
        
        # Evaluator 초기화
        self.evaluator = None
        if EVALUATOR_ENABLED:
            try:
                self.evaluator = KpopSentenceEvaluator()
                print("   ✅ 문장 평가기 활성화")
            except Exception as e:
                print(f"   ℹ️ 평가기 초기화 실패: {e}")
                self.evaluator = None
        
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

    def retrieve_kpop(self, state: GraphState) -> GraphState:
        """K-pop 문장 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        kpop_docs = self.kpop_retriever.invoke(query, level)
        return {"kpop_docs": kpop_docs}
    
    def retrieve_grammar(self, state: GraphState) -> GraphState:
        """문법 검색 노드"""
        level = state['difficulty_level']
        query = state['input_text']
        grammar_docs = self.grammar_retriever.invoke(query, level)
        return {"grammar_docs": grammar_docs}

    def generate_sentences(self, state: GraphState) -> GraphState:
        """검색된 단어와 문법을 활용한 문장 생성 (K-pop 정보 포함)"""
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        
        # K-pop 정보 추출
        kpop_references = []
        kpop_context_text = ""
        
        if 'kpop_docs' in state and state['kpop_docs']:
            print(f"[참조] K-pop 문서 개수: {len(state['kpop_docs'])}")
            
            for doc in state['kpop_docs'][:3]:
                sentence = doc.metadata.get('sentence', '')
                song = doc.metadata.get('song', '')
                group = doc.metadata.get('group', '')
                
                if sentence:
                    kpop_references.append({
                        "sentence": sentence,
                        "song": song,
                        "group": group,
                    })
                    kpop_context_text += f'- "{sentence}" ({song} - {group})\n'
        
        print(f"[참조] K-pop 참조 개수: {len(kpop_references)}")
        
        # 문법과 grade 정보 추출
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 어휘 포맷팅
        words_formatted = []
        vocab_list = []  # 평가용
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
            vocab_list.append(word)
        
        if grammar_info:
            random_grammar_item = random.choice(grammar_info)
            target_grammar = random_grammar_item['grammar']
            target_grade = random_grammar_item['grade']
            print("grammar : ", target_grammar)
            print("grade : ", target_grade)
        else:
            target_grammar = "기본 문법"
            target_grade = 1
        
        # 난이도별 프롬프트 생성
        difficulty = state['difficulty_level']
        difficulty_guide = {
            "basic": "초급 학습자 (TOPIK 1-2급): 짧고 간단한 문장, 기본 시제 사용",
            "intermediate": "중급 학습자 (TOPIK 3-4급): 다양한 연결어미, 자연스러운 일상 대화 표현",
            "advanced": "고급 학습자 (TOPIK 5-6급): 복잡한 문장 구조, 격식체나 문어체 가능"
        }
        
        prompt = self._build_generation_prompt(
            difficulty, 
            target_grade, 
            words_formatted, 
            target_grammar, 
            kpop_context_text,
            difficulty_guide
        )
        
        # 문장 생성 (3개)
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()][:3]
        
        # 평가 수행 (있을 경우)
        critique_summary = self._evaluate_sentences(
            sentences, 
            target_grammar, 
            vocab_list
        )
        
        # JSON 저장 데이터
        save_data = {
            "level": target_grade,
            "target_grammar": target_grammar,
            "kpop_references": kpop_references,
            "critique_summary": critique_summary
        }
        
        messages = [
            ("user", state['input_text']),
            ("assistant", "\n".join(sentences))
        ]
        
        return {
            "generated_sentences": sentences,
            "messages": messages,
            "sentence_data": save_data,
            "target_grade": target_grade
        }
    
    def _build_generation_prompt(self, difficulty, target_grade, words_formatted, 
                                target_grammar, kpop_context_text, difficulty_guide):
        """프롬프트 템플릿 생성 (난이도별)"""
        prompt_templates = {
            "basic": """
[ROLE]
너는 한국어를 배우는 초급 학습자를 위한 친절한 한국어 선생님이야.

[INSTRUCTIONS]
- 학습 수준: {difficulty_level} (Grade {target_grade})
- 오늘의 단어: {words_formatted}
- 오늘의 문법: {target_grammar}
- K-pop 참고: {kpop_context_text}

[SENTENCE RULES]
1. 짧고 간단한 문장 (10-15 단어)
2. 기본 시제만 사용
3. 문법 패턴 {target_grammar} 필수 포함
4. 제시된 단어 최소 3개 포함

형식: 번호 없이 문장 3개만
""",
            "intermediate": """
[ROLE]
너는 중급 한국어 학습자를 위한 경험 많은 한국어 교사야.

[INSTRUCTIONS]
- 학습 수준: {difficulty_level} (Grade {target_grade})
- 핵심 어휘: {words_formatted}
- 목표 문법: {target_grammar}
- K-pop 참고: {kpop_context_text}

[REQUIREMENTS]
1. 자연스러운 대화체
2. 문법 {target_grammar} 활용
3. 제시된 어휘 3-4개 포함
4. 실생활 상황 반영

출력: 예문 3개만 (번호 없이)
""",
            "advanced": """
[ROLE]
너는 고급 한국어 학습자를 위한 전문 교수다.

[INSTRUCTIONS]
- 학습 수준: {difficulty_level} (Grade {target_grade})
- 핵심 어휘: {words_formatted}
- 핵심 문법: {target_grammar}
- K-pop 참고: {kpop_context_text}

[REQUIREMENTS]
1. 복잡한 문장 구조
2. 문법 {target_grammar} 심화 활용
3. 고급 어휘 사용
4. 문어체 또는 격식체

출력: 예문 3개만
"""
        }
        
        template = prompt_templates.get(difficulty, prompt_templates["intermediate"])
        return template.format(
            difficulty_level=difficulty_guide.get(difficulty, difficulty),
            target_grade=target_grade,
            words_formatted=', '.join(words_formatted),
            target_grammar=target_grammar,
            kpop_context_text=kpop_context_text if kpop_context_text else "없음"
        )
    
    def _evaluate_sentences(self, sentences, target_grammar, vocab_list):
        """생성된 문장 평가"""
        if self.evaluator and sentences:
            try:
                print("\n   📊 생성된 문장 품질 평가 중...")
                evaluation_results = self.evaluator.evaluate_batch(
                    sentences,
                    grammar=target_grammar,
                    vocab=vocab_list
                )
                
                # 평가 결과를 critique_summary에 포함
                critique_summary = []
                for sent, eval_res in zip(sentences, evaluation_results):
                    critique_summary.append({
                        "sentence": sent,
                        "grammar_ok": eval_res.get("grammar_ok", False),
                        "vocab_ok": eval_res.get("vocab_ok", False)
                    })
                
                return critique_summary
                
            except Exception as e:
                print(f"   ⚠️ 평가 중 오류: {e}")
        
        # 평가 없이 기본 형식
        return [{"sentence": s} for s in sentences]
    
    def format_output(self, state: GraphState) -> GraphState:
        """최종 출력 포맷팅 및 JSON 저장"""
        output = f"=== 한국어 학습 문제 생성 결과 ===\n"
        output += f"난이도: {state['difficulty_level']}\n"
        
        if 'target_grade' in state:
            output += f"문법 Grade: {state['target_grade']}\n"
        
        output += "\n생성된 예문:\n"
        for i, sentence in enumerate(state['generated_sentences'], 1):
            output += f"{i}. {sentence}\n"
        
        # JSON 파일 저장
        if 'sentence_data' in state and state['sentence_data']:
            saved_file = self._save_to_json(state['sentence_data'])
            output += f"\n예문이 저장되었습니다: {saved_file}\n"
        
        return {"final_output": output}
    
    def _save_to_json(self, sentence_data: dict) -> str:
        """JSON 파일로 저장"""
        out_dir = Path(SENTENCE_SAVE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        level = sentence_data.get("level", "grade1")
        title = sentence_data.get("title", "untitled")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base = f"sentences_{level}_{title}_{timestamp}"
        safe_base = sanitize_filename(base)
        filepath = out_dir / f"{safe_base}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sentence_data, f, ensure_ascii=False, indent=2)

        return str(filepath)


# Agentic RAG 구현
class AgenticKoreanLearningNodes(KoreanLearningNodes):
    """
    Agentic RAG 노드 (KoreanLearningNodes 상속)
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
        
        # Agentic 에이전트 추가
        self.query_agent = QueryAnalysisAgent(llm)
        self.quality_agent = QualityCheckAgent(llm)
    
    def analyze_query_agent(self, state: GraphState) -> GraphState:
        """쿼리 분석 에이전트 노드"""
        print("\n🔍 [Agent] Query Analysis")
        analysis = self.query_agent.analyze(state['input_text'])
        
        print(f"   Difficulty: {analysis['difficulty']}")
        print(f"   Topic: {analysis['topic']}")
        print(f"   Needs K-pop: {analysis.get('needs_kpop', False)}")
        print(f"   K-pop Groups: {analysis.get('kpop_groups', [])}")
        
        return {
            "difficulty_level": analysis['difficulty'],
            "query_analysis": analysis
        }
    
    def retrieve_kpop_mixed(self, state: GraphState) -> GraphState:
        """K-pop 검색 노드 (DB 전용)"""
        print("\n🎵 [Agent] K-pop Retrieval (DB Only)")
        
        level = state['difficulty_level']
        query = state['input_text']
        
        kpop_db_docs = self.kpop_retriever.invoke(query, level)
        kpop_db_docs = kpop_db_docs[:5]
        print(f"   DB 검색: {len(kpop_db_docs)}개")
        
        return {"kpop_docs": kpop_db_docs}
    
    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드"""
        print("\n✅ [Agent] 품질 검증")
        
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        result = self.quality_agent.check(
            vocab_count=len(state.get('vocabulary_docs', [])),
            grammar_count=len(state.get('grammar_docs', [])),
            kpop_db_count=len(state.get('kpop_docs', [])),
            needs_kpop=needs_kpop
        )
        
        print(f"   어휘: {result['vocab_count']}개")
        print(f"   문법: {result['grammar_count']}개")
        print(f"   K-pop: {result['kpop_db_count']}개")
        print(f"   상태: {result['message']}")
        
        return {"quality_check": result}
    
    def generate_sentences_with_kpop(self, state):
        """
        K-pop 메타데이터를 활용한 한국어 학습 문장 생성
        3개 생성 → 평가 수행
        """
        print("\n✏️ [Agent] 한국어 학습 문장 생성 (K-pop 통합)")
        
        from utils import extract_words_from_docs, extract_grammar_with_grade
        
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 쿼리 분석 정보
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        specified_groups = query_analysis.get('kpop_groups', [])
        
        print(f"   쿼리 분석: needs_kpop={needs_kpop}, 지정 그룹={specified_groups}")
        
        # K-pop 메타데이터 처리
        kpop_metadata, kpop_context_text, kpop_groups = self._process_kpop_docs(
            state.get('kpop_docs', []),
            specified_groups
        )
        
        has_kpop = len(kpop_metadata) > 0
        
        if has_kpop:
            print(f"   K-pop 정보: {len(kpop_metadata)}개 - {kpop_groups}")
        else:
            print(f"   K-pop 정보: 없음")
        
        # 어휘/문법 준비
        words_formatted = []
        vocab_list = []  # 평가용
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
            vocab_list.append(word)
        
        if grammar_info:
            random_grammar_item = random.choice(grammar_info)
            target_grammar = random_grammar_item['grammar']
            target_grade = random_grammar_item['grade']
        else:
            target_grammar = "기본 문법"
            target_grade = 1
        
        difficulty = state['difficulty_level']
        
        # 프롬프트 생성
        prompt = self._build_kpop_prompt(
            difficulty,
            target_grade,
            target_grammar,
            words_formatted,
            has_kpop,
            needs_kpop,
            kpop_context_text,
            kpop_groups
        )
        
        print(f"\n   🎯 타겟: 문법 '{target_grammar}' + 어휘 {len(words_formatted)}개")
        
        # 문장 생성 (3개)
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()][:3]
        
        print(f"   ✅ {len(sentences)}개 문장 생성 완료")
        
        # 평가 수행
        critique_summary = self._evaluate_sentences(
            sentences,
            target_grammar,
            vocab_list
        )
        
        # JSON 저장 데이터
        save_data = {
            "level": target_grade,
            "target_grammar": target_grammar,
            "kpop_references": kpop_metadata,
            "specified_groups": specified_groups,
            "critique_summary": critique_summary
        }
        
        messages = [
            ("user", state['input_text']),
            ("assistant", "\n".join(sentences))
        ]
        
        return {
            "generated_sentences": sentences,
            "messages": messages,
            "sentence_data": save_data,
            "target_grade": target_grade
        }
    
    def _process_kpop_docs(self, kpop_docs, specified_groups):
        """K-pop 문서 처리 및 필터링"""
        kpop_metadata = []
        kpop_context_text = ""
        kpop_groups = []
        
        if not kpop_docs:
            return kpop_metadata, kpop_context_text, kpop_groups
        
        # 필터링
        filtered_docs = []
        if specified_groups:
            for doc in kpop_docs:
                group = doc.metadata.get('group', '')
                if any(g.upper() == group.upper() for g in specified_groups):
                    filtered_docs.append(doc)
            
            if not filtered_docs:
                filtered_docs = kpop_docs[:3]
        else:
            filtered_docs = kpop_docs[:3]
        
        # 메타데이터 추출
        for doc in filtered_docs:
            group = doc.metadata.get('group', '')
            if group:
                kpop_groups.append(group)
                
                meta = {
                    "group": group,
                    "agency": doc.metadata.get('agency', ''),
                    "fandom": doc.metadata.get('fandom', ''),
                    "concepts": doc.metadata.get('concepts', []),
                    "members": [m.get("name", "") for m in doc.metadata.get('members', [])[:4]]
                }
                kpop_metadata.append(meta)
                
                # 컨텍스트 텍스트
                kpop_context_text += f"【{group}】\n"
                if meta['agency']:
                    kpop_context_text += f"  소속사: {meta['agency']}\n"
                if meta['fandom']:
                    kpop_context_text += f"  팬덤: {meta['fandom']}\n"
                if meta['concepts']:
                    kpop_context_text += f"  컨셉: {', '.join(meta['concepts'])}\n"
                if meta['members']:
                    kpop_context_text += f"  멤버: {', '.join(meta['members'])}\n"
                kpop_context_text += "\n"
        
        return kpop_metadata, kpop_context_text, kpop_groups
    
    def _build_kpop_prompt(self, difficulty, target_grade, target_grammar, 
                          words_formatted, has_kpop, needs_kpop, 
                          kpop_context_text, kpop_groups):
        """K-pop 통합 프롬프트 생성"""
        difficulty_guide = {
            "basic": "초급 (TOPIK 1-2급): 짧고 간단한 문장",
            "intermediate": "중급 (TOPIK 3-4급): 자연스러운 일상 표현",
            "advanced": "고급 (TOPIK 5-6급): 복잡한 문장 구조"
        }
        
        # K-pop 지시사항
        kpop_instruction = ""
        kpop_requirement = ""
        
        if has_kpop and needs_kpop:
            groups_text = ', '.join(kpop_groups)
            kpop_instruction = f"""
【K-pop 그룹 정보】
{kpop_context_text}

⚠️ K-pop 필수 규칙:
- 위 그룹({groups_text})만 사용
- 영어는 한국어로: "BLACKPINK"→"블랙핑크"
- 3개 문장 모두 K-pop 포함
"""
            kpop_requirement = f"필수: {groups_text} 내용 포함"
            
        elif has_kpop:
            groups_text = ', '.join(kpop_groups)
            kpop_instruction = f"""
【K-pop 정보 (선택)】
{kpop_context_text}
"""
            kpop_requirement = f"선택: {groups_text} 활용 가능"
        
        prompt = f"""한국어 학습용 예문을 정확히 3개 생성하세요.

【학습 정보】
- 수준: {difficulty_guide.get(difficulty)}
- 문법: {target_grammar} (Grade {target_grade})
- 어휘: {', '.join(words_formatted)}
{kpop_instruction}
【생성 규칙】
1. 문법 '{target_grammar}' 필수 사용
2. 제시 어휘 중 3개 이상 포함
3. 문장 3개만, 번호 없이
{f'4. {kpop_requirement}' if kpop_requirement else ''}

예문:
"""
        return prompt
    
    def format_output_agentic(self, state: GraphState) -> GraphState:
        """Agentic RAG 출력 포맷팅"""
        print("\n📄 [Agent] 최종 출력")
        
        output = "=" * 80 + "\n"
        output += "🎓 한국어 학습 문제 생성 (Agentic RAG)\n"
        output += "=" * 80 + "\n\n"
        
        # 생성된 문장
        sentences = state.get('generated_sentences', [])
        if sentences:
            output += "【생성된 학습 예문】\n"
            for i, sentence in enumerate(sentences, 1):
                output += f"   {i}. {sentence}\n"
        
        # 파일 저장
        if 'sentence_data' in state and state['sentence_data']:
            saved_file = self._save_to_json(state['sentence_data'])
            output += f"\n💾 저장: {saved_file}\n"
        
        output += "\n" + "=" * 80 + "\n"
        
        return {"final_output": output}