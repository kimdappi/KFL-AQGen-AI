"""
LangGraph 노드 정의 (개선된 재생성 로직)
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

# Evaluator 임포트
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
        
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 어휘 포맷팅
        words_formatted = []
        vocab_list = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
            vocab_list.append(word)
        
        target_grammar = grammar_info[0]['grammar'] if grammar_info else "기본 문법"
        target_grade = grammar_info[0]['grade'] if grammar_info else 1
        
        print("grammar : ", target_grammar)
        print("grade : ", target_grade)
        
        difficulty = state['difficulty_level']
        difficulty_guide = {
            "basic": "초급 학습자 (TOPIK 1-2급): 짧고 간단한 문장, 기본 시제 사용",
            "intermediate": "중급 학습자 (TOPIK 3-4급): 다양한 연결어미, 자연스러운 일상 대화 표현",
            "advanced": "고급 학습자 (TOPIK 5-6급): 복잡한 문장 구조, 격식체나 문어체 가능"
        }
        
        # 각 문장에 서로 다른 어휘를 강제할당 (최대 3개 사용)
        selected_words = vocab_list[:3]

        # K-pop 컨텍스트가 있으면 상위 5개에서 3개를 고유하게 선택하여 문장별 강제 할당
        assigned_kpop = []  # [{group, song, members[], concepts[]}]
        if 'kpop_docs' in state and state['kpop_docs']:
            pool = state['kpop_docs'][:5]
            seen = set()
            for d in pool:
                group = (d.metadata.get('group', '') or '').strip()
                song = (d.metadata.get('song', '') or '').strip()
                members = [m.get('name', '').strip() for m in (d.metadata.get('members', []) or []) if m.get('name')]
                concepts = [c.strip() for c in (d.metadata.get('concepts', []) or []) if isinstance(c, str) and c.strip()]
                key = group.lower() if group else (song.lower() if song else None)
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                assigned_kpop.append({
                    "group": group,
                    "song": song,
                    "members": members[:3],
                    "concepts": concepts[:3]
                })
                if len(assigned_kpop) >= 3:
                    break

        prompt = self._build_generation_prompt(
            difficulty,
            target_grade,
            words_formatted,
            target_grammar,
            kpop_context_text,
            difficulty_guide,
            selected_words,
            vocab_list,
            assigned_kpop
        )

        # 문장 생성 (검증 포함, 최대 2회 재시도)
        max_attempts = 2
        sentences = []
        for attempt in range(max_attempts):
            response = self.llm.predict(prompt)
            candidates = response.strip().split('\n')
            candidates = [s.strip() for s in candidates if s.strip()][:3]

            # 3문장 확보 실패 시 재시도
            if len(candidates) < 3:
                continue

            # 어휘 강제 사용 검증: 문장1→selected_words[0], 문장2→...[1], 문장3→...[2]
            ok = True
            for idx, word in enumerate(selected_words):
                if idx >= 3:
                    break
                if word.lower() not in candidates[idx].lower():
                    ok = False
                    break

            # K-pop 컨텍스트 강제 검증 (있을 때만): 문장1→ctx1, 문장2→ctx2, 문장3→ctx3
            if ok and assigned_kpop:
                for idx, ctx in enumerate(assigned_kpop):
                    if idx >= 3:
                        break
                    group = (ctx.get('group') or '').lower()
                    song = (ctx.get('song') or '').lower()
                    members = [(m or '').lower() for m in (ctx.get('members') or [])]
                    concepts = [(c or '').lower() for c in (ctx.get('concepts') or [])]
                    sent_lower = candidates[idx].lower()
                    included = False
                    if group and group in sent_lower:
                        included = True
                    if not included and song and song in sent_lower:
                        included = True
                    if not included and any(m and m in sent_lower for m in members):
                        included = True
                    if not included and any(c and c in sent_lower for c in concepts):
                        included = True
                    if not included:
                        ok = False
                        break

            if ok:
                sentences = candidates
                break

            # 실패 시 프롬프트를 더 강하게 보강하여 재시도
            missing_idx = idx + 1
            strengthen_note = f"\n[강제 규칙 재확인] 문장{missing_idx}에 반드시 '{selected_words[idx]}'를 포함하세요."
            if assigned_kpop and idx < len(assigned_kpop):
                g = assigned_kpop[idx].get('group')
                s = assigned_kpop[idx].get('song')
                ms = assigned_kpop[idx].get('members') or []
                cs = assigned_kpop[idx].get('concepts') or []
                options = []
                if g:
                    options.append(f"그룹 '{g}'")
                if s:
                    options.append(f"곡명 '{s}'")
                if ms:
                    options.append("멤버 " + ", ".join([f"'{m}'" for m in ms]))
                if cs:
                    options.append("컨셉 " + ", ".join([f"'{c}'" for c in cs]))
                if options:
                    strengthen_note += f" 또한 문장{missing_idx}에 K-pop 관련 요소 ({' 또는 '.join(options)}) 중 하나를 반드시 포함하세요."
            prompt = prompt + strengthen_note

        # 마지막 시도까지 실패한 경우라도 최신 candidates 사용
        if not sentences:
            sentences = candidates if 'candidates' in locals() else []
        
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
                                target_grammar, kpop_context_text, difficulty_guide,
                                selected_words=None, vocab_raw=None, assigned_kpop=None):
        """프롬프트 템플릿 생성"""
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
2. 문법 패턴 {target_grammar} 필수 포함
3. 제시된 단어 문장 하나당 최소 1개씩 겹치지 않게 필수 포함
4. 아래 어휘·K-pop 강제 할당을 반드시 지키기:
   문장1: '{w1}' 포함{kc1}
   문장2: '{w2}' 포함{kc2}
   문장3: '{w3}' 포함{kc3}

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
1. 중급 수준의 문장 생성
2. 문법 {target_grammar} 필수 포함
3. 제시된 어휘 문장당 최소 1개씩 겹치지 않게 필수 포함
4. 아래 어휘·K-pop 강제 할당을 반드시 지키기:
   문장1: '{w1}' 포함{kc1}
   문장2: '{w2}' 포함{kc2}
   문장3: '{w3}' 포함{kc3}

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
2. 문법 {target_grammar} 필수 포함해서 심화 활용
3. 제시된 어휘 중 문장당 최소 1개 겹치지 않게 필수 포함
4. 아래 어휘·K-pop 강제 할당을 반드시 지키기:
   문장1: '{w1}' 포함{kc1}
   문장2: '{w2}' 포함{kc2}
   문장3: '{w3}' 포함{kc3}

출력: 예문 3개만
"""
        }
        
        template = prompt_templates.get(difficulty, prompt_templates["intermediate"])
        # 강제 할당 단어/K-pop 준비
        w1 = (selected_words[0] if selected_words and len(selected_words) > 0 else '')
        w2 = (selected_words[1] if selected_words and len(selected_words) > 1 else w1)
        w3 = (selected_words[2] if selected_words and len(selected_words) > 2 else w2)

        def make_kpop_clause(idx):
            if not assigned_kpop or len(assigned_kpop) <= idx:
                return ""
            g = assigned_kpop[idx].get('group') or ''
            s = assigned_kpop[idx].get('song') or ''
            ms = assigned_kpop[idx].get('members') or []
            cs = assigned_kpop[idx].get('concepts') or []
            parts = []
            if g:
                parts.append(f"그룹 '{g}'")
            if s:
                parts.append(f"곡명 '{s}'")
            if ms:
                parts.append("멤버 " + ", ".join([f"'{m}'" for m in ms]))
            if cs:
                parts.append("컨셉 " + ", ".join([f"'{c}'" for c in cs]))
            if not parts:
                return ""
            return ", K-pop 관련 요소 (" + " 또는 ".join(parts) + ") 중 하나 포함"

        kc1 = make_kpop_clause(0)
        kc2 = make_kpop_clause(1)
        kc3 = make_kpop_clause(2)

        return template.format(
            difficulty_level=difficulty_guide.get(difficulty, difficulty),
            target_grade=target_grade,
            words_formatted=', '.join(words_formatted),
            target_grammar=target_grammar,
            kpop_context_text=kpop_context_text if kpop_context_text else "없음",
            w1=w1,
            w2=w2,
            w3=w3,
            kc1=kc1,
            kc2=kc2,
            kc3=kc3
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


# =====================================
# Agentic RAG 구현 (개선 버전)
# =====================================
class AgenticKoreanLearningNodes(KoreanLearningNodes):
    """Agentic RAG 노드 - 재생성 로직 최적화"""
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
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
        """K-pop 검색 노드"""
        print("\n🎵 [Agent] K-pop Retrieval")
        
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
        개선된 문장 생성 - 3개 보장 및 리소스 분배
        """
        import random
        print("\n✏️ [Agent] 한국어 학습 문장 생성 (3개 보장)")
        
        from utils import extract_words_from_docs, extract_grammar_with_grade
        
        # 데이터 추출 (정확히 3개씩)
        words_info = extract_words_from_docs(state['vocabulary_docs'])[:3]
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])[:1]
        
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        specified_groups = query_analysis.get('kpop_groups', [])
        
        # K-pop 정보 처리 (개선된 버전)
        kpop_metadata, kpop_contexts = self._process_kpop_docs_enhanced(
            state.get('kpop_docs', [])[:3],
            specified_groups
        )
        
        # 기본 정보 설정
        vocab_list = [word for word, _ in words_info]
        # 어휘 부족시 채우기
        while len(vocab_list) < 3:
            vocab_list.append(f"학습단어{len(vocab_list)+1}")
        vocab_list = vocab_list[:3]  # 정확히 3개
        
        target_grammar = grammar_info[0]['grammar'] if grammar_info else "기본 문법"
        target_grade = grammar_info[0]['grade'] if grammar_info else 1
        difficulty = state['difficulty_level']
        
        print(f"   타겟: 문법 '{target_grammar}' + 어휘 {vocab_list}")
        if needs_kpop and kpop_contexts:
            print(f"   K-pop 컨텍스트: {len(kpop_contexts)}개")
        
        # 3개 문장 개별 생성
        generated_sentences = []
        
        for i in range(3):
            # 각 문장별 리소스 할당
            vocab = vocab_list[i] if i < len(vocab_list) else vocab_list[0]
            kpop_ctx = kpop_contexts[i] if i < len(kpop_contexts) else None
            
            # 개별 문장 프롬프트
            prompt = f"""한국어 학습용 문장 1개를 생성하세요.

【필수 조건】
- 수준: {difficulty} (TOPIK {target_grade})
- 문법: '{target_grammar}' 반드시 포함
- 어휘: '{vocab}' 반드시 포함"""
            
            if kpop_ctx:
                prompt += f"\n- K-pop: {kpop_ctx['display']} 자연스럽게 포함"
            
            prompt += """

【요구사항】
- 10-20자 길이
- 자연스럽고 실용적인 문장
- 번호나 기호 없이 문장만

문장:"""
            
            try:
                response = self.llm.predict(prompt)
                sentence = response.strip().lstrip('0123456789.-) ').strip()
                
                if sentence and len(sentence) > 5:
                    generated_sentences.append(sentence)
                    print(f"      문장{i+1}: {sentence}")
                else:
                    # 백업 문장
                    if kpop_ctx:
                        fallback = f"{kpop_ctx['group']}의 {vocab}{target_grammar} 좋아해요."
                    else:
                        fallback = f"{vocab}{target_grammar} 연습해요."
                    generated_sentences.append(fallback)
                    print(f"      문장{i+1} (대체): {fallback}")
            except Exception as e:
                print(f"      문장{i+1} 생성 오류: {e}")
                fallback = f"{vocab}{target_grammar} 공부합니다."
                generated_sentences.append(fallback)
        
        # 정확히 3개 보장
        while len(generated_sentences) < 3:
            fallback = f"{target_grammar} 패턴 예문입니다."
            generated_sentences.append(fallback)
        generated_sentences = generated_sentences[:3]
        
        print(f"   ✅ 최종 생성: {len(generated_sentences)}개 문장")
        
        # 데이터 저장
        sentence_data = {
            "level": f"grade{target_grade}",
            "title": sanitize_filename(state['input_text'][:50]),
            "target_grammar": target_grammar,
            "vocabulary": vocab_list,
            "critique_summary": [
                {
                    "sentence": sent,
                    "vocab_used": vocab_list[i] if i < len(vocab_list) else "",
                    "kpop_context": kpop_contexts[i]['display'] if i < len(kpop_contexts) else ""
                }
                for i, sent in enumerate(generated_sentences)
            ]
        }
        
        if kpop_metadata:
            sentence_data["kpop_references"] = kpop_metadata
        
        return {
            "generated_sentences": generated_sentences,
            "sentence_data": sentence_data,
            "target_grade": target_grade
        }

    def _process_kpop_docs_enhanced(self, kpop_docs, specified_groups):
        """K-pop 문서 처리 - 다양한 필드 활용 버전"""
        import random
        
        kpop_metadata = []
        kpop_contexts = []  # 각 문장별 K-pop 컨텍스트
        
        if not kpop_docs:
            return kpop_metadata, kpop_contexts
        
        # 필터링 (specified_groups가 있으면 해당 그룹만)
        filtered_docs = kpop_docs[:3]
        if specified_groups:
            filtered = []
            for doc in kpop_docs:
                group = doc.metadata.get('group', '')
                if any(g.upper() == group.upper() for g in specified_groups):
                    filtered.append(doc)
            if filtered:
                filtered_docs = filtered[:3]
        
        # 각 문서별로 다양한 컨텍스트 생성
        for doc in filtered_docs:
            meta = doc.metadata
            group = meta.get('group', '')
            
            if not group:
                continue
            
            # 메타데이터 저장
            full_meta = {
                "group": group,
                "agency": meta.get('agency', ''),
                "fandom": meta.get('fandom', ''),
                "concepts": meta.get('concepts', []),
                "members": [m.get("name", "") for m in meta.get('members', [])[:3]]
            }
            kpop_metadata.append(full_meta)
            
            # 다양한 컨텍스트 옵션 생성
            context_options = []
            
            # 1. 그룹명 컨텍스트
            context_options.append({
                'type': 'group',
                'display': f"{group}",
                'group': group
            })
            
            # 2. 멤버 컨텍스트
            members = meta.get('members', [])
            if members:
                member = random.choice(members[:5])
                member_name = member.get('name', '')
                if member_name:
                    context_options.append({
                        'type': 'member',
                        'display': f"{group}의 {member_name}",
                        'group': group
                    })
            
            # 3. 팬덤 컨텍스트
            fandom = meta.get('fandom', '')
            if fandom:
                context_options.append({
                    'type': 'fandom',
                    'display': f"{group} 팬덤 {fandom}",
                    'group': group
                })
            
            # 4. 소속사 컨텍스트
            agency = meta.get('agency', '')
            if agency:
                context_options.append({
                    'type': 'agency',
                    'display': f"{agency} 소속 {group}",
                    'group': group
                })
            
            # 5. 컨셉 컨텍스트
            concepts = meta.get('concepts', [])
            if concepts:
                concept = random.choice(concepts)
                context_options.append({
                    'type': 'concept',
                    'display': f"{concept} 컨셉의 {group}",
                    'group': group
                })
            
            # 랜덤하게 하나 선택
            if context_options:
                selected = random.choice(context_options)
                kpop_contexts.append(selected)
        
        # 3개 맞추기 (부족하면 반복)
        while len(kpop_contexts) < 3 and kpop_contexts:
            kpop_contexts.append(random.choice(kpop_contexts))
        
        return kpop_metadata, kpop_contexts


    def _calculate_score(self, critique, kpop_ok):
        # 우선 쿼리에서 식별된 그룹으로 필터링, 없으면 상위 결과 사용
        if kpop_groups:
            kpool = [d for d in kdocs_all if (d.metadata.get('group', '') or '').upper() in {g.upper() for g in kpop_groups}][:5]
        else:
            kpool = kdocs_all[:5]
        seen_keys = set()
        for d in kpool:
            group = (d.metadata.get('group', '') or '').strip()
            song = (d.metadata.get('song', '') or '').strip()
            members = [m.get('name', '').strip() for m in (d.metadata.get('members', []) or []) if m.get('name')]
            concepts = [c.strip() for c in (d.metadata.get('concepts', []) or []) if isinstance(c, str) and c.strip()]
            key = group.lower() if group else (song.lower() if song else None)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            assigned_kpop.append({
                'group': group,
                'song': song,
                'members': members[:3],
                'concepts': concepts[:3]
            })
            if len(assigned_kpop) >= 3:
                break
        
        # ===================================
        # 3회 시도, 가장 좋은 결과 선택
        # ===================================
        max_attempts = 3
        all_attempts = []
        
        for attempt in range(max_attempts):
            print(f"\n   📝 시도 {attempt + 1}/{max_attempts}")
            
            # 점진적으로 강화된 프롬프트 생성
            prompt = self._build_progressive_prompt(
                attempt,
                difficulty,
                target_grade,
                target_grammar,
                vocab_list,
                kpop_groups,
                kpop_context_text,
                needs_kpop,
                assigned_kpop,
                all_attempts  # 이전 실패 정보
            )
            
            # 문장 생성
            response = self.llm.predict(prompt)
            sentences = [s.strip() for s in response.strip().split('\n') if s.strip()][:3]
            
            # 평가 수행
            critique = self._evaluate_sentences(sentences, target_grammar, vocab_list)
            
            # K-pop 포함 체크 (문장별 할당 기준)
            if needs_kpop and assigned_kpop:
                kpop_ok = self._check_kpop_assigned(sentences, assigned_kpop)
            elif needs_kpop:
                kpop_ok = self._check_kpop_inclusion(sentences, kpop_groups)
            else:
                kpop_ok = True
            
            # 점수 계산
            score = self._calculate_score(critique, kpop_ok)
            
            all_attempts.append({
                'sentences': sentences,
                'critique': critique,
                'score': score,
                'kpop_ok': kpop_ok
            })
            
            print(f"      점수: {score}/3 (문법+어휘+K-pop)")
            
            # 완벽한 결과면 즉시 종료
            if score == 3:
                print(f"   ✅ 완벽한 문장 생성!")
                break
        
        # 가장 좋은 결과 선택
        best_attempt = max(all_attempts, key=lambda x: x['score'])
        final_sentences = best_attempt['sentences']
        critique_summary = best_attempt['critique']
        
        print(f"\n   🏆 최종 선택: 점수 {best_attempt['score']}/3")
        
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
            ("assistant", "\n".join(final_sentences))
        ]
        
        return {
            "generated_sentences": final_sentences,
            "messages": messages,
            "sentence_data": save_data,
            "target_grade": target_grade
        }
    
    def _calculate_score(self, critique, kpop_ok):
        """
        문장 품질 점수 계산
        - 문법 충족: +1점
        - 어휘 충족: +1점
        - K-pop 포함 (필요시): +1점
        """
        grammar_pass = sum(1 for c in critique if c.get('grammar_ok', False))
        vocab_pass = sum(1 for c in critique if c.get('vocab_ok', False))
        
        score = 0
        if grammar_pass == 3:  # 3개 문장 모두 문법 충족
            score += 1
        if vocab_pass == 3:    # 3개 문장 모두 어휘 충족
            score += 1
        if kpop_ok:             # K-pop 조건 충족
            score += 1
        
        return score
    
    def _build_progressive_prompt(self, *args, **kwargs):
        """더 이상 사용하지 않음 - generate_sentences_with_kpop에서 직접 프롬프트 생성"""
        return ""
    
    def _process_kpop_docs(self, kpop_docs, specified_groups):
        """K-pop 문서 처리 및 필터링"""
        kpop_metadata = []
        kpop_context_text = ""
        kpop_groups = []
        
        if not kpop_docs:
            return kpop_metadata, kpop_context_text, kpop_groups
        
        # 필터링
        filtered_docs = kpop_docs[:3]
        if specified_groups:
            filtered = []
            for doc in kpop_docs:
                group = doc.metadata.get('group', '')
                if any(g.upper() == group.upper() for g in specified_groups):
                    filtered.append(doc)
            
            if filtered:
                filtered_docs = filtered[:3]
        
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
    
    def _check_kpop_inclusion(self, sentences, kpop_groups):
        """K-pop 그룹명 포함 여부 체크"""
        if not kpop_groups:
            return True
        
        # 영어 그룹명의 한글 변환 매핑
        korean_names = {
            "BLACKPINK": "블랙핑크",
            "BTS": "방탄소년단",
            "TWICE": "트와이스",
            "NewJeans": "뉴진스",
            "EXO": "엑소",
            "Stray Kids": "스트레이키즈",
            "aespa": "에스파",
            "SEVENTEEN": "세븐틴"
        }
        
        # 모든 문장에서 K-pop 정보 포함 확인
        for sentence in sentences:
            has_kpop = False
            for group in kpop_groups:
                # 영어명 또는 한글명 체크
                if (group.lower() in sentence.lower() or 
                    korean_names.get(group, "").lower() in sentence.lower()):
                    has_kpop = True
                    break
            
            if not has_kpop:
                return False
        
        return True

    def _check_kpop_assigned(self, sentences, assigned_kpop):
        """문장별로 할당된 K-pop 요소(그룹/곡명/멤버/컨셉) 포함 여부 체크"""
        if not sentences or not assigned_kpop:
            return True
        for idx, sentence in enumerate(sentences[:3]):
            if idx >= len(assigned_kpop):
                continue
            ctx = assigned_kpop[idx]
            sent_lower = sentence.lower()
            group = (ctx.get('group') or '').lower()
            song = (ctx.get('song') or '').lower()
            members = [(m or '').lower() for m in (ctx.get('members') or [])]
            concepts = [(c or '').lower() for c in (ctx.get('concepts') or [])]

            included = False
            if group and group in sent_lower:
                included = True
            if not included and song and song in sent_lower:
                included = True
            if not included and any(m and m in sent_lower for m in members):
                included = True
            if not included and any(c and c in sent_lower for c in concepts):
                included = True
            if not included:
                return False
        return True

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