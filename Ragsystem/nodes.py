# =====================================
# nodes.py (개선 버전) - Evaluator 기반 최적화
# =====================================
"""
LangGraph 노드 정의 (개선된 재생성 로직)
- 3회 시도 후 가장 좋은 결과 자동 선택
- 점진적 프롬프트 강화
- 명확한 어휘 할당
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
                                target_grammar, kpop_context_text, difficulty_guide):
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
        개선된 문장 생성 로직
        - 3회 시도 후 가장 좋은 결과 선택
        - 점진적 프롬프트 강화
        """
        print("\n✏️ [Agent] 한국어 학습 문장 생성 (최적화)")
        
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
        
        print(f"   K-pop 정보: {len(kpop_metadata)}개 - {kpop_groups}" if kpop_metadata else "   K-pop 정보: 없음")
        
        # 어휘/문법 준비
        vocab_list = [word for word, _ in words_info[:5]]
        target_grammar = grammar_info[0]['grammar'] if grammar_info else "기본 문법"
        target_grade = grammar_info[0]['grade'] if grammar_info else 1
        difficulty = state['difficulty_level']
        
        print(f"\n   🎯 타겟: 문법 '{target_grammar}' + 어휘 {vocab_list}")
        
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
                all_attempts  # 이전 실패 정보
            )
            
            # 문장 생성
            response = self.llm.predict(prompt)
            sentences = [s.strip() for s in response.strip().split('\n') if s.strip()][:3]
            
            # 평가 수행
            critique = self._evaluate_sentences(sentences, target_grammar, vocab_list)
            
            # K-pop 포함 체크
            kpop_ok = self._check_kpop_inclusion(sentences, kpop_groups) if needs_kpop else True
            
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
    
    def _build_progressive_prompt(self, attempt, difficulty, target_grade, 
                                  target_grammar, vocab_list, kpop_groups,
                                  kpop_context_text, needs_kpop, previous_attempts):
        """
        점진적으로 강화되는 프롬프트 생성
        - attempt 0: 기본 프롬프트
        - attempt 1: 강화된 프롬프트 + 이전 실패 정보
        - attempt 2: 최대 강화 + 구체적인 어휘 할당
        """
        difficulty_guide = {
            "basic": "초급 (TOPIK 1-2급)",
            "intermediate": "중급 (TOPIK 3-4급)",
            "advanced": "고급 (TOPIK 5-6급)"
        }
        
        # 기본 정보
        base_info = f"""【학습 정보】
수준: {difficulty_guide.get(difficulty)}
문법: {target_grammar} (Grade {target_grade})
어휘: {', '.join(vocab_list)}
"""
        
        # K-pop 정보
        kpop_info = ""
        if kpop_groups and needs_kpop:
            kpop_info = f"""
【K-pop 정보】
{kpop_context_text}
그룹: {', '.join(kpop_groups)}
"""
        
        # 시도별 프롬프트
        if attempt == 0:
            # 첫 시도: 기본 프롬프트
            prompt = f"""한국어 학습용 예문을 정확히 3개 생성하세요.

{base_info}{kpop_info}
【생성 규칙】
1. 문법 '{target_grammar}' 3개 문장 모두에 필수 사용
2. 제시 어휘 중 각 문장마다 최소 1개 이상 포함 (겹치지 않게)
3. 자연스러운 한국어 문장
4. 번호 없이 문장 3개만

예문:
"""
        
        elif attempt == 1:
            # 두 번째 시도: 강화 + 이전 실패 분석
            prev = previous_attempts[0]
            failed_items = []
            
            if prev['score'] < 3:
                critique = prev['critique']
                grammar_fail = sum(1 for c in critique if not c.get('grammar_ok', False))
                vocab_fail = sum(1 for c in critique if not c.get('vocab_ok', False))
                
                if grammar_fail > 0:
                    failed_items.append(f"- 문법 '{target_grammar}' 미포함: {grammar_fail}개 문장")
                if vocab_fail > 0:
                    failed_items.append(f"- 어휘 미포함: {vocab_fail}개 문장")
                if not prev['kpop_ok'] and needs_kpop:
                    failed_items.append(f"- K-pop 정보 미포함")
            
            fail_text = "\n".join(failed_items) if failed_items else "일부 조건 미충족"
            
            prompt = f"""⚠️ 이전 시도 실패 - 반드시 모든 조건을 충족하세요!

{base_info}{kpop_info}
【이전 실패 원인】
{fail_text}

【필수 조건】
✅ 문법 '{target_grammar}' - 3개 문장 모두 반드시 포함!
✅ 어휘 {', '.join(vocab_list)} - 각 문장마다 최소 1개 겹치지 않게 포함!
{f"✅ K-pop '{', '.join(kpop_groups)}' - 3개 문장 모두 포함!" if needs_kpop and kpop_groups else ""}

【생성 규칙】
1. 문법 패턴을 명확하게 사용
2. 각 문장마다 다른 어휘 사용
3. 자연스럽고 실용적인 문장

예문:
"""
        
        else:  # attempt == 2
            # 세 번째 시도: 최대 강화 + 명확한 어휘 할당
            vocab_assignment = ""
            for i, word in enumerate(vocab_list[:3], 1):
                vocab_assignment += f"   문장{i}: '{word}' 반드시 포함\n"
            
            prompt = f"""🚨 최종 시도 - 아래 지시사항을 정확히 따르세요!

{base_info}{kpop_info}
【명확한 어휘 할당】
{vocab_assignment}

【절대 규칙】
1. 문법 '{target_grammar}' - 3개 문장 모두 명확하게 사용
2. 위 어휘 할당표대로 각 문장에 지정된 어휘 반드시 포함
3. 자연스럽고 문법적으로 완벽한 문장
{f"4. K-pop 그룹 '{', '.join(kpop_groups)}' 반드시 포함 (영어→한글)" if needs_kpop and kpop_groups else ""}

【예시 형식】
문장1: [어휘1 + 문법 + K-pop]
문장2: [어휘2 + 문법 + K-pop]
문장3: [어휘3 + 문법 + K-pop]

예문:
"""
        
        return prompt
    
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