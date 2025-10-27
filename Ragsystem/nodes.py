# =====================================
# nodes.py (Updated) - grade를 level로 사용
# 수정 완료
# =====================================
"""
LangGraph 노드 정의 (문장 저장 기능 및 grade 사용)
"""
import json
import os
import re
import random
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from typing import Any
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from Ragsystem.schema import GraphState #디렉토리 구조 정리
from utils import (#사용하는 함수만 선언해도 됨
    detect_difficulty_from_text,
    extract_words_from_docs,
    extract_grammar_with_grade  
    #extract_grammars_from_docs, format_docs # 현재 안쓰는 함수
)
from config import LLM_CONFIG,SENTENCE_SAVE_DIR #문장 저장경로 변경
from agents import QueryAnalysisAgent, QualityCheckAgent

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
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        self.vocabulary_retriever = vocabulary_retriever
        self.grammar_retriever = grammar_retriever
        self.kpop_retriever = kpop_retriever  # ✅ 추가
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

        # ✅ K-pop 정보 추출 및 포맷팅
        kpop_references = []
        kpop_context_text = ""
        
        if 'kpop_docs' in state and state['kpop_docs']:
            print(f"[참조] K-pop 문서 개수: {len(state['kpop_docs'])}")
            
            for doc in state['kpop_docs'][:3]:  # 상위 3개만
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
        
        # 문법과 grade 정보 함께 추출
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # 단어와 품사 정보 포맷팅
        words_formatted = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
        
        if grammar_info:
            # 검색된 문법 리스트에서 무작위로 하나를 선택합니다.
            random_grammar_item = random.choice(grammar_info)
            target_grammar = random_grammar_item['grammar']
            target_grade = random_grammar_item['grade']
            print("grammer : ", target_grammar)
            print("grade : ", target_grade)
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
                5. 사용자 지정 관심사인 {kpop_context_text} 반영해서 문장 생성
                
                예문 (번호 없이 문장만):
                """
                
                # --- START: 난이도별 프롬프트 분기 로직 ---
        difficulty = state['difficulty_level']

                # 난이도별 프롬프트 템플릿 정의
        prompt_templates = {
                    "basic": """
        [ROLE]
        너는 이제 막 한국어를 배우기 시작한 7살 외국인 아이들을 가르치는 **아주 친절하고 상냥한 유치원 선생님**이야. 아이들의 눈높이에 맞춰, 세상에서 가장 쉽고 재미있는 한국어 문장을 만들어줘야 해.

        [INSTRUCTIONS]
        - **미션**: 아래의 단어와 문법으로, 아이들이 "한국어 정말 재미있다!"라고 느낄 만한 예문 3개를 만들어줘.
        - **학습 수준**: {difficulty_level} (TOPIK 1~2급, Grade {target_grade})
        - **오늘의 단어**: {words_formatted}
        - **오늘의 문법**: `{target_grammar}`
        - **아이들의 관심사 (K-pop)**: {kpop_context_text}

        [SENTENCE RULES]
        1.  **쉬운 단어만!**: '오늘의 단어' 외에는 아이들도 아는 아주 기본적인 단어만 사용해. (예: 사과, 가다, 먹다, 크다)
        2.  **짧은 문장!**: 문장은 무조건 짧고 간단하게 만들어줘. (예: 주어 + 목적어 + 동사)
        3.  **재미있게!**: 아이들의 관심사인 K-pop 가수 이름이나 노래 제목을 넣어서 재미있게 만들어줘.
        4.  **정확한 문법!**: '오늘의 문법'인 `{target_grammar}`를 정확하게 사용해야 해. 만약 문법이 동사를 필요로 하면, 주어진 단어와 어울리는 동사(예: 공부하다, 숙제하다)를 찾아서 문장을 만들어줘.
        5.  **자연스럽게!**: 단어들을 억지로 조합하지 말고, 실제 한국인들이 사용할 법한 자연스러운 문장을 만들어줘.

        [OUTPUT FORMAT]
        - 다른 설명은 절대 하지 말고, 예문 3개만 한 줄씩 바로 출력해줘.

        [예문 시작]
        """,
            "intermediate": """
        [ROLE]
        너는 한국어학당에서 중급 회화 수업을 담당하는 **실력 있고 경험 많은 한국어 교사**야. 학생들이 수업이 끝나고 바로 실생활에서 써먹을 수 있는 유용한 문장을 만드는 것이 너의 역할이야.

        [INSTRUCTIONS]
        - **목표**: 아래 정보를 활용하여, 중급 학습자(TOPIK 3~4급)가 친구와 대화하거나 일상생활에서 겪는 상황에 맞는 실용적인 예문 3개를 생성해줘.
        - **학습 수준**: {difficulty_level} (TOPIK 3~4급, Grade {target_grade})
        - **핵심 어휘**: {words_formatted}
        - **목표 문법**: `{target_grammar}`
        - **학생 관심사 (K-pop)**: {kpop_context_text}

        [SENTENCE REQUIREMENTS]
        0.  **문법 준수**: 생성하는 모든 문장에는 목표 문법인 `{target_grammar}`가 **반드시** 포함되어야 한다.
        1.  **자연스러운 대화체**: 실제 한국인들이 친구와 대화할 때 사용하는 자연스러운 말투와 억양을 살려서 문장을 만들어줘.
        2.  **문법 활용**: `{target_grammar}` 문법의 의미와 쓰임이 명확하게 드러나는 문맥을 제시해줘.
        3.  **문맥의 구체성**: K-pop 노래를 듣고 감상을 말하거나, 콘서트에 가는 계획을 세우는 등 구체적인 상황을 설정해서 문장을 만들어줘.
        4.  **적절한 복잡도**: 두 문장을 자연스럽게 연결하는 등 Grade {target_grade} 수준에 맞는 문장 구조를 사용해줘.

        [OUTPUT FORMAT]
        - 번호나 부가 설명 없이, 생성된 예문 3개만 한 줄씩 출력해줘.

        [예문 시작]
        """,
            "advanced": """
        [ROLE]
        너는 한국학을 전공하는 외국인 석박사 과정 학생들의 논문 지도를 담당하는 **매우 전문적이고 논리적인 국어국문학과 교수**다. 너의 목표는 학생들이 복잡한 생각과 주장을 한국어로 명확하고 깊이 있게 표현하도록 돕는 것이다.

        [INSTRUCTIONS]
        - **과제**: 아래의 핵심 어휘와 문법을 바탕으로, 고급 학습자(TOPIK 5~6급)가 학술적인 토론이나 격식 있는 글쓰기에서 사용할 만한 수준 높은 예문 3개를 작성하라.
        - **학습 수준**: {difficulty_level} (TOPIK 5~6급, Grade {target_grade})
        - **핵심 어휘**: {words_formatted}
        - **핵심 문법**: `{target_grammar}`
        - **참고 자료 (K-pop)**: {kpop_context_text}

        [SENTENCE REQUIREMENTS]
        0.  **문법 준수**: 생성하는 모든 문장에는 목표 문법인 `{target_grammar}`가 **반드시** 포함되어야 한다.
        1.  **격식과 논리**: 문어체 혹은 격식체를 사용하여 논리적이고 객관적인 사실이나 주장을 서술하는 문장을 구성하라.
        2.  **어휘 수준**: 제시된 어휘 외에도 해당 주제를 논의하는 데 필요한 고급 어휘나 적절한 한자어를 사용하라.
        3.  **문법의 깊이**: `{target_grammar}` 문법이 가진 미묘한 뉘앙스나 심화된 쓰임을 보여줄 수 있는 복합적인 문장을 만들어라.
        4.  **주제의 심층 분석**: 참고 자료인 K-pop을 단순한 흥밋거리가 아닌, 하나의 사회·문화적 현상으로 분석하거나 비평하는 관점의 문장을 제시하라.

        [OUTPUT FORMAT]
        - 서론이나 결론 없이, 완성된 예문 3개만 한 줄씩 출력하라.

        [예문 시작]
        """
        }

        # 난이도에 맞는 프롬프트 선택 (기본값: intermediate)
        prompt_template = prompt_templates.get(difficulty, prompt_templates["intermediate"])
        prompt = prompt_template.format(
            difficulty_level=difficulty,
            target_grade=target_grade,
            words_formatted=', '.join(words_formatted),
            target_grammar=target_grammar,
            kpop_context_text=kpop_context_text if kpop_context_text else "특별한 관심사 없음" # kpop_context_text가 비어있을 경우 대비
            )
        # --- END: 난이도별 프롬프트 분기 로직 ---

        
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # JSON 형식으로 저장할 데이터 생성 (grade를 level로 사용)
        save_data = {
            "level": target_grade,  # grade 값을 level로 사용 1-6
            "target_grammar": target_grammar,
            "kpop_references": kpop_references,
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
        out_dir = Path(SENTENCE_SAVE_DIR)
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

#Agent RAG 구현 추가
class AgenticKoreanLearningNodes(KoreanLearningNodes):
    """
    Agentic RAG 노드 (기존 KoreanLearningNodes 상속)
    기존 기능을 모두 유지하면서 Agentic 기능 추가
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        # 기존 초기화
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
        
        return {
            "difficulty_level": analysis['difficulty'],
            "query_analysis": analysis
        }
    
    def retrieve_kpop_mixed(self, state: GraphState) -> GraphState:
        """
        K-pop 검색 노드 (DB 전용)
        데이터베이스에서만 K-pop 학습 자료 검색
        """
        print("\n🎵 [Agent] K-pop Retrieval (DB Only)")
        
        level = state['difficulty_level']
        query = state['input_text']
        
        # 1. 기존 DB에서 검색 (최대 5개)
        kpop_db_docs = self.kpop_retriever.invoke(query, level)
        kpop_db_docs = kpop_db_docs[:5]  # 5개로 제한
        print(f"   DB 검색: {len(kpop_db_docs)}개")
        
        return {
            "kpop_docs": kpop_db_docs
        }
    
    
    """
    Updated check_quality_agent method for nodes.py
    Replace the existing method in AgenticKoreanLearningNodes class
    """

    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드"""
        print("\n✅ [Agent] 품질 검증")
        
        # 쿼리 분석에서 K-pop 필요 여부 확인
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        result = self.quality_agent.check(
            vocab_count=len(state.get('vocabulary_docs', [])),
            grammar_count=len(state.get('grammar_docs', [])),
            kpop_db_count=len(state.get('kpop_docs', [])),
            needs_kpop=needs_kpop  # K-pop 필요 여부 전달
        )
        
        print(f"   어휘: {result['vocab_count']}개")
        print(f"   문법: {result['grammar_count']}개")
        if needs_kpop:
            print(f"   K-pop DB: {result['kpop_db_count']}개 (필요)")
        else:
            print(f"   K-pop DB: {result['kpop_db_count']}개 (불필요)")
        print(f"   상태: {result['message']}")
        
        return {"quality_check": result}
    """
    Updated generate_sentences_with_kpop method for nodes.py
    Replace the existing method in AgenticKoreanLearningNodes class
    """

    def generate_sentences_with_kpop(self, state: GraphState) -> GraphState:
        """
        K-pop 정보를 활용한 한국어 학습 문장 생성 (조건부)
        K-pop 검색 결과가 있을 때만 K-pop 맥락 포함
        """
        print("\n✏️ [Agent] 한국어 학습 문장 생성")
        
        words_info = extract_words_from_docs(state['vocabulary_docs'])
        grammar_info = extract_grammar_with_grade(state['grammar_docs'])
        
        # K-pop 정보 통합 (있을 때만)
        kpop_references = []
        kpop_context_text = ""
        has_kpop = False
        kpop_groups = []  # K-pop 그룹 리스트
        kpop_songs = []   # K-pop 노래 리스트
        
        # DB에서 가져온 K-pop 문장들
        kpop_db_docs = state.get('kpop_docs', [])
        if kpop_db_docs:
            has_kpop = True
            for doc in kpop_db_docs[:5]:
                sentence = doc.metadata.get('sentence', '')
                song = doc.metadata.get('song', '')
                group = doc.metadata.get('group', '')
                context = doc.metadata.get('context', '')
                
                if sentence:
                    kpop_references.append({
                        "sentence": sentence,
                        "song": song,
                        "group": group,
                        "context": context,
                        "source": "database"
                    })
                    kpop_context_text += f'- "{sentence}" ({song} - {group})\n'
                    
                    # 그룹명과 노래 제목 수집
                    if group and group not in kpop_groups:
                        kpop_groups.append(group)
                    if song and song not in kpop_songs:
                        kpop_songs.append(song)
        
        # needs_kpop 확인 (쿼리에 K-pop 키워드가 있는지)
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        if has_kpop:
            print(f"   K-pop 참조 문장: {len(kpop_references)}개 (DB)")
            if needs_kpop:
                print(f"   ⚠️ K-pop 쿼리 감지: K-pop 내용 필수 포함")
        else:
            print(f"   K-pop 참조 문장: 없음 (일반 예문 생성)")
        
        # 어휘 포맷팅
        words_formatted = []
        for word, wordclass in words_info[:5]:
            words_formatted.append(f"{word}({wordclass})")
        
        # 문법 선택
        if grammar_info:
            import random
            random_grammar_item = random.choice(grammar_info)
            target_grammar = random_grammar_item['grammar']
            target_grade = random_grammar_item['grade']
        else:
            target_grammar = "기본 문법"
            target_grade = 1
        
        # 난이도별 설명
        difficulty_guide = {
            "basic": "초급 학습자 (TOPIK 1-2급): 짧고 간단한 문장, 기본 시제 사용",
            "intermediate": "중급 학습자 (TOPIK 3-4급): 다양한 연결어미, 자연스러운 일상 대화 표현",
            "advanced": "고급 학습자 (TOPIK 5-6급): 복잡한 문장 구조, 격식체나 문어체 가능"
        }
        
        difficulty = state['difficulty_level']
        
        # K-pop 유무에 따른 프롬프트 생성
        if has_kpop and needs_kpop:
            # K-pop 쿼리이고 참조가 있을 때 - K-pop 내용 필수
            kpop_groups_text = ', '.join(kpop_groups[:3]) if kpop_groups else ""
            kpop_songs_text = ', '.join(kpop_songs[:3]) if kpop_songs else ""
            
            kpop_instruction = f"""
    【K-pop 참조 자료 ({len(kpop_references)}개)】
    {kpop_context_text}
    
    🎵 K-pop 그룹: {kpop_groups_text}
    🎵 노래 제목: {kpop_songs_text}

    **⚠️ K-pop 필수 포함 규칙**:
    - 위에 제시된 K-pop 그룹명(예: {kpop_groups[0] if kpop_groups else 'BTS'})을 **반드시** 3개 문장 모두에 포함해야 합니다
    - K-pop 참조 문장의 내용을 자연스럽게 활용하세요
    - 예시: "{kpop_groups[0] if kpop_groups else 'BLACKPINK'}처럼 춤추고 싶어요"
    - K-pop 관련 내용이 모든 문장에 포함되어야 합니다.
    """
            kpop_requirement = f"**필수**: K-pop 관련 내용(그룹명, 노래 등)이 3개 문장 모두에 포함되어야 합니다."
        elif has_kpop and not needs_kpop:
            # K-pop 참조는 있지만 필수는 아닐 때
            kpop_instruction = f"""
    【K-pop 참조 문장 ({len(kpop_references)}개)】
    {kpop_context_text}

    **K-pop 활용 규칙**:
    - 위의 K-pop 참조 문장 내용을 자연스럽게 활용할 수 있습니다
    - K-pop 아티스트, 노래, 문화를 언급하면 학습자에게 더 흥미로울 수 있습니다
    """
            kpop_requirement = "K-pop 관련 내용이 자연스럽게 포함되면 좋지만 필수는 아닙니다"
        else:
            # K-pop 참조가 없을 때
            kpop_instruction = ""
            kpop_requirement = ""
        
        # 문장 생성 프롬프트
        prompt = f"""당신은 외국인을 위한 한국어 교육 문제 생성 전문가입니다.
    다음 조건을 활용하여 한국어 학습용 예문을 **정확히 3개** 생성해주세요.

    【학습자 정보】
    - 수준: {difficulty_guide.get(difficulty, '일반')}
    - 목표 문법: {target_grammar} (등급 {target_grade})

    【활용 어휘】
    {', '.join(words_formatted)}
    {kpop_instruction}
    【문장 생성 규칙】
    1. **필수**: 제시된 어휘 중 최소 3개 이상 포함
    2. **필수**: 목표 문법 '{target_grammar}' 반드시 사용
    3. 문법 등급 {target_grade}에 적합한 난이도
    4. 외국인이 한국어를 사용시 실제로 사용 가능한 표현
    5. 자연스러운 한국어 
    6. 한국 문화적으로 적절해야 한다
    {f'7. ✅ {kpop_requirement}' if kpop_requirement else ''}

    【출력 형식】
    - 정확히 3개 문장만 출력
    - 번호나 설명 없이 문장만 출력
    - 각 문장은 새 줄에 작성

    예문 3개:
    """
        
        response = self.llm.predict(prompt)
        sentences = response.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()][:3]  # 정확히 3개
        
        print(f"   생성 완료: {len(sentences)}개 문장")
        
        # JSON 저장 데이터
        save_data = {
            "level": target_grade,
            "target_grammar": target_grammar,
            "kpop_references": kpop_references,  # 있을 때만 포함
            "critique_summary": [{"sentence": s} for s in sentences]
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


    def format_output_agentic(self, state: GraphState) -> GraphState:
        """출력 포맷팅 (Agentic 버전 - 한국어 교육 중심)"""
        print("\n📄 [Agent] 최종 출력 포맷팅")
        
        difficulty_kr = {
            "basic": "초급 (TOPIK 1-2급)",
            "intermediate": "중급 (TOPIK 3-4급)",
            "advanced": "고급 (TOPIK 5-6급)"
        }
        
        output = f"=" * 80 + "\n"
        output += "🎓 외국인을 위한 한국어 학습 문제 생성 결과 (Agentic RAG)\n"
        output += "=" * 80 + "\n\n"
        
        # 1. 학습자 정보
        difficulty = state.get('difficulty_level', 'basic')
        output += f"【학습자 수준】\n"
        output += f"   난이도: {difficulty_kr.get(difficulty, difficulty)}\n"
        
        if 'target_grade' in state:
            output += f"   문법 등급: Grade {state['target_grade']}\n"
        
        # 2. 검색된 어휘
        vocab_docs = state.get('vocabulary_docs', [])
        if vocab_docs:
            output += f"\n【선택된 학습 어휘】 (상위 10개)\n"
            for i, doc in enumerate(vocab_docs[:10], 1):
                word = doc.metadata.get('word', 'N/A')
                wordclass = doc.metadata.get('wordclass', 'N/A')
                guide = doc.metadata.get('guide', 'N/A')
                topik_level = doc.metadata.get('topik_level', 'N/A')
                output += f"   {i}. {word} ({wordclass}) - {guide[:40]}... [TOPIK{topik_level}]\n"
        
        # 3. 검색된 문법
        grammar_docs = state.get('grammar_docs', [])
        if grammar_docs:
            output += f"\n【선택된 학습 문법】 (등급 낮은 순)\n"
            for i, doc in enumerate(grammar_docs[:5], 1):
                grammar = doc.metadata.get('grammar', 'N/A')
                grade = doc.metadata.get('grade', 'N/A')
                output += f"   {i}. {grammar} (등급: {grade})\n"
        
        # 4. K-pop 참조 (있을 때만 표시)
        kpop_db_docs = state.get('kpop_docs', [])
        if kpop_db_docs:
            output += f"\n【K-pop 학습 자료】 데이터베이스: {len(kpop_db_docs)}개\n"
            for i, doc in enumerate(kpop_db_docs[:5], 1):
                sentence = doc.metadata.get('sentence', 'N/A')
                song = doc.metadata.get('song', 'N/A')
                group = doc.metadata.get('group', 'N/A')
                output += f'   {i}. "{sentence}"\n'
                output += f'       └─ {song} - {group}\n'
        
        # 5. 생성된 학습 예문
        sentences = state.get('generated_sentences', [])
        if sentences:
            kpop_label = " (K-pop 맥락 포함)" if kpop_db_docs else ""
            output += f"\n【생성된 학습 예문】{kpop_label}\n"
            for i, sentence in enumerate(sentences, 1):
                output += f"   {i}. {sentence}\n"
        
        # 6. 파일 저장 정보
        if 'sentence_data' in state and state['sentence_data']:
            saved_file = self._save_to_json(state['sentence_data'])
            output += f"\n💾 학습 자료 저장 위치: {saved_file}\n"
        
        output += "\n" + "=" * 80 + "\n"
        
        return {"final_output": output}