"""
worksheet_agent.py - 한국어 학습 워크시트/문제지 생성 에이전트 
K-POP 에이전트에서 생성된 문장을 활용하여 문제 생성
"""
from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import json
import logging
from datetime import datetime
import random
import os
import re

logger = logging.getLogger(__name__)

class WorksheetAgent(BaseAgent):
    """한국어 학습 워크시트 생성 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="WorksheetAgent")
        
        # 문제 유형별 스키마 (간소화)
        self.question_schemas = {
            'multiple_choice': {
                'type': 'multiple_choice',
                'question': str,
                'options': list,
                'answer': str,
                'explanation': str,
                'points': int,
                'source_sentence': str
            },
            'fill_blank': {
                'type': 'fill_blank',
                'question': str,  # 빈칸이 포함된 문장
                'answer': str,
                'explanation': str,
                'points': int,
                'source_sentence': str
            },
            'true_false': {
                'type': 'true_false',
                'question': str,  # 판단할 문장
                'answer': bool,
                'explanation': str,
                'points': int,
                'source_sentence': str
            }
        }
        
        # 난이도별 설정 (3가지 유형만)
        self.difficulty_configs = {
            'beginner': {
                'multiple_choice': 50,
                'fill_blank': 25,
                'true_false': 25,
                'total_questions': 10
            },
            'intermediate': {
                'multiple_choice': 40,
                'fill_blank': 35,
                'true_false': 25,
                'total_questions': 15
            },
            'advanced': {
                'multiple_choice': 35,
                'fill_blank': 40,
                'true_false': 25,
                'total_questions': 20
            }
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        워크시트 생성 처리 - K-POP 콘텐츠 기반
        
        Args:
            input_data: {
                'content': List[str],  # K-POP 에이전트에서 생성된 문장들
                'difficulty': str,
                'interest': str,
                'age_group': str,
                'questions': List[Dict] (optional)
            }
        """
        if not self.validate_input(input_data, ['content', 'difficulty', 'interest']):
            return {'error': '필수 입력 누락'}
        
        content = input_data['content']
        difficulty = input_data['difficulty']
        interest = input_data['interest']
        age_group = input_data.get('age_group', '20대')
        existing_questions = input_data.get('questions', None)
        
        logger.info(f"📝 {age_group} 대상 {difficulty} 난이도 {interest} 워크시트 생성 중...")
        logger.info(f"입력된 문장 수: {len(content)}개")
        
        # 기존 문제가 있으면 그대로 사용, 없으면 새로 생성
        if existing_questions:
            questions = existing_questions
            logger.info(f"기존 문제 {len(questions)}개 사용")
        else:
            questions = self._generate_questions_from_content(
                content, difficulty, interest, age_group
            )
            logger.info(f"새로운 문제 {len(questions)}개 생성")
        
        # 결과 파일 생성
        output_path = self._create_output(questions, difficulty, interest, age_group)
        
        return {
            'questions': questions,
            'output_path': output_path,
            'metadata': {
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'total_questions': len(questions),
                'source_sentences': len(content),
                'created_at': datetime.now().isoformat()
            }
        }
    
    def _generate_questions_from_content(self, content: List[str], 
                                        difficulty: str, interest: str, 
                                        age_group: str) -> List[Dict]:
        """K-POP 콘텐츠 기반 문제 생성"""
        questions = []
        config = self.difficulty_configs[difficulty]
        
        # 문제 유형별 개수 계산
        total = min(config['total_questions'], len(content))
        type_counts = {
            'multiple_choice': int(total * config['multiple_choice'] / 100),
            'fill_blank': int(total * config['fill_blank'] / 100),
            'true_false': int(total * config['true_false'] / 100)
        }
        
        # 남은 문제 수 조정
        current_total = sum(type_counts.values())
        if current_total < total:
            type_counts['multiple_choice'] += total - current_total
        
        # 문장을 섞어서 다양한 문제 생성
        shuffled_content = content.copy()
        random.shuffle(shuffled_content)
        
        sentence_index = 0
        
        # 각 유형별 문제 생성
        for q_type, count in type_counts.items():
            for _ in range(count):
                if sentence_index < len(shuffled_content):
                    question = self._create_question_from_sentence(
                        q_type,
                        shuffled_content[sentence_index],
                        difficulty,
                        interest,
                        age_group
                    )
                    if question:
                        questions.append(question)
                    sentence_index += 1
        
        return questions
    
    def _create_question_from_sentence(self, q_type: str, sentence: str, 
                                      difficulty: str, interest: str, 
                                      age_group: str) -> Dict:
        """특정 유형의 문제 생성 - K-POP 문장 활용"""
        
        if q_type == 'multiple_choice':
            return self._create_multiple_choice_kpop(sentence, difficulty, interest, age_group)
        elif q_type == 'fill_blank':
            return self._create_fill_blank_kpop(sentence, difficulty)
        elif q_type == 'true_false':
            return self._create_true_false_kpop(sentence, difficulty)
        
        return None
    
    def _create_multiple_choice_kpop(self, sentence: str, difficulty: str, 
                                    interest: str, age_group: str) -> Dict:
        """K-POP 문장 기반 객관식 문제 생성"""
        
        # K-POP 관련 키워드 추출
        kpop_elements = self._extract_kpop_elements(sentence)
        
        # LLM을 사용한 문제 생성
        prompt = f"""
한국어 학습용 객관식 문제를 만들어주세요.

원문: {sentence}
난이도: {difficulty}
대상: {age_group}

다음 형식으로 작성:
질문: [문장 내용 이해 질문]
정답: [정답 선택지]
오답1: [틀린 선택지]
오답2: [틀린 선택지]
오답3: [틀린 선택지]
해설: [정답 설명]

문장의 핵심 내용을 묻는 구체적인 질문을 만드세요.
"""
        
        response = self.generate_response(prompt, max_new_tokens=200)
        
        # 응답 파싱
        parsed = self._parse_multiple_choice_response(response, sentence, kpop_elements)
        
        return {
            'type': '객관식',
            'question': parsed['question'],
            'options': parsed['options'],
            'answer': parsed['answer'],
            'explanation': parsed['explanation'],
            'points': 5 if difficulty == 'beginner' else (7 if difficulty == 'intermediate' else 10),
            'source_sentence': sentence
        }
    
    def _create_fill_blank_kpop(self, sentence: str, difficulty: str) -> Dict:
        """K-POP 문장 기반 빈칸 채우기 문제"""
        
        # K-POP 관련 중요 단어 찾기
        important_words = self._find_important_words(sentence)
        blank_sentence = sentence
        
        if not important_words:
            # 기본 방식: 랜덤 단어 선택
            words = sentence.split()
            if len(words) < 3:
                return None
            blank_idx = random.randint(1, len(words) - 1)
            answer = words[blank_idx]
            words[blank_idx] = "_____"
            blank_sentence = " ".join(words)
        else:
            # 중요 단어 중 하나를 빈칸으로
            answer = random.choice(important_words)
            blank_sentence = sentence.replace(answer, "_____", 1)
        
        # 해설 생성
        explanation = self._generate_fill_blank_explanation(sentence, answer, difficulty)
        
        return {
            'type': '빈칸채우기',
            'question': blank_sentence,
            'answer': answer,
            'explanation': explanation,
            'points': 3 if difficulty == 'beginner' else (5 if difficulty == 'intermediate' else 7),
            'source_sentence': sentence
        }
    
    def _create_true_false_kpop(self, sentence: str, difficulty: str) -> Dict:
        """K-POP 문장 기반 참/거짓 문제"""
        
        is_true = random.choice([True, False])
        modified_sentence = sentence
        
        if not is_true:
            # 문장을 의미가 반대되도록 변형
            modifications = [
                ('는', '는 않'),
                ('했어요', '하지 않았어요'),
                ('했습니다', '하지 않았습니다'),
                ('있습니다', '없습니다'),
                ('있어요', '없어요'),
                ('좋아해요', '싫어해요'),
                ('유명합니다', '유명하지 않습니다'),
                ('인기를', '인기가 없음을')
            ]
            
            modified = False
            for original, replacement in modifications:
                if original in sentence:
                    modified_sentence = sentence.replace(original, replacement)
                    modified = True
                    break
            
            # 변형이 안 됐으면 다른 방법 시도
            if not modified:
                # 아티스트 이름 바꾸기
                kpop_artists = ['BTS', 'BLACKPINK', 'NCT', 'SEVENTEEN', 'Stray Kids', 
                               'TWICE', 'EXO', 'ENHYPEN', 'NewJeans', 'IVE']
                for artist in kpop_artists:
                    if artist in sentence:
                        other_artist = random.choice([a for a in kpop_artists if a != artist])
                        modified_sentence = sentence.replace(artist, other_artist)
                        break
                
                # 년도 바꾸기
                year_pattern = r'\d{4}년'
                years = re.findall(year_pattern, sentence)
                if years:
                    original_year = years[0]
                    new_year = str(random.randint(2010, 2024)) + "년"
                    modified_sentence = sentence.replace(original_year, new_year)
        
        # 해설 생성
        if is_true:
            explanation = "제시된 문장은 원문과 동일한 내용입니다."
        else:
            explanation = f"원문: {sentence}\n변경된 부분을 확인하세요."
        
        return {
            'type': '참거짓',
            'question': modified_sentence,
            'answer': is_true,
            'explanation': explanation,
            'points': 3,
            'source_sentence': sentence
        }
    
    def _extract_kpop_elements(self, sentence: str) -> Dict:
        """문장에서 K-POP 관련 요소 추출"""
        elements = {
            'artists': [],
            'keywords': [],
            'years': []
        }
        
        # 아티스트 이름 (영어 대문자로 시작)
        artist_pattern = r'[A-Z][A-Za-z0-9]+'
        elements['artists'] = re.findall(artist_pattern, sentence)
        
        # 년도
        year_pattern = r'\d{4}년'
        elements['years'] = re.findall(year_pattern, sentence)
        
        # K-POP 키워드
        kpop_keywords = ['데뷔', '컴백', '앨범', '콘서트', '팬', '활동', '노래', 
                        '무대', '팬미팅', '음원', '차트', '빌보드', '그래미']
        elements['keywords'] = [kw for kw in kpop_keywords if kw in sentence]
        
        return elements
    
    def _find_important_words(self, sentence: str) -> List[str]:
        """문장에서 중요한 단어 찾기"""
        important_words = []
        
        # K-POP 관련 중요 패턴
        patterns = [
            r'[A-Z][A-Za-z0-9]+',  # 영어 아티스트명
            r'\d{4}년',  # 년도
            r'데뷔|컴백|앨범|콘서트|팬미팅|활동|음원|차트',  # K-POP 용어
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            important_words.extend(matches)
        
        # 중복 제거
        return list(set(important_words))
    
    def _generate_fill_blank_explanation(self, original: str, answer: str, difficulty: str) -> str:
        """빈칸 채우기 문제 해설 생성"""
        
        explanation = f"정답: {answer}\n"
        
        if difficulty == 'beginner':
            explanation += f"이 단어는 문장에서 핵심 정보를 나타냅니다."
        elif difficulty == 'intermediate':
            explanation += f"문맥상 '{answer}'가 가장 적절한 답입니다."
        else:
            explanation += f"문장 구조와 의미를 고려하면 '{answer}'가 정답입니다."
        
        return explanation
    
    def _parse_multiple_choice_response(self, response: str, 
                                       original_sentence: str, 
                                       kpop_elements: Dict) -> Dict:
        """LLM 응답을 파싱하여 객관식 문제 구성"""
        
        # 기본값 설정
        default_question = "다음 문장의 내용으로 올바른 것은?"
        
        # K-POP 요소 기반 기본 선택지
        if kpop_elements['artists']:
            artist = kpop_elements['artists'][0]
            default_options = [
                f"{artist}에 대한 내용이다",
                f"{artist}와 관련이 없다",
                "다른 아티스트에 대한 내용이다",
                "K-POP과 무관한 내용이다"
            ]
        else:
            default_options = [
                "K-POP 활동에 관한 내용이다",
                "팬덤 문화에 관한 내용이다",
                "음악과 무관한 내용이다",
                "모든 내용이 맞다"
            ]
        
        try:
            lines = response.split('\n')
            question = ""
            answer_text = ""
            wrong_answers = []
            explanation = ""
            
            for line in lines:
                line = line.strip()
                if '질문:' in line:
                    question = line.split('질문:', 1)[1].strip()
                elif '정답:' in line:
                    answer_text = line.split('정답:', 1)[1].strip()
                elif '오답' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        wrong_answers.append(parts[1].strip())
                elif '해설:' in line:
                    explanation = line.split('해설:', 1)[1].strip()
            
            # 선택지 구성
            if answer_text and len(wrong_answers) >= 3:
                options = [answer_text] + wrong_answers[:3]
                random.shuffle(options)
                answer_index = str(options.index(answer_text) + 1)
            else:
                options = default_options
                answer_index = "1"
            
            return {
                'question': question or default_question,
                'options': options,
                'answer': answer_index,
                'explanation': explanation or f"정답은 {answer_index}번입니다. 원문을 잘 읽어보세요."
            }
            
        except Exception as e:
            logger.warning(f"파싱 실패: {e}")
            return {
                'question': default_question,
                'options': default_options,
                'answer': "1",
                'explanation': "정답은 1번입니다."
            }
    
    def _create_output(self, questions: List[Dict], difficulty: str, 
                      interest: str, age_group: str) -> str:
        """결과 출력 파일 생성"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"worksheet_{interest}_{age_group}_{difficulty}_{timestamp}.json"
        filepath = os.path.join("output", filename)
        
        os.makedirs("output", exist_ok=True)
        
        output_data = {
            'metadata': {
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'created_at': datetime.now().isoformat(),
                'total_questions': len(questions)
            },
            'questions': questions,
            'answer_key': self._create_answer_key(questions)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 워크시트 저장: {filepath}")
        return filepath
    
    def _create_answer_key(self, questions: List[Dict]) -> List[Dict]:
        """정답지 생성 - 간소화된 형식"""
        answer_key = []
        
        for i, q in enumerate(questions, 1):
            answer_entry = {
                '번호': i,
                '유형': q['type'],
                '질문': q.get('question', ''),
                '정답': '',
                '해설': q.get('explanation', '')
            }
            
            # 유형별 정답 형식
            if q['type'] == '객관식':
                answer_entry['정답'] = f"{q['answer']}번"
                answer_entry['선지'] = q.get('options', [])
            elif q['type'] == '빈칸채우기':
                answer_entry['정답'] = q['answer']
            elif q['type'] == '참거짓':
                answer_entry['정답'] = "참" if q['answer'] else "거짓"
            
            # 원본 문장 추가
            if 'source_sentence' in q:
                answer_entry['원본'] = q['source_sentence']
            
            answer_key.append(answer_entry)
        
        return answer_key