"""
worksheet_agent.py - 한국어 학습 워크시트/문제지 생성 에이전트
"""
from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import json
import logging
from datetime import datetime
import random
import os

logger = logging.getLogger(__name__)

class WorksheetAgent(BaseAgent):
    """한국어 학습 워크시트 생성 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="WorksheetAgent")
        
        # 문제 유형별 스키마
        self.question_schemas = {
            'multiple_choice': {
                'type': 'multiple_choice',
                'question': str,
                'options': list,
                'answer': str,
                'explanation': str,
                'points': int
            },
            'fill_blank': {
                'type': 'fill_blank',
                'sentence': str,
                'answer': str,
                'hints': list,
                'points': int
            },
            'true_false': {
                'type': 'true_false',
                'statement': str,
                'answer': bool,
                'explanation': str,
                'points': int
            }
        }
        
        # 난이도별 설정
        self.difficulty_configs = {
            'beginner': {
                'multiple_choice': 50,
                'fill_blank': 30,
                'true_false': 20,
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
        워크시트 생성 처리 - BaseAgent의 추상 메서드 구현
        
        Args:
            input_data: {
                'content': List[str],  # 생성된 문장들
                'difficulty': str,
                'interest': str,
                'age_group': str,
                'questions': List[Dict] (optional) # 기존 문제들
            }
        
        Returns:
            {
                'questions': List[Dict],
                'pdf_path': str,
                'metadata': Dict
            }
        """
        if not self.validate_input(input_data, ['content', 'difficulty', 'interest']):
            return {'error': '필수 입력 누락'}
        
        content = input_data['content']
        difficulty = input_data['difficulty']
        interest = input_data['interest']
        age_group = input_data.get('age_group', '20대')
        existing_questions = input_data.get('questions', None)
        
        logger.info(f"📝 {difficulty} 난이도 워크시트 생성 중...")
        
        # 기존 문제가 있으면 그대로 사용, 없으면 새로 생성
        if existing_questions:
            questions = existing_questions
            logger.info(f"기존 문제 {len(questions)}개 사용")
        else:
            questions = self._generate_questions(content, difficulty, interest)
            logger.info(f"새로운 문제 {len(questions)}개 생성")
        
        # PDF 생성 (또는 JSON으로 저장)
        pdf_path = self._create_output(questions, difficulty, interest, age_group)
        
        return {
            'questions': questions,
            'pdf_path': pdf_path,
            'metadata': {
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'total_questions': len(questions),
                'created_at': datetime.now().isoformat()
            }
        }
    
    def _generate_questions(self, content: List[str], difficulty: str, interest: str) -> List[Dict]:
        """콘텐츠 기반 문제 생성"""
        questions = []
        config = self.difficulty_configs[difficulty]
        
        # 문제 유형별 개수 계산
        total = config['total_questions']
        type_counts = {
            'multiple_choice': int(total * config['multiple_choice'] / 100),
            'fill_blank': int(total * config['fill_blank'] / 100),
            'true_false': int(total * config['true_false'] / 100)
        }
        
        # 각 유형별 문제 생성
        for q_type, count in type_counts.items():
            for i in range(count):
                if i < len(content):
                    question = self._create_question(
                        q_type,
                        content[i % len(content)],
                        difficulty,
                        interest
                    )
                    if question:
                        questions.append(question)
        
        return questions
    
    def _create_question(self, q_type: str, sentence: str, difficulty: str, interest: str) -> Dict:
        """특정 유형의 문제 생성"""
        
        if q_type == 'multiple_choice':
            return self._create_multiple_choice(sentence, difficulty, interest)
        elif q_type == 'fill_blank':
            return self._create_fill_blank(sentence, difficulty)
        elif q_type == 'true_false':
            return self._create_true_false(sentence, interest)
        
        return None
    
    def _create_multiple_choice(self, sentence: str, difficulty: str, interest: str) -> Dict:
        """객관식 문제 생성"""
        
        # LLM을 사용한 문제 생성
        prompt = f"""
한국어 학습 객관식 문제를 만들어주세요.

원문: {sentence}
난이도: {difficulty}
주제: {interest}

다음 형식으로 작성:
질문: [문장 이해 관련 질문]
정답: [정답 선택지]
오답1: [그럴듯한 오답]
오답2: [그럴듯한 오답]
오답3: [그럴듯한 오답]
"""
        
        response = self.generate_response(prompt, max_new_tokens=150)
        
        # 파싱 및 기본값 설정
        options = [
            f"{interest}와 관련이 있다",
            f"{interest}와 관련이 없다",
            "내용을 이해하기 어렵다",
            "모두 맞다"
        ]
        
        return {
            'type': 'multiple_choice',
            'question': f"다음 문장의 의미는?: '{sentence[:50]}...'",
            'options': options,
            'answer': "1",
            'explanation': f"이 문장은 {interest}에 관한 내용입니다.",
            'points': 5 if difficulty == 'beginner' else 7
        }
    
    def _create_fill_blank(self, sentence: str, difficulty: str) -> Dict:
        """빈칸 채우기 문제 생성"""
        
        words = sentence.split()
        if len(words) < 3:
            return None
        
        # 빈칸 위치 선택
        blank_idx = random.randint(1, len(words) - 1)
        answer = words[blank_idx]
        words[blank_idx] = "_____"
        blank_sentence = " ".join(words)
        
        return {
            'type': 'fill_blank',
            'sentence': blank_sentence,
            'answer': answer,
            'hints': [f"글자 수: {len(answer)}"],
            'points': 3 if difficulty == 'beginner' else 5
        }
    
    def _create_true_false(self, sentence: str, interest: str) -> Dict:
        """참/거짓 문제 생성"""
        
        is_true = random.choice([True, False])
        
        if not is_true:
            # 문장을 약간 변형
            sentence = sentence.replace("는", "는 않")
        
        return {
            'type': 'true_false',
            'statement': sentence,
            'answer': is_true,
            'explanation': "원문과 비교하여 판단하세요.",
            'points': 3
        }
    
    def _create_output(self, questions: List[Dict], difficulty: str, 
                      interest: str, age_group: str) -> str:
        """결과 출력 파일 생성"""
        
        # JSON 형식으로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"worksheet_{difficulty}_{interest}_{timestamp}.json"
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
        """정답지 생성"""
        answer_key = []
        
        for i, q in enumerate(questions, 1):
            answer = {
                'number': i,
                'type': q['type']
            }
            
            if q['type'] == 'multiple_choice':
                answer['correct'] = q['answer']
                answer['explanation'] = q.get('explanation', '')
            elif q['type'] == 'fill_blank':
                answer['correct'] = q['answer']
                answer['hints'] = q.get('hints', [])
            elif q['type'] == 'true_false':
                answer['correct'] = "참" if q['answer'] else "거짓"
                answer['explanation'] = q.get('explanation', '')
            
            answer_key.append(answer)
        
        return answer_key
