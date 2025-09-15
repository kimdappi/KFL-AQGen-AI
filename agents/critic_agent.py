"""
critic_agent.py - 생성된 문제의 품질과 적절성을 검증하는 에이전트
"""
from base_agent import BaseAgent
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """문제 품질 검증 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="CriticAgent")
        
        # 평가 기준
        self.evaluation_criteria = {
            'difficulty_match': {
                'weight': 0.25,
                'description': '난이도 적절성'
            },
            'content_relevance': {
                'weight': 0.25,
                'description': '관심사 관련성'
            },
            'linguistic_quality': {
                'weight': 0.20,
                'description': '언어적 품질'
            },
            'educational_value': {
                'weight': 0.20,
                'description': '교육적 가치'
            },
            'format_correctness': {
                'weight': 0.10,
                'description': '형식 정확성'
            }
        }
        
        # 난이도별 기준
        self.difficulty_standards = {
            'beginner': {
                'vocab_count': 500,  # 기초 어휘 수
                'sentence_complexity': 'simple',  # 단문 위주
                'grammar_patterns': ['present', 'past', 'basic_particles'],
                'max_sentence_length': 30
            },
            'intermediate': {
                'vocab_count': 2000,
                'sentence_complexity': 'moderate',  # 복문 포함
                'grammar_patterns': ['all_tenses', 'conjunctions', 'honorifics'],
                'max_sentence_length': 50
            },
            'advanced': {
                'vocab_count': 5000,
                'sentence_complexity': 'complex',  # 복잡한 구조
                'grammar_patterns': ['passive', 'causative', 'idioms', 'proverbs'],
                'max_sentence_length': 70
            }
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        문제 검증 처리
        
        Args:
            input_data: {
                'questions': List[Dict],  # 검증할 문제들
                'difficulty': str,  # 목표 난이도
                'interest': str,  # 관심사
                'age_group': str,  # 나이대
                'content': List[str]  # 원본 콘텐츠
            }
        
        Returns:
            {
                'evaluation': Dict,  # 평가 결과
                'approved_questions': List[Dict],  # 승인된 문제
                'rejected_questions': List[Dict],  # 거부된 문제
                'suggestions': List[str],  # 개선 제안
                'overall_score': float  # 전체 점수
            }
        """
        if not self.validate_input(input_data, ['questions', 'difficulty', 'interest']):
            return {'error': '필수 입력 누락'}
        
        questions = input_data['questions']
        difficulty = input_data['difficulty']
        interest = input_data['interest']
        age_group = input_data.get('age_group', '20대')
        content = input_data.get('content', [])
        
        logger.info(f"🔍 {len(questions)}개 문제 검증 시작...")
        
        # 1. 개별 문제 평가
        evaluated_questions = []
        for q in questions:
            evaluation = self._evaluate_question(q, difficulty, interest, age_group)
            evaluated_questions.append({
                'question': q,
                'evaluation': evaluation,
                'score': evaluation['total_score']
            })
        
        # 2. 문제 분류 (승인/거부)
        threshold = 0.6  # 60점 이상 승인
        approved = [eq for eq in evaluated_questions if eq['score'] >= threshold]
        rejected = [eq for eq in evaluated_questions if eq['score'] < threshold]
        
        # 3. 전체 평가
        overall_evaluation = self._evaluate_overall(
            evaluated_questions, difficulty, interest
        )
        
        # 4. 개선 제안 생성
        suggestions = self._generate_suggestions(
            evaluated_questions, difficulty, interest, overall_evaluation
        )
        
        # 5. 결과 반환
        return {
            'evaluation': overall_evaluation,
            'approved_questions': [eq['question'] for eq in approved],
            'rejected_questions': [eq['question'] for eq in rejected],
            'suggestions': suggestions,
            'overall_score': overall_evaluation['average_score'],
            'metadata': {
                'total_evaluated': len(questions),
                'approved_count': len(approved),
                'rejected_count': len(rejected),
                'evaluation_time': datetime.now().isoformat()
            }
        }
    
    def _evaluate_question(self, question: Dict, difficulty: str, 
                          interest: str, age_group: str) -> Dict:
        """개별 문제 평가"""
        
        scores = {}
        feedback = []
        
        # 1. 난이도 적절성 평가
        difficulty_score = self._check_difficulty_match(question, difficulty)
        scores['difficulty_match'] = difficulty_score
        if difficulty_score < 0.7:
            feedback.append(f"난이도가 {difficulty} 수준에 맞지 않음")
        
        # 2. 관심사 관련성 평가
        relevance_score = self._check_content_relevance(question, interest)
        scores['content_relevance'] = relevance_score
        if relevance_score < 0.7:
            feedback.append(f"{interest} 주제와 관련성 부족")
        
        # 3. 언어적 품질 평가
        linguistic_score = self._check_linguistic_quality(question)
        scores['linguistic_quality'] = linguistic_score
        if linguistic_score < 0.7:
            feedback.append("문법이나 철자 오류 가능성")
        
        # 4. 교육적 가치 평가
        educational_score = self._check_educational_value(question, difficulty)
        scores['educational_value'] = educational_score
        if educational_score < 0.7:
            feedback.append("교육적 가치 부족")
        
        # 5. 형식 정확성 평가
        format_score = self._check_format_correctness(question)
        scores['format_correctness'] = format_score
        if format_score < 0.7:
            feedback.append("문제 형식 오류")
        
        # 가중 평균 계산
        total_score = sum(
            scores[criterion] * self.evaluation_criteria[criterion]['weight']
            for criterion in scores
        )
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': feedback,
            'passed': total_score >= 0.6
        }
    
    def _check_difficulty_match(self, question: Dict, difficulty: str) -> float:
        """난이도 적절성 확인"""
        
        standards = self.difficulty_standards[difficulty]
        score = 1.0
        
        # 문제 텍스트 추출
        text = ""
        if question['type'] == 'multiple_choice':
            text = question.get('question', '')
        elif question['type'] == 'fill_blank':
            text = question.get('sentence', '')
        elif question['type'] == 'true_false':
            text = question.get('statement', '')
        elif question['type'] == 'translation':
            text = question.get('source', '')
        elif question['type'] == 'reading_comprehension':
            text = question.get('passage', '')
        
        # 문장 길이 체크
        if len(text) > standards['max_sentence_length'] * 1.5:
            score -= 0.3
        elif len(text) < standards['max_sentence_length'] * 0.3:
            score -= 0.2
        
        # 복잡도 체크 (간단한 휴리스틱)
        if difficulty == 'beginner':
            # 초급은 단순해야 함
            if text.count(',') > 2 or text.count('.') > 2:
                score -= 0.2
        elif difficulty == 'intermediate':
            # 중급은 적당한 복잡도
            if text.count(',') < 1 and text.count('.') < 1:
                score -= 0.2
        elif difficulty == 'advanced':
            # 고급은 복잡해야 함
            if text.count(',') < 2 and len(text) < 50:
                score -= 0.3
        
        return max(0, min(1, score))
    
    def _check_content_relevance(self, question: Dict, interest: str) -> float:
        """관심사 관련성 확인"""
        
        # 문제에서 텍스트 추출
        all_text = []
        for key in ['question', 'sentence', 'statement', 'source', 'passage']:
            if key in question:
                all_text.append(str(question[key]))
        
        combined_text = ' '.join(all_text).lower()
        
        # 관심사 키워드 체크
        interest_keywords = {
            'kpop': ['아이돌', '가수', '음악', '노래', '댄스', 'k-pop', 'kpop', '그룹'],
            'kdrama': ['드라마', '배우', '연기', '시청률', '방송'],
            'korean_food': ['음식', '요리', '맛', '김치', '밥', '반찬'],
            'korean_culture': ['문화', '전통', '한복', '명절', '예절'],
            'technology': ['기술', '스마트폰', '컴퓨터', '인터넷', 'IT'],
            'sports': ['스포츠', '운동', '경기', '선수', '팀']
        }
        
        keywords = interest_keywords.get(interest, [interest])
        matches = sum(1 for keyword in keywords if keyword in combined_text)
        
        # 매칭 비율로 점수 계산
        score = min(1.0, matches / max(1, len(keywords) * 0.3))
        
        return score
    
    def _check_linguistic_quality(self, question: Dict) -> float:
        """언어적 품질 확인"""
        
        # LLM을 사용한 품질 평가
        text_samples = []
        for key in ['question', 'sentence', 'statement', 'source']:
            if key in question:
                text_samples.append(question[key])
        
        if not text_samples:
            return 0.5
        
        sample = text_samples[0][:100]  # 첫 100자만 평가
        
        prompt = f"""
다음 한국어 문장의 문법과 철자를 평가해주세요.
문장: {sample}

평가 (0-10점):
- 문법 정확성:
- 철자 정확성:
- 자연스러움:

종합 점수 (0-10):
"""
        
        response = self.generate_response(prompt, max_new_tokens=50)
        
        # 점수 추출 시도
        try:
            if '종합' in response:
                score_text = response.split('종합')[1]
                for word in score_text.split():
                    if word.replace('.', '').isdigit():
                        return float(word) / 10
        except:
            pass
        
        # 기본 점수
        return 0.7
    
    def _check_educational_value(self, question: Dict, difficulty: str) -> float:
        """교육적 가치 평가"""
        
        score = 0.7  # 기본 점수
        
        # 문제 유형별 가치 평가
        q_type = question.get('type')
        
        if q_type == 'multiple_choice':
            # 선택지가 교육적인지 확인
            options = question.get('options', [])
            if len(options) >= 4:
                score += 0.1
            if question.get('explanation'):
                score += 0.2
        
        elif q_type == 'fill_blank':
            # 힌트가 있으면 가치 상승
            if question.get('hints'):
                score += 0.2
        
        elif q_type == 'translation':
            # 대체 번역이 있으면 가치 상승
            if question.get('alternatives'):
                score += 0.15
        
        elif q_type == 'reading_comprehension':
            # 독해 문제는 기본적으로 높은 가치
            score = 0.9
            if len(question.get('questions', [])) >= 2:
                score = 1.0
        
        # 난이도에 맞는 포인트 배점 확인
        expected_points = {
            'beginner': {'min': 3, 'max': 7},
            'intermediate': {'min': 5, 'max': 10},
            'advanced': {'min': 7, 'max': 15}
        }
        
        points = question.get('points', 0)
        if expected_points[difficulty]['min'] <= points <= expected_points[difficulty]['max']:
            score += 0.1
        
        return min(1.0, score)
    
    def _check_format_correctness(self, question: Dict) -> float:
        """형식 정확성 확인"""
        
        score = 1.0
        q_type = question.get('type')
        
        # 필수 필드 체크
        required_fields = {
            'multiple_choice': ['question', 'options', 'answer'],
            'fill_blank': ['sentence', 'answer'],
            'true_false': ['statement', 'answer'],
            'translation': ['source', 'answer'],
            'reading_comprehension': ['passage', 'questions']
        }
        
        if q_type in required_fields:
            for field in required_fields[q_type]:
                if field not in question:
                    score -= 0.3
        else:
            score = 0.5  # 알 수 없는 유형
        
        # 특정 유형별 추가 체크
        if q_type == 'multiple_choice':
            options = question.get('options', [])
            if len(options) < 2:
                score -= 0.5
            elif len(options) > 6:
                score -= 0.2
        
        elif q_type == 'fill_blank':
            sentence = question.get('sentence', '')
            if '_____' not in sentence and '___' not in sentence:
                score -= 0.5
        
        return max(0, score)
    
    def _evaluate_overall(self, evaluated_questions: List[Dict], 
                         difficulty: str, interest: str) -> Dict:
        """전체 문제 세트 평가"""
        
        # 평균 점수 계산
        total_scores = [eq['score'] for eq in evaluated_questions]
        average_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        # 문제 유형 분포 분석
        type_distribution = {}
        for eq in evaluated_questions:
            q_type = eq['question'].get('type', 'unknown')
            type_distribution[q_type] = type_distribution.get(q_type, 0) + 1
        
        # 난이도 일관성 분석
        difficulty_scores = [
            eq['evaluation']['scores'].get('difficulty_match', 0) 
            for eq in evaluated_questions
        ]
        difficulty_consistency = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 0
        
        # 관심사 관련성 분석
        relevance_scores = [
            eq['evaluation']['scores'].get('content_relevance', 0)
            for eq in evaluated_questions
        ]
        content_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        return {
            'average_score': average_score,
            'type_distribution': type_distribution,
            'difficulty_consistency': difficulty_consistency,
            'content_relevance': content_relevance,
            'total_questions': len(evaluated_questions),
            'passed_questions': sum(1 for eq in evaluated_questions if eq['score'] >= 0.6),
            'failed_questions': sum(1 for eq in evaluated_questions if eq['score'] < 0.6)
        }
    
    def _generate_suggestions(self, evaluated_questions: List[Dict], 
                            difficulty: str, interest: str, 
                            overall_evaluation: Dict) -> List[str]:
        """개선 제안 생성"""
        
        suggestions = []
        
        # 1. 전체 점수 기반 제안
        if overall_evaluation['average_score'] < 0.6:
            suggestions.append("⚠️ 전반적인 문제 품질 개선이 필요합니다.")
        elif overall_evaluation['average_score'] < 0.8:
            suggestions.append("📝 문제 품질이 양호하나 일부 개선이 필요합니다.")
        else:
            suggestions.append("✅ 문제 품질이 우수합니다.")
        
        # 2. 난이도 일관성 제안
        if overall_evaluation['difficulty_consistency'] < 0.7:
            suggestions.append(f"🎯 {difficulty} 난이도에 맞게 문제를 조정하세요.")
            if difficulty == 'beginner':
                suggestions.append("   - 문장을 더 간단하게 만드세요")
                suggestions.append("   - 기초 어휘만 사용하세요")
            elif difficulty == 'intermediate':
                suggestions.append("   - 적절한 복잡도를 유지하세요")
                suggestions.append("   - 다양한 문법 구조를 포함하세요")
            else:
                suggestions.append("   - 고급 어휘와 관용구를 추가하세요")
                suggestions.append("   - 복잡한 문장 구조를 사용하세요")
        
        # 3. 관심사 관련성 제안
        if overall_evaluation['content_relevance'] < 0.7:
            suggestions.append(f"🎭 {interest} 주제와 더 관련된 내용을 포함하세요.")
            suggestions.append(f"   - {interest} 관련 키워드를 더 많이 사용하세요")
            suggestions.append(f"   - 실제 {interest} 예시를 활용하세요")
        
        # 4. 문제 유형 분포 제안
        type_dist = overall_evaluation['type_distribution']
        total = overall_evaluation['total_questions']
        
        # 예상 분포와 비교
        expected_dist = self.difficulty_configs[difficulty]
        
        for q_type in ['multiple_choice', 'fill_blank', 'true_false', 'translation', 'reading_comprehension']:
            actual_ratio = type_dist.get(q_type, 0) / total * 100 if total > 0 else 0
            expected_ratio = expected_dist.get(q_type, 0)
            
            if abs(actual_ratio - expected_ratio) > 15:  # 15% 이상 차이
                if actual_ratio < expected_ratio:
                    suggestions.append(f"📊 {q_type} 문제를 더 추가하세요 (현재: {actual_ratio:.0f}%, 권장: {expected_ratio}%)")
                else:
                    suggestions.append(f"📊 {q_type} 문제를 줄이세요 (현재: {actual_ratio:.0f}%, 권장: {expected_ratio}%)")
        
        # 5. 개별 문제 피드백 종합
        common_issues = {}
        for eq in evaluated_questions:
            for feedback in eq['evaluation']['feedback']:
                common_issues[feedback] = common_issues.get(feedback, 0) + 1
        
        # 가장 빈번한 문제 top 3
        if common_issues:
            sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)
            suggestions.append("\n🔍 주요 개선 사항:")
            for issue, count in sorted_issues[:3]:
                suggestions.append(f"   - {issue} ({count}개 문제)")
        
        # 6. 실패한 문제들에 대한 구체적 제안
        failed_questions = [eq for eq in evaluated_questions if eq['score'] < 0.6]
        if failed_questions:
            suggestions.append(f"\n❌ {len(failed_questions)}개 문제가 기준 미달:")
            for i, fq in enumerate(failed_questions[:3], 1):  # 상위 3개만
                q_type = fq['question'].get('type', 'unknown')
                score = fq['score']
                suggestions.append(f"   {i}. {q_type} 문제 (점수: {score:.2f})")
                if fq['evaluation']['feedback']:
                    suggestions.append(f"      - {fq['evaluation']['feedback'][0]}")
        
        return suggestions