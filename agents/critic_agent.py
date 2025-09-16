"""
critic_agent.py - K-POP 세대별 콘텐츠 및 문제 검증 에이전트 (재생성 요청 기능 추가)
"""
from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """K-POP 세대별 콘텐츠 검증 에이전트 (재생성 요청 기능 포함)"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="CriticAgent")
        
        # K-POP 세대별 검증 기준
        self.kpop_age_criteria = {
            '10대': {
                'must_have_artists': ['NewJeans', 'IVE', 'LE SSERAFIM', 'ENHYPEN', 'Stray Kids'],
                'era': '4세대 (2020-2024)',
                'platforms': ['TikTok', '위버스', '버블', '유튜브 쇼츠'],
                'forbidden_artists': ['H.O.T', '젝스키스', 'S.E.S'],
                'key_terms': ['챌린지', '직캠', '포카', '스밍', '컴백'],
                'cultural_refs': ['음방 1위', '아육대', '팬싸', '영통']
            },
            '20대': {
                'must_have_artists': ['BTS', 'BLACKPINK', 'SEVENTEEN', 'NCT', 'aespa'],
                'era': '3.5-4세대 (2015-2024)',
                'platforms': ['유튜브', '트위터', '위버스', '브이라이브'],
                'forbidden_artists': ['H.O.T', '젝스키스'],
                'key_terms': ['월드투어', '빌보드', '그래미', '정규앨범', '유닛'],
                'cultural_refs': ['스타디움 콘서트', '팬미팅', '시즌그리팅', '자컨']
            },
            '30대': {
                'must_have_artists': ['BIGBANG', 'EXO', '소녀시대', 'SHINee', '2NE1'],
                'era': '2-3세대 (2010-2020)',
                'platforms': ['팬카페', '멜론', '음악방송', '유튜브'],
                'forbidden_artists': ['NewJeans', 'IVE'],
                'key_terms': ['컴백', '입대', '제대', '재계약', '솔로'],
                'cultural_refs': ['응원봉', '팬클럽', '연말시상식', '가요대전']
            },
            '40대+': {
                'must_have_artists': ['H.O.T', 'S.E.S', '핑클', '신화', 'god'],
                'era': '1-2세대 (1996-2010)',
                'platforms': ['팬카페', 'CD', '카세트', '음반'],
                'forbidden_artists': ['NewJeans', 'IVE', 'LE SSERAFIM', 'ENHYPEN'],
                'key_terms': ['데뷔', '해체', '재결합', '1집', '팬클럽 창단'],
                'cultural_refs': ['가요톱텐', '뮤직뱅크', '팬레터', '사인회']
            }
        }
        
        # 검증 임계값 설정
        self.sentence_threshold = 0.7  # 문장 승인 기준
        self.question_threshold = 0.65  # 문제 승인 기준
        self.regeneration_threshold = 0.6  # 재생성 요청 기준 (60% 미만 승인시)
        self.max_regeneration_attempts = 3  # 최대 재생성 시도 횟수
        
        # 검증 기준 가중치
        self.sentence_criteria = {
            'generation_accuracy': {'weight': 0.2, 'description': '세대 정확성'},
            'cultural_relevance': {'weight': 0.2, 'description': '문화적 관련성'},
            'difficulty_match': {'weight': 0.2, 'description': '난이도 적절성'},
            'linguistic_quality': {'weight': 0.4, 'description': '언어적 품질 및 한국어 정확성'}
        }
        
        self.question_criteria = {
            'content_relevance': {'weight': 0.2, 'description': 'K-POP 세대 관련성'},
            'difficulty_match': {'weight': 0.3, 'description': '난이도 적절성'},
            'educational_value': {'weight': 0.3, 'description': '한국어 교육 목적에 적합'},
            'format_correctness': {'weight': 0.2, 'description': '형식 정확성'}
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        검증 처리 - 검증 유형에 따라 분기하고 필요시 재생성 요청
        """
        validation_type = input_data.get('validation_type', 'questions')
        
        if validation_type == 'sentences':
            return self._validate_sentences_with_regeneration(input_data)
        else:
            return self._validate_questions_with_regeneration(input_data)
    
    def _validate_sentences_with_regeneration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """문장 검증 및 재생성 요청 로직"""
        
        sentences = input_data.get('sentences', input_data.get('content', []))
        age_group = input_data['age_group']
        difficulty = input_data['difficulty']
        attempt = input_data.get('attempt', 1)
        
        logger.info(f"🔍 문장 검증 시작 (시도 {attempt}/{self.max_regeneration_attempts})")
        
        # 기본 검증 수행
        validation_result = self._validate_sentences(input_data)
        
        # 승인률 계산
        total_sentences = len(sentences)
        approved_count = len(validation_result['approved_sentences'])
        approval_rate = approved_count / total_sentences if total_sentences > 0 else 0
        
        logger.info(f"📊 승인률: {approval_rate:.2%} ({approved_count}/{total_sentences})")
        
        # 재생성 필요 여부 판단
        needs_regeneration = (
            approval_rate < self.regeneration_threshold and 
            attempt < self.max_regeneration_attempts
        )
        
        if needs_regeneration:
            logger.warning(f"⚠️ 승인률 {approval_rate:.2%} < {self.regeneration_threshold:.0%}, 재생성 필요")
            
            # 재생성 가이드라인 생성
            regeneration_guide = self._generate_sentence_regeneration_guide(
                validation_result, age_group, difficulty
            )
            
            validation_result.update({
                'needs_regeneration': True,
                'regeneration_reason': f"승인률 {approval_rate:.2%}로 기준 {self.regeneration_threshold:.0%} 미달",
                'regeneration_guide': regeneration_guide,
                'attempt': attempt,
                'max_attempts': self.max_regeneration_attempts
            })
        else:
            if attempt >= self.max_regeneration_attempts:
                logger.warning(f"⚠️ 최대 재생성 시도 횟수 {self.max_regeneration_attempts}회 도달")
                validation_result['regeneration_reason'] = "최대 시도 횟수 도달"
            else:
                logger.info(f"✅ 승인률 {approval_rate:.2%} 만족, 검증 통과")
            
            validation_result['needs_regeneration'] = False
        
        return validation_result
    
    def _validate_questions_with_regeneration(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """문제 검증 및 재생성 요청 로직"""
        
        questions = input_data.get('questions', [])
        attempt = input_data.get('attempt', 1)
        
        logger.info(f"🔍 문제 검증 시작 (시도 {attempt}/{self.max_regeneration_attempts})")
        
        # 기본 검증 수행
        validation_result = self._validate_questions(input_data)
        
        # 승인률 계산
        total_questions = len(questions)
        approved_count = len(validation_result['approved_questions'])
        approval_rate = approved_count / total_questions if total_questions > 0 else 0
        
        logger.info(f"📊 승인률: {approval_rate:.2%} ({approved_count}/{total_questions})")
        
        # 중복 검사 추가
        duplicate_issues = self._check_question_duplicates(questions)
        
        # 재생성 필요 여부 판단
        needs_regeneration = (
            (approval_rate < self.regeneration_threshold or len(duplicate_issues) > 0) and 
            attempt < self.max_regeneration_attempts
        )
        
        if needs_regeneration:
            reasons = []
            if approval_rate < self.regeneration_threshold:
                reasons.append(f"승인률 {approval_rate:.2%} < {self.regeneration_threshold:.0%}")
            if duplicate_issues:
                reasons.append(f"중복 문제 {len(duplicate_issues)}개 발견")
            
            logger.warning(f"⚠️ 재생성 필요: {', '.join(reasons)}")
            
            # 재생성 가이드라인 생성
            regeneration_guide = self._generate_question_regeneration_guide(
                validation_result, duplicate_issues, input_data
            )
            
            validation_result.update({
                'needs_regeneration': True,
                'regeneration_reason': ', '.join(reasons),
                'regeneration_guide': regeneration_guide,
                'duplicate_issues': duplicate_issues,
                'attempt': attempt,
                'max_attempts': self.max_regeneration_attempts
            })
        else:
            if attempt >= self.max_regeneration_attempts:
                logger.warning(f"⚠️ 최대 재생성 시도 횟수 {self.max_regeneration_attempts}회 도달")
                validation_result['regeneration_reason'] = "최대 시도 횟수 도달"
            else:
                logger.info(f"✅ 승인률 {approval_rate:.2%} 만족, 중복 없음, 검증 통과")
            
            validation_result['needs_regeneration'] = False
        
        return validation_result
    
    def _check_question_duplicates(self, questions: List[Dict]) -> List[Dict]:
        """문제 중복 검사"""
        
        duplicates = []
        seen_questions = {}
        seen_answers = {}
        
        for i, q in enumerate(questions):
            q_text = str(q.get('question', q.get('statement', q.get('sentence', ''))))
            q_answer = str(q.get('answer', ''))
            
            # 문제 내용 중복 검사
            if q_text in seen_questions:
                duplicates.append({
                    'type': 'question_duplicate',
                    'indices': [seen_questions[q_text], i],
                    'content': q_text
                })
            else:
                seen_questions[q_text] = i
            
            # 답 중복 검사 (객관식의 경우)
            if q.get('type') == 'multiple_choice' and q_answer:
                if q_answer in seen_answers:
                    duplicates.append({
                        'type': 'answer_duplicate',
                        'indices': [seen_answers[q_answer], i],
                        'content': q_answer
                    })
                else:
                    seen_answers[q_answer] = i
            
            # 선택지 내부 중복 검사
            if 'options' in q and isinstance(q['options'], list):
                option_set = set()
                for opt in q['options']:
                    if opt in option_set:
                        duplicates.append({
                            'type': 'option_duplicate',
                            'question_index': i,
                            'content': opt
                        })
                    option_set.add(opt)
        
        return duplicates
    
    def _generate_sentence_regeneration_guide(self, validation_result: Dict, 
                                            age_group: str, difficulty: str) -> Dict:
        """문장 재생성 가이드라인 생성"""
        
        age_criteria = self.kpop_age_criteria[age_group]
        rejected_sentences = validation_result.get('rejected_sentences', [])
        
        # 주요 문제점 분석
        common_issues = []
        if len(rejected_sentences) > 0:
            # 여기서는 간단한 분석만 수행
            common_issues = [
                f"{age_group} 세대에 맞지 않는 아티스트 사용",
                f"문화적 맥락 부족",
                f"{difficulty} 난이도에 부적절한 문장 길이"
            ]
        
        guide = {
            'target_sentence_count': len(validation_result.get('approved_sentences', [])) + len(rejected_sentences),
            'common_issues': common_issues,
            'recommendations': [
                f"반드시 포함할 아티스트: {', '.join(age_criteria['must_have_artists'][:3])}",
                f"사용 금지 아티스트: {', '.join(age_criteria['forbidden_artists'][:2])}",
                f"권장 키워드: {', '.join(age_criteria['key_terms'][:3])}",
                f"문화적 요소: {', '.join(age_criteria['cultural_refs'][:2])}"
            ],
            'difficulty_guide': self._get_difficulty_guide(difficulty),
            'examples_to_avoid': rejected_sentences[:3]  # 상위 3개 실패 예시
        }
        
        return guide
    
    def _generate_question_regeneration_guide(self, validation_result: Dict, 
                                            duplicate_issues: List[Dict], 
                                            input_data: Dict) -> Dict:
        """문제 재생성 가이드라인 생성"""
        
        age_group = input_data.get('age_group', '20대')
        difficulty = input_data.get('difficulty', 'beginner')
        age_criteria = self.kpop_age_criteria[age_group]
        
        guide = {
            'target_question_count': len(input_data.get('questions', [])),
            'duplicate_prevention': {
                'avoid_duplicate_questions': len([d for d in duplicate_issues if d['type'] == 'question_duplicate']),
                'avoid_duplicate_answers': len([d for d in duplicate_issues if d['type'] == 'answer_duplicate']),
                'avoid_duplicate_options': len([d for d in duplicate_issues if d['type'] == 'option_duplicate'])
            },
            'quality_requirements': {
                'min_approval_rate': f"{self.regeneration_threshold:.0%}",
                'required_question_types': ['multiple_choice', 'fill_blank', 'true_false'],
                'age_appropriate_content': age_criteria['era']
            },
            'content_guidelines': [
                f"사용할 아티스트: {', '.join(age_criteria['must_have_artists'][:3])}",
                f"피할 아티스트: {', '.join(age_criteria['forbidden_artists'][:2])}",
                f"포함할 키워드: {', '.join(age_criteria['key_terms'][:3])}"
            ],
            'format_requirements': {
                'multiple_choice': "4개 선택지, 명확한 정답 1개",
                'fill_blank': "적절한 난이도의 빈칸, 명확한 답",
                'true_false': "명확한 참/거짓 판단 가능한 문장"
            },
            'difficulty_guide': self._get_difficulty_guide(difficulty),
            'rejected_examples': [q['question'] for q in validation_result.get('rejected_questions', [])[:3]]
        }
        
        return guide
    
    def _get_difficulty_guide(self, difficulty: str) -> Dict:
        """난이도별 가이드라인"""
        
        guides = {
            'beginner': {
                'sentence_length': '20-40자',
                'vocabulary': '기본 어휘 중심',
                'grammar': '단순한 문장 구조',
                'question_types': ['multiple_choice', 'true_false']
            },
            'intermediate': {
                'sentence_length': '40-60자',
                'vocabulary': '중급 어휘 포함',
                'grammar': '복합 문장 구조',
                'question_types': ['multiple_choice', 'fill_blank', 'true_false']
            },
            'advanced': {
                'sentence_length': '60자 이상',
                'vocabulary': '고급 어휘 및 관용 표현',
                'grammar': '복잡한 문장 구조, 연결어미',
                'question_types': ['reading_comprehension', 'translation', 'fill_blank']
            }
        }
        
        return guides.get(difficulty, guides['beginner'])
    
    # 기존 검증 메서드들은 그대로 유지
    def _validate_sentences(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """K-POP 세대별 문장 검증 (기존 코드와 동일)"""
        
        sentences = input_data.get('sentences', input_data.get('content', []))
        age_group = input_data['age_group']
        difficulty = input_data['difficulty']
        
        logger.info(f"🔍 {age_group} 대상 K-POP 문장 {len(sentences)}개 검증 시작...")
        
        age_criteria = self.kpop_age_criteria[age_group]
        evaluated_sentences = []
        
        for sentence in sentences:
            evaluation = self._evaluate_kpop_sentence(sentence, age_group, difficulty, age_criteria)
            evaluated_sentences.append({
                'sentence': sentence,
                'evaluation': evaluation,
                'score': evaluation['total_score']
            })
        
        # 문장 분류
        approved = [es for es in evaluated_sentences if es['score'] >= self.sentence_threshold]
        rejected = [es for es in evaluated_sentences if es['score'] < self.sentence_threshold]
        
        # 전체 평가
        average_score = sum(es['score'] for es in evaluated_sentences) / len(evaluated_sentences) if evaluated_sentences else 0
        
        # 개선 제안
        suggestions = self._generate_kpop_suggestions(evaluated_sentences, age_group, age_criteria)
        
        return {
            'approved_sentences': [es['sentence'] for es in approved],
            'rejected_sentences': [es['sentence'] for es in rejected],
            'suggestions': suggestions,
            'overall_score': average_score,
            'metadata': {
                'age_group': age_group,
                'era': age_criteria['era'],
                'total_evaluated': len(sentences),
                'approved_count': len(approved),
                'rejected_count': len(rejected),
                'evaluation_time': datetime.now().isoformat()
            }
        }
    
    # 나머지 기존 메서드들도 그대로 유지 (공간상 생략)
    def _evaluate_kpop_sentence(self, sentence: str, age_group: str, 
                                difficulty: str, age_criteria: Dict) -> Dict:
        """개별 K-POP 문장 평가"""
        
        scores = {}
        feedback = []
        
        # 1. 세대 정확성 평가
        generation_score = self._check_generation_accuracy(sentence, age_criteria)
        scores['generation_accuracy'] = generation_score
        if generation_score < 0.7:
            feedback.append(f"{age_group} K-POP 세대와 맞지 않는 내용")
        
        # 2. 문화적 관련성 평가
        cultural_score = self._check_cultural_relevance(sentence, age_criteria)
        scores['cultural_relevance'] = cultural_score
        if cultural_score < 0.7:
            feedback.append(f"{age_criteria['era']} 팬덤 문화와 맞지 않음")
        
        # 3. 난이도 적절성
        difficulty_score = self._check_difficulty_appropriateness(sentence, difficulty)
        scores['difficulty_match'] = difficulty_score
        if difficulty_score < 0.7:
            feedback.append(f"{difficulty} 난이도에 부적절")
        
        # 4. 언어적 품질
        linguistic_score = self._check_linguistic_quality(sentence)
        scores['linguistic_quality'] = linguistic_score
        if linguistic_score < 0.7:
            feedback.append("문법 또는 철자 오류")
        
        # 가중 평균
        total_score = sum(
            scores[criterion] * self.sentence_criteria[criterion]['weight']
            for criterion in scores
        )
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': feedback,
            'passed': total_score >= 0.7
        }
    
    def _check_generation_accuracy(self, sentence: str, age_criteria: Dict) -> float:
        """K-POP 세대 정확성 확인"""
        
        score = 0.5  # 기본 점수
        sentence_lower = sentence.lower()
        
        # 필수 아티스트 체크
        for artist in age_criteria['must_have_artists']:
            if artist.lower() in sentence_lower:
                score += 0.3
                break
        
        # 금지된 아티스트 체크 (다른 세대)
        for artist in age_criteria['forbidden_artists']:
            if artist.lower() in sentence_lower:
                score -= 0.4
                break
        
        # 핵심 용어 체크
        for term in age_criteria['key_terms']:
            if term in sentence:
                score += 0.2
        
        return max(0, min(1, score))
    
    def _check_cultural_relevance(self, sentence: str, age_criteria: Dict) -> float:
        """문화적 관련성 확인"""
        
        score = 0.6  # 기본 점수
        
        # 플랫폼 언급 체크
        for platform in age_criteria['platforms']:
            if platform in sentence:
                score += 0.2
                break
        
        # 문화적 레퍼런스 체크
        for ref in age_criteria['cultural_refs']:
            if ref in sentence:
                score += 0.2
        
        return min(1, score)
    
    def _check_difficulty_appropriateness(self, sentence: str, difficulty: str) -> float:
        """난이도 적절성 확인"""
        
        score = 1.0
        
        if difficulty == 'beginner':
            if len(sentence) > 40:
                score -= 0.3
            if sentence.count(',') > 2:
                score -= 0.2
        elif difficulty == 'intermediate':
            if len(sentence) < 20 or len(sentence) > 60:
                score -= 0.2
        elif difficulty == 'advanced':
            if len(sentence) < 40:
                score -= 0.3
        
        return max(0, score)
    
    def _check_linguistic_quality(self, sentence: str) -> float:
        """언어적 품질 확인"""
        
        if not sentence or len(sentence) < 5:
            return 0.1
        
        if not sentence.endswith(('.', '!', '?', '다', '요', '까')):
            return 0.6
        
        return 0.8
    
    def _generate_kpop_suggestions(self, evaluated_sentences: List[Dict], 
                                   age_group: str, age_criteria: Dict) -> List[str]:
        """K-POP 세대별 개선 제안"""
        
        suggestions = []
        avg_score = sum(es['score'] for es in evaluated_sentences) / len(evaluated_sentences) if evaluated_sentences else 0
        
        if avg_score < 0.7:
            suggestions.append(f"⚠️ {age_group} K-POP 팬에게 적합하지 않은 내용입니다. 재생성 필요.")
            suggestions.append(f"💡 {age_criteria['era']} 아티스트를 더 많이 언급하세요.")
            suggestions.append(f"💡 추천 아티스트: {', '.join(age_criteria['must_have_artists'][:3])}")
        elif avg_score < 0.85:
            suggestions.append(f"📝 {age_group} 대상으로는 적절하나 일부 개선 필요")
        else:
            suggestions.append(f"✅ {age_group} K-POP 팬에게 완벽한 콘텐츠입니다!")
        
        return suggestions
    
    def _validate_questions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """K-POP 문제 검증 (기존 코드 간소화 버전)"""
        
        questions = input_data.get('questions', [])
        age_group = input_data.get('age_group', '20대')
        difficulty = input_data['difficulty']
        
        logger.info(f"🔍 K-POP 문제 {len(questions)}개 검증 시작...")
        
        age_criteria = self.kpop_age_criteria[age_group]
        evaluated_questions = []
        
        for q in questions:
            evaluation = self._evaluate_kpop_question(q, age_group, difficulty, age_criteria)
            evaluated_questions.append({
                'question': q,
                'evaluation': evaluation,
                'score': evaluation['total_score']
            })
        
        # 문제 분류
        approved = [eq for eq in evaluated_questions if eq['score'] >= self.question_threshold]
        rejected = [eq for eq in evaluated_questions if eq['score'] < self.question_threshold]
        
        # 전체 평가
        average_score = sum(eq['score'] for eq in evaluated_questions) / len(evaluated_questions) if evaluated_questions else 0
        
        # 개선 제안
        suggestions = self._generate_question_suggestions(evaluated_questions, age_group)
        
        return {
            'evaluation': {
                'average_score': average_score,
                'total_questions': len(questions),
                'passed_questions': len(approved),
                'failed_questions': len(rejected)
            },
            'approved_questions': [eq['question'] for eq in approved],
            'rejected_questions': [eq['question'] for eq in rejected],
            'suggestions': suggestions,
            'overall_score': average_score,
            'metadata': {
                'age_group': age_group,
                'total_evaluated': len(questions),
                'approved_count': len(approved),
                'rejected_count': len(rejected),
                'evaluation_time': datetime.now().isoformat()
            }
        }
    
    def _evaluate_kpop_question(self, question: Dict, age_group: str,
                                difficulty: str, age_criteria: Dict) -> Dict:
        """개별 K-POP 문제 평가 (간소화)"""
        
        scores = {
            'content_relevance': 0.8,
            'difficulty_match': 0.7,
            'educational_value': 0.8,
            'format_correctness': 0.9
        }
        
        # 간단한 검증 로직
        all_text = str(question).lower()
        
        # K-POP 관련성 체크
        if any(artist.lower() in all_text for artist in age_criteria['must_have_artists']):
            scores['content_relevance'] += 0.1
        
        if any(artist.lower() in all_text for artist in age_criteria['forbidden_artists']):
            scores['content_relevance'] -= 0.3
        
        # 가중 평균
        total_score = sum(
            scores[criterion] * self.question_criteria[criterion]['weight']
            for criterion in scores
        )
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': [],
            'passed': total_score >= self.question_threshold
        }
    
    def _generate_question_suggestions(self, evaluated_questions: List[Dict], 
                                      age_group: str) -> List[str]:
        """문제 개선 제안"""
        
        suggestions = []
        avg_score = sum(eq['score'] for eq in evaluated_questions) / len(evaluated_questions) if evaluated_questions else 0
        age_criteria = self.kpop_age_criteria[age_group]
        
        if avg_score < self.question_threshold:
            suggestions.append(f"문제가 {age_group} K-POP 학습에 부적합합니다.")
            suggestions.append(f"{age_criteria['era']} 콘텐츠를 더 반영하세요.")
        elif avg_score < 0.8:
            suggestions.append("문제 품질이 양호하나 개선 여지가 있습니다.")
        else:
            suggestions.append("훌륭한 K-POP 학습 문제입니다!")
        
        # K-POP 관련성이 낮은 문제들 확인
        low_relevance = [eq for eq in evaluated_questions 
                        if eq['evaluation']['scores'].get('content_relevance', 0) < 0.7]
        
        if low_relevance:
            suggestions.append(f"{len(low_relevance)}개 문제에 K-POP 요소가 부족합니다.")
            suggestions.append(f"추천: {', '.join(age_criteria['must_have_artists'][:2])} 관련 내용 추가")
        
        return suggestions
    
    def _generate_suggestions(self, evaluated_questions: List[Dict[str, Any]], 
                             difficulty: str, interest: str, 
                             overall_evaluation: Dict[str, Any]) -> List[str]:
        """기존 코드 호환성을 위한 메서드"""
        
        age_group = overall_evaluation.get("age_group", "20대") if isinstance(overall_evaluation, dict) else "20대"
        suggestions = self._generate_question_suggestions(evaluated_questions, age_group)
        
        # 난이도별 힌트 추가
        if difficulty == "beginner":
            suggestions.append("초급: 선택지 수를 3-4개로 유지하고 문장을 40자 내로 줄여보세요.")
        elif difficulty == "intermediate":
            suggestions.append("중급: 빈칸 채우기/객관식을 적절히 섞고 어휘 난이도를 약간 높여보세요.")
        elif difficulty == "advanced":
            suggestions.append("고급: 해설에 문법/담화 표지를 추가하고 장문 독해를 더 포함해 보세요.")
        
        # 관심사별 힌트
        if interest == "kpop":
            suggestions.append("K-POP 용어(컴백/스밍/직캠 등) 노출을 늘려 실제 맥락을 강화하세요.")
        
        return suggestions