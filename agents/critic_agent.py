"""
critic_agent.py - K-POP 세대별 콘텐츠 및 문제 검증 에이전트
"""
from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CriticAgent(BaseAgent):
    """K-POP 세대별 콘텐츠 검증 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="CriticAgent")
        
        # K-POP 세대별 검증 기준
        self.kpop_age_criteria = {
            '10대': {
                'must_have_artists': ['NewJeans', 'IVE', 'LE SSERAFIM', 'ENHYPEN', 'Stray Kids'],
                'era': '4세대 (2020-2024)',
                'platforms': ['TikTok', '위버스', '버블', '유튜브 쇼츠'],
                'forbidden_artists': ['H.O.T', '젝스키스', 'S.E.S'],  # 너무 오래된 그룹
                'key_terms': ['챌린지', '직캠', '포카', '스밍', '컴백'],
                'cultural_refs': ['음방 1위', '아육대', '팬싸', '영통']
            },
            '20대': {
                'must_have_artists': ['BTS', 'BLACKPINK', 'SEVENTEEN', 'NCT', 'aespa'],
                'era': '3.5-4세대 (2015-2024)',
                'platforms': ['유튜브', '트위터', '위버스', '브이라이브'],
                'forbidden_artists': ['H.O.T', '젝스키스'],  # 1세대는 부적절
                'key_terms': ['월드투어', '빌보드', '그래미', '정규앨범', '유닛'],
                'cultural_refs': ['스타디움 콘서트', '팬미팅', '시즌그리팅', '자컨']
            },
            '30대': {
                'must_have_artists': ['BIGBANG', 'EXO', '소녀시대', 'SHINee', '2NE1'],
                'era': '2-3세대 (2010-2020)',
                'platforms': ['팬카페', '멜론', '음악방송', '유튜브'],
                'forbidden_artists': ['NewJeans', 'IVE'],  # 너무 최신 그룹
                'key_terms': ['컴백', '입대', '제대', '재계약', '솔로'],
                'cultural_refs': ['응원봉', '팬클럽', '연말시상식', '가요대전']
            },
            '40대+': {
                'must_have_artists': ['H.O.T', 'S.E.S', '핑클', '신화', 'god'],
                'era': '1-2세대 (1996-2010)',
                'platforms': ['팬카페', 'CD', '카세트', '음반'],
                'forbidden_artists': ['NewJeans', 'IVE', 'LE SSERAFIM', 'ENHYPEN'],  # 4세대 부적절
                'key_terms': ['데뷔', '해체', '재결합', '1집', '팬클럽 창단'],
                'cultural_refs': ['가요톱텐', '뮤직뱅크', '팬레터', '사인회']
            }
        }
        
        # 문장 평가 기준
        self.sentence_criteria = {
            'generation_accuracy': {
                'weight': 0.2,
                'description': '세대 정확성'
            },
            'cultural_relevance': {
                'weight': 0.2,
                'description': '문화적 관련성'
            },
            'difficulty_match': {
                'weight': 0.2,
                'description': '난이도 적절성'
            },
            'linguistic_quality': {
                'weight': 0.4,
                'description': '언어적 품질'
            }
        }
        
        # 문제 평가 기준
        self.question_criteria = {
            'content_relevance': {
                'weight': 0.2,
                'description': 'K-POP 세대 관련성'
            },
            'difficulty_match': {
                'weight': 0.3,
                'description': '난이도 적절성'
            },
            'educational_value': {
                'weight': 0.3,
                'description': '한국어 교육 목적에 적합'
            },
            'format_correctness': {
                'weight': 0.2,
                'description': '형식 정확성'
            }
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        검증 처리 - 검증 유형에 따라 분기
        """
        validation_type = input_data.get('validation_type', 'questions')
        
        if validation_type == 'sentences':
            return self._validate_sentences(input_data)
        else:
            return self._validate_questions(input_data)
    
    def _validate_sentences(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """K-POP 세대별 문장 검증"""
        
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
        threshold = 0.7  # K-POP 정확성이 중요하므로 높은 기준
        approved = [es for es in evaluated_sentences if es['score'] >= threshold]
        rejected = [es for es in evaluated_sentences if es['score'] < threshold]
        
        # 전체 평가
        average_score = sum(es['score'] for es in evaluated_sentences) / len(evaluated_sentences) if evaluated_sentences else 0
        needs_regeneration = len(approved) < len(sentences) * 0.6  # 60% 미만이면 재생성
        
        # 개선 제안
        suggestions = self._generate_kpop_suggestions(evaluated_sentences, age_group, age_criteria)
        
        return {
            'approved_sentences': [es['sentence'] for es in approved],
            'rejected_sentences': [es['sentence'] for es in rejected],
            'suggestions': suggestions,
            'needs_regeneration': needs_regeneration,
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
        
        # 시대 키워드 체크
        era_keywords = age_criteria['era'].split('(')[1].rstrip(')').split('-')
        for year in era_keywords:
            if year in sentence:
                score += 0.1
        
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
        
        # LLM을 통한 추가 평가
        prompt = f"""
다음 문장이 {age_criteria['era']} K-POP 팬덤 문화를 잘 반영하는지 평가해주세요.

문장: {sentence}
시대: {age_criteria['era']}
주요 아티스트: {', '.join(age_criteria['must_have_artists'][:3])}

평가 (0-10점):
"""
        
        response = self.generate_response(prompt, max_new_tokens=50)
        
        try:
            for word in response.split():
                if word.replace('.', '').replace(':', '').isdigit():
                    num = float(word.replace('.', '').replace(':', ''))
                    if num <= 10:
                        llm_score = num / 10
                        score = (score + llm_score) / 2
                        break
        except:
            pass
        
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
            if '때문에' not in sentence and '통해' not in sentence and '위해' not in sentence:
                score -= 0.2
        
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
        
        # 세부 분석
        generation_scores = [es['evaluation']['scores'].get('generation_accuracy', 0) for es in evaluated_sentences]
        avg_generation = sum(generation_scores) / len(generation_scores) if generation_scores else 0
        
        if avg_generation < 0.7:
            suggestions.append(f"🎯 {age_criteria['era']} 시대 특성을 더 반영하세요.")
            suggestions.append(f"   - 사용 금지: {', '.join(age_criteria['forbidden_artists'][:2])}")
            suggestions.append(f"   - 권장 키워드: {', '.join(age_criteria['key_terms'][:3])}")
        
        return suggestions
    
    def _validate_questions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """K-POP 문제 검증"""
        
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
        threshold = 0.65
        approved = [eq for eq in evaluated_questions if eq['score'] >= threshold]
        rejected = [eq for eq in evaluated_questions if eq['score'] < threshold]
        
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
        """개별 K-POP 문제 평가"""
        
        scores = {}
        feedback = []
        
        # 1. K-POP 세대 관련성
        relevance_score = self._check_question_kpop_relevance(question, age_criteria)
        scores['content_relevance'] = relevance_score
        if relevance_score < 0.7:
            feedback.append(f"{age_group} K-POP 세대와 맞지 않는 문제")
        
        # 2. 난이도 적절성
        difficulty_score = self._check_question_difficulty(question, difficulty)
        scores['difficulty_match'] = difficulty_score
        if difficulty_score < 0.7:
            feedback.append(f"{difficulty} 난이도에 부적절")
        
        # 3. 교육적 가치
        educational_score = self._check_educational_value(question)
        scores['educational_value'] = educational_score
        if educational_score < 0.7:
            feedback.append("교육적 가치 부족")
        
        # 4. 형식 정확성
        format_score = self._check_format_correctness(question)
        scores['format_correctness'] = format_score
        if format_score < 0.7:
            feedback.append("문제 형식 오류")
        
        # 가중 평균
        total_score = sum(
            scores[criterion] * self.question_criteria[criterion]['weight']
            for criterion in scores
        )
        
        return {
            'scores': scores,
            'total_score': total_score,
            'feedback': feedback,
            'passed': total_score >= 0.65
        }
    
    def _check_question_kpop_relevance(self, question: Dict, age_criteria: Dict) -> float:
        """문제의 K-POP 세대 관련성 확인"""
        
        # 문제 텍스트 추출
        all_text = []
        for key in ['question', 'sentence', 'statement', 'source', 'passage', 'options']:
            if key in question:
                if isinstance(question[key], list):
                    all_text.extend(question[key])
                else:
                    all_text.append(str(question[key]))
        
        combined_text = ' '.join(all_text).lower()
        score = 0.5
        
        # 필수 아티스트 체크
        for artist in age_criteria['must_have_artists']:
            if artist.lower() in combined_text:
                score += 0.2
        
        # 금지 아티스트 체크
        for artist in age_criteria['forbidden_artists']:
            if artist.lower() in combined_text:
                score -= 0.3
        
        # 문화 요소 체크
        for ref in age_criteria['cultural_refs']:
            if ref in combined_text:
                score += 0.1
        
        return max(0, min(1, score))
    
    def _check_question_difficulty(self, question: Dict, difficulty: str) -> float:
        """문제 난이도 확인"""
        
        q_type = question.get('type', '')
        score = 0.7
        
        if difficulty == 'beginner':
            if q_type in ['multiple_choice', 'true_false']:
                score += 0.2
            elif q_type in ['translation', 'reading_comprehension']:
                score -= 0.2
        elif difficulty == 'intermediate':
            if q_type in ['fill_blank', 'multiple_choice']:
                score += 0.1
        elif difficulty == 'advanced':
            if q_type in ['translation', 'reading_comprehension']:
                score += 0.2
            elif q_type == 'true_false':
                score -= 0.2
        
        return max(0, min(1, score))
    
    def _check_educational_value(self, question: Dict) -> float:
        """교육적 가치 평가"""
        
        score = 0.7
        
        # 설명이 있으면 가치 상승
        if question.get('explanation'):
            score += 0.15
        
        # 힌트가 있으면 가치 상승
        if question.get('hints'):
            score += 0.1
        
        # 포인트가 적절하면 가치 상승
        if question.get('points', 0) > 0:
            score += 0.05
        
        return min(1, score)
    
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
            score = 0.5
        
        return max(0, score)
    
    def _generate_question_suggestions(self, evaluated_questions: List[Dict], 
                                      age_group: str) -> List[str]:
        """문제 개선 제안"""
        
        suggestions = []
        avg_score = sum(eq['score'] for eq in evaluated_questions) / len(evaluated_questions) if evaluated_questions else 0
        age_criteria = self.kpop_age_criteria[age_group]
        
        if avg_score < 0.65:
            suggestions.append(f"⚠️ 문제가 {age_group} K-POP 학습에 부적합합니다.")
            suggestions.append(f"💡 {age_criteria['era']} 콘텐츠를 더 반영하세요.")
        elif avg_score < 0.8:
            suggestions.append("📝 문제 품질이 양호하나 개선 여지가 있습니다.")
        else:
            suggestions.append("✅ 훌륭한 K-POP 학습 문제입니다!")
        
        # K-POP 관련성이 낮은 문제들 확인
        low_relevance = [eq for eq in evaluated_questions 
                        if eq['evaluation']['scores'].get('content_relevance', 0) < 0.7]
        
        if low_relevance:
            suggestions.append(f"🎵 {len(low_relevance)}개 문제에 K-POP 요소가 부족합니다.")
            suggestions.append(f"   추천: {', '.join(age_criteria['must_have_artists'][:2])} 관련 내용 추가")
        
        return suggestions
    
    def _generate_suggestions(self, evaluated_questions: List[Dict[str, Any]],difficulty: str,interest: str,overall_evaluation: Dict[str, Any]) -> List[str]:
        """
        기존 코드가 기대하는 시그니처를 유지.
        내부에서는 difficulty_configs 없이 동작하도록 구성.
        """
        # overall_evaluation에서 age_group 있으면 사용
        age_group = "20대"
        if isinstance(overall_evaluation, dict):
            age_group = overall_evaluation.get("age_group", age_group)

    # 1) 기본 제안: 이미 구현된 question 기반 제안 활용
        suggestions = self._generate_question_suggestions(evaluated_questions, age_group)

    # 2) 난이도별 힌트 몇 개 추가(선택)
        if difficulty == "beginner":
            suggestions.append("🔰 초급: 선택지 수를 3~4개로 유지하고 문장을 40자 내로 줄여보세요.")
        elif difficulty == "intermediate":
            suggestions.append("⚖️ 중급: 빈칸 채우기/객관식을 적절히 섞고 어휘 난이도를 약간 높여보세요.")
        elif difficulty == "advanced":
            suggestions.append("🏁 고급: 해설에 문법/담화 표지를 추가하고 장문 독해를 더 포함해 보세요.")

    # 3) 관심사별 힌트 (K-POP 예시)
        if interest == "kpop":
            suggestions.append("🎵 K-POP 용어(컴백/스밍/직캠 등) 노출을 늘려 실제 맥락을 강화하세요.")

        return suggestions
