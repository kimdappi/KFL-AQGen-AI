"""
main.py - 한국어 학습 문제지 생성 시스템 메인 파일 (재생성 피드백 루프 포함)
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any

# 에이전트 임포트
from agents.base_agent import BaseAgent
from agents.kpop_agent import InterestAgent
from agents.worksheet_agent import WorksheetAgent  
from agents.critic_agent import CriticAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('korean_test_generator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KoreanTestGenerator:
    """한국어 학습 문제지 생성 시스템 (재생성 피드백 루프 포함)"""
    
    def __init__(self, model_name: str = None):
        """
        시스템 초기화
        
        Args:
            model_name: 사용할 LLM 모델 (기본: skt/kogpt2-base-v2)
        """
        logger.info("한국어 학습 문제지 생성 시스템 시작")
        
        # 에이전트 초기화
        self.interest_agent = InterestAgent(model_name)
        self.worksheet_agent = WorksheetAgent(model_name)
        self.critic_agent = CriticAgent(model_name)
        
        # 결과 저장 디렉토리
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 재생성 설정
        self.max_regeneration_attempts = 3
        
        logger.info("모든 에이전트 초기화 완료")
    
    def generate_test(self, interest: str, age_group: str, difficulty: str) -> Dict[str, Any]:
        """
        문제지 생성 전체 프로세스 (재생성 피드백 루프 포함)
        
        Args:
            interest: 관심사 (kpop, kdrama, korean_food 등)
            age_group: 나이대 (10대, 20대, 30대, 40대+)
            difficulty: 난이도 (beginner, intermediate, advanced)
        
        Returns:
            생성 결과 딕셔너리
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"문제지 생성 시작")
        logger.info(f"  - 관심사: {interest}")
        logger.info(f"  - 나이대: {age_group}")
        logger.info(f"  - 난이도: {difficulty}")
        logger.info(f"{'='*50}\n")
        
        try:
            # 1단계: 관심사 기반 콘텐츠 생성 (재생성 루프)
            logger.info("1단계: 관심사 기반 콘텐츠 생성 중...")
            interest_result = self._generate_content_with_feedback(interest, age_group, difficulty)
            
            if 'error' in interest_result:
                raise Exception(f"콘텐츠 생성 실패: {interest_result['error']}")
            
            # 생성된 문장 출력
            self._print_generated_sentences(interest_result)
            
            # 2단계: 문제지 생성 (재생성 루프)
            logger.info("\n2단계: 문제지 생성 중...")
            worksheet_result = self._generate_questions_with_feedback(
                interest_result['content'], interest, age_group, difficulty
            )
            
            if 'error' in worksheet_result:
                raise Exception(f"문제지 생성 실패: {worksheet_result['error']}")
            
            logger.info(f"최종적으로 {len(worksheet_result['questions'])}개 문제 생성 완료")
            
            # 3단계: 최종 문제지 생성
            logger.info("\n3단계: 최종 문제지 생성 중...")
            final_result = self.worksheet_agent.process({
                'content': interest_result['content'],
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'questions': worksheet_result['questions']
            })
            
            # 4단계: 결과 저장
            result = self._save_results({
                'interest_content': interest_result,
                'worksheet': worksheet_result,
                'evaluation': worksheet_result.get('final_evaluation', {}),
                'pdf_path': final_result.get('pdf_path', ''),
                'metadata': {
                    'interest': interest,
                    'age_group': age_group,
                    'difficulty': difficulty,
                    'generated_at': datetime.now().isoformat(),
                    'total_regeneration_attempts': interest_result.get('total_attempts', 1) + worksheet_result.get('total_attempts', 1) - 2
                }
            })
            
            # 결과 요약 출력
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"문제지 생성 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _generate_content_with_feedback(self, interest: str, age_group: str, difficulty: str) -> Dict[str, Any]:
        """문장 생성 및 검증 피드백 루프"""
        
        attempt = 1
        regeneration_guide = None
        
        while attempt <= self.max_regeneration_attempts:
            logger.info(f"문장 생성 시도 {attempt}/{self.max_regeneration_attempts}")
            
            # 문장 생성
            generation_input = {
                'interest': interest,
                'age_group': age_group,
                'difficulty': difficulty
            }
            
            # 재생성 가이드라인이 있으면 추가
            if regeneration_guide:
                generation_input['regeneration_guide'] = regeneration_guide
                logger.info("재생성 가이드라인 적용")
            
            interest_result = self.interest_agent.process(generation_input)
            
            if 'error' in interest_result:
                if attempt == self.max_regeneration_attempts:
                    return interest_result
                attempt += 1
                continue
            
            # 문장 검증
            validation_input = {
                'validation_type': 'sentences',
                'sentences': interest_result['content'],
                'age_group': age_group,
                'difficulty': difficulty,
                'attempt': attempt
            }
            
            validation_result = self.critic_agent.process(validation_input)
            
            if not validation_result.get('needs_regeneration', False):
                logger.info("문장 검증 통과")
                interest_result['validation_result'] = validation_result
                interest_result['total_attempts'] = attempt
                return interest_result
            
            # 재생성 필요
            logger.warning(f"문장 재생성 필요: {validation_result.get('regeneration_reason', '품질 기준 미달')}")
            regeneration_guide = validation_result.get('regeneration_guide', {})
            
            if attempt < self.max_regeneration_attempts:
                self._print_regeneration_feedback(validation_result, "문장")
            
            attempt += 1
        
        # 최대 시도 횟수 도달
        logger.warning(f"최대 재생성 시도 {self.max_regeneration_attempts}회 도달. 현재 결과를 사용합니다.")
        interest_result['validation_result'] = validation_result
        interest_result['total_attempts'] = attempt - 1
        return interest_result
    
    def _generate_questions_with_feedback(self, content: list, interest: str, 
                                        age_group: str, difficulty: str) -> Dict[str, Any]:
        """문제 생성 및 검증 피드백 루프"""
        
        attempt = 1
        regeneration_guide = None
        
        while attempt <= self.max_regeneration_attempts:
            logger.info(f"문제 생성 시도 {attempt}/{self.max_regeneration_attempts}")
            
            # 문제 생성
            generation_input = {
                'content': content,
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group
            }
            
            # 재생성 가이드라인이 있으면 추가
            if regeneration_guide:
                generation_input['regeneration_guide'] = regeneration_guide
                logger.info("재생성 가이드라인 적용")
            
            worksheet_result = self.worksheet_agent.process(generation_input)
            
            if 'error' in worksheet_result:
                if attempt == self.max_regeneration_attempts:
                    return worksheet_result
                attempt += 1
                continue
            
            # 문제 검증
            validation_input = {
                'validation_type': 'questions',
                'questions': worksheet_result['questions'],
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'attempt': attempt
            }
            
            validation_result = self.critic_agent.process(validation_input)
            
            if not validation_result.get('needs_regeneration', False):
                logger.info("문제 검증 통과")
                worksheet_result['questions'] = validation_result['approved_questions']
                worksheet_result['final_evaluation'] = validation_result
                worksheet_result['total_attempts'] = attempt
                return worksheet_result
            
            # 재생성 필요
            logger.warning(f"문제 재생성 필요: {validation_result.get('regeneration_reason', '품질 기준 미달')}")
            regeneration_guide = validation_result.get('regeneration_guide', {})
            
            # 중복 문제 정보 출력
            duplicate_issues = validation_result.get('duplicate_issues', [])
            if duplicate_issues:
                logger.warning(f"중복 문제 {len(duplicate_issues)}개 발견")
                for issue in duplicate_issues[:3]:  # 처음 3개만 출력
                    logger.warning(f"  - {issue['type']}: {issue.get('content', '')[:50]}...")
            
            if attempt < self.max_regeneration_attempts:
                self._print_regeneration_feedback(validation_result, "문제")
            
            attempt += 1
        
        # 최대 시도 횟수 도달 - 승인된 문제만이라도 사용
        logger.warning(f"최대 재생성 시도 {self.max_regeneration_attempts}회 도달.")
        
        if validation_result.get('approved_questions'):
            logger.info(f"승인된 문제 {len(validation_result['approved_questions'])}개를 사용합니다.")
            worksheet_result['questions'] = validation_result['approved_questions']
        else:
            logger.warning("승인된 문제가 없어 원본 문제를 사용합니다.")
        
        worksheet_result['final_evaluation'] = validation_result
        worksheet_result['total_attempts'] = attempt - 1
        return worksheet_result
    
    def _print_generated_sentences(self, interest_result: Dict):
        """생성된 문장 출력"""
        
        print("\n" + "="*60)
        print("생성된 학습 문장들:")
        print("="*60)
        
        for i, sentence in enumerate(interest_result.get('content', []), 1):
            print(f"{i}. {sentence}")
        
        print("="*60)
        
        # 검증 결과가 있으면 출력
        if 'validation_result' in interest_result:
            validation = interest_result['validation_result']
            approved_count = len(validation.get('approved_sentences', []))
            total_count = len(interest_result.get('content', []))
            print(f"\n검증 결과: {approved_count}/{total_count} 문장 승인 ({approved_count/total_count*100:.1f}%)")
        
        # 출처 정보 출력
        if 'sources' in interest_result:
            print("\n참고 출처:")
            for source in interest_result['sources']:
                print(f"  - {source.get('title', 'Unknown')} ({source.get('source', '')})")
        
        print()
    
    def _print_regeneration_feedback(self, validation_result: Dict, content_type: str):
        """재생성 피드백 출력"""
        
        print(f"\n{'='*50}")
        print(f"{content_type} 재생성 필요")
        print(f"{'='*50}")
        
        reason = validation_result.get('regeneration_reason', '품질 기준 미달')
        print(f"사유: {reason}")
        
        # 재생성 가이드라인 출력
        guide = validation_result.get('regeneration_guide', {})
        if guide:
            print("\n개선 사항:")
            
            if 'recommendations' in guide:
                for rec in guide['recommendations'][:3]:  # 상위 3개만
                    print(f"  - {rec}")
            
            if 'common_issues' in guide:
                print(f"\n주요 문제점:")
                for issue in guide['common_issues'][:3]:
                    print(f"  - {issue}")
            
            if 'duplicate_prevention' in guide:
                dup = guide['duplicate_prevention']
                if any(dup.values()):
                    print(f"\n중복 방지:")
                    if dup.get('avoid_duplicate_questions', 0) > 0:
                        print(f"  - 중복 문제 {dup['avoid_duplicate_questions']}개 수정 필요")
                    if dup.get('avoid_duplicate_answers', 0) > 0:
                        print(f"  - 중복 답 {dup['avoid_duplicate_answers']}개 수정 필요")
        
        attempt = validation_result.get('attempt', 1)
        max_attempts = validation_result.get('max_attempts', self.max_regeneration_attempts)
        print(f"\n재생성 시도: {attempt}/{max_attempts}")
        print("="*50)
    
    def _save_results(self, results: Dict) -> Dict:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 파일로 상세 결과 저장
        json_path = os.path.join(
            self.output_dir,
            f"test_result_{results['metadata']['interest']}_{timestamp}.json"
        )
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"결과 저장: {json_path}")
        
        results['json_path'] = json_path
        return results
    
    def _print_summary(self, result: Dict):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("문제지 생성 완료!")
        print("="*60)
        
        metadata = result.get('metadata', {})
        evaluation = result.get('evaluation', {})
        
        print(f"\n기본 정보:")
        print(f"  - 관심사: {metadata.get('interest')}")
        print(f"  - 나이대: {metadata.get('age_group')}")
        print(f"  - 난이도: {metadata.get('difficulty')}")
        
        print(f"\n생성 결과:")
        content_count = len(result.get('interest_content', {}).get('content', []))
        question_count = len(result.get('worksheet', {}).get('questions', []))
        print(f"  - 생성된 콘텐츠: {content_count}개")
        print(f"  - 최종 문제 수: {question_count}개")
        
        # 재생성 정보
        total_attempts = metadata.get('total_regeneration_attempts', 0)
        if total_attempts > 0:
            print(f"  - 총 재생성 시도: {total_attempts}회")
        
        print(f"\n품질 평가:")
        overall_score = evaluation.get('overall_score', 0)
        print(f"  - 전체 점수: {overall_score:.2f}/1.0")
        
        eval_data = evaluation.get('evaluation', {})
        if eval_data:
            print(f"  - 승인된 문제: {eval_data.get('passed_questions', 0)}개")
            print(f"  - 거부된 문제: {eval_data.get('failed_questions', 0)}개")
        
        # 개선 제안
        suggestions = evaluation.get('suggestions', [])
        if suggestions:
            print(f"\n개선 제안:")
            for suggestion in suggestions[:3]:  # 처음 3개만
                print(f"  {suggestion}")
        
        print(f"\n저장 위치:")
        if result.get('pdf_path'):
            print(f"  - PDF: {result['pdf_path']}")
        print(f"  - JSON: {result.get('json_path')}")
        
        print("\n" + "="*60)


def get_user_input():
    """사용자 입력 받기"""
    print("\n한국어 학습 문제지 생성 시스템")
    print("="*50)
    
    # 관심사 선택
    print("\n관심사를 선택하세요:")
    interests = {
        '1': 'kpop',
        '2': 'kdrama',
        '3': 'korean_food',
        '4': 'korean_culture',
        '5': 'technology',
        '6': 'sports'
    }
    
    for key, value in interests.items():
        print(f"  {key}. {value}")
    
    interest_choice = input("\n선택 (1-6): ").strip()
    interest = interests.get(interest_choice, 'kpop')
    
    # 나이대 선택
    print("\n학습자 나이대를 선택하세요:")
    age_groups = {
        '1': '10대',
        '2': '20대',
        '3': '30대',
        '4': '40대+'
    }
    
    for key, value in age_groups.items():
        print(f"  {key}. {value}")
    
    age_choice = input("\n선택 (1-4): ").strip()
    age_group = age_groups.get(age_choice, '20대')
    
    # 난이도 선택
    print("\n난이도를 선택하세요:")
    difficulties = {
        '1': 'beginner',
        '2': 'intermediate',
        '3': 'advanced'
    }
    
    for key, value in difficulties.items():
        print(f"  {key}. {value}")
    
    difficulty_choice = input("\n선택 (1-3): ").strip()
    difficulty = difficulties.get(difficulty_choice, 'beginner')
    
    return interest, age_group, difficulty


def main():
    """메인 실행 함수"""
    try:
        # 모델 선택 (선택사항)
        print("\n사용할 모델을 선택하세요:")
        print("  1. KoGPT2 (가장 빠름, 125M)")
        print("  2. Polyglot-Ko (균형, 1.3B)")
        print("  3. 기본값 사용")
        
        model_choice = input("\n선택 (1-3): ").strip()
        
        model_map = {
            '1': 'skt/kogpt2-base-v2',
            '2': 'EleutherAI/polyglot-ko-1.3b',
            '3': None
        }
        
        model_name = model_map.get(model_choice)
        
        # 시스템 초기화
        print("\n시스템 초기화 중...")
        generator = KoreanTestGenerator(model_name)
        
        while True:
            # 사용자 입력
            interest, age_group, difficulty = get_user_input()
            
            print(f"\n선택 확인:")
            print(f"  - 관심사: {interest}")
            print(f"  - 나이대: {age_group}")
            print(f"  - 난이도: {difficulty}")
            
            confirm = input("\n진행하시겠습니까? (y/n): ").strip().lower()
            
            if confirm == 'y':
                # 문제지 생성
                result = generator.generate_test(interest, age_group, difficulty)
                
                if 'error' not in result:
                    print("\n문제지 생성이 완료되었습니다!")
                else:
                    print(f"\n오류 발생: {result['error']}")
            
            # 계속 여부
            continue_choice = input("\n다른 문제지를 생성하시겠습니까? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        
        print("\n프로그램을 종료합니다. 감사합니다!")
        
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"시스템 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n시스템 오류가 발생했습니다: {e}")
        print("\n도움이 필요하면 로그 파일을 확인하세요: korean_test_generator.log")


if __name__ == "__main__":
    main()