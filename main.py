"""
main.py - 한국어 학습 문제지 생성 시스템 메인 파일
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
    """한국어 학습 문제지 생성 시스템"""
    
    def __init__(self, model_name: str = None):
        """
        시스템 초기화
        
        Args:
            model_name: 사용할 LLM 모델 (기본: skt/kogpt2-base-v2)
        """
        logger.info("🚀 한국어 학습 문제지 생성 시스템 시작")
        
        # 에이전트 초기화
        self.interest_agent = InterestAgent(model_name)
        self.testpaper_agent = TestPaperAgent(model_name)
        self.critic_agent = CriticAgent(model_name)
        
        # 결과 저장 디렉토리
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("✅ 모든 에이전트 초기화 완료")
    
    def generate_test(self, interest: str, age_group: str, difficulty: str) -> Dict[str, Any]:
        """
        문제지 생성 전체 프로세스
        
        Args:
            interest: 관심사 (kpop, kdrama, korean_food 등)
            age_group: 나이대 (10대, 20대, 30대, 40대+)
            difficulty: 난이도 (beginner, intermediate, advanced)
        
        Returns:
            생성 결과 딕셔너리
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"📚 문제지 생성 시작")
        logger.info(f"  - 관심사: {interest}")
        logger.info(f"  - 나이대: {age_group}")
        logger.info(f"  - 난이도: {difficulty}")
        logger.info(f"{'='*50}\n")
        
        try:
            # 1단계: 관심사 기반 콘텐츠 생성
            logger.info("📝 1단계: 관심사 기반 콘텐츠 생성 중...")
            interest_result = self.interest_agent.process({
                'interest': interest,
                'age_group': age_group,
                'difficulty': difficulty
            })
            
            if 'error' in interest_result:
                raise Exception(f"콘텐츠 생성 실패: {interest_result['error']}")
            
            logger.info(f"✅ {len(interest_result['content'])}개 문장 생성 완료")
            
            # 2단계: 문제지 생성
            logger.info("\n📝 2단계: 문제지 생성 중...")
            testpaper_result = self.testpaper_agent.process({
                'content': interest_result['content'],
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group
            })
            
            if 'error' in testpaper_result:
                raise Exception(f"문제지 생성 실패: {testpaper_result['error']}")
            
            logger.info(f"✅ {len(testpaper_result['questions'])}개 문제 생성 완료")
            
            # 3단계: 문제 검증
            logger.info("\n📝 3단계: 문제 품질 검증 중...")
            critic_result = self.critic_agent.process({
                'questions': testpaper_result['questions'],
                'difficulty': difficulty,
                'interest': interest,
                'age_group': age_group,
                'content': interest_result['content']
            })
            
            if 'error' in critic_result:
                raise Exception(f"검증 실패: {critic_result['error']}")
            
            logger.info(f"✅ 검증 완료 - 전체 점수: {critic_result['overall_score']:.2f}")
            
            # 4단계: 최종 문제지 생성 (검증 통과 문제만)
            if critic_result['approved_questions']:
                logger.info("\n📝 4단계: 최종 문제지 생성 중...")
                final_result = self.testpaper_agent.process({
                    'content': interest_result['content'],
                    'difficulty': difficulty,
                    'interest': interest,
                    'age_group': age_group,
                    'questions': critic_result['approved_questions']  # 승인된 문제만 사용
                })
                
                pdf_path = final_result.get('pdf_path', testpaper_result.get('pdf_path'))
            else:
                logger.warning("⚠️ 승인된 문제가 없어 원본 문제지를 사용합니다.")
                pdf_path = testpaper_result.get('pdf_path')
            
            # 5단계: 결과 저장
            result = self._save_results({
                'interest_content': interest_result,
                'testpaper': testpaper_result,
                'evaluation': critic_result,
                'pdf_path': pdf_path,
                'metadata': {
                    'interest': interest,
                    'age_group': age_group,
                    'difficulty': difficulty,
                    'generated_at': datetime.now().isoformat()
                }
            })
            
            # 결과 요약 출력
            self._print_summary(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 문제지 생성 실패: {e}")
            return {'error': str(e)}
    
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
        
        logger.info(f"💾 결과 저장: {json_path}")
        
        results['json_path'] = json_path
        return results
    
    def _print_summary(self, result: Dict):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 문제지 생성 완료!")
        print("="*60)
        
        metadata = result.get('metadata', {})
        evaluation = result.get('evaluation', {})
        
        print(f"\n📌 기본 정보:")
        print(f"  - 관심사: {metadata.get('interest')}")
        print(f"  - 나이대: {metadata.get('age_group')}")
        print(f"  - 난이도: {metadata.get('difficulty')}")
        
        print(f"\n📈 생성 결과:")
        print(f"  - 생성된 콘텐츠: {len(result.get('interest_content', {}).get('content', []))}개")
        print(f"  - 생성된 문제: {len(result.get('testpaper', {}).get('questions', []))}개")
        print(f"  - 승인된 문제: {len(evaluation.get('approved_questions', []))}개")
        print(f"  - 거부된 문제: {len(evaluation.get('rejected_questions', []))}개")
        
        print(f"\n⭐ 품질 평가:")
        print(f"  - 전체 점수: {evaluation.get('overall_score', 0):.2f}/1.0")
        eval_data = evaluation.get('evaluation', {})
        print(f"  - 난이도 일관성: {eval_data.get('difficulty_consistency', 0):.2f}")
        print(f"  - 콘텐츠 관련성: {eval_data.get('content_relevance', 0):.2f}")
        
        print(f"\n💡 개선 제안:")
        for i, suggestion in enumerate(evaluation.get('suggestions', [])[:5], 1):
            print(f"  {suggestion}")
        
        print(f"\n📁 저장 위치:")
        if result.get('pdf_path'):
            print(f"  - PDF: {result['pdf_path']}")
        print(f"  - JSON: {result.get('json_path')}")
        
        print("\n" + "="*60)


def get_user_input():
    """사용자 입력 받기"""
    print("\n🎯 한국어 학습 문제지 생성 시스템")
    print("="*50)
    
    # 관심사 선택
    print("\n📌 관심사를 선택하세요:")
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
    print("\n📌 학습자 나이대를 선택하세요:")
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
    print("\n📌 난이도를 선택하세요:")
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
        print("\n🤖 사용할 모델을 선택하세요:")
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
        print("\n⏳ 시스템 초기화 중...")
        generator = KoreanTestGenerator(model_name)
        
        while True:
            # 사용자 입력
            interest, age_group, difficulty = get_user_input()
            
            print(f"\n✅ 선택 확인:")
            print(f"  - 관심사: {interest}")
            print(f"  - 나이대: {age_group}")
            print(f"  - 난이도: {difficulty}")
            
            confirm = input("\n진행하시겠습니까? (y/n): ").strip().lower()
            
            if confirm == 'y':
                # 문제지 생성
                result = generator.generate_test(interest, age_group, difficulty)
                
                if 'error' not in result:
                    print("\n✅ 문제지 생성이 완료되었습니다!")
                else:
                    print(f"\n❌ 오류 발생: {result['error']}")
            
            # 계속 여부
            continue_choice = input("\n다른 문제지를 생성하시겠습니까? (y/n): ").strip().lower()
            if continue_choice != 'y':
                break
        
        print("\n👋 프로그램을 종료합니다. 감사합니다!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        print(f"\n❌ 시스템 오류가 발생했습니다: {e}")
        print("\n도움이 필요하면 로그 파일을 확인하세요: korean_test_generator.log")


if __name__ == "__main__":
    main()