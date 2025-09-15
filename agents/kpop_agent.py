"""
interest_agent.py - 사용자 관심사 기반 최신 콘텐츠 검색 및 문장 생성
"""
from base_agent import BaseAgent
from typing import Dict, Any, List
import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class InterestAgent(BaseAgent):
    """사용자 관심사 기반 최신 정보 검색 및 문장 생성 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="InterestAgent")
        
        # 검색 API 설정 (무료 API 사용)
        self.search_apis = {
            'news': 'https://newsapi.org/v2/everything',  # News API (무료 계정 필요)
            'wiki': 'https://ko.wikipedia.org/api/rest_v1/page/summary/'
        }
        
        # 관심사별 키워드 매핑
        self.interest_keywords = {
            'kpop': ['케이팝', 'K-POP', '아이돌', 'BTS', 'BLACKPINK', 'Stray Kids', 'SEVENTEEN', 'NewJeans', 'NCT', 'ENHYPEN', 'LE SSERAFIM', 'IVE', 'aespa'],
        }
        
        # 나이대별 언어 스타일
        self.age_styles = {
            '10대': {'formality': 'casual', 'complexity': 'simple'},
            '20대': {'formality': 'casual', 'complexity': 'medium'},
            '30대': {'formality': 'polite', 'complexity': 'medium'},
            '40대+': {'formality': 'formal', 'complexity': 'complex'}
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        관심사 기반 콘텐츠 처리
        
        Args:
            input_data: {
                'interest': str,  # 관심사 (예: 'kpop')
                'age_group': str,  # 나이대 (예: '20대')
                'difficulty': str  # 난이도 (beginner/intermediate/advanced)
            }
        
        Returns:
            {
                'content': List[str],  # 생성된 문장들
                'sources': List[str],  # 출처 정보
                'metadata': Dict  # 메타데이터
            }
        """
        # 입력 검증
        if not self.validate_input(input_data, ['interest', 'age_group', 'difficulty']):
            return {'error': '필수 입력 누락'}
        
        interest = input_data['interest']
        age_group = input_data['age_group']
        difficulty = input_data['difficulty']
        
        logger.info(f"📚 {interest} 관련 콘텐츠 검색 중...")
        
        # 1. 최신 정보 검색
        search_results = self._search_online_content(interest)
        
        # 2. 난이도별 문장 생성
        generated_content = self._generate_sentences(
            search_results, 
            interest, 
            age_group, 
            difficulty
        )
        
        # 3. 결과 반환
        return {
            'content': generated_content['sentences'],
            'sources': generated_content['sources'],
            'metadata': {
                'interest': interest,
                'age_group': age_group,
                'difficulty': difficulty,
                'timestamp': datetime.now().isoformat(),
                'total_sentences': len(generated_content['sentences'])
            }
        }
    
    def _search_online_content(self, interest: str) -> List[Dict]:
        """온라인에서 관심사 관련 최신 정보 검색"""
        results = []
        keywords = self.interest_keywords.get(interest, [interest])
        
        for keyword in keywords[:3]:  # 상위 3개 키워드만 검색
            try:
                # Wikipedia 검색 (무료, API 키 불필요)
                wiki_url = f"https://ko.wikipedia.org/api/rest_v1/page/summary/{keyword}"
                response = requests.get(wiki_url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        'source': 'Wikipedia',
                        'title': data.get('title', ''),
                        'content': data.get('extract', ''),
                        'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                    })
                
                # 간단한 웹 스크래핑 (예시 - 실제로는 더 많은 소스 필요)
                if interest == 'kpop':
                    # K-pop 관련 정적 데이터 (실제로는 API나 크롤링 필요)
                    results.append({
                        'source': 'K-pop',
                        'title': f'{keyword} 최신 뉴스',
                        'content': f'{keyword}가 새로운 앨범을 발표했습니다. 이번 앨범은 다양한 장르의 음악을 선보이며 팬들의 큰 관심을 받고 있습니다.',
                        'url': 'example.com'
                    })
                
                time.sleep(0.5)  # API 호출 제한 방지
                
            except Exception as e:
                logger.warning(f"검색 오류 ({keyword}): {e}")
                continue
        
        # 검색 결과가 없으면 기본 콘텐츠 제공
        if not results:
            results = self._get_default_content(interest)
        
        return results
    
    def _get_default_content(self, interest: str) -> List[Dict]:
        """기본 콘텐츠 제공 (검색 실패시)"""
        default_contents = {
            'kpop': [
                {
                    'source': 'Default',
                    'title': 'K-POP의 세계적 인기',
                    'content': 'K-POP은 한국의 대중음악으로, 전 세계적으로 큰 인기를 얻고 있습니다. BTS, BLACKPINK 등 많은 그룹이 빌보드 차트에서 좋은 성적을 거두고 있습니다.',
                    'url': ''
                },
                {
                    'source': 'Default',
                    'title': 'K-POP 댄스 문화',
                    'content': 'K-POP의 특징 중 하나는 화려한 퍼포먼스입니다. 정교한 안무와 무대 연출로 팬들을 매료시킵니다.',
                    'url': ''
                }
            ],
        }
        
        return default_contents.get(interest, [
            {
                'source': 'Default',
                'title': f'{interest} 관련 정보',
                'content': f'{interest}는 한국 문화의 중요한 부분입니다.',
                'url': ''
            }
        ])
    
    def _generate_sentences(self, search_results: List[Dict], interest: str, 
                           age_group: str, difficulty: str) -> Dict:
        """검색 결과를 바탕으로 난이도별 문장 생성"""
        
        sentences = []
        sources = []
        
        # 난이도별 문장 복잡도 설정
        sentence_configs = {
            'beginner': {
                'max_length': 30,
                'vocab_level': 'basic',
                'grammar': 'simple',
                'count': 10
            },
            'intermediate': {
                'max_length': 50,
                'vocab_level': 'intermediate',
                'grammar': 'moderate',
                'count': 8
            },
            'advanced': {
                'max_length': 70,
                'vocab_level': 'advanced',
                'grammar': 'complex',
                'count': 6
            }
        }
        
        config = sentence_configs.get(difficulty, sentence_configs['beginner'])
        style = self.age_styles.get(age_group, self.age_styles['20대'])
        
        for result in search_results[:3]:  # 상위 3개 결과만 사용
            content = result['content'][:500]  # 내용 제한
            
            # 프롬프트 생성
            prompt = self._create_generation_prompt(
                content, interest, config, style
            )
            
            # LLM을 통한 문장 생성
            generated = self.generate_response(
                prompt,
                max_new_tokens=200,
                temperature=0.7
            )
            
            # 생성된 문장 파싱
            parsed_sentences = self._parse_generated_sentences(generated, config)
            sentences.extend(parsed_sentences)
            
            # 출처 추가
            sources.append({
                'title': result['title'],
                'source': result['source'],
                'url': result['url']
            })
            
            if len(sentences) >= config['count']:
                break
        
        # 문장 수가 부족하면 추가 생성
        while len(sentences) < config['count']:
            additional_prompt = f"{interest}에 대한 {difficulty} 수준의 한국어 학습 문장을 만들어주세요."
            additional = self.generate_response(additional_prompt, max_new_tokens=100)
            sentences.append(additional.strip())
        
        return {
            'sentences': sentences[:config['count']],
            'sources': sources
        }
    
    def _create_generation_prompt(self, content: str, interest: str, 
                                 config: Dict, style: Dict) -> str:
        """문장 생성을 위한 프롬프트 생성"""
        
        vocab_guides = {
            'basic': '한국어 배운지 6개월 이내 되는 사람을 위한 문제이다. 기초 단어만 사용',
            'intermediate': '한국어 배운지 1년 이내 되는 사람을 위한 문제이다. 중급 단어 포함',
            'advanced': '한국어 배운지 1년 이상 되는 사람을 위한 문제이다. 고급 어휘와 관용구 사용'
        }
        
        grammar_guides = {
            'simple': '현재형과 과거형 위주',
            'moderate': '다양한 시제와 연결어미 사용',
            'complex': '복잡한 문법 구조와 피동/사동 포함'
        }
        
        formality_guides = {
            'casual': '반말체 (해요체)',
            'polite': '존댓말 (합니다체)',
            'formal': '격식체 (하십시오체)'
        }
        
        prompt = f"""
다음 내용을 바탕으로 {interest} 관련 한국어 학습 문장을 만들어주세요.

원문 내용: {content}

요구사항:
- 난이도: {vocab_guides[config['vocab_level']]}
- 문법: {grammar_guides[config['grammar']]}
- 문체: {formality_guides[style['formality']]}
- 문장 길이: 최대 {config['max_length']}자
- 문장 개수: 3개

각 문장은 다음 형식으로 작성하세요:
1. [문장]
2. [문장]
3. [문장]
"""
        
        return prompt
    
    def _parse_generated_sentences(self, generated: str, config: Dict) -> List[str]:
        """생성된 텍스트에서 문장 추출"""
        sentences = []
        
        # 번호가 매겨진 문장 추출
        lines = generated.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # 번호나 기호 제거
                sentence = line.lstrip('0123456789.-) ').strip()
                if sentence and len(sentence) <= config['max_length'] * 2:
                    sentences.append(sentence)
        
        # 문장이 없으면 전체 텍스트를 문장으로 분할
        if not sentences:
            for sentence in generated.split('.'):
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:
                    sentences.append(sentence + '.')
        
        return sentences