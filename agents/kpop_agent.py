"""
kpop_agent.py - K-POP 세대별 맞춤 콘텐츠 생성 에이전트
"""
from agents.base_agent import BaseAgent
from typing import Dict, Any, List
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class InterestAgent(BaseAgent):
    """K-POP 세대별 맞춤 콘텐츠 생성 에이전트"""
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name, agent_name="KpopAgent")
        
        # K-POP 세대별 아티스트 및 특징
        self.kpop_generations = {
            '10대': {
                'artists': ['NewJeans', 'IVE', 'LE SSERAFIM', 'ENHYPEN', 'Stray Kids', 
                           'NMIXX', 'TREASURE', 'ATEEZ', 'THE BOYZ', 'ITZY'],
                'topics': ['챌린지', 'TikTok', '음방 1위', '컴백', '포토카드', 
                          '팬싸인회', '버블', '위버스', '직캠', '아육대'],
                'keywords': ['스밍', '올팬', '덕질', '최애', '입덕', '탈덕', 
                            '컴백', '티저', '뮤비', '안무'],
                'years': '2020-2024',
                'description': '4세대 K-POP, SNS 중심 팬덤 문화'
            },
            '20대': {
                'artists': ['BTS', 'SEVENTEEN', 'NCT', 'BLACKPINK', 'TWICE', 
                           'Stray Kids', 'ATEEZ', 'TXT', 'aespa', 'ENHYPEN'],
                'topics': ['월드투어', '빌보드', '그래미', '스타디움 콘서트', '팬미팅',
                          '시즌그리팅', '유튜브', '브이라이브', '팬덤', '굿즈'],
                'keywords': ['컴백', '정규앨범', '리패키지', '솔로데뷔', '유닛',
                            '음원차트', '뮤직비디오', '안무영상', '비하인드', '자컨'],
                'years': '2015-2024',
                'description': '3.5-4세대 K-POP, 글로벌 진출 세대'
            },
            '30대': {
                'artists': ['BIGBANG', 'EXO', '소녀시대', 'SHINee', 'INFINITE',
                           '2NE1', 'f(x)', 'BEAST', 'BTOB', 'APINK'],
                'topics': ['팬클럽', '응원봉', '음악방송', '팬카페', '콘서트',
                          '정규앨범', '리패키지', '컴백무대', '연말시상식', '가요대전'],
                'keywords': ['컴백', '활동', '휴식기', '입대', '제대', '재계약',
                            '솔로활동', '드라마OST', '예능출연', '팬미팅'],
                'years': '2010-2020',
                'description': '2-3세대 K-POP, 한류 확산 시대'
            },
            '40대+': {
                'artists': ['H.O.T', '젝스키스', 'S.E.S', '핑클', '신화',
                           'god', '플라이투더스카이', '보아', '동방신기', 'SS501'],
                'topics': ['데뷔', '해체', '재결합', '팬클럽 창단', '1집',
                          'CD', '카세트테이프', '음반', '가요톱텐', '뮤직뱅크'],
                'keywords': ['데뷔무대', '컴백', '정규앨범', '팬레터', '팬클럽',
                            '콘서트', '팬미팅', '사인회', '음반판매량', '가요프로그램'],
                'years': '1996-2010',
                'description': '1-2세대 K-POP, K-POP의 시작'
            }
        }
        
        # 난이도별 문장 템플릿
        self.difficulty_templates = {
            'beginner': {
                'patterns': [
                    "{artist}는 {year}년에 데뷔했어요.",
                    "{artist}의 새 노래가 나왔어요.",
                    "저는 {artist}를 좋아해요.",
                    "{artist} 콘서트에 가고 싶어요.",
                    "{topic}에서 {artist}를 봤어요."
                ],
                'max_length': 30
            },
            'intermediate': {
                'patterns': [
                    "{artist}가 {topic}에서 {keyword} 활동을 시작했습니다.",
                    "{year}년에 데뷔한 {artist}는 현재 {topic}로 유명합니다.",
                    "많은 팬들이 {artist}의 {keyword}를 기다리고 있습니다.",
                    "{artist}의 {keyword}는 {topic}에서 큰 인기를 얻었습니다.",
                    "요즘 {artist}가 {topic} 관련 활동을 활발히 하고 있어요."
                ],
                'max_length': 50
            },
            'advanced': {
                'patterns': [
                    "{artist}는 {year}년대 {description}을 대표하는 그룹으로, {topic} 분야에서 큰 성과를 거두었습니다.",
                    "{keyword} 활동을 통해 {artist}는 {topic} 시장에서 독보적인 위치를 차지하게 되었습니다.",
                    "{year}년대 K-POP의 특징인 {description}는 {artist}의 {keyword}를 통해 잘 나타납니다.",
                    "팬덤 문화의 변화와 함께 {artist}의 {topic} 관련 활동도 {keyword} 중심으로 진화했습니다."
                ],
                'max_length': 70
            }
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        K-POP 세대별 콘텐츠 생성
        
        Args:
            input_data: {
                'interest': 'kpop' (고정),
                'age_group': str,
                'difficulty': str
            }
        """
        # 입력 검증
        if not self.validate_input(input_data, ['age_group', 'difficulty']):
            return {'error': '필수 입력 누락'}
        
        age_group = input_data['age_group']
        difficulty = input_data['difficulty']
        
        logger.info(f"🎵 {age_group} 대상 K-POP 콘텐츠 생성 중...")
        
        # 세대별 데이터 가져오기
        generation_data = self.kpop_generations.get(age_group, self.kpop_generations['20대'])
        
        # 문장 생성
        sentences = self._generate_kpop_sentences(generation_data, difficulty)
        
        # 메타데이터 생성
        metadata = {
            'age_group': age_group,
            'difficulty': difficulty,
            'generation': generation_data['description'],
            'years': generation_data['years'],
            'timestamp': datetime.now().isoformat()
        }
        
        # 소스 정보
        sources = self._create_sources(generation_data)
        
        return {
            'content': sentences,
            'sources': sources,
            'metadata': metadata
        }
    
    def _generate_kpop_sentences(self, generation_data: Dict, difficulty: str) -> List[str]:
        """세대별 K-POP 문장 생성"""
        
        sentences = []
        templates = self.difficulty_templates[difficulty]
        
        # 템플릿 기반 문장 생성
        for _ in range(10):  # 10개 문장 생성 (조절 가능)
            template = random.choice(templates['patterns'])
            
            # 데이터 선택
            artist = random.choice(generation_data['artists'])
            topic = random.choice(generation_data['topics'])
            keyword = random.choice(generation_data['keywords'])
            year = random.randint(*map(int, generation_data['years'].split('-')))
            description = generation_data['description']
            
            # 템플릿 채우기
            sentence = template.format(
                artist=artist,
                topic=topic,
                keyword=keyword,
                year=year,
                description=description
            )
            
            sentences.append(sentence)
        
        # LLM을 통한 추가 문장 생성
        llm_sentences = self._generate_with_llm(generation_data, difficulty)
        sentences.extend(llm_sentences)
        
        return sentences[:12]  # 최대 12개 반환
    
    def _generate_with_llm(self, generation_data: Dict, difficulty: str) -> List[str]:
        """LLM을 사용한 세대별 문장 생성"""
        
        artists_sample = random.sample(generation_data['artists'], min(3, len(generation_data['artists'])))
        topics_sample = random.sample(generation_data['topics'], min(3, len(generation_data['topics'])))
        
        prompt = f"""
한국어 학습용 K-POP 문장을 만들어주세요.

대상: {generation_data['description']}
아티스트: {', '.join(artists_sample)}
주요 토픽: {', '.join(topics_sample)}
활동 시기: {generation_data['years']}
난이도: {difficulty}

요구사항:
- 실제 K-POP 팬덤 문화를 반영
- 해당 세대가 공감할 수 있는 내용
- {self.difficulty_templates[difficulty]['max_length']}자 이내
- 3개의 문장 생성

형식:
1. [문장]
2. [문장]  
3. [문장]
"""
        
        response = self.generate_response(prompt, max_new_tokens=200)
        
        # 응답 파싱
        sentences = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                sentence = line.lstrip('0123456789.-) ').strip()
                if sentence:
                    sentences.append(sentence)
        
        return sentences
    
    def _create_sources(self, generation_data: Dict) -> List[Dict]:
        """소스 정보 생성"""
        
        sources = []
        
        # 주요 아티스트 정보
        for artist in generation_data['artists'][:3]:
            sources.append({
                'title': f'{artist} 프로필',
                'source': 'K-POP Database',
                'type': 'artist_info',
                'url': f'https://kpop.example.com/{artist}'
            })
        
        # 시대별 정보
        sources.append({
            'title': f"{generation_data['years']} K-POP 역사",
            'source': 'K-POP History',
            'type': 'historical',
            'description': generation_data['description']
        })
        
        return sources