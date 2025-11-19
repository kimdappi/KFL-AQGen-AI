"""
KFL-AQGen-AI Agentic RAG 에이전트
쿼리 분석 및 품질 검증 기능 제공
K-pop 그룹 필터링 지원 추가
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
import json


class QueryAnalysisAgent:
    """
    쿼리 분석 에이전트
    사용자 쿼리를 분석하여 난이도, 주제, 검색 필요성, K-pop 그룹 파악
    임베딩 기반 그룹명 자동 매칭 지원
    """
    
    def __init__(self, llm=None, kpop_retriever=None):
        self.llm = llm or ChatOpenAI(model="gpt-5", temperature=0)
        self.kpop_retriever = kpop_retriever  # 임베딩 기반 매칭을 위해 필요
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and return structured analysis
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with:
            - difficulty: basic/intermediate/advanced
            - topic: Main topic
            - needs_kpop: Whether K-pop content is relevant (true/false)
            - kpop_groups: List of specific K-pop groups mentioned 
        """
        # DB에서 실제 그룹명 리스트 가져오기 (프롬프트 개선용)
        available_groups_list = []
        if self.kpop_retriever and hasattr(self.kpop_retriever, 'kpop_data'):
            available_groups_list = [d.metadata.get('group', '') for d in self.kpop_retriever.kpop_data if d.metadata.get('group')]
        
        available_groups_str = ', '.join(available_groups_list) if available_groups_list else "BLACKPINK, BTS, TWICE, NewJeans, EXO, Stray Kids, aespa, SEVENTEEN, IVE, Red Velvet, LE SSERAFIM"
        
        prompt = f"""Analyze the following Korean language learning query and dynamically extract ALL relevant K-pop filter conditions. Respond in JSON format:

Query: "{query}"

Available K-pop groups in database: {available_groups_str}
(Always use the exact group name from this list when extracting group names from the query.)

Available K-pop metadata fields for filtering:
- group: Group name (e.g., "BLACKPINK", "BTS", "TWICE")
- members: Member names (e.g., "Jisoo", "RM", "Nayeon")
- member_roles: Member roles (e.g., "rapper", "vocal", "dancer", "leader")
- agency: Agency name (e.g., "YG Entertainment", "BIGHIT MUSIC")
- fandom: Fandom name (e.g., "BLINK", "ARMY", "ONCE")
- concepts: Concepts (e.g., "girl crush", "hip-hop", "self-love", "youth")
- debut_year: Debut year extracted from member debut dates (e.g., 2013, 2016)
- group_type: "girl_group" or "boy_group" (infer from group name if not explicit)

Analysis items:
1. difficulty: basic/intermediate/advanced (check for keywords: basic, middle, intermediate, advanced, beginner, 초급, 중급, 고급)
   - Korean keywords: "초급" or "기초" → "basic", "중급" → "intermediate", "고급" or "상급" → "advanced"
   - If no difficulty keywords are found, default to "basic"
2. topic: Main topic (e.g., restaurant, travel, school, K-pop, daily life)
3. needs_kpop: true if query mentions K-pop, Korean music, K-pop artists, concepts, agencies, fandoms, members, or group types
4. kpop_filters: Dynamic filter conditions object - extract ALL mentioned conditions from the query

**IMPORTANT for needs_kpop**:
- Set to true if query contains: k-pop, kpop, K-pop, any group name, member name, agency, fandom, concept, or group type
- Set to true if query mentions Korean music, Korean songs, or Korean idols
- Set to false for general Korean learning queries without K-pop references

**IMPORTANT for kpop_filters**:
Extract ALL filter conditions mentioned in the query. Only include fields that are explicitly mentioned or clearly implied.

- groups: List of group names (e.g., ["BLACKPINK"], ["BTS", "TWICE"], or [])
  * Use EXACT group names from the database list above
  * If query mentions a group in Korean (e.g., "아이브", "블랙핑크"), extract it and it will be automatically matched to the standard English name
  * Examples: "아이브" → will be matched to "IVE", "블랙핑크" → will be matched to "BLACKPINK"
  
- members: List of member names mentioned (e.g., ["Jisoo", "RM"], or [])
  * Extract individual member names: "지수", "Jisoo", "RM", "제니" → ["Jisoo", "Jennie"]
  
- member_roles: List of member roles mentioned (e.g., ["rapper", "vocal"], or [])
  * Extract if mentioned: "래퍼", "rapper", "보컬", "vocal", "댄서", "dancer", "리더", "leader"
  * Korean → English: "래퍼" → "rapper", "보컬" → "vocal", "댄서" → "dancer", "리더" → "leader"
  * Examples: "걸그룹 래퍼들" → ["rapper"], "보컬 멤버들" → ["vocal"]
  
- agencies: List of agency names (e.g., ["YG Entertainment"], or [])
  * Extract if mentioned: "YG", "SM", "JYP", "BIGHIT" etc.
  
- fandoms: List of fandom names (e.g., ["BLINK", "ARMY"], or [])
  * Extract if mentioned: "BLINK", "ARMY", "ONCE" etc.
  
- concepts: List of concepts (e.g., ["girl crush"], ["hip-hop"], or [])
  * Common: "girl crush", "hip-hop", "confidence", "self-love", "youth", "storytelling", "bright", "cute", "energetic"
  * Korean → English: "걸크러시" → "girl crush", "힙합" → "hip-hop"
  
- debut_year: Year number or null
  * Extract from: "2013년 데뷔" → 2013, "debut in 2016" → 2016, "2013년" → 2013
  
- group_type: "girl_group", "boy_group", or null
  * Infer from context: "걸그룹" → "girl_group", "보이그룹" → "boy_group"
  * Or infer from group names: BLACKPINK, TWICE, Red Velvet, IVE, NewJeans, LE SSERAFIM → "girl_group"
  * BTS, EXO, SEVENTEEN, Stray Kids → "boy_group"

**Examples**:
- "지수 관련 문제" → {{"members": ["Jisoo"]}}
- "걸그룹 문제" → {{"group_type": "girl_group"}}
- "YG 소속사 그룹" → {{"agencies": ["YG Entertainment"]}}
- "BLINK 팬덤" → {{"fandoms": ["BLINK"]}}
- "걸크러시 컨셉의 2016년 데뷔 그룹" → {{"concepts": ["girl crush"], "debut_year": 2016}}
- "블랙핑크의 지수와 제니" → {{"groups": ["BLACKPINK"], "members": ["Jisoo", "Jennie"]}}
- "걸그룹 래퍼들" → {{"group_type": "girl_group", "member_roles": ["rapper"]}}
- "보컬 멤버들" → {{"member_roles": ["vocal"]}}

JSON format:
{{
  "difficulty": "basic/intermediate/advanced",
  "topic": "topic name",
  "needs_kpop": true/false,
  "kpop_filters": {{
    "groups": ["GROUP1", "GROUP2"] or [],
    "members": ["MEMBER1", "MEMBER2"] or [],
    "member_roles": ["rapper", "vocal", "dancer", "leader"] or [],
    "agencies": ["AGENCY1"] or [],
    "fandoms": ["FANDOM1"] or [],
    "concepts": ["concept1", "concept2"] or [],
    "debut_year": 2013 or null,
    "group_type": "girl_group" or "boy_group" or null
  }}
}}

Respond ONLY with valid JSON, no additional text.
"""
        
        response = self.llm.predict(prompt)

        try:
            result = json.loads(response)
            # Ensure keys
            if 'kpop_filters' not in result:
                result['kpop_filters'] = {}
            
            filters = result.get('kpop_filters', {})
            
            # 그룹명 표준화 - 임베딩 기반 자동 매칭
            if 'groups' not in filters:
                filters['groups'] = []
            
            normalized_groups = set()
            
            # 1. LLM이 추출한 그룹명들을 임베딩 기반으로 표준화
            extracted_groups = filters.get('groups', [])
            for g in extracted_groups:
                name = g.strip()
                if not name:
                    continue
                
                # 임베딩 기반 매칭 시도
                standardized = self._normalize_group_name(name)
                if standardized:
                    normalized_groups.add(standardized)
                else:
                    # 매칭 실패 시 원본 사용 (LLM이 이미 표준 이름으로 추출했을 수 있음)
                    normalized_groups.add(name)
            
            # 2. 쿼리 전체에서 그룹명 직접 감지 (임베딩 기반)
            if self.kpop_retriever:
                query_groups = self._extract_groups_from_query(query)
                normalized_groups.update(query_groups)
            
            filters['groups'] = list(normalized_groups)
            
            # 멤버 이름 표준화 (K-pop 데이터에서 동적 추출)
            if 'members' not in filters:
                filters['members'] = []
            normalized_members = []
            if self.kpop_retriever and hasattr(self.kpop_retriever, 'kpop_data'):
                # K-pop 데이터에서 모든 멤버 이름 추출
                all_member_names = set()
                for doc in self.kpop_retriever.kpop_data:
                    member_names = doc.metadata.get('member_names', [])
                    all_member_names.update([name.lower() for name in member_names if name])
                
                # 쿼리에서 추출한 멤버명을 실제 데이터와 매칭
                for m in filters.get('members', []):
                    m_lower = m.lower().strip()
                    # 정확히 일치하는 경우
                    if m_lower in all_member_names:
                        # 원본 대소문자 찾기
                        for doc in self.kpop_retriever.kpop_data:
                            member_names = doc.metadata.get('member_names', [])
                            for name in member_names:
                                if name.lower() == m_lower:
                                    normalized_members.append(name)
                                    break
                            if normalized_members and normalized_members[-1].lower() == m_lower:
                                break
                    else:
                        # 일치하지 않으면 원본 사용 (LLM이 이미 표준 이름으로 추출했을 수 있음)
                        normalized_members.append(m)
            else:
                # kpop_retriever가 없으면 원본 사용
                normalized_members = filters.get('members', [])
            filters['members'] = list(set(normalized_members))
            
            # 소속사 표준화 (K-pop 데이터에서 동적 추출)
            if 'agencies' not in filters:
                filters['agencies'] = []
            normalized_agencies = []
            if self.kpop_retriever and hasattr(self.kpop_retriever, 'kpop_data'):
                # K-pop 데이터에서 모든 소속사 추출
                all_agencies = set()
                for doc in self.kpop_retriever.kpop_data:
                    agency = doc.metadata.get('agency', '')
                    if agency:
                        all_agencies.add(agency.lower())
                
                # 쿼리에서 추출한 소속사를 실제 데이터와 매칭
                for a in filters.get('agencies', []):
                    a_lower = a.lower().strip()
                    # 부분 일치 검색
                    matched = False
                    for agency_lower in all_agencies:
                        if a_lower in agency_lower or agency_lower in a_lower:
                            # 원본 대소문자 찾기
                            for doc in self.kpop_retriever.kpop_data:
                                agency = doc.metadata.get('agency', '')
                                if agency and agency.lower() == agency_lower:
                                    normalized_agencies.append(agency)
                                    matched = True
                                    break
                            if matched:
                                break
                    if not matched:
                        # 일치하지 않으면 원본 사용
                        normalized_agencies.append(a)
            else:
                # kpop_retriever가 없으면 원본 사용
                normalized_agencies = filters.get('agencies', [])
            filters['agencies'] = list(set(normalized_agencies))
            
            # 팬덤 표준화
            if 'fandoms' not in filters:
                filters['fandoms'] = []
            filters['fandoms'] = [f.strip() for f in filters.get('fandoms', []) if f.strip()]
            
            # 컨셉 표준화 (K-pop 데이터에서 동적 추출)
            if 'concepts' not in filters:
                filters['concepts'] = []
            normalized_concepts = []
            if self.kpop_retriever and hasattr(self.kpop_retriever, 'kpop_data'):
                # K-pop 데이터에서 모든 컨셉 추출
                all_concepts = set()
                for doc in self.kpop_retriever.kpop_data:
                    concepts = doc.metadata.get('concepts', [])
                    if isinstance(concepts, list):
                        all_concepts.update([c.lower() for c in concepts if isinstance(c, str)])
                
                # 쿼리에서 추출한 컨셉을 실제 데이터와 매칭
                for concept in filters.get('concepts', []):
                    concept_lower = concept.lower().strip()
                    # 정확히 일치하는 경우
                    if concept_lower in all_concepts:
                        # 원본 대소문자 찾기
                        for doc in self.kpop_retriever.kpop_data:
                            concepts = doc.metadata.get('concepts', [])
                            if isinstance(concepts, list):
                                for c in concepts:
                                    if isinstance(c, str) and c.lower() == concept_lower:
                                        normalized_concepts.append(c)
                                        break
                            if normalized_concepts and normalized_concepts[-1].lower() == concept_lower:
                                break
                    else:
                        # 일치하지 않으면 원본 사용 (LLM이 이미 표준 이름으로 추출했을 수 있음)
                        normalized_concepts.append(concept)
            else:
                # kpop_retriever가 없으면 원본 사용
                normalized_concepts = filters.get('concepts', [])
            filters['concepts'] = list(set(normalized_concepts))
            
            # 데뷔 연도 정수 변환
            if 'debut_year' not in filters:
                filters['debut_year'] = None
            if filters.get('debut_year'):
                try:
                    filters['debut_year'] = int(filters['debut_year'])
                except (ValueError, TypeError):
                    filters['debut_year'] = None
            
            # 그룹 타입 표준화
            if 'group_type' not in filters:
                filters['group_type'] = None
            if filters.get('group_type'):
                gt = filters['group_type'].lower().strip()
                if gt in ['girl_group', 'girl group', '걸그룹']:
                    filters['group_type'] = 'girl_group'
                elif gt in ['boy_group', 'boy group', '보이그룹']:
                    filters['group_type'] = 'boy_group'
                else:
                    filters['group_type'] = None
            
            result['kpop_filters'] = filters
            return result
        except json.JSONDecodeError:
            # Default fallback
            return {
                "difficulty": "basic",
                "topic": "general",
                "needs_kpop": False,
                "kpop_filters": {
                    "groups": [],
                    "members": [],
                    "agencies": [],
                    "fandoms": [],
                    "concepts": [],
                    "debut_year": None,
                    "group_type": None
                }
            }
    
    def _normalize_group_name(self, group_name: str) -> str | None:
        """
        임베딩 기반으로 그룹명을 표준화
        kpop_retriever의 _match_groups_by_query() 활용
        한글 그룹명도 더 잘 인식하도록 개선
        """
        if not self.kpop_retriever or not group_name:
            return None
        
        try:
            # retriever의 임베딩 매칭 메서드 사용
            ranked = self.kpop_retriever._match_groups_by_query(group_name)
            if ranked and len(ranked) > 0:
                # 상위 1개 중 임계치 이상이면 반환 (상위 1위는 0.60으로 낮춤)
                best_match, score = ranked[0]
                threshold = 0.60  # 한글 그룹명 인식을 위해 임계치 낮춤
                if score >= threshold:
                    return best_match
        except Exception as e:
            # 임베딩 매칭 실패 시 None 반환 (원본 사용)
            pass
        
        return None
    
    def _extract_groups_from_query(self, query: str) -> list[str]:
        """
        쿼리 전체에서 그룹명을 임베딩 기반으로 추출
        한글 그룹명도 더 잘 인식하도록 개선
        """
        if not self.kpop_retriever:
            return []
        
        try:
            # 쿼리 전체를 임베딩 매칭에 사용
            ranked = self.kpop_retriever._match_groups_by_query(query)
            extracted = []
            threshold = self.kpop_retriever.group_match_threshold if self.kpop_retriever else 0.75
            
            # 상위 3개까지 확인
            for i, (name, score) in enumerate(ranked[:3]):
                # 상위 1위는 임계치를 0.60으로 낮춰서 더 관대하게 매칭
                if i == 0 and score >= 0.60:
                    extracted.append(name)
                elif score >= threshold:
                    extracted.append(name)
            
            return extracted
        except Exception:
            return []


class QualityCheckAgent:
    """
    품질 검증 에이전트
    검색 결과의 충분성 검증
    """
    
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-5", temperature=0)
    
    def check(
        self, 
        vocab_count: int,
        grammar_count: int,
        kpop_db_count: int,
        needs_kpop: bool = False,
        kpop_only: bool = False
    ) -> Dict[str, Any]:
        """
        검색 결과가 충분한지 확인
        
        Args:
            vocab_count: 검색된 어휘 항목 수
            grammar_count: 검색된 문법 항목 수
            kpop_db_count: 데이터베이스에서 검색된 K-pop 항목 수
            needs_kpop: K-pop 검색이 필요한지 여부
            kpop_only: K-pop 전용 쿼리인지 여부
            
        Returns:
            Dictionary with:
            - sufficient: 결과가 충분한지 여부 (Boolean)
            - vocab_count, grammar_count, kpop_db_count: 각 항목 수
            - total_kpop: 총 K-pop 항목 수
            - message: 상태 메시지
        """
        
        # 기본 최소 요구사항
        basic_sufficient = (
            vocab_count >= 5 and
            grammar_count >= 1
        )
        
        # K-pop이 필요한 경우 추가 체크
        if needs_kpop:
            sufficient = basic_sufficient and kpop_db_count >= 3
        else:
            sufficient = basic_sufficient
        
        return {
            "sufficient": sufficient,
            "vocab_count": vocab_count,
            "grammar_count": grammar_count,
            "kpop_db_count": kpop_db_count,
            "total_kpop": kpop_db_count,
            "needs_kpop": needs_kpop,
            "message": "충분함" if sufficient else "추가 검색 필요"
        }