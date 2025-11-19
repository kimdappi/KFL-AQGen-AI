"""
Router-Integrated Nodes for KFL-AQGen-AI
Extends AgenticKoreanLearningNodes with intelligent routing capabilities
"""

from typing import Any
import re
from Ragsystem.schema import GraphState
from Ragsystem.nodes import AgenticKoreanLearningNodes
from Ragsystem.router import IntelligentRouter, format_routing_summary, RetrieverType
from utils import get_group_type


class RouterIntegratedNodes(AgenticKoreanLearningNodes):
    """
    Router-Integrated Nodes
    Combines all features from AgenticKoreanLearningNodes + Intelligent Routing
    """
    
    def __init__(self, vocabulary_retriever, grammar_retriever, kpop_retriever, llm=None):
        # Initialize parent class (all existing features)
        super().__init__(vocabulary_retriever, grammar_retriever, kpop_retriever, llm)
        
        # Add intelligent router
        self.router = IntelligentRouter(llm=llm)
        print("✅ [Router] IntelligentRouter initialized (DB only mode)")
    
    def routing_node(self, state: GraphState) -> GraphState:
        """
        라우팅 노드: 쿼리 분석 후 검색 전략 결정
        analyze_query_agent 노드 다음에 실행됨
        """
        print("\n" + "="*70)
        print("🔀 [라우터] 한국어 학습 자료 검색 전략 수립")
        print("="*70)
        
        # 쿼리 분석 결과 추출
        query = state.get("input_text", "")
        difficulty = state.get("difficulty_level", "intermediate")
        query_analysis = state.get("query_analysis", {})
        
        topic = query_analysis.get("topic", "")
        
        # 라우팅 결정
        decision = self.router.route(
            query=query,
            difficulty=difficulty,
            topic=topic,
            query_analysis=query_analysis
        )
        
        # 결과 출력
        print(format_routing_summary(decision))
        print("="*70)
        
        # 상태 업데이트
        return {
            "routing_decision": decision,
            "search_strategies": [s.to_dict() for s in decision.strategies]
        }
    
    def retrieve_vocabulary_routed(self, state: GraphState) -> GraphState:
        """라우터 기반 어휘 검색"""
        decision = state.get("routing_decision")
        
        # 라우팅 결정이 없으면 기본 방식 사용
        if not decision:
            print("   ⚠️ 라우팅 정보 없음, 기본 검색 실행")
            return super().retrieve_vocabulary(state)
        
        # Vocabulary 전략 찾기
        strategy = decision.get_strategy(RetrieverType.VOCABULARY)
        if not strategy:
            print("   ⏭️  어휘 검색 스킵됨 (라우터 결정)")
            return {"vocabulary_docs": []}
        
        # 전략에 따른 검색 실행
        print(f"\n📚 [어휘 검색] TOPIK 어휘 데이터베이스")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        vocab_docs = self.vocabulary_retriever.invoke(strategy.query, level)
        
        # limit 적용
        limit = strategy.params.get("limit", 10)
        vocab_docs = vocab_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(vocab_docs)}개 어휘")
        
        return {"vocabulary_docs": vocab_docs}
    
    def retrieve_grammar_routed(self, state: GraphState) -> GraphState:
        """라우터 기반 문법 검색"""
        decision = state.get("routing_decision")
        
        if not decision:
            print("   ⚠️ 라우팅 정보 없음, 기본 검색 실행")
            return super().retrieve_grammar(state)
        
        strategy = decision.get_strategy(RetrieverType.GRAMMAR)
        if not strategy:
            print("   ⏭️  문법 검색 스킵됨 (라우터 결정)")
            return {"grammar_docs": []}
        
        print(f"\n📖 [문법 검색] 한국어 문법 패턴 데이터베이스")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        grammar_docs = self.grammar_retriever.invoke(strategy.query, level)
        
        limit = strategy.params.get("limit", 5)
        grammar_docs = grammar_docs[:limit]
        
        print(f"   ✅ 검색 완료: {len(grammar_docs)}개 문법 패턴")
        
        return {"grammar_docs": grammar_docs}
    
    def retrieve_kpop_routed(self, state: GraphState) -> GraphState:
        """
        라우터 기반 K-pop 검색 (조건부 - 쿼리에 K-pop 키워드 있을 때만)
        웹 검색 없음 - 데이터베이스만 사용
        """
        decision = state.get("routing_decision")
        
        if not decision:
            print("   ⚠️ 라우팅 정보 없음")
            return {"kpop_docs": []}
        
        strategy = decision.get_strategy(RetrieverType.KPOP)
        if not strategy:
            print("   ⏭️  K-pop 검색 스킵 (쿼리에 K-pop 키워드 없음)")
            return {"kpop_docs": []}
        
        print(f"\n🎵 [K-pop 검색] 한국어 학습용 K-pop 문장 (DB 전용)")
        print(f"   검색어: '{strategy.query}'")
        print(f"   학습자 수준: {strategy.params.get('level')}")
        print(f"   재시도: {strategy.retry_count}회")
        
        level = strategy.params.get("level", state['difficulty_level'])
        
        # DB 검색만 수행
        db_limit = strategy.params.get("db_limit", 5)
        
        # 동적 필터링: kpop_filters 객체를 기반으로 모든 메타데이터 필드 필터링
        qa = state.get('query_analysis', {})
        kpop_filters = qa.get('kpop_filters', {}) if qa else {}
        
        # 필터링 조건이 있으면 적용
        has_filters = any([
            kpop_filters.get('groups'),
            kpop_filters.get('members'),
            kpop_filters.get('member_roles'),
            kpop_filters.get('agencies'),
            kpop_filters.get('fandoms'),
            kpop_filters.get('concepts'),
            kpop_filters.get('debut_year'),
            kpop_filters.get('group_type')
        ])
        
        # 필터링 조건이 있으면 그룹명을 직접 사용하여 검색 (더 정확)
        if has_filters and kpop_filters.get('groups'):
            # 그룹명이 있으면 해당 그룹 문서만 직접 가져오기
            specified_groups = [g.strip() for g in kpop_filters['groups'] if g]
            specified_groups_lower = {g.lower() for g in specified_groups}
            print(f"   🔍 필터링 조건 감지: 그룹 {specified_groups}")
            
            # 모든 K-pop 데이터에서 지정된 그룹만 필터링 (대소문자 무시)
            all_kpop_docs = self.kpop_retriever.kpop_data if hasattr(self.kpop_retriever, 'kpop_data') else []
            kpop_db_docs = []
            for doc in all_kpop_docs:
                doc_group = (doc.metadata.get('group', '') or '').strip()
                doc_group_lower = doc_group.lower()
                # 정확 일치 확인 (대소문자 무시)
                if doc_group in specified_groups or doc_group_lower in specified_groups_lower:
                    kpop_db_docs.append(doc)
            
            print(f"   ✅ 그룹 필터링 결과: {len(kpop_db_docs)}개 문서 (그룹: {specified_groups})")
        else:
            # 필터링 조건이 없거나 그룹명이 없으면 일반 검색
            kpop_db_docs = self.kpop_retriever.invoke(strategy.query, level)
        
        filtered = []
        filter_reasons = []
        
        if has_filters:
            for d in kpop_db_docs:
                match = True
                doc_reasons = []
                
                # 1. 그룹 필터링 (대소문자 무시, 공백 제거)
                if kpop_filters.get('groups'):
                    doc_group = (d.metadata.get('group', '') or '').strip()
                    doc_group_lower = doc_group.lower()
                    specified_groups = [g.strip() for g in kpop_filters['groups'] if g]
                    specified_groups_lower = {g.lower() for g in specified_groups}
                    
                    # 정확 일치 또는 부분 일치 확인
                    if doc_group_lower in specified_groups_lower or doc_group in specified_groups:
                        doc_reasons.append(f"그룹: {doc_group}")
                    else:
                        match = False
                        continue
                
                # 2. 멤버 필터링
                if match and kpop_filters.get('members'):
                    doc_member_names = [m.lower() for m in (d.metadata.get('member_names', []) or [])]
                    specified_members = [m.lower() for m in kpop_filters['members']]
                    member_match = any(sm in doc_member_names for sm in specified_members)
                    if member_match:
                        matched_members = [sm for sm in specified_members if sm in doc_member_names]
                        doc_reasons.append(f"멤버: {', '.join(matched_members)}")
                    else:
                        match = False
                        continue
                
                # 2-1. 멤버 role 필터링 (래퍼, 보컬, 댄서, 리더 등)
                if match and kpop_filters.get('member_roles'):
                    doc_members = d.metadata.get('members', [])
                    doc_roles = [m.get('role', '').lower() for m in doc_members if isinstance(m, dict) and m.get('role')]
                    specified_roles = [r.lower() for r in kpop_filters['member_roles']]
                    role_match = any(sr in doc_roles for sr in specified_roles)
                    if role_match:
                        matched_roles = [sr for sr in specified_roles if sr in doc_roles]
                        doc_reasons.append(f"역할: {', '.join(matched_roles)}")
                    else:
                        match = False
                        continue
                
                # 3. 소속사 필터링
                if match and kpop_filters.get('agencies'):
                    doc_agency = (d.metadata.get('agency', '') or '').lower()
                    specified_agencies = [a.lower() for a in kpop_filters['agencies']]
                    agency_match = any(sa in doc_agency for sa in specified_agencies)
                    if agency_match:
                        matched_agency = [sa for sa in specified_agencies if sa in doc_agency][0]
                        doc_reasons.append(f"소속사: {d.metadata.get('agency')}")
                    else:
                        match = False
                        continue
                
                # 4. 팬덤 필터링
                if match and kpop_filters.get('fandoms'):
                    doc_fandom = (d.metadata.get('fandom', '') or '').lower()
                    specified_fandoms = [f.lower() for f in kpop_filters['fandoms']]
                    fandom_match = any(sf in doc_fandom for sf in specified_fandoms)
                    if fandom_match:
                        matched_fandom = [sf for sf in specified_fandoms if sf in doc_fandom][0]
                        doc_reasons.append(f"팬덤: {d.metadata.get('fandom')}")
                    else:
                        match = False
                        continue
                
                # 5. 컨셉 필터링
                if match and kpop_filters.get('concepts'):
                    doc_concepts = [c.lower() for c in (d.metadata.get('concepts', []) or []) if isinstance(c, str)]
                    specified_concepts = [c.lower() for c in kpop_filters['concepts']]
                    concept_match = any(sc in doc_concepts for sc in specified_concepts)
                    if concept_match:
                        matched_concepts = [sc for sc in specified_concepts if sc in doc_concepts]
                        doc_reasons.append(f"컨셉: {', '.join(matched_concepts)}")
                    else:
                        match = False
                        continue
                
                # 6. 데뷔 연도 필터링
                if match and kpop_filters.get('debut_year'):
                    members = d.metadata.get('members', [])
                    doc_debut_years = set()
                    for m in members:
                        debut = m.get('debut', '')
                        if debut and len(debut) >= 4:
                            try:
                                year = int(debut[:4])
                                doc_debut_years.add(year)
                            except ValueError:
                                pass
                    
                    if kpop_filters['debut_year'] in doc_debut_years:
                        doc_reasons.append(f"데뷔: {kpop_filters['debut_year']}년")
                    else:
                        match = False
                        continue
                
                # 7. 그룹 타입 필터링 (걸그룹/보이그룹)
                if match and kpop_filters.get('group_type'):
                    group_name = d.metadata.get('group', '')
                    doc_group_type = get_group_type(group_name, self.kpop_retriever)
                    
                    if doc_group_type == kpop_filters['group_type']:
                        doc_reasons.append(f"타입: {kpop_filters['group_type']}")
                    else:
                        match = False
                        continue
                
                if match:
                    filtered.append(d)
                    if doc_reasons:
                        filter_reasons.extend(doc_reasons)
            
            if filtered:
                kpop_db_docs = filtered
                if filter_reasons:
                    print(f"   🔍 필터링 적용: {', '.join(set(filter_reasons))}")
                print(f"   ✅ 필터링 결과: {len(kpop_db_docs)}개 문서")
            else:
                # 필터링 조건에 맞는 문서가 없으면 디버깅 정보 출력
                print(f"   ⚠️ 필터링 조건에 맞는 문서를 찾을 수 없습니다.")
                print(f"   📋 필터링 조건:")
                if kpop_filters.get('groups'):
                    print(f"      - 그룹: {kpop_filters['groups']}")
                if kpop_filters.get('members'):
                    print(f"      - 멤버: {kpop_filters['members']}")
                if kpop_filters.get('member_roles'):
                    print(f"      - 역할: {kpop_filters['member_roles']}")
                if kpop_filters.get('agencies'):
                    print(f"      - 소속사: {kpop_filters['agencies']}")
                if kpop_filters.get('fandoms'):
                    print(f"      - 팬덤: {kpop_filters['fandoms']}")
                if kpop_filters.get('concepts'):
                    print(f"      - 컨셉: {kpop_filters['concepts']}")
                if kpop_filters.get('debut_year'):
                    print(f"      - 데뷔 연도: {kpop_filters['debut_year']}")
                if kpop_filters.get('group_type'):
                    print(f"      - 그룹 타입: {kpop_filters['group_type']}")
                print(f"   📊 검색된 문서 수: {len(kpop_db_docs)}개")
                if kpop_db_docs:
                    print(f"   📄 검색된 문서의 그룹: {[d.metadata.get('group', '') for d in kpop_db_docs[:5]]}")
                
                # 필터링 실패 시에도 원본 검색 결과를 반환 (최소한의 정보라도 제공)
                # 단, 그룹 필터만 있는 경우는 빈 리스트 반환 (정확도 보장)
                if kpop_filters.get('groups') and not any([
                    kpop_filters.get('members'),
                    kpop_filters.get('member_roles'),
                    kpop_filters.get('agencies'),
                    kpop_filters.get('fandoms'),
                    kpop_filters.get('concepts'),
                    kpop_filters.get('debut_year'),
                    kpop_filters.get('group_type')
                ]):
                    # 그룹 필터만 있고 매칭 실패 → 빈 리스트
                    kpop_db_docs = []
                else:
                    # 다른 필터도 있으면 원본 검색 결과 반환 (부분 매칭 허용)
                    print(f"   ⚠️ 필터링 실패했지만 원본 검색 결과 {len(kpop_db_docs)}개 반환")
                    kpop_db_docs = kpop_db_docs[:db_limit]
        else:
            # 필터링 조건이 없으면 토큰 기반 매칭
            raw_query = state.get('input_text', '')
            q_tokens = set([t.strip().lower() for t in re.split(r"[^\w가-힣]+", raw_query) if len(t.strip()) >= 2])
            for d in kpop_db_docs:
                group = (d.metadata.get('group', '') or '').lower()
                song = (d.metadata.get('song', '') or '').lower()
                member_names = [m.lower() for m in (d.metadata.get('member_names', []) or [])]
                concepts = [c.lower() for c in (d.metadata.get('concepts', []) or []) if isinstance(c, str)]
                fields = set()
                if group:
                    fields.add(group)
                if song:
                    fields.add(song)
                fields.update(member_names)
                fields.update(concepts)
                if any(tok in fields for tok in q_tokens):
                    filtered.append(d)
            if filtered:
                kpop_db_docs = filtered

        # 최종적으로 최대 5개만 반환
        kpop_db_docs = kpop_db_docs[:db_limit]
        
        # 검증: 반환되는 문서 정보 확인 및 그룹 필터 검증
        if has_filters:
            returned_groups = set()
            returned_members = set()
            returned_roles = set()
            returned_agencies = set()
            returned_fandoms = set()
            returned_concepts = set()
            returned_years = set()
            returned_types = set()
            
            for d in kpop_db_docs:
                g = d.metadata.get('group', '')
                if g:
                    returned_groups.add(g)
                
                # 멤버 role 추출
                members = d.metadata.get('members', [])
                for m in members:
                    if isinstance(m, dict):
                        role = m.get('role', '')
                        if role:
                            returned_roles.add(role.lower())
                        debut = m.get('debut', '')
                        if debut and len(debut) >= 4:
                            try:
                                returned_years.add(int(debut[:4]))
                            except ValueError:
                                pass
                
                agency = d.metadata.get('agency', '')
                if agency:
                    returned_agencies.add(agency)
                
                fandom = d.metadata.get('fandom', '')
                if fandom:
                    returned_fandoms.add(fandom)
                
                concepts = d.metadata.get('concepts', [])
                returned_concepts.update([c.lower() for c in concepts if isinstance(c, str)])
                
                # 그룹 타입 추론
                group_type = get_group_type(g, self.kpop_retriever)
                if group_type:
                    returned_types.add(group_type)
            
            # 그룹 필터가 있으면 반환된 그룹이 모두 필터 조건에 맞는지 검증
            if kpop_filters.get('groups'):
                specified_groups = [g.strip() for g in kpop_filters['groups'] if g]
                specified_groups_lower = {g.lower() for g in specified_groups}
                invalid_groups = []
                for returned_group in returned_groups:
                    returned_group_lower = returned_group.lower()
                    if returned_group not in specified_groups and returned_group_lower not in specified_groups_lower:
                        invalid_groups.append(returned_group)
                
                if invalid_groups:
                    print(f"   ⚠️ 경고: 필터 조건에 맞지 않는 그룹이 포함됨: {invalid_groups}")
                    # 필터 조건에 맞지 않는 그룹 제거
                    kpop_db_docs = [d for d in kpop_db_docs 
                                   if (d.metadata.get('group', '') in specified_groups or 
                                       d.metadata.get('group', '').lower() in specified_groups_lower)]
                    returned_groups = {g for g in returned_groups 
                                     if (g in specified_groups or g.lower() in specified_groups_lower)}
                    print(f"   ✅ 필터링 재적용: {len(kpop_db_docs)}개 문서만 반환 (그룹: {list(returned_groups)})")
            
            print(f"   ✅ DB 검색 완료: {len(kpop_db_docs)}개 K-pop 문장")
            if returned_groups:
                print(f"   📋 반환된 그룹: {list(returned_groups)}")
            if returned_members:
                print(f"   📋 반환된 멤버: {list(returned_members)[:5]}")
            if returned_roles:
                print(f"   📋 반환된 역할: {list(returned_roles)}")
            if returned_agencies:
                print(f"   📋 반환된 소속사: {list(returned_agencies)}")
            if returned_fandoms:
                print(f"   📋 반환된 팬덤: {list(returned_fandoms)}")
            if returned_concepts:
                print(f"   📋 반환된 컨셉: {list(returned_concepts)}")
            if returned_years:
                print(f"   📋 반환된 데뷔 연도: {sorted(list(returned_years))}")
            if returned_types:
                print(f"   📋 반환된 그룹 타입: {list(returned_types)}")
        else:
            print(f"   ✅ DB 검색 완료: {len(kpop_db_docs)}개 K-pop 문장")
        
        return {
            "kpop_docs": kpop_db_docs
        }

    def check_quality_agent(self, state: GraphState) -> GraphState:
        """품질 검증 에이전트 노드 - 간소화"""
        print("\n✅ [Agent] 품질 검증")
        
        query_analysis = state.get('query_analysis', {})
        needs_kpop = query_analysis.get('needs_kpop', False)
        
        # 간소화된 기준: 어휘 3개, 문법 1개, K-pop 3개
        vocab_count = len(state.get('vocabulary_docs', []))
        grammar_count = len(state.get('grammar_docs', []))
        kpop_count = len(state.get('kpop_docs', []))
        
        sufficient = (vocab_count >= 3 and grammar_count >= 1)
        if needs_kpop:
            sufficient = sufficient and (kpop_count >= 3)
        
        result = {
            "sufficient": sufficient,
            "vocab_count": vocab_count,
            "grammar_count": grammar_count,
            "kpop_db_count": kpop_count,
            "needs_kpop": needs_kpop,
            "message": "충분함" if sufficient else "추가 검색 필요"
        }
        
        print(f"   어휘: {vocab_count}개 (목표 3개)")
        print(f"   문법: {grammar_count}개 (목표 1개)")
        if needs_kpop:
            print(f"   K-pop: {kpop_count}개 (목표 3개)")
        
        return {"quality_check": result}
    
    def rerank_node(self, state: GraphState) -> GraphState:
        """재검색 노드 - 간소화"""
        print("\n🔄 [재검색] 품질 개선을 위한 재검색 (1회만)")
        
        quality_check = state.get("quality_check", {})
        current_count = state.get("rerank_count", 0)
        new_count = current_count + 1
        
        # 간단한 재검색: 어휘 5개, 문법 3개, K-pop 5개 추가 검색
        level = state.get("difficulty_level", "intermediate")
        query = state.get("input_text", "")
        
        # 어휘 재검색
        if quality_check.get("vocab_count", 0) < 3:
            print(f"   📚 어휘 재검색 (현재 {quality_check.get('vocab_count')}개)")
            vocab_docs = self.vocabulary_retriever.invoke(query, level)[:5]
            state["vocabulary_docs"] = vocab_docs
        
        # 문법 재검색
        if quality_check.get("grammar_count", 0) < 1:
            print(f"   📖 문법 재검색 (현재 {quality_check.get('grammar_count')}개)")
            grammar_docs = self.grammar_retriever.invoke(query, level)[:3]
            state["grammar_docs"] = grammar_docs
        
        # K-pop 재검색 (필요시)
        if quality_check.get("needs_kpop") and quality_check.get("kpop_db_count", 0) < 3:
            print(f"   🎵 K-pop 재검색 (현재 {quality_check.get('kpop_db_count')}개)")
            kpop_docs = self.kpop_retriever.invoke(query, level)[:5]
            state["kpop_docs"] = kpop_docs
        
        print(f"   ✅ 재검색 완료 (카운터: {new_count})")
        
        return {
            "rerank_count": new_count
        }