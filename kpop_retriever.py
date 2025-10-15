# =====================================
# kpop_retriever.py
# =====================================
"""
K-pop 문장 컨텍스트 제공자 (Context Provider)
- 난이도 구분 없이 JSON 파일 기반
- 문장 생성 시 아이돌 이름, 그룹, 콘셉트 등을 프롬프트에 반영하기 위한 데이터 제공
"""

import json
import random
from typing import Dict


class KpopContextProvider:
    """K-pop 그룹/멤버 정보를 불러와 문장 생성용 컨텍스트를 제공"""

    def __init__(self, json_path: str = "data/kpop_db.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            raise ValueError(f"❌ JSON 파일 로드 실패: {e}")

    def get_random_member_context(self) -> Dict[str, str]:
        """랜덤 멤버 기반 문장 컨텍스트 반환"""
        member = random.choice(self.data["members"])
        context = {
            "group": self.data["group"],
            "member": member["name"],
            "role": member["role"],
            "agency": self.data["agency"],
            "concept": random.choice(self.data["concepts"]),
            "fandom": self.data["fandom"],
        }
        return context

    def format_context_as_prompt(self, grammar: str = None, vocab: str = None) -> str:
        """문법/어휘 주제를 받아 자연스러운 프롬프트 문장으로 변환"""
        context = self.get_random_member_context()
        prompt = (
            f"{context['group']}의 {context['member']}는 {context['concept']} 콘셉트로 활동하는 "
            f"{context['role']}입니다. "
            f"이 정보를 반영해 문장을 만들어 보세요."
        )
        if grammar:
            prompt += f" 반드시 문법 '{grammar}'를 포함해야 합니다."
        if vocab:
            prompt += f" 또한 어휘 '{vocab}'를 사용하세요."
        return prompt
