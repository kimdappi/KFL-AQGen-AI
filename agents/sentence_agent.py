# agents/sentence_agent.py
from typing import List
from dataclasses import dataclass
import re
from .base import BaseAgent, AgentResult, UserProfile

# 난이도별 힌트(LLM 프롬프트에만 사용)
LEVEL_HINT = {
    "beginner":     {"len_note": "각 문장은 6~12어절, 기초 어휘 중심"},
    "intermediate": {"len_note": "각 문장은 10~18어절, 연결어 활용"},
    "advanced":     {"len_note": "각 문장은 15~25어절, 고급 어휘/문형"},
}

@dataclass
class SentenceGenConfig:
    n: int = 6                   # 생성 문장 개수
    locale: str = "ko"           # 출력 언어(고정: 한국어)
    retry_once_if_bad: bool = True  # 개수/형식 불일치 시 1회 재시도

class SentenceAgent(BaseAgent):
    """
    인구통계 + 관심사를 반영하여 개인화 예문을 LLM으로만 생성.
    - LLM 미주입 시 RuntimeError 발생
    - 한 줄당 정확히 한 문장, 총 N줄을 목표로 생성
    """
    def _build_prompt(self, profile: UserProfile, cfg: SentenceGenConfig) -> str:
        interests_str = ", ".join(profile.interests[:5]) if profile.interests else "일상"
        lvl = profile.level if profile.level in LEVEL_HINT else "beginner"
        note = LEVEL_HINT[lvl]["len_note"]
        return (
            "당신은 한국어 교육용 예문 작성 전문가입니다.\n"
            "아래 학습자 프로필에 맞춘 한국어 예문을 생성하세요.\n\n"
            f"[학습자 프로필]\n- 관심사: {interests_str}\n- 국적: {profile.nationality}\n"
            f"- 나이: {profile.age}\n- 성별: {profile.gender}\n- 수준: {profile.level}\n\n"
            "[요구 사항]\n"
            f"- 총 {cfg.n}개 문장을 생성하세요.\n"
            "- 출력 형식: 번호/불릿/따옴표/코드블록 없이 오직 문장만, 줄바꿈으로 구분\n"
            "- 한 줄에 딱 한 문장만 작성\n"
            f"- {note}\n"
            "- 문화적으로 중립/친절, 편견·비하·혐오 표현 금지\n"
            "- 학습자가 공감할 수 있도록 관심사 키워드를 자연스럽게 포함\n"
            "- 한국어 종결형(마침표/다/요/?)로 끝맺음\n"
        )

    def _postprocess_lines(self, raw: str) -> List[str]:
        # 코드펜스/따옴표/불릿/번호 제거 후 라인 파싱
        text = re.sub(r"^```.*?```", "", raw, flags=re.S)  # 코드블록 제거
        lines = [l.strip() for l in text.splitlines()]
        cleaned = []
        for l in lines:
            if not l:
                continue
            l = re.sub(r'^\s*[\-\*\d\.\)\]]+\s*', '', l)  # 불릿/번호 제거
            l = l.strip('“”"\'·•\t ').strip()
            if l:
                cleaned.append(l)

        # 중복 제거, 공백 라인 제거
        uniq, seen = [], set()
        for s in cleaned:
            if s not in seen:
                seen.add(s)
                uniq.append(s)

        # 종결부호/종결형 없으면 마침표 보정
        fixed = []
        for s in uniq:
            if re.search(r'[.?!…]$|[다요]$', s):
                fixed.append(s)
            else:
                fixed.append(s + ".")
        return fixed

    def generate(self, profile: UserProfile, cfg: SentenceGenConfig) -> AgentResult:
        if self.llm is None:
            raise RuntimeError("SentenceAgent에는 LLM이 필요합니다. SentenceAgent(llm=...)로 주입하세요.")

        prompt = self._build_prompt(profile, cfg)
        raw = self._call_llm(prompt)
        lines = self._postprocess_lines(raw)

        # 개수 정합성 체크 → 부족/초과 처리
        if len(lines) != cfg.n and cfg.retry_once_if_bad:
            # 더 엄격한 재요청
            strict_prompt = (
                prompt
                + "\n[중요]\n- 반드시 정확히 "
                + str(cfg.n)
                + "줄을 출력하세요.\n- 한 줄에 한 문장, 불릿/번호/설명 금지.\n"
            )
            raw2 = self._call_llm(strict_prompt)
            lines2 = self._postprocess_lines(raw2)
            if len(lines2) == cfg.n:
                lines = lines2

        # 최종 개수 맞추기(초과는 자르기, 부족은 있는 만큼 반환)
        lines = lines[: cfg.n]

        return AgentResult(outputs=lines, meta={"count": len(lines), "profile": profile.__dict__})
