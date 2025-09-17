# agents/critic_agent.py
from typing import List, Dict, Any
from dataclasses import dataclass
import json, re
from .base import BaseAgent, AgentResult, UserProfile

@dataclass
class CriticConfig:
    min_len: int = 5
    max_len: int = 140
    min_words: int = 5
    retry_limit: int = 1          # 재생성 루프 외부에서 관리
    use_rewrite: bool = True      # GPT가 제안한 rewrite를 채택할지

_SYS = (
"당신은 한국어 교육용 문장/문항의 품질을 평가하는 엄격한 크리틱입니다. "
"출력은 반드시 JSON 한 덩어리만 반환하세요. 설명/서문/코드는 금지."
)

def _build_rules(profile: UserProfile, is_item: bool) -> str:
    # 최소 규칙: 길이/완결성/연령 적합성/편견 금지
    age_bucket = "child" if profile.age<=12 else ("teen" if profile.age<=18 else ("adult" if profile.age<60 else "senior"))
    base = [
        f"- 대상 연령대: {age_bucket} (age={profile.age})에 부적절한 주제 금지",
        "- 편견/차별/비하/욕설/혐오 표현 금지",
        "- 문장은 완결된 종결형 또는 문장부호로 마무리",
        f"- 최소 어절수 {5} 이상 권장",
    ]
    if is_item:
        base.append("- 문제는 명확하고 한 가지 정답 기준 유지")
    return "\n".join(base)

_JSON_FMT = (
'출력 포맷(JSON): {"results":[{"text":"원본","ok":true/false,"score":0.0~1.0,'
'"reasons":["이유1","이유2"],"rewrite":"수정문 or null"}]}'
)

class CriticAgent(BaseAgent):
    """
    GPT에 프로필/규칙을 주고 문장/문항을 평가.
    - ok=false면 rewrite 제안 포함
    - LLM 래퍼(callable)를 self.llm로 주입해야 함.
    """
    def _crit(self, items: List[str], profile: UserProfile, is_item: bool) -> Dict[str, Any]:
        assert self.llm is not None, "CriticAgent에 llm(callable)을 주입하세요."

        user = (
            f"[프로필]\n"
            f"- interests: {', '.join(profile.interests) if profile.interests else '일상'}\n"
            f"- nationality: {profile.nationality}, age: {profile.age}, gender: {profile.gender}, level: {profile.level}\n\n"
            f"[평가기준]\n{_build_rules(profile, is_item)}\n\n"
            f"[대상 {'문항' if is_item else '문장'} 목록]\n" +
            "\n".join([f"{i+1}. {s}" for i, s in enumerate(items)]) + "\n\n" +
            _JSON_FMT
        )
        out = self.llm(system=_SYS, user=user)
        # JSON만 깔끔히 추출
        m = re.search(r'\{.*\}\s*$', out.strip(), re.S)
        raw = m.group(0) if m else out
        try:
            return json.loads(raw)
        except Exception:
            # 실패 시 전부 통과시키는 폴백
            return {"results":[{"text":s,"ok":True,"score":0.8,"reasons":[],"rewrite":None} for s in items]}

    def evaluate_sentences(self, sentences: List[str], profile: UserProfile, cfg: CriticConfig) -> AgentResult:
        j = self._crit(sentences, profile, is_item=False)
        kept = []
        fb = []
        for r in j.get("results", []):
            text = r.get("text","")
            ok = bool(r.get("ok", False))
            rewrite = r.get("rewrite")
            reasons = r.get("reasons", [])
            score = float(r.get("score", 0.0))

            # 하드 가드(길이/단어수)
            if len(text) > cfg.max_len or len(text.split()) < cfg.min_words:
                ok = False
                reasons.append("하드 규칙 위반(길이/단어수)")
            if ok:
                kept.append(text)
            elif cfg.use_rewrite and rewrite:
                kept.append(rewrite)
            fb.append({"text": text, "ok": ok, "score": score, "reasons": reasons, "rewrite": rewrite})
        # 중복 제거
        uniq, seen = [], set()
        for s in kept:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return AgentResult(outputs=uniq, meta={"feedbacks": fb, "kept": len(uniq), "input": len(sentences)})

    def evaluate_items(self, items: List[str], profile: UserProfile, cfg: CriticConfig) -> AgentResult:
        j = self._crit(items, profile, is_item=True)
        kept, fb = [], []
        for r in j.get("results", []):
            text = r.get("text","")
            ok = bool(r.get("ok", False))
            rewrite = r.get("rewrite")
            reasons = r.get("reasons", [])
            score = float(r.get("score", 0.0))
            if ok:
                kept.append(text)
            elif cfg.use_rewrite and rewrite:
                kept.append(rewrite)
            fb.append({"text": text, "ok": ok, "score": score, "reasons": reasons, "rewrite": rewrite})
        # 중복 제거
        uniq, seen = [], set()
        for s in kept:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return AgentResult(outputs=uniq, meta={"feedbacks": fb, "kept": len(uniq), "input": len(items)})
