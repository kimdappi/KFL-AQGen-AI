# agents/question_agent.py
from typing import List
from dataclasses import dataclass
import random
from .base import BaseAgent, AgentResult, UserProfile
from utils.template_store import TemplateStore

@dataclass
class QuestionConfig:
    qtype: str = "mcq"           # "mcq"|"short"|"fill"
    difficulty: str = "beginner" # 혹은 profile.level
    locale: str = "ko"

class QuestionAgent(BaseAgent):
    def __init__(self, llm=None, template_path="templates.json"):
        super().__init__(llm)
        self.store = TemplateStore(template_path)

    def _mk_mcq(self, sentence: str, choice_count:int):
        answer = sentence
        choices = [answer] + [f"오답 보기 {i+1}" for i in range(choice_count-1)]
        random.shuffle(choices)
        answer_idx = 1 + choices.index(answer)
        body = "\n".join([f"{i+1}) {c}" for i,c in enumerate(choices)])
        return body, answer_idx

    def _mk_fill(self, sentence: str):
        toks = sentence.split()
        if len(toks) > 4:
            idx = max(1, len(toks)//3)
            ans = toks[idx]
            toks[idx] = "____"
            return " ".join(toks), ans
        return sentence, "(단어)"

    def generate(self, sentences: List[str], profile: UserProfile, cfg: QuestionConfig) -> AgentResult:
        tpl = self.store.get(cfg.qtype, cfg.locale)
        items = []
        for s in sentences:
            if cfg.qtype == "mcq":
                body, ans_idx = self._mk_mcq(s, tpl.get("choice_count",4))
                text = tpl["template_text"].format(difficulty=cfg.difficulty, sentence=s, choices=body, answer_idx=ans_idx)
            elif cfg.qtype == "fill":
                masked, ans = self._mk_fill(s)
                text = tpl["template_text"].format(difficulty=cfg.difficulty, sentence=s, masked=masked, answer=ans)
            else:
                text = tpl["template_text"].format(difficulty=cfg.difficulty, sentence=s)
            items.append(text)
        return AgentResult(outputs=items, meta={"count": len(items), "qtype": cfg.qtype})
