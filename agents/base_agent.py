# agents/base.py
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class UserProfile:
    interests: List[str]          # ["sports","k-drama", ...]
    nationality: str              # "JP","KR","US", ...
    age: int
    gender: str                   # "F","M","N"
    level: str                    # "beginner"|"intermediate"|"advanced"

@dataclass
class AgentResult:
    outputs: List[str]
    meta: Dict[str, Any]

class BaseAgent:
    def __init__(self, llm: Optional[callable] = None):
        self.llm = llm  # llm(prompt)->str

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return f"[MOCK LLM]\n{prompt[:200]}"
        return self.llm(prompt)
