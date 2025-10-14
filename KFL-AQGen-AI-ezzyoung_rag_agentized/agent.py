# =====================================
# agent.py (패치: 모든 StructuredTool에 description 보장)
# =====================================
import json
from textwrap import dedent
from typing import List, Dict
from functools import partial
# === agent.py (상단 import 일부) ===
from typing import List, Union, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool  # 권장: core에서 가져오기
from langchain.tools.render import render_text_description
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict  # pydantic v2

from tools import (
    DifficultyInput, VocabularyInput, GrammarInput, GeneratorInput,
    detect_difficulty_func, retrieve_vocabulary_func,
    retrieve_grammar_func, korean_sentence_generator_func
)

# 각 Input 모델이 추가 키를 허용하지 않도록(필수)
# 만약 tools.py의 Input들이 이미 extra="forbid"라면 이 단계는 생략 가능.

# ---------------------------
# 1) 빈 오브젝트 (추가 키 금지)
# ---------------------------
class EmptyObj(BaseModel):
    model_config = ConfigDict(extra="forbid")

# -----------------------------------------
# 2) "플래닝 전용" Args (모두 문자열로 설계)
#    - 플레이스홀더([step_0_output])를 허용하려면 str 이어야 함
# -----------------------------------------
class PlanDetectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_query: str

class PlanVocabArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    level: str

class PlanGrammarArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    level: str
    keyword: Optional[str] = None

class PlanGeneratorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    difficulty_level: str
    vocabulary_docs: str   # ← 플레이스홀더를 문자열로 받는다
    grammar_docs: str      # ← 플레이스홀더를 문자열로 받는다

# -----------------------------------------
# 3) 도구별 Step (플래닝 전용) — Discriminated Union
# -----------------------------------------
class DetectStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["detect_difficulty"]
    args: PlanDetectArgs

class VocabStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["retrieve_vocabulary"]
    args: PlanVocabArgs

class GrammarStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["retrieve_grammar"]
    args: PlanGrammarArgs

class GeneratorStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["korean_sentence_generator"]
    args: PlanGeneratorArgs

class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["detect_difficulty", "retrieve_vocabulary", "retrieve_grammar", "korean_sentence_generator"]
    # 도구 인자들은 JSON 문자열로 받는다 (placeholder 허용)
    args_json: str = Field(
        description="이 단계에서 호출할 도구의 인자들을 담은 JSON 문자열. 예: '{\"user_query\":\"...\"}'"
    )

# -----------------------------------------
# 4) 최상위 Plan — metadata를 "빈 오브젝트"로 명시
# -----------------------------------------
class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: List[PlanStep]
    # OpenAI 엄격 스키마가 요구하는 최상단 metadata (빈 오브젝트, 추가 키 금지)
    metadata: EmptyObj = Field(default_factory=EmptyObj, description="Reserved; leave empty.")



def _safe_desc(fn, fallback: str) -> str:
    """함수 docstring 없을 때도 항상 description 이 생기도록 보조 함수"""
    doc = (fn.__doc__ or "").strip()
    return doc if doc else fallback

class Agent:
    def __init__(self, llm, vocab_retriever, grammar_retriever):
        self.llm = llm

        # 1) 난이도 판별 도구 (의존성 없음) — 반드시 description 제공
        detect_tool = StructuredTool.from_function(
            func=detect_difficulty_func,
            name="detect_difficulty",
            args_schema=DifficultyInput,
            description=_safe_desc(
                detect_difficulty_func,
                "사용자 요청에서 목표 난이도(TOPIK 1–6)를 추정합니다."
            ),
        )

        # 2) 어휘/문법/문장생성 — partial로 의존성 주입 + description 보장
        vocab_tool = StructuredTool.from_function(
            func=partial(retrieve_vocabulary_func, retriever=vocab_retriever),
            name="retrieve_vocabulary",
            args_schema=VocabularyInput,
            description=_safe_desc(
                retrieve_vocabulary_func,
                "질의와 난이도에 맞는 어휘 문서를 검색합니다."
            ),
        )

        grammar_tool = StructuredTool.from_function(
            func=partial(retrieve_grammar_func, retriever=grammar_retriever),
            name="retrieve_grammar",
            args_schema=GrammarInput,
            description=_safe_desc(
                retrieve_grammar_func,
                "질의와 난이도에 맞는 문법 문서를 검색합니다."
            ),
        )

        generator_tool = StructuredTool.from_function(
            func=partial(korean_sentence_generator_func, llm=self.llm),
            name="korean_sentence_generator",
            args_schema=GeneratorInput,
            description=_safe_desc(
                korean_sentence_generator_func,
                "난이도/어휘/문법 자료를 바탕으로 교육용 한국어 문장을 생성합니다."
            ),
        )

        self.tools = [detect_tool, vocab_tool, grammar_tool, generator_tool]
        # ==========================================================

    def planner_node(self, state: dict) -> dict:
        # (이하 함수들은 수정할 필요 없습니다)
        print("--- 1. 계획 수립 (Planner) ---")
        rendered_tools = render_text_description(self.tools)
        system_prompt_template = dedent("""
            # 역할
            당신은 한국어 교육 콘텐츠 생성을 위한 전문가 '플래너' 에이전트입니다.
            ...

            # 출력 형식 (반드시 아래 JSON 스키마를 따를 것)
            {{ 
                "steps": [
                    {{
                    "tool": "detect_difficulty",
                    "args_json": "{{\"user_query\":\"사용자 요청 원문\"}}"
                    }},
                    {{
                    "tool": "retrieve_vocabulary",
                    "args_json": "{{\"query\":\"관련 어휘 검색어\",\"level\":\"[step_0_output]\"}}"
                    }},
                    {{
                    "tool": "retrieve_grammar",
                    "args_json": "{{\"query\":\"요청에 맞는 문법\",\"level\":\"[step_0_output]\",\"keyword\":null}}"
                    }},
                    {{
                    "tool": "korean_sentence_generator",
                    "args_json": "{{\"difficulty_level\":\"[step_0_output]\",\"vocabulary_docs\":\"[step_1_output]\",\"grammar_docs\":\"[step_2_output]\"}}"
                    }}
                ],
                "metadata": {{}}
                }}
                            """)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("user", "{input_text}"),
        ])
        structured_llm = self.llm.with_structured_output(Plan, method="json_schema")

        chain = prompt | structured_llm  
        response_plan_object = chain.invoke({"input_text": state["input_text"], "tools": rendered_tools})
        plan_steps_as_dicts = [step.model_dump() for step in response_plan_object.steps]
        return {"plan": plan_steps_as_dicts, "current_step": 0}


    def _subst_placeholders(self, obj, tool_outputs):
        """dict/list/string 재귀적으로 [step_n_output] 치환"""
        if isinstance(obj, dict):
            return {k: self._subst_placeholders(v, tool_outputs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._subst_placeholders(v, tool_outputs) for v in obj]
        elif isinstance(obj, str) and obj.startswith("[step_") and obj.endswith("_output]"):
            idx = int(obj.split("_")[1])
            return tool_outputs[idx]["output"]
        else:
            return obj

    def tool_executor_node(self, state: dict) -> dict:
        print(f"--- 2. 도구 실행 (Step {state['current_step']}) ---")
        current = state["plan"][state["current_step"]]
        tool_name = current["tool"]
        args_json = current["args_json"]

        try:
            raw_args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError as e:
            raise ValueError(f"플래너가 잘못된 args_json을 반환했습니다: {args_json}\n{e}")

        # 기존
        # tool_args = _subst_placeholders(raw_args, state.get("tool_outputs", []))

        # 플레이스홀더 치환 (재귀)
        tool_args = self._subst_placeholders(raw_args, state.get("tool_outputs", []))


        target_tool = next((t for t in self.tools if t.name == tool_name), None)
        if not target_tool:
            raise ValueError(f"도구를 찾을 수 없습니다: {tool_name}")

        output = target_tool.invoke(tool_args)  # 각 도구가 Pydantic 스키마로 검증
        print(f"   - 도구: {tool_name}")
        print(f"   - 결과 (일부): {str(output)[:200]}...")

        new_tool_outputs = state.get("tool_outputs", [])
        new_tool_outputs.append({"tool": tool_name, "output": output})
        return {"tool_outputs": new_tool_outputs}


    def final_output_node(self, state: dict) -> dict:
        print("--- 3. 최종 결과 포맷팅 ---")
        generator_output_json = ""
        for out in reversed(state["tool_outputs"]):
            if out["tool"] == "korean_sentence_generator":
                generator_output_json = out["output"]
                break
        if not generator_output_json:
            return {"final_output": "문장 생성에 실패했습니다."}
        try:
            data = json.loads(generator_output_json)
            sentences = data.get("sentences", [])
            output = "=== 한국어 학습 문제 생성 결과 ===\n\n"
            for s in sentences:
                output += f" 역할: {s.get('role', '')}\n"
                output += f"  - 문장: {s.get('sentence', '')}\n"
                output += f"  - 교육 의도: {s.get('pedagogical_rationale', '')}\n\n"
            return {"final_output": output.strip()}
        except json.JSONDecodeError:
            return {"final_output": f"생성된 결과가 유효한 JSON이 아닙니다:\n{generator_output_json}"}


