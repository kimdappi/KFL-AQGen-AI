# =====================================
# graph.py (Updated for Agentic RAG)
# =====================================
"""
Agentic RAG 워크플로우 정의
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema import AgentState
from agent import Agent
from config import LLM_CONFIG
from langchain_openai import ChatOpenAI



class AgenticKoreanLearningGraph:
    def __init__(self, vocabulary_retriever, grammar_retriever):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=LLM_CONFIG.get('temperature', 0.1), # 계획 수립의 일관성을 위해 온도를 낮춤
            max_tokens=LLM_CONFIG.get('max_tokens', 2000)
        )
        self.agent = Agent(llm, vocabulary_retriever, grammar_retriever)
        self.workflow = None
        self.app = None
        self._build_graph()

    def _should_continue(self, state: AgentState) -> str:
        """계획의 다음 단계를 실행할지, 끝낼지 결정합니다."""
        if state["current_step"] < len(state["plan"]):
            return "continue"
        else:
            return "end"

    def _after_tool_execution(self, state: AgentState) -> dict:
        """도구 실행 후, 다음 단계를 위해 step 카운터를 증가시킵니다."""
        return {"current_step": state["current_step"] + 1}
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("planner", self.agent.planner_node)
        workflow.add_node("tool_executor", self.agent.tool_executor_node)
        workflow.add_node("after_tool_execution", self._after_tool_execution)
        workflow.add_node("final_output_formatter", self.agent.final_output_node)
        
        # 엣지 연결
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "tool_executor")
        
        # 조건부 엣지: 계획이 남았는지 확인
        workflow.add_conditional_edges(
            "after_tool_execution",
            self._should_continue,
            {
                "continue": "tool_executor", # 계획이 남았으면 다시 도구 실행
                "end": "final_output_formatter" # 끝났으면 최종 출력
            }
        )
        workflow.add_edge("tool_executor", "after_tool_execution")
        workflow.add_edge("final_output_formatter", END)
        
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def invoke(self, input_text: str, config=None):
        inputs = AgentState(
            input_text=input_text,
            plan=[],
            tool_outputs=[],
            current_step=0,
            generated_sentences_json="",
            final_output="",
            messages=[("user", input_text)]
        )
        result = self.app.invoke(inputs, config)
        return result['final_output']