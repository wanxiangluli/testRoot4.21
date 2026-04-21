# backend/agents/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from backend.agents.state import AgentState
from backend.agents.nodes.planner import plan_node
from backend.agents.nodes.executor import execute_node
from backend.agents.nodes.reflector import reflect_node
from loguru import logger

# === 构建状态图 ===
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_node)
workflow.add_node("reflector", reflect_node)

# 设置入口点
workflow.set_entry_point("planner")

# === 定义条件边逻辑 ===

def should_continue(state: AgentState) -> str:
    """
    判断规划节点后是否需要执行工具
    """
    last_message = state["messages"][-1]
    # 如果最后一条消息包含 tool_calls，则继续执行
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    # 否则直接结束 (直接回答)
    return "end"

def check_reflection(state: AgentState) -> str:
    """
    判断反思节点后是结束还是重新规划
    """
    # 如果已经有最终答案，则结束
    if state.get("final_answer"):
        return "end"
    # 否则返回规划节点重新思考
    return "retry"

# 添加边
workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "continue": "executor",
        "end": END
    }
)

workflow.add_edge("executor", "reflector")

workflow.add_conditional_edges(
    "reflector",
    check_reflection,
    {
        "retry": "planner",
        "end": END
    }
)

# === 编译图 (带记忆检查点) ===
# MemorySaver 用于在内存中保存会话状态，实现多轮对话记忆
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

logger.info("LangGraph Agent 状态图编译完成")