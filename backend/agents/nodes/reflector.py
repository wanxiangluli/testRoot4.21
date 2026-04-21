# backend/agents/nodes/reflector.py
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from backend.core.config import settings
from backend.core.prompts import Prompts
from backend.agents.state import AgentState
from loguru import logger

# 初始化 LLM (不需要绑定工具)
llm = ChatOpenAI(
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
    model=settings.LLM_MODEL_NAME,
    temperature=0
)

def reflect_node(state: AgentState) -> dict:
    """
    反思节点：评估执行结果，决定是继续还是结束
    """
    logger.info("--- 进入反思节点 ---")
    
    # 构建评估 Prompt
    prompt = Prompts.REFLECTOR_PROMPT.format(
        user_input=state["user_input"],
        tool_name=state["messages"][-2].tool_calls[0]["name"] if hasattr(state["messages"][-2], "tool_calls") else "N/A",
        tool_args=state["messages"][-2].tool_calls[0]["args"] if hasattr(state["messages"][-2], "tool_calls") else {},
        tool_output=state["messages"][-1].content
    )
    
    # 调用 LLM 进行评估
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content
    
    logger.info(f"反思决策: {decision}")
    
    # 简单的逻辑：如果包含 FINISH 或迭代次数超过限制，则结束
    if "FINISH" in decision or state["iterations"] >= 5:
        # 生成最终答案
        # 这里我们可以让 LLM 基于历史生成最终回答
        final_response = llm.invoke(state["messages"])
        return {
            "final_answer": final_response.content,
            "messages": [final_response] # 添加最终回答到历史
        }
    else:
        # 如果需要改进，可以将反思结果加入上下文，重新规划
        # 这里我们返回一个指导性消息，让 Planner 重新思考
        feedback_msg = HumanMessage(content=f"反思反馈：{decision}。请根据此反馈调整你的策略。")
        return {"messages": [feedback_msg]}