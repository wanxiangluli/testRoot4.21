# backend/agents/state.py
from typing import TypedDict, List, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    Agent 状态定义
    """
    # 关键修改：使用 Annotated[List[BaseMessage], add_messages]
    # 这告诉 LangGraph：每次返回新的 messages 时，不要覆盖，而是追加到历史记录后面
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 用户原始输入 (仅当前轮次有效，无需追加)
    user_input: str
    
    # 当前迭代次数
    iterations: int
    
    # 最终生成的答案
    final_answer: Optional[str]
    
    # 中间步骤记录
    intermediate_steps: List[Any]