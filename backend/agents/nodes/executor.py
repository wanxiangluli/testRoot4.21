# backend/agents/nodes/executor.py
import json
from langchain_core.messages import ToolMessage
from backend.agents.state import AgentState
from backend.tools.rag_tool import hybrid_rag_tool
from backend.tools.db_tool import db_tool
from backend.tools.chart_tool import chart_tool
from loguru import logger

# 工具映射表
TOOL_MAPPING = {
    "knowledge_search": hybrid_rag_tool.run,
    "database_query": db_tool.run,
    "generate_chart": chart_tool.run,
}

def execute_node(state: AgentState) -> dict:
    """
    执行节点：运行工具并返回结果
    """
    logger.info("--- 进入执行节点 ---")
    
    # 获取最后一条 AI 消息
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    # 可能有多个工具调用
    tool_calls = last_message.tool_calls
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        logger.info(f"正在调用工具: {tool_name}, 参数: {tool_args}")
        
        # 查找并执行工具
        tool_func = TOOL_MAPPING.get(tool_name)
        
        if tool_func:
            try:
                # 执行工具 (注意参数解包)
                # 这里假设工具接受 Pydantic Model 或 Dict
                # 我们的 rag_tool.run 接受 query: str, 但 tool_args 是 dict
                # 需要调整工具接口适配，这里做个简单适配
                if tool_name == "knowledge_search":
                    result = tool_func(query=tool_args["query"])
                elif tool_name == "database_query":
                    result = tool_func(input_data=tool_args) # db_tool 接受 input_data
                elif tool_name == "generate_chart":
                    result = tool_func(input_data=tool_args) # chart_tool 接受 input_data
                else:
                    result = {"error": "Unknown tool"}
                
                # 如果结果是 Pydantic 对象，转为 Dict
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                    
            except Exception as e:
                result = {"error": str(e)}
                logger.error(f"工具执行错误: {e}")
        else:
            result = {"error": "Tool not found"}
            
        # 构建 ToolMessage
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tool_id
            )
        )

    return {"messages": tool_messages}