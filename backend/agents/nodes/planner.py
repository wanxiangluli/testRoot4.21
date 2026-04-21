# backend/agents/nodes/planner.py
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from backend.core.config import settings
from backend.core.prompts import Prompts
from backend.agents.state import AgentState
from backend.tools.rag_tool import hybrid_rag_tool
from backend.tools.db_tool import db_tool
from backend.tools.chart_tool import chart_tool
from loguru import logger

# 初始化大模型 (绑定工具)
llm = ChatOpenAI(
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
    model=settings.LLM_MODEL_NAME,
    temperature=0.1
)

# 定义工具列表 (符合 OpenAI Function Calling 格式)
tools = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_search",
            "description": "查询企业知识库，获取公司制度、产品手册、技术文档等非结构化信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "查询业务数据库，获取销售额、订单量、用户数等结构化数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {"type": "string", "description": "自然语言查询描述"}
                },
                "required": ["query_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "根据数据生成图表配置。必须在 database_query 之后调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "数据列表"},
                    "chart_type": {"type": "string", "description": "图表类型"},
                    "title": {"type": "string", "description": "图表标题"}
                },
                "required": ["data"]
            }
        }
    }
]

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)

def plan_node(state: AgentState) -> dict:
    """
    规划节点：决定下一步行动
    """
    logger.info(f"--- 进入规划节点 (Iteration: {state['iterations']}) ---")
    
    # 构建消息历史
    messages = state["messages"]
    
    # 如果是第一次进入，添加系统提示
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        messages.insert(0, SystemMessage(content=Prompts.SYSTEM_PROMPT))
    
    # 调用 LLM 进行推理
    response = llm_with_tools.invoke(messages)
    
    # 返回需要更新的状态
    return {
        "messages": [response], 
        "iterations": state["iterations"] + 1
    }