# backend/tools/db_tool.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger
import random

class DBQueryInput(BaseModel):
    """数据库查询工具输入"""
    query_text: str = Field(..., description="自然语言查询描述，如'上个月销售额'")
    # 实际项目中这里应该是 SQL 或具体的查询参数

class DBTool:
    """
    模拟数据库查询工具：
    在实际项目中，这里会连接 MySQL/Postgres/ClickHouse 执行 SQL。
    目前为了演示 ToolUse，返回模拟数据。
    """
    def __init__(self):
        # 模拟数据库连接
        pass

    def query_sales_data(self, month: str = "2023-10") -> List[Dict]:
        """模拟查询销售数据"""
        # 模拟数据返回
        return [
            {"product": "企业版A", "sales": random.randint(100, 500), "month": month},
            {"product": "企业版B", "sales": random.randint(50, 300), "month": month},
            {"product": "个人版", "sales": random.randint(200, 800), "month": month},
        ]

    def run(self, input_data: DBQueryInput) -> Dict[str, Any]:
        """
        执行查询
        """
        logger.info(f"执行数据库查询: {input_data.query_text}")
        
        # 这里应该有 Text-to-SQL 的逻辑，暂时用简单的关键词匹配模拟
        if "销售" in input_data.query_text:
            data = self.query_sales_data()
            return {
                "status": "success",
                "data": data,
                "message": f"查询到 {len(data)} 条销售数据"
            }
        else:
            return {
                "status": "fail",
                "data": [],
                "message": "未找到相关数据表或字段"
            }

# 实例化
db_tool = DBTool()