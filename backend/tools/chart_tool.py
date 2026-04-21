# backend/tools/chart_tool.py
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

# 引用我们自己的配置和提示词
from backend.core.config import settings
from backend.core.prompts import Prompts

# 简单的 OpenAI 客户端封装（为了复用配置）
from openai import OpenAI

class ChartInput(BaseModel):
    """图表生成工具的输入"""
    data: List[Dict[str, Any]] = Field(..., description="要可视化的数据，列表套字典格式")
    chart_type: str = Field(default="bar", description="图表类型，如 bar, line, pie")
    title: str = Field(default="数据分析图表", description="图表标题")

class ChartTool:
    """
    动态图表生成工具：
    1. 接收数据
    2. 调用 LLM 生成 ECharts/Pyecharts 的 option JSON
    3. 返回前端可直接渲染的配置
    """
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL
        )
        self.model = settings.LLM_MODEL_NAME

    def generate_chart_option(self, data: List[Dict], chart_type: str, title: str) -> Dict:
        """
        核心方法：将数据转换为图表配置
        """
        if not data:
            return {"error": "没有数据无法生成图表"}

        # 构造提示词，让大模型生成配置
        prompt = f"""
请根据以下数据生成一个 {chart_type} 图表的 Pyecharts 配置 JSON。
数据如下：
{json.dumps(data, ensure_ascii=False, indent=2)}

要求：
1. 包含 title, tooltip, legend, xAxis, yAxis, series 等必要配置。
2. 标题设置为: {title}
3. 直接返回 JSON 字符串，不要包裹在 Markdown 代码块中。
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": Prompts.CHART_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        content = response.choices[0].message.content.strip()
        
        # 清理可能存在的 Markdown 包裹
        if content.startswith("```"):
            content = content.strip("`").replace("json", "").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"图表配置解析失败: {content}")
            return {"error": "图表配置生成失败", "raw_content": content}

    def run(self, input_data: ChartInput) -> Dict:
        """对外暴露的执行接口"""
        logger.info(f"正在生成图表: 类型={input_data.chart_type}, 标题={input_data.title}")
        option = self.generate_chart_option(input_data.data, input_data.chart_type, input_data.title)
        return {
            "type": "chart",
            "chart_option": option
        }

# 实例化供 Agent 调用
chart_tool = ChartTool()