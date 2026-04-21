# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router
from backend.core.config import settings
from loguru import logger
import uvicorn

# === 初始化 FastAPI 应用 ===
app = FastAPI(
    title="Enterprise Agent Assistant",
    description="基于 RAG 与 Agent 的企业级智能数据分析平台",
    version="1.0.0"
)

# === 配置 CORS (允许前端 Streamlit 跨域访问) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境应替换为具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 注册路由 ===
app.include_router(api_router, prefix="/api")

# === 启动事件 ===
@app.on_event("startup")
async def startup_event():
    logger.info(f"服务启动中... 正在加载模型: {settings.LLM_MODEL_NAME}")
    # 这里可以预加载一些模型或检查连接
    logger.info("后端服务就绪")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )