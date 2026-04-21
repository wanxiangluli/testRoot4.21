# backend/core/config.py
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # === 百炼 API 配置 ===
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_BASE_URL: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "qwen-plus-1220")

    # === Embedding 配置 (改用阿里云百炼云端模型) ===
    # 百炼支持的文本嵌入模型名称，通常是 text-embedding-v1 或 text-embedding-v2
    EMBEDDING_MODEL: str = "text-embedding-v2" 
    
    # === Reranker 配置 (改用本地 Base 模型) ===
    # base 模型体积小、速度快，适合开发调试
    RERANKER_MODEL: str = "qwen3-rerank"

    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "enterprise_knowledge"

    RETRIEVE_TOP_K: int = 8
    BM25_TOP_K: int = 8
    RERANK_TOP_K: int = 4

    # 查询指令 (云端 Embedding 通常不需要指令前缀，但百炼有些模型可能需要，
    # 为了通用性，我们在代码里处理，这里先留空)
    QUERY_INSTRUCTION: str = "" 

    class Config:
        env_file = ".env"
        extra = "ignore"

    def check_api_key(self) -> None:
        if not self.DASHSCOPE_API_KEY:
            raise ValueError("未检测到 DASHSCOPE_API_KEY，请检查环境变量或 .env 文件。")

settings = Settings()
settings.check_api_key()