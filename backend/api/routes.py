# backend/api/routes.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from backend.agents.graph import app_graph
from backend.tools.rag_tool import hybrid_rag_tool, RetrievedDoc
from backend.core.config import settings
from loguru import logger
# === 2. 文本分块 ===
from langchain_text_splitters import RecursiveCharacterTextSplitter

import uuid
import pdfplumber  # 引入 PDF 解析库
import os

router = APIRouter()

# === 请求/响应模型 ===

class ChatRequest(BaseModel):
    user_input: str
    thread_id: Optional[str] = "default-thread"  # 用于区分会话

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]] = []
    chart_data: Optional[Dict[str, Any]] = None

# === 接口定义 ===

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    智能对话接口：处理用户问题，返回 Agent 响应
    """
    logger.info(f"收到用户请求 [Thread: {request.thread_id}]: {request.user_input}")

    try:
        # 1. 构造初始状态
        initial_state = {
            "messages": [HumanMessage(content=request.user_input)],
            "user_input": request.user_input,
            "iterations": 0,
            "final_answer": None,
            "intermediate_steps": []
        }

        # 2. 调用 LangGraph 执行
        config = {"configurable": {"thread_id": request.thread_id}}
        final_state = app_graph.invoke(initial_state, config=config)

        # === 3. 解析结果 (关键修改) ===
        
        # 优先获取显式的 final_answer
        final_answer = final_state.get("final_answer")
        
        # 兜底逻辑：如果 final_answer 为空 (例如 Agent 直接回答，未进入反思节点)
        # 则从 messages 中提取最后一条 AI 消息
        if not final_answer:
            for msg in reversed(final_state.get("messages", [])):
                # 寻找最后一条 AI 消息
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    logger.info("从消息历史中提取到 AI 回答作为兜底")
                    break
        
        # 最终兜底：如果实在没有，返回默认字符串
        if not final_answer:
            final_answer = "抱歉，我暂时无法回答这个问题。"
            
        # 提取图表数据与来源文档
        chart_data = None
        source_documents = []
        
        # ... (后续提取 source_documents 和 chart_data 的代码保持不变) ...
        for msg in final_state.get("messages", []):
            if msg.type == "tool":
                try:
                    import json
                    content = json.loads(msg.content)
                    if isinstance(content, dict) and content.get("docs"):
                        source_documents = [
                            doc.model_dump() if isinstance(doc, RetrievedDoc) else doc 
                            for doc in content["docs"]
                        ]
                except:
                    pass

        return ChatResponse(
            answer=final_answer,
            source_documents=source_documents,
            chart_data=chart_data
        )

    except Exception as e:
        logger.error(f"Chat Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/upload")
async def upload_documents(file: UploadFile = File(...)):
    """
    文档上传接口：
    1. 解析文档内容
    2. 文本分块
    3. 写入向量库与 BM25 索引
    """
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(status_code=400, detail="目前仅支持 PDF, TXT, MD 格式")

    try:
        logger.info(f"开始处理文件: {file.filename}")
        
        # === 1. 读取与解析内容 ===
        full_text = ""
        
        if file.filename.endswith('.pdf'):
            # 使用 pdfplumber 解析 PDF
            # 注意：UploadFile 的 file 对象是 SpooledTemporaryFile，pdfplumber 需要.seek(0)
            file.file.seek(0)
            with pdfplumber.open(file.file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                        
        elif file.filename.endswith(('.txt', '.md')):
            # 直接读取文本
            content = await file.read()
            # 尝试常见编码解码
            try:
                full_text = content.decode('utf-8')
            except UnicodeDecodeError:
                full_text = content.decode('gbk')
        
        if not full_text.strip():
            return {"status": "error", "message": "文件内容为空或无法解析"}


        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # 每块最大 500 字符
            chunk_overlap=50,     # 块之间重叠 50 字符，防止语义截断
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        logger.info(f"文件被切分为 {len(chunks)} 个片段")

        # === 3. 构建文档对象 ===
        docs = []
        for idx, chunk in enumerate(chunks):
            # 这里可以加入更复杂的逻辑，比如记录 chunk 属于哪一页
            # 目前简化为记录来源文件名和块 ID
            doc = RetrievedDoc(
                doc_id=f"{file.filename}_chunk_{idx}",
                text=chunk,
                source=file.filename,
                # page=... # 如果需要页码，需要在解析阶段保留 page 信息
            )
            docs.append(doc)
        
        # === 4. 写入向量库 ===
        hybrid_rag_tool.add_documents(docs)
        
        return {
            "status": "success", 
            "message": f"文件 {file.filename} 已成功入库，共生成 {len(docs)} 个知识片段。"
        }

    except Exception as e:
        logger.error(f"Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))