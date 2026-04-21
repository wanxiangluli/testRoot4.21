# frontend/app.py
import streamlit as st
import requests
import json
import time
from streamlit_echarts import st_echarts

# === 配置 ===
st.set_page_config(page_title="企业智能客服助手", layout="wide")
BACKEND_URL = "http://localhost:8000/api"  # FastAPI 后端地址

# === 初始化 Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = [] # 存储对话历史 [{"role": "user/assistant", "content": "..."}]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_" + str(int(time.time()))

# === 侧边栏：知识库管理 ===
with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传企业文档 (PDF/TXT/MD)", type=["pdf", "txt", "md"])
    
    if uploaded_file is not None:
        with st.spinner("正在解析并入库..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success(f"文件 {uploaded_file.name} 上传成功！")
                else:
                    st.error(f"上传失败: {response.text}")
            except Exception as e:
                st.error(f"连接后端服务失败: {e}")

    st.divider()
    st.markdown("### ℹ️ 使用说明")
    st.markdown("""
    1. 上传企业文档构建知识库。
    2. 在对话框提问，系统将自动：
       - 检索知识库回答问题。
       - 查询业务数据并生成图表。
       - 标注信息来源，实现溯源。
    """)

# === 主区域：智能对话 ===
st.title("🤖 企业知识库智能客服")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 如果是 AI 消息，可能包含图表和引用
        if message["role"] == "assistant":
            # 渲染文本
            st.markdown(message["content"])
            
            # 渲染图表 (如果有)
            if message.get("chart_data"):
                st_echarts(options=message["chart_data"], height="400px")
            
            # 渲染引用来源 (如果有)
            if message.get("sources"):
                with st.expander("查看引用来源"):
                    for idx, doc in enumerate(message["sources"]):
                        st.caption(f"**来源 {idx+1}**: {doc.get('source', '未知文件')} - 第 {doc.get('page', '?')} 页")
                        st.text(doc.get("text", "")[:200] + "...")
        else:
            st.markdown(message["content"])

# === 用户输入处理 ===
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端 API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("正在思考中... 🧠")
        
        try:
            payload = {
                "user_input": prompt,
                "thread_id": st.session_state.thread_id
            }
            response = requests.post(f"{BACKEND_URL}/chat", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                ai_answer = result.get("answer", "抱歉，出现了一个错误。")
                sources = result.get("source_documents", [])
                chart_data = result.get("chart_data")
                
                # 模拟流式输出效果 (可选)
                # message_placeholder.markdown(ai_answer)
                
                # 更新 Session State
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer,
                    "sources": sources,
                    "chart_data": chart_data
                })
                
                # 渲染最终结果
                st.markdown(ai_answer)
                if chart_data:
                    st_echarts(options=chart_data, height="400px")
                if sources:
                    with st.expander("查看引用来源"):
                        for idx, doc in enumerate(sources):
                            st.caption(f"**来源 {idx+1}**: {doc.get('source', '未知')} - 第 {doc.get('page', '?')} 页")
                            st.code(doc.get("text", "")[:300], language="markdown")
            else:
                st.error(f"后端服务错误: {response.text}")
                
        except Exception as e:
            st.error(f"无法连接到后端服务: {e}")