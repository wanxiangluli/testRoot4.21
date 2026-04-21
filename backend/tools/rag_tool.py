# backend/tools/rag_tool.py
import os
# === 关键修改：设置 HuggingFace 镜像源 (必须在导入 FlagEmbedding 之前) ===
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import requests
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
import jieba
from loguru import logger
from pydantic import BaseModel, Field

# 移除 SentenceTransformer 导入
# from sentence_transformers import SentenceTransformer 

# 保留 FlagEmbedding 用于 Rerank
from FlagEmbedding import FlagReranker

# 引入 OpenAI 客户端用于调用百炼 Embedding
from openai import OpenAI

from backend.core.config import settings

# ---------- 数据结构 ----------

class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None
    source: Optional[str] = None
    score: float = 0.0

class RAGResult(BaseModel):
    answer: str
    docs: List[RetrievedDoc]
    chart_spec: Optional[Dict[str, Any]] = None

# ---------- BM25 分词器 ----------

class ChineseBM25Tokenizer:
    def __init__(self):
        self.tokenizer = jieba

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer.lcut(text)
        return [t for t in tokens if t.strip()]

# ---------- 向量检索 (改用百炼 API) ----------

class VectorRetriever:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        
        # 初始化 OpenAI 客户端，指向百炼
        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL
        )

        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB loaded. Using Cloud Embedding: {self.model_name}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        调用阿里云百炼 API 获取向量 (增加分批处理)
        """
        all_embeddings = []
        batch_size = 25  # 阿里云百炼限制单次请求最大条数为 25
        
        try:
            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                # 提取当前批次的向量
                # 注意：为了防止顺序混乱，最好按 index 排序（虽然通常返回顺序一致）
                batch_embeddings = sorted(response.data, key=lambda e: e.index)
                embeddings = [item.embedding for item in batch_embeddings]
                
                all_embeddings.extend(embeddings)
                
            return all_embeddings
            
        except Exception as e:
            logger.error(f"调用百炼 Embedding API 失败: {e}")
            # 返回空向量防止崩溃，实际生产需要重试机制
            return [[0.0] * 1024 for _ in texts]
    def add_documents(self, docs: List[RetrievedDoc], embeddings: Optional[List[List[float]]] = None):
        if not docs:
            return

        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        metadatas = []
        for d in docs:
            meta = {
                "page": d.page or -1,
                "bbox": json.dumps(d.bbox) if d.bbox else "",
                "source": d.source or "",
            }
            metadatas.append(meta)

        if embeddings is None:
            # 调用 API 生成向量
            embeddings = self.get_embeddings(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(docs)} documents to ChromaDB via Cloud API.")

    def query(self, query_text: str, top_k: int = None) -> List[RetrievedDoc]:
        top_k = top_k or settings.RETRIEVE_TOP_K
        
        # 1. 将查询文本转向量
        query_embedding = self.get_embeddings([query_text])[0]

        # 2. 检索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        docs = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            doc = RetrievedDoc(
                doc_id=meta.get("id", ""), 
                text=text,
                page=meta.get("page"),
                bbox=json.loads(meta["bbox"]) if meta.get("bbox") else None,
                source=meta.get("source"),
                score=1.0 - dist
            )
            docs.append(doc)
        return docs

# ---------- BM25 检索 ----------
# ... BM25Retriever 代码保持不变 ...

class BM25Retriever:
    def __init__(self):
        self.tokenizer = ChineseBM25Tokenizer()
        self.corpus: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_documents(self, docs: List[RetrievedDoc]):
        if not docs:
            return
        for d in docs:
            self.corpus.append(d.text)
            self.doc_ids.append(d.doc_id)

        tokenized_corpus = [self.tokenizer.tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 indexed {len(docs)} documents.")

    def query(self, query_text: str, top_k: int = None) -> List[RetrievedDoc]:
        if not self.bm25:
            return []

        top_k = top_k or settings.BM25_TOP_K
        tokenized_query = self.tokenizer.tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            results.append(RetrievedDoc(
                doc_id=self.doc_ids[idx],
                text=self.corpus[idx],
                score=float(scores[idx])
            ))
        return results

# ---------- RRF 混合合并 ----------
# ... reciprocal_rank_fusion 代码保持不变 ...

def reciprocal_rank_fusion(
    list1: List[RetrievedDoc],
    list2: List[RetrievedDoc],
    k: int = 60
) -> List[RetrievedDoc]:
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, RetrievedDoc] = {}

    for rank, doc in enumerate(list1, start=1):
        rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc.doc_id] = doc

    for rank, doc in enumerate(list2, start=1):
        rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc.doc_id] = doc

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in sorted_ids:
        d = doc_map[doc_id]
        d.score = score
        results.append(d)
    return results

# ---------- Reranker (更新模型名称) ----------

class Reranker:
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model_name = settings.RERANKER_MODEL
        # 百炼 Rerank API 的专用端点 (注意：这不是 OpenAI 兼容接口，是原生的)
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        logger.info(f"初始化云端 Reranker: {self.model_name}")

    def rerank(self, query: str, docs: List[RetrievedDoc], top_k: int = None) -> List[RetrievedDoc]:
        if not docs:
            return []

        top_k = top_k or settings.RERANK_TOP_K
        
        # 1. 准备请求数据
        # 文档列表只需要文本内容
        documents_text = [d.text for d in docs]
        
        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": documents_text
            },
            "parameters": {
                "top_n": top_k # 让 API 直接返回 top_n 个结果
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            # 2. 发送请求
            response = requests.post(self.url, json=payload, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Rerank API 调用失败: {response.status_code} - {response.text}")
                # 降级：如果API失败，直接返回前 top_k 个原始结果
                return docs[:top_k]

            result = response.json()
            
            # 3. 解析结果
            # 百炼返回格式通常是: {"output": [{"index": 0, "relevance_score": 0.9}, ...]}
            rerank_results = result.get("output", {}).get("results", [])
            
            # 按分数降序排序 (API通常已排序，但为了保险)
            rerank_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            # 4. 重构 RetrievedDoc 列表
            final_docs = []
            for item in rerank_results[:top_k]:
                idx = item["index"]
                score = item["relevance_score"]
                
                # 复制原 doc 对象并更新分数
                original_doc = docs[idx]
                # Pydantic 对象如果不可变可以用 .model_copy()
                # 这里我们简单修改属性 (如果是不可变模式需调整)
                original_doc.score = score 
                final_docs.append(original_doc)
            
            logger.info(f"云端 Rerank 完成，返回 {len(final_docs)} 个文档。")
            return final_docs

        except Exception as e:
            logger.error(f"Rerank 过程发生异常: {e}")
            return docs[:top_k]
# ---------- 主工具类 ----------
# ... HybridRAGTool 代码保持不变 ...

class HybridRAGTool:
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
        self.reranker = Reranker()
        
        # 启动时尝试从 ChromaDB 恢复 BM25 数据
        self._load_bm25_from_chroma()

    def _load_bm25_from_chroma(self):
        """从 ChromaDB 加载已有数据到 BM25 内存索引"""
        try:
            existing_data = self.vector_retriever.collection.get(
                include=["documents", "metadatas"]
            )
            
            if existing_data and existing_data['ids']:
                logger.info(f"检测到历史数据，正在恢复 {len(existing_data['ids'])} 条记录到 BM25...")
                
                docs = []
                for id_, text, meta in zip(
                    existing_data['ids'], 
                    existing_data['documents'], 
                    existing_data['metadatas']
                ):
                    docs.append(RetrievedDoc(
                        doc_id=id_,
                        text=text,
                        source=meta.get("source", ""),
                        page=meta.get("page")
                    ))
                
                self.bm25_retriever.add_documents(docs)
                logger.info("BM25 索引恢复完成。")
            else:
                logger.info("ChromaDB 为空，无需恢复 BM25。")
                
        except Exception as e:
            logger.warning(f"恢复 BM25 数据失败: {e}")

    def add_documents(self, docs: List[RetrievedDoc]):
        """添加文档到检索系统"""
        self.vector_retriever.add_documents(docs)
        self.bm25_retriever.add_documents(docs)

    def retrieve(self, query: str) -> List[RetrievedDoc]:
        """混合检索 + Rerank"""
        vec_docs = self.vector_retriever.query(query, top_k=settings.RETRIEVE_TOP_K)
        bm25_docs = self.bm25_retriever.query(query, top_k=settings.BM25_TOP_K)
        merged_docs = reciprocal_rank_fusion(vec_docs, bm25_docs)
        reranked_docs = self.reranker.rerank(query, merged_docs, top_k=settings.RERANK_TOP_K)

        logger.info(f"Retrieved {len(vec_docs)} vector docs, {len(bm25_docs)} BM25 docs, "
                    f"merged {len(merged_docs)}, reranked {len(reranked_docs)}.")
        return reranked_docs

    def build_prompt(self, query: str, docs: List[RetrievedDoc]) -> str:
        context = "\n\n".join(
            f"【文档{i+1}】(来源: {d.source or '未知'}, 页码: {d.page or '未知'})\n{d.text}"
            for i, d in enumerate(docs)
        )
        prompt = (
            "你是一位经验丰富的企业高级顾问。请基于提供的【参考文档】回答用户问题。\n\n"
            
            "### 核心要求：\n"
            "1. **深度整合**：不要简单复述文档原文，请将信息提炼、整合成逻辑通顺的回答。\n"
            "2. **适度扩展**：在确保核心信息来自文档的前提下，你可以利用你的通用知识对文档中的概念、术语进行解释，或者提供相关的背景知识，帮助用户更好地理解。\n"
            "3. **结构清晰**：如果回答较长，请使用分点、小标题等形式进行排版，保持可读性。\n"
            "4. **诚实原则**：如果文档中完全没有提到用户问题的核心内容，请明确告知，不要强行编造。\n\n"
            
            f"### 参考文档：\n{context}\n\n"
            
            f"### 用户问题：\n{query}\n\n"
            
            "### 你的回答："
        )
        return prompt

    def run(self, query: str, return_prompt: bool = False) -> RAGResult:
        """对外暴露的主接口"""
        # 1. 检索
        docs = self.retrieve(query)
        if not docs:
            return RAGResult(answer="抱歉，知识库中没有找到相关内容。", docs=[])

        # 2. 构建 Prompt
        prompt = self.build_prompt(query, docs)

        # 3. 调用百炼 LLM
        client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.DASHSCOPE_BASE_URL,
        )
        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()

        # 4. 封装结果
        result = RAGResult(answer=answer, docs=docs, chart_spec=None)

        if return_prompt:
            return result, prompt
        return result
    
hybrid_rag_tool = HybridRAGTool()