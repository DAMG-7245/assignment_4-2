import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import pinecone
import chromadb
import sys
from .gemini import generate_answer_with_gemini
from dotenv import load_dotenv
load_dotenv()
# 假设使用 sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")  # 示例

# Initialize embedding model (lazy load)
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def compute_embeddings(texts: List[str]) -> List[np.ndarray]:
    """
    用 sentence-transformers 生成向量.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# 1) manual approach
def rag_manual(chunks: List[str], user_query: str, top_k: int = 3) -> List[str]:
    """
    手动计算嵌入, 对所有 chunk 做相似度排序, 返回最相似的 top_k chunks.
    """
    # 先把 chunks 和 user_query 全部算 embedding
    chunk_embeddings = compute_embeddings(chunks)
    query_embedding = compute_embeddings([user_query])[0]  # shape=(768,) for example

    # 计算余弦相似度
    scores = []
    for idx, emb in enumerate(chunk_embeddings):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        scores.append((idx, sim))

    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunks[idx] for (idx, _) in scores[:top_k]]
    return top_chunks

# 2) Pinecone approach
def rag_pinecone(index_name: str, chunks: List[str], user_query: str, top_k: int = 3) -> List[str]:
    """
    将 chunks 上传到 Pinecone (若需要), 再搜索 user_query.
    - 这里做简单示例，每次都新建 index 并 upsert chunks, 你可以改成离线流程.
    """
    # 初始化 pinecone
    if PINECONE_API_KEY:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    else:
        raise ValueError("PINECONE_API_KEY not set.")
    
    # 若 index 不存在则创建
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=384)  # depends on your model dimension
    index = pinecone.Index(index_name)

    # 计算 embedding
    chunk_embeddings = compute_embeddings(chunks)

    # 先清空index (仅演示)
    index.delete(deleteAll=True)

    # upsert
    vectors = [(f"id-{i}", chunk_embeddings[i].tolist(), {"text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(vectors=vectors)

    # user_query 向量
    query_emb = compute_embeddings([user_query])[0].tolist()

    # query pinecone
    result = index.query(vector=query_emb, top_k=top_k, include_metadata=True)
    # 返回最相关 chunks
    top_chunks = [m["metadata"]["text"] for m in result.matches]
    return top_chunks

# 3) ChromaDB approach
def rag_chromadb(collection_name: str, chunks: List[str], user_query: str, top_k: int = 3) -> List[str]:
    """
    使用 ChromaDB 做向量检索.
    同理, 简化演示: 每次都新建 DB, insert chunks, query user_query.
    """
    client = chromadb.Client()
    # 如果 collection 已存在，可先 drop 再创建(演示)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    coll = client.create_collection(name=collection_name)

    # 计算 embedding
    chunk_embeddings = compute_embeddings(chunks)

    # Insert
    ids = [f"id-{i}" for i in range(len(chunks))]
    embeddings_list = [emb.tolist() for emb in chunk_embeddings]
    metadata = [{"original_text": chunks[i]} for i in range(len(chunks))]
    coll.add(documents=chunks, embeddings=embeddings_list, ids=ids, metadatas=metadata)

    # query
    query_emb = compute_embeddings([user_query])[0].tolist()
    results = coll.query(query_embeddings=[query_emb], n_results=top_k)

    # results 的结构一般是:
    # {
    #   'ids': [...],
    #   'embeddings': [...],
    #   'documents': [...],
    #   'metadatas': [...]
    # }
    top_chunks = results["documents"][0]  # top_k documents for the single query
    return top_chunks


def rag_retrieval(chunks: List[str], user_query: str, rag_type: str = "manual", top_k: int = 3) -> List[str]:
    """
    统一调用入口.
    rag_type in {"manual", "pinecone", "chromadb"}
    """
    if rag_type == "manual":
        return rag_manual(chunks, user_query, top_k=top_k)
    elif rag_type == "pinecone":
        # 你需要给一个 Pinecone index name
        return rag_pinecone("my_pinecone_index", chunks, user_query, top_k=top_k)
    elif rag_type == "chromadb":
        # 指定 collection name
        return rag_chromadb("my_chroma_collection", chunks, user_query, top_k=top_k)
    else:
        raise ValueError(f"Unknown RAG type: {rag_type}")


def rag_query_with_gemini(chunks: List[str], user_query: str, rag_type: str = "manual", top_k: int = 3) -> str:
    """
    1) 用 RAG 检索 (manual/pinecone/chromadb)获取 top_k chunk
    2) 拼接这些 chunk 成为 context
    3) 调用 Gemini 生成最终回答
    """
    top_chunks = rag_retrieval(chunks, user_query, rag_type, top_k)
    context = "\n\n".join(top_chunks)

    # 调用 Gemini
    final_answer = generate_answer_with_gemini(context, user_query, model="gemini-2.0-flash")
    return final_answer
