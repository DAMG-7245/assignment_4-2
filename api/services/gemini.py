import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("请设置 GEMINI_API_KEY 环境变量，以便调用 Gemini API")

# 创建一个全局的 genai.Client 实例
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def generate_answer_with_gemini(context: str, question: str, model: str = "gemini-2.0-flash") -> str:
    """
    调用Google Gemini模型，结合 context（RAG检索到的文本片段）和 question 生成答案。
    context: 检索到的文本片段拼接
    question: 用户问题
    model: Gemini 模型名称, 如 "gemini-2.0-flash"
    """
    prompt = (
        "You are a helpful AI assistant. "
        "Use the following context to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = _gemini_client.models.generate_content(
        model=model,
        contents=prompt
    )

    # 如果官方文档说明 response.text 是回答内容，则直接用它
    return response.text
