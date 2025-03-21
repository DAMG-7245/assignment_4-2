import streamlit as st
import requests

def s3_pdf_selector(base_url: str) -> str:
    """
    组件1: 从后端获取S3中的PDF列表, 让用户在前端选择一个PDF key.
    :param base_url: 后端API地址 (如 http://localhost:8000/api)
    :return: 用户所选的pdf_key (str)
    """
    pdf_keys = []
    with st.spinner("加载 PDF 列表..."):
        try:
            resp = requests.get(f"{base_url}/list-pdfs")
            resp.raise_for_status()
            pdf_keys = resp.json().get("pdf_keys", [])
        except Exception as e:
            st.error(f"获取PDF列表失败: {e}")

    if not pdf_keys:
        st.warning("S3中没有找到任何PDF文件。请检查或等待Airflow爬取。")
        return ""

    selected_pdf = st.selectbox("选择一个 S3 PDF Key:", pdf_keys)
    return selected_pdf


def parser_selector() -> str:
    """
    组件2: 让用户选择 PDF 解析方式 (basic, docling, mistral_ocr).
    :return: 用户选定的解析方式
    """
    parser_type = st.selectbox("选择解析器", ["basic", "docling", "mistral_ocr"])
    return parser_type


def chunking_selector() -> str:
    """
    组件3: 让用户选择 chunking 策略 (fixed, paragraph, sentence).
    :return: 选定的策略
    """
    chunk_strategy = st.selectbox("选择 Chunking 策略", ["fixed", "paragraph", "sentence"])
    return chunk_strategy


def parse_and_chunk_button(base_url: str, pdf_key: str, parser_type: str, chunk_strategy: str):
    """
    组件4: 当用户点击按钮时, 调用后端的 parse-and-chunk 接口
    并将返回的 chunks 存储到 session_state.
    """
    if st.button("解析并切分"):
        if not pdf_key:
            st.warning("请先选择PDF")
            return
        with st.spinner("正在解析并切分..."):
            try:
                parse_resp = requests.get(
                    f"{base_url}/parse-and-chunk",
                    params={
                        "pdf_key": pdf_key,
                        "parser_type": parser_type,
                        "chunk_strategy": chunk_strategy,
                    }
                )
                parse_resp.raise_for_status()
                data = parse_resp.json()
                st.session_state["chunks"] = data["chunks"]
                st.session_state["chunks_count"] = data["chunks_count"]
                st.success(f"解析成功，获得 {data['chunks_count']} 个文本片段！")
            except Exception as e:
                st.error(f"解析失败: {e}")


def rag_query_and_answer(base_url: str):
    """
    组件5: 让用户输入查询, 选择RAG方案, 并可选择是否使用Gemini做最终回答.
    如果只想看检索片段, 就不选 Gemini; 如果想要 LLM回答, 选 Gemini.
    """
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.info("请先执行 解析并切分, 以获取 chunks.")
        return

    st.subheader("查询与回答")
    user_query = st.text_input("请输入你的问题:", "")
    rag_type = st.selectbox("RAG 方法", ["manual", "pinecone", "chromadb"])
    top_k = st.slider("检索Top K", 1, 5, 3)

    # 是否调用Gemini
    use_gemini = st.checkbox("使用 Gemini 生成最终回答")

    if st.button("查询"):
        if use_gemini:
            # 调用 /rag-ask-gemini
            with st.spinner("RAG检索 + Gemini生成中..."):
                payload = {
                    "user_query": user_query,
                    "chunks": chunks,
                    "rag_type": rag_type,
                    "top_k": top_k
                }
                try:
                    resp = requests.post(f"{base_url}/rag-ask-gemini", json=payload)
                    resp.raise_for_status()
                    answer = resp.json().get("answer", "")
                    st.write("### Gemini 给出的最终回答：")
                    st.write(answer)
                except Exception as e:
                    st.error(f"调用 /rag-ask-gemini 失败: {e}")
        else:
            # 不用Gemini，只做 RAG 检索
            with st.spinner("RAG检索中..."):
                payload = {
                    "user_query": user_query,
                    "chunks": chunks,
                    "rag_type": rag_type,
                    "top_k": top_k
                }
                try:
                    resp = requests.post(f"{base_url}/rag-query", json=payload)
                    resp.raise_for_status()
                    result_data = resp.json()
                    top_chunks = result_data.get("top_chunks", [])
                    if not top_chunks:
                        st.warning("未检索到任何相关片段。")
                    else:
                        st.write("### RAG检索到的文本片段:")
                        for i, c in enumerate(top_chunks):
                            st.markdown(f"**结果 {i+1}**:\n```\n{c}\n```")
                except Exception as e:
                    st.error(f"调用 /rag-query 失败: {e}")
