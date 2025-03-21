import os
import json
import base64
import tempfile
import requests
from docling.document_converter import DocumentConverter
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

# 从环境变量获取 API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
JINA_AUTH_TOKEN = os.getenv("JINA_AUTH_TOKEN", "jina_9df6578702b74931aab7876a813b316fxrct62iLrDKQS0V7O-fNFsRFwkr3")

###########################################################
# 1) Mistral OCR: 官方示例 - 使用 document_url
#    如果 PDF 是公开的URL, 可直接让 Mistral去抓取
###########################################################
def mistral_ocr_from_url(document_url=None, pdf_bytes=None):
    """
    Process a document using Mistral OCR API with improved response handling.
    
    Args:
        document_url (str, optional): URL to a PDF document
        pdf_bytes (bytes, optional): Raw PDF data as bytes
        
    Returns:
        str: Extracted text concatenated from all pages
    """
    # 参数校验
    if not document_url and not pdf_bytes:
        raise ValueError("Either document_url or pdf_bytes must be provided")
    
    # 初始化客户端
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY environment variable is not set.")
    client = Mistral(api_key=MISTRAL_API_KEY)

    try:
        # 构建文档参数
        document = {
            "type": "document_url",
            "document_url": document_url
        } if document_url else {
            "type": "document_data",
            "document_data": base64.b64encode(pdf_bytes).decode("utf-8")
        }

        # 调用OCR API
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document,
            include_image_base64=False
        )

        # 转换响应为字典
        try:
            response_dict = ocr_response.model_dump()
        except AttributeError:
            response_dict = dict(ocr_response) if hasattr(ocr_response, "__dict__") else {}

        # 调试：打印原始响应结构
        print("原始响应结构:", json.dumps(response_dict, indent=2)[:500] + "...")

        # 解析页面数据
        pages_data = None
        if "pages" in response_dict:  # 优先检查顶层pages
            pages_data = response_dict["pages"]
        elif "data" in response_dict and "pages" in response_dict["data"]:  # 检查data下的pages
            pages_data = response_dict["data"]["pages"]
        elif hasattr(ocr_response, "pages"):  # 直接访问对象属性
            pages_data = ocr_response.pages
        elif hasattr(ocr_response, "data") and hasattr(ocr_response.data, "pages"):
            pages_data = ocr_response.data.pages

        # 提取文本内容
        full_text = []
        if pages_data:
            for page in pages_data:
                # 自动处理字典或对象类型
                page_dict = page if isinstance(page, dict) else vars(page)
                
                # 优先提取markdown字段
                if "markdown" in page_dict:
                    full_text.append(str(page_dict["markdown"]))
                elif "text" in page_dict:
                    full_text.append(str(page_dict["text"]))
                elif "content" in page_dict:
                    full_text.append(str(page_dict["content"]))

        return "\n\n".join(full_text) if full_text else ""

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Mistral OCR 处理失败: {str(e)}")

###########################################################
# 2) Docling: 官方示例 - 既可传本地路径, 也支持网络URL
###########################################################
def docling_from_url_or_path(source: str) -> str:
    """
    Docling 可以直接传一个 URL (网络PDF) 或者本地文件路径.
    官方示例:
      source = "https://arxiv.org/pdf/2408.09869"
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    # 这里导出为 Markdown, 也可改成 export_to_text() 等
    return result.document.export_to_markdown()

###########################################################
# 3) Requests + Jina: 将PDF (bytes) 发送至 https://r.jina.ai/
#    需要先把 PDF 编码为 base64
###########################################################
def jina_parse_pdf(pdf_bytes: bytes) -> str:
    """
    官方示例中使用requests对Jina接口进行调用。
    需要:
      - JINA_AUTH_TOKEN (Bearer)
      - URL: 'https://r.jina.ai/'
    """
    if not JINA_AUTH_TOKEN:
        raise ValueError("JINA_AUTH_TOKEN not set.")

    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    url = 'https://r.jina.ai/'  # 修正后的URL
    headers = {
        'Authorization': f'Bearer {JINA_AUTH_TOKEN}',
        'X-Return-Format': 'markdown',  # 若想要Markdown格式
        'Content-Type': 'application/json'
    }
    data = {
        # 这里 'url' 不一定要真实可访问, Jina文档里是示例
        'url': 'https://example.com',
        'pdf': encoded_pdf
    }
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code != 200:
        raise RuntimeError(f"Jina parse failed, status={resp.status_code}, resp={resp.text}")

    # 如果Jina返回json可解析,可用 resp.json()["text"] 之类
    return resp.text


###########################################################
# 4) 统一示例：根据 parser_type, 执行不同处理
#    A) "mistral_ocr_url"
#    B) "docling_url"
#    C) "jina_bytes"
###########################################################
def parse_pdf_any(pdf_url: str = "", parser_type: str = "", pdf_bytes: bytes = b"") -> str:
    if parser_type == "mistral_ocr_url":
        if not pdf_url:
            raise ValueError("pdf_url is required for mistral_ocr_url")
        return mistral_ocr_from_url(pdf_url)
    elif parser_type == "docling_url":
        if not pdf_url:
            raise ValueError("pdf_url is required for docling_url")
        return docling_from_url_or_path(pdf_url)
    elif parser_type == "jina_bytes":
        if not pdf_bytes:
            raise ValueError("pdf_bytes is required for jina_bytes parser")
        return jina_parse_pdf(pdf_bytes)
    else:
        raise ValueError(f"Unknown parser_type: {parser_type}")

###########################################################
#   以下是演示如何调用官方示例 & 进行简单测试
###########################################################
if __name__ == "__main__":
    print("===== [Demo 1] Mistral OCR from URL =====")
    try:
        text_ocr = parse_pdf_any(
            pdf_url="https://arxiv.org/pdf/2201.04234",
            parser_type="mistral_ocr_url"
        )
        print("Mistral OCR result (first 300 chars):\n", text_ocr[:300], "...")
    except Exception as e:
        print("Mistral OCR failed:", e)

    print("\n===== [Demo 2] Docling from URL =====")
    try:
        docling_text = parse_pdf_any(
            pdf_url="https://arxiv.org/pdf/2408.09869",
            parser_type="docling_url"
        )
        print("Docling result (first 300 chars):\n", docling_text[:300], "...")
    except Exception as e:
        print("Docling parse failed:", e)

    print("\n===== [Demo 3] Jina parse from PDF bytes =====")
    try:
        # 这里演示从本地读取一个PDF
        local_pdf_path = "some_local_file.pdf"
        if os.path.exists(local_pdf_path):
            with open(local_pdf_path, "rb") as f:
                pdf_data = f.read()
            jina_text = parse_pdf_any(parser_type="jina_bytes", pdf_bytes=pdf_data)
            print("Jina parse result (first 300 chars):\n", jina_text[:300], "...")
        else:
            print(f"Skip Jina parse: {local_pdf_path} not found.")
    except Exception as e:
        print("Jina parse failed:", e)

    print("\n===== [Demo 4] Simple GET request example (unrelated) =====")
    # 仅示例 requests.get + 自定义header
    # 如果想获取 https://r.jina.ai/ 直接访问
    try:
        test_url = "https://r.jina.ai/"
        test_headers = {
            'Authorization': f'Bearer {JINA_AUTH_TOKEN}',
            'X-Return-Format': 'markdown'
        }
        resp = requests.get(test_url, headers=test_headers)
        print("Status:", resp.status_code)
        print("Resp text (first 200 chars):", resp.text[:200], "...")
    except Exception as e:
        print("GET request failed:", e)
