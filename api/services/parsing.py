import os
import tempfile
import fitz  # PyMuPDF
from docling.document_converter import DocumentConverter
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

def parse_pdf_basic(pdf_bytes: bytes) -> str:
    """
    使用 PyMuPDF 进行简单解析 (只对文本型PDF有效)
    """
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def parse_pdf_docling(pdf_bytes: bytes) -> str:
    """
    使用 Docling 解析, 可以获取更丰富结构.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        converter = DocumentConverter()
        result = converter.convert(temp_path)
        # 导出为 Markdown 或纯文本都可以
        text = result.document.export_to_markdown()
        return text
    finally:
        os.remove(temp_path)

def parse_pdf_mistral_ocr(pdf_bytes: bytes) -> str:
    """
    使用 Mistral OCR 对图片/扫描版 PDF 做 OCR.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY not set. Please set env variable.")

    client = Mistral(api_key=MISTRAL_API_KEY)

    # 1) 先把 pdf_bytes 写到临时文件
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        temp_path = tmp.name

    try:
        # 2) 上传文件到 Mistral
        with open(temp_path, "rb") as f:
            uploaded_pdf = client.files.upload(
                file={
                    "file_name": os.path.basename(temp_path),
                    "content": f,
                },
                purpose="ocr"
            )

        # 3) 获取签名URL
        signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

        # 4) 调用 OCR
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )

        # 5) 拼接结果
        extracted_text = ""
        if ocr_response and 'data' in ocr_response and 'pages' in ocr_response['data']:
            pages = ocr_response['data']['pages']
            extracted_text = "\n".join(page['text'] for page in pages if 'text' in page)
        return extracted_text

    finally:
        os.remove(temp_path)


def parse_pdf(pdf_bytes: bytes, parser_type: str) -> str:
    """
    统一入口：根据 parser_type 调用对应解析函数
    parser_type in {"basic", "docling", "mistral_ocr"}
    """
    if parser_type == "basic":
        return parse_pdf_basic(pdf_bytes)
    elif parser_type == "docling":
        return parse_pdf_docling(pdf_bytes)
    elif parser_type == "mistral_ocr":
        return parse_pdf_mistral_ocr(pdf_bytes)
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
