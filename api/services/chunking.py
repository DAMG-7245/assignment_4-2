# api/services/chunking_service.py

import re
from typing import List

def chunk_text_fixed_size(text: str, chunk_size: int = 500) -> List[str]:
    """
    按固定长度 chunk_size 进行切分.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def chunk_text_by_paragraph(text: str) -> List[str]:
    """
    按段落切分 (以空行或换行符为界).
    """
    # 先把 \r\n 转成 \n
    paragraphs = text.replace('\r\n', '\n').split('\n\n')
    # 再将换行过多的段落做 strip
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def chunk_text_by_sentence(text: str) -> List[str]:
    """
    简易按句子切分, 遇到句号/问号/感叹号后断开.
    """
    # 这里用正则 split, 仅作示例
    sentence_end_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_end_pattern, text)
    # strip 并过滤空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def chunk_text(text: str, strategy: str = "fixed") -> List[str]:
    """
    根据指定 strategy 进行文本切分.
    strategy in {"fixed", "paragraph", "sentence"}
    """
    if strategy == "fixed":
        return chunk_text_fixed_size(text, chunk_size=500)
    elif strategy == "paragraph":
        return chunk_text_by_paragraph(text)
    elif strategy == "sentence":
        return chunk_text_by_sentence(text)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
