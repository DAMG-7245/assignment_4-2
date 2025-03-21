import os
import io
import boto3
from typing import List
from dotenv import load_dotenv
load_dotenv()
# 你可以把密钥放在环境变量 / AWS CLI 配置 / IAM Role
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # 或其他地区
S3_BUCKET = os.getenv("S3_BUCKET", "bigdata-project4-storage")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def list_pdfs_in_s3(prefix: str = "data/raw") -> List[str]:
    """
    列出 S3 中 prefix 下的所有 PDF 文件 (Key).
    返回 key 的列表，例如: ["data/raw/nvidia_2020_Q1.pdf", ...]
    """
    pdf_keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    pdf_keys.append(key)
    return pdf_keys

def get_pdf_content_from_s3(key: str) -> bytes:
    """
    从 S3 下载指定 key 的 PDF 文件, 返回 bytes.
    """
    response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
    pdf_bytes = response["Body"].read()
    return pdf_bytes
