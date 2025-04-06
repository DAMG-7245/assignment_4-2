import os
import io
import boto3
import pandas as pd
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "bigdata-project4-storage")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def get_quarterly_report_mapping_from_s3(
    excel_key: str = "nvidia_reports.xlsx"
) -> Dict[str, str]:
    """
    从S3拉取Excel并返回 { YearQuarter: PDF URL } 字典。
    如果出现错误，则抛出异常。
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=excel_key)
    except Exception as e:
        raise RuntimeError(f"从S3获取对象失败: {e}")

    excel_bytes = response["Body"].read()

    try:
        df = pd.read_excel(io.BytesIO(excel_bytes))
    except Exception as e:
        raise RuntimeError(f"读取Excel文件失败: {e}")

    # 检查必要的列是否存在
    if not {"Year_Quarter", "Link"}.issubset(df.columns):
        raise ValueError("Excel文件缺少必需的 'YearQuarter' 或 'URL' 列")

    # 构造字典，转换成字符串并去除空格
    mapping = df.set_index("Year_Quarter")["Link"].apply(lambda x: str(x).strip()).to_dict()

    return mapping
def get_s3_presigned_url(pdf_key: str, expires_in: int = 3600) -> str:
    """
    生成一个预签名 URL，用于外部服务（例如解析服务）访问 S3 上指定 pdf_key 的对象。
    
    参数：
      - pdf_key: S3 对象的 Key（例如 "data/nvidia_2024Q1.pdf"）
      - expires_in: URL 过期时间（单位秒），默认1小时

    返回：
      - 预签名 URL 字符串，如果失败则返回空字符串。
    """
    try:
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': S3_BUCKET, 'Key': pdf_key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        print(f"生成预签名 URL 失败: {e}")
        return ""
