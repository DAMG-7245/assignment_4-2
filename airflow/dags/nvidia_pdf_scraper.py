import os
import requests
import boto3
from bs4 import BeautifulSoup
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError
from airflow.providers.amazon.aws.hooks.s3 import S3Hook


BASE_NEWS_URL = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-{quarter}-quarter-and-fiscal-{year}"
QUARTERS = ["first", "second", "third", "fourth"]
YEARS_BACK = 5

# AWS S3 Configuration
S3_BUCKET = "bigdata-project4-storage"
S3_PREFIX = "data/raw"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize boto3 S3 client
s3_client = boto3.client("s3")

def get_pdf_links_from_article(url, year, quarter_name):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf"):
                if href.startswith("/"):
                    href = "https://nvidianews.nvidia.com" + href
                filename = f"nvidia_{year}_{quarter_name.upper()}.pdf"
                pdf_links.append({"url": href, "filename": filename})
        return pdf_links

    except Exception as e:
        print(f"[ERROR] Failed to parse article for {year} {quarter_name.title()} Quarter: {e}")
        return []

def fetch_all_pdf_links():
    current_year = datetime.now().year
    all_links = []

    for year in range(current_year - YEARS_BACK, current_year):
        for quarter_name in QUARTERS:
            news_url = BASE_NEWS_URL.format(quarter=quarter_name, year=year)
            print(f"[INFO] Checking news article: {news_url}")
            pdf_links = get_pdf_links_from_article(news_url, year, quarter_name)
            all_links.extend(pdf_links)
    return all_links

def download_and_upload_pdfs_to_s3(pdf_links, aws_conn_id="aws_default", s3_bucket="mybucketname", s3_prefix="data/raw"):
    # Get S3Hook and boto3 client from Airflow connection
    s3_hook = S3Hook(aws_conn_id=aws_conn_id)
    s3_client = s3_hook.get_conn()

    for item in pdf_links:
        url = item["url"]
        filename = item["filename"]
        s3_key = f"{s3_prefix}/{filename}"

        try:
            print(f"[DOWNLOADING] {filename} from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            print(f"[UPLOADING] to s3://{s3_bucket}/{s3_key}")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=response.content,
                ContentType="application/pdf"
            )

            print(f"[SUCCESS] Uploaded {filename} to S3")

        except Exception as e:
            print(f"[ERROR] Failed to download or upload {filename}: {e}")

if __name__ == "__main__":
    pdf_links = fetch_all_pdf_links()
    print(f"\n✅ Found {len(pdf_links)} PDF links.")
    download_and_upload_pdfs_to_s3(pdf_links)
