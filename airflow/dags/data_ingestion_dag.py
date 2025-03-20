from datetime import datetime, timedelta
import os
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

DATA_DIR = "/airflow/data/raw"
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']

@task
def get_nvidia_pdf_urls():
    urls = []
    current_year = datetime.now().year
    for year in range(current_year - 5, current_year):
        for quarter in QUARTERS:
            url = f"https://investor.nvidia.com/financials/{year}/{quarter}/NVIDIA-{year}-{quarter}-Report.pdf"
            urls.append({"url": url, "filename": f"nvidia_{year}_{quarter}.pdf"})
    return urls

@task
def download_pdf(obj):
    url = obj["url"]
    filename = obj["filename"]
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(output_path):
        print(f"{output_path} already exists. Skipping.")
        return output_path

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

with DAG(
    dag_id='nvidia_quarterly_reports_ingestion',
    default_args=default_args,
    description='Download NVIDIA quarterly reports using dynamic task mapping',
    schedule=timedelta(days=7),  # ✅ updated from `schedule_interval` → `schedule`
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=['nvidia', 'rag'],
) as dag:
    
    start = EmptyOperator(task_id="start")

    urls = get_nvidia_pdf_urls()
    download_tasks = download_pdf.expand(obj=urls)

    start >> urls >> download_tasks
