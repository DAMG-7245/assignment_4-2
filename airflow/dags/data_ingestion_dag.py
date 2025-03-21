from airflow import DAG
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
from nvidia_pdf_scraper import fetch_all_pdf_links, download_and_upload_pdfs_to_s3

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="nvidia_quarterly_pdf_scraper_dag",
    default_args=default_args,
    schedule=timedelta(days=7),
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=["nvidia", "scraper"],
) as dag:

    start = EmptyOperator(task_id="start")

    @task
    def get_all_links():
        return fetch_all_pdf_links()

    @task
    def download_all_pdfs(pdf_objs):
        # ✅ You can configure these as needed
        download_and_upload_pdfs_to_s3(
            pdf_objs,
            aws_conn_id="aws_default",
            s3_bucket="bigdata-project4-storage",
            s3_prefix="data/raw"
        )

    links = get_all_links()
    download = download_all_pdfs(links)

    start >> links >> download
