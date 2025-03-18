from datetime import datetime, timedelta
import os
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

NVIDIA_REPORTS_BASE_URL = "https://investor.nvidia.com/financial-info/quarterly-reports/default.aspx"
DATA_DIR = "/opt/airflow/data/raw"
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']

# This would need to be improved with actual web scraping logic to get the real PDFs
def get_nvidia_quarterly_report_urls():
    # This is a placeholder. In a real implementation, 
    # you would scrape the NVIDIA investor relations page
    # to get the actual PDF URLs for the past 5 years
    urls = []
    current_year = datetime.now().year
    
    # Get reports for the past 5 years
    for year in range(current_year - 5, current_year):
        for quarter in QUARTERS:
            # This is a placeholder URL pattern - would need to be replaced with real URLs
            url = f"https://investor.nvidia.com/financials/{year}/{quarter}/NVIDIA-{year}-{quarter}-Report.pdf"
            urls.append((url, f"{year}_{quarter}"))
    
    return urls

def download_pdf(url, filename, **kwargs):
    """Download a PDF from a URL and save it to the data directory"""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return output_path
        
    try:
        # In a real implementation, you'd handle authentication, redirects, etc.
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded {url} to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise

with DAG(
    'nvidia_quarterly_reports_ingestion',
    default_args=default_args,
    description='Download NVIDIA quarterly reports for the past 5 years',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=['nvidia', 'rag'],
) as dag:
    
    # Task to get the list of PDF URLs to download
    get_urls_task = PythonOperator(
        task_id='get_pdf_urls',
        python_callable=get_nvidia_quarterly_report_urls,
    )
    
    # Dynamically create download tasks based on the URLs returned
    def create_download_tasks(urls_with_names):
        download_tasks = []
        for url, name in urls_with_names:
            task = PythonOperator(
                task_id=f'download_{name}',
                python_callable=download_pdf,
                op_kwargs={'url': url, 'filename': f'nvidia_{name}.pdf'},
            )
            download_tasks.append(task)
        return download_tasks
    
    # Task dependencies
    download_tasks = create_download_tasks(get_urls_task.output)
    get_urls_task >> download_tasks