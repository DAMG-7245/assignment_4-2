from datetime import datetime, timedelta
import os
import json
from airflow.decorators import dag, task
from airflow.models import Variable
import pypdf
import pdfplumber

# ------------------------------
# DAG Config
# ------------------------------
RAW_DATA_DIR = "/airflow/data/raw"
PROCESSED_DATA_DIR = "/airflow/data/processed"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ------------------------------
# Tasks
# ------------------------------
@task
def get_pdf_files():
    pdf_files = []
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith('.pdf') and file.startswith('nvidia_'):
            pdf_files.append(os.path.join(RAW_DATA_DIR, file))
    return pdf_files


@task
def parse_with_pypdf(pdf_path):
    output_dir = os.path.join(PROCESSED_DATA_DIR, "pypdf")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))

    try:
        pdf_content = []
        with open(pdf_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            metadata = reader.metadata
            year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                pdf_content.append({'page': page_num + 1, 'text': text})

        result = {
            'source': pdf_path,
            'year_quarter': year_quarter,
            'parser': 'pypdf',
            'metadata': {k: str(v) for k, v in metadata.items()} if metadata else {},
            'content': pdf_content
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        return output_path
    except Exception as e:
        print(f"PyPDF Error on {pdf_path}: {e}")
        raise


@task
def parse_with_docling(pdf_path):
    output_dir = os.path.join(PROCESSED_DATA_DIR, "docling")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))

    try:
        year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')
        pdf_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                tables = [table for table in page.extract_tables()]
                pdf_content.append({'page': page_num + 1, 'text': text, 'tables': tables})

        result = {
            'source': pdf_path,
            'year_quarter': year_quarter,
            'parser': 'docling',
            'content': pdf_content
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        return output_path
    except Exception as e:
        print(f"Docling-mock Error on {pdf_path}: {e}")
        raise


@task
def parse_with_mistral_ocr(pdf_path):
    output_dir = os.path.join(PROCESSED_DATA_DIR, "mistral_ocr")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))

    try:
        year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')

        class Page:
            def __init__(self):
                self.text = "[MOCK OCR TEXT]"
                self.tables = []
                self.form_fields = {}

        class OCRResult:
            def __init__(self):
                self.pages = [Page()]
                self.metadata = {'mock': True}

        ocr_result = OCRResult()
        pdf_content = []
        for idx, page in enumerate(ocr_result.pages):
            pdf_content.append({
                'page': idx + 1,
                'text': page.text,
                'tables': page.tables,
                'form_fields': page.form_fields
            })

        result = {
            'source': pdf_path,
            'year_quarter': year_quarter,
            'parser': 'mistral_ocr',
            'metadata': ocr_result.metadata,
            'content': pdf_content
        }
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        return output_path
    except Exception as e:
        print(f"Mistral OCR mock error on {pdf_path}: {e}")
        raise


# ------------------------------
# DAG Definition
# ------------------------------
@dag(
    dag_id='pdf_parsing_taskflow',
    default_args=default_args,
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=['nvidia', 'rag', 'taskflow']
)
def parsing_dag_taskflow():
    pdf_files = get_pdf_files()
    parse_with_pypdf.expand(pdf_path=pdf_files)
    parse_with_docling.expand(pdf_path=pdf_files)
    parse_with_mistral_ocr.expand(pdf_path=pdf_files)


dag_instance = parsing_dag_taskflow()
