from datetime import datetime, timedelta
import os
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import pdfplumber

# Import parsing strategies
#from docling import PDFExtractor  # Assuming Docling has this interface
from mistralai.ocr import MistralOCR  # Assuming Mistral OCR client
import pypdf  # Basic PDF extraction from Assignment 1

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

RAW_DATA_DIR = "/airflow/data/raw"
PROCESSED_DATA_DIR = "/airflow/data/processed"

# Strategy 1: Basic PyPDF extraction (from Assignment 1)
def parse_with_pypdf(pdf_path, **kwargs):
    """Parse PDF using PyPDF library"""
    output_dir = os.path.join(PROCESSED_DATA_DIR, "pypdf")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
    
    try:
        pdf_content = []
        with open(pdf_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            metadata = pdf_reader.metadata
            
            # Extract year and quarter from filename
            # Assuming filename format: nvidia_YYYY_QN.pdf
            year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                pdf_content.append({
                    'page': page_num + 1,
                    'text': page_text,
                })
        
        # Save the extracted content
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
        print(f"Error parsing {pdf_path} with PyPDF: {e}")
        raise

# Strategy 2: Docling parsing
"""
def parse_with_docling(pdf_path, **kwargs):
    '''Parse PDF using Docling library'''
    output_dir = os.path.join(PROCESSED_DATA_DIR, "docling")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
    
    try:
        # Extract year and quarter from filename
        year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')
        
        # Use Docling to extract text
        extractor = PDFExtractor(pdf_path)
        document = extractor.extract()
        
        # Format the extracted content
        pdf_content = []
        for idx, page in enumerate(document.pages):
            pdf_content.append({
                'page': idx + 1,
                'text': page.text,
                'tables': [table.to_dict() for table in page.tables],
            })
        
        # Save the extracted content
        result = {
            'source': pdf_path,
            'year_quarter': year_quarter,
            'parser': 'docling',
            'metadata': document.metadata,
            'content': pdf_content
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return output_path
    except Exception as e:
        print(f"Error parsing {pdf_path} with Docling: {e}")
        raise
"""
def parse_with_docling(pdf_path, **kwargs):
    """Parse PDF using pdfplumber (as Docling strategy replacement)"""
    output_dir = os.path.join(PROCESSED_DATA_DIR, "docling")  # 沿用原本 docling 資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))

    try:
        # Extract year and quarter from filename
        year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')

        pdf_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                tables = [table.extract() for table in page.extract_tables()]
                
                pdf_content.append({
                    'page': page_num + 1,
                    'text': page_text,
                    'tables': tables
                })

        result = {
            'source': pdf_path,
            'year_quarter': year_quarter,
            'parser': 'pdfplumber',
            'content': pdf_content
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return output_path

    except Exception as e:
        print(f"Error parsing {pdf_path} with pdfplumber: {e}")
        raise
# Strategy 3: Mistral OCR
def parse_with_mistral_ocr(pdf_path, **kwargs):
    """Parse PDF using Mistral OCR"""
    output_dir = os.path.join(PROCESSED_DATA_DIR, "mistral_ocr")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(pdf_path)
    output_path = os.path.join(output_dir, filename.replace('.pdf', '.json'))
    
    try:
        # Extract year and quarter from filename
        year_quarter = filename.replace('nvidia_', '').replace('.pdf', '')
        
        # Initialize Mistral OCR client
        # This is a placeholder - you'd need to implement with the actual Mistral OCR API
        ocr_client = MistralOCR(api_key=Variable.get("MISTRAL_API_KEY"))
        
        with open(pdf_path, 'rb') as f:
            file_content = f.read()
            
        # Process the document with Mistral OCR
        ocr_result = ocr_client.process_document(
            file_content=file_content,
            file_name=filename
        )
        
        # Format the extracted content
        pdf_content = []
        for idx, page in enumerate(ocr_result.pages):
            pdf_content.append({
                'page': idx + 1,
                'text': page.text,
                'tables': page.tables,
                'form_fields': page.form_fields
            })
        
        # Save the extracted content
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
        print(f"Error parsing {pdf_path} with Mistral OCR: {e}")
        raise

def get_pdf_files():
    """Get all PDF files in the raw data directory"""
    pdf_files = []
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith('.pdf') and file.startswith('nvidia_'):
            pdf_files.append(os.path.join(RAW_DATA_DIR, file))
    return pdf_files

with DAG(
    'pdf_parsing',
    default_args=default_args,
    description='Parse NVIDIA quarterly report PDFs using multiple strategies',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 3, 1),
    catchup=False,
    tags=['nvidia', 'rag', 'pdf'],
) as dag:
    
    # Task to get the list of PDF files to process
    get_pdfs_task = PythonOperator(
        task_id='get_pdf_files',
        python_callable=get_pdf_files,
    )
    
    # Create a task group for each PDF parsing strategy
    def create_parsing_tasks(pdf_paths):
        parsing_tasks = []
        
        for pdf_path in pdf_paths:
            # Use the basename of the PDF as part of the task ID
            pdf_basename = os.path.basename(pdf_path).replace('.pdf', '')
            
            with TaskGroup(group_id=f'parse_{pdf_basename}') as pdf_task_group:
                # Task for PyPDF parsing
                pypdf_task = PythonOperator(
                    task_id=f'pypdf_{pdf_basename}',
                    python_callable=parse_with_pypdf,
                    op_kwargs={'pdf_path': pdf_path},
                )
                
                # Task for Docling parsing
                docling_task = PythonOperator(
                    task_id=f'docling_{pdf_basename}',
                    python_callable=parse_with_docling,
                    op_kwargs={'pdf_path': pdf_path},
                )
                
                # Task for Mistral OCR parsing
                mistral_task = PythonOperator(
                    task_id=f'mistral_{pdf_basename}',
                    python_callable=parse_with_mistral_ocr,
                    op_kwargs={'pdf_path': pdf_path},
                )
                
                # Define task dependencies within the group (parallel execution)
                # No dependencies needed as they run in parallel
            
            parsing_tasks.append(pdf_task_group)
        
        return parsing_tasks
    
    # Task dependencies
    parsing_task_groups = create_parsing_tasks(get_pdfs_task.output)
    get_pdfs_task >> parsing_task_groups