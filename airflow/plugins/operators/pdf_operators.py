import os
import tempfile
from typing import Dict, List, Any, Optional
import requests
import boto3
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

class PDFDownloadOperator(BaseOperator):
    """
    Custom operator to download PDF files from a URL or S3
    """
    
    @apply_defaults
    def __init__(
        self,
        source: str,
        destination_path: str,
        source_type: str = 'url',  # 'url' or 's3'
        s3_bucket: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.source = source
        self.destination_path = destination_path
        self.source_type = source_type
        self.s3_bucket = s3_bucket
        self.headers = headers or {}
        
    def execute(self, context: Dict[str, Any]) -> str:
        """
        Execute the operator
        
        Args:
            context: Airflow context
            
        Returns:
            Path where the PDF was saved
        """
        self.log.info(f"Downloading PDF from {self.source} to {self.destination_path}")
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(self.destination_path), exist_ok=True)
        
        if self.source_type == 'url':
            self._download_from_url()
        elif self.source_type == 's3':
            self._download_from_s3()
        else:
            raise AirflowException(f"Invalid source_type: {self.source_type}")
        
        self.log.info(f"PDF downloaded successfully to {self.destination_path}")
        return self.destination_path
    
    def _download_from_url(self) -> None:
        """Download PDF from URL"""
        try:
            response = requests.get(self.source, headers=self.headers, stream=True)
            response.raise_for_status()
            
            with open(self.destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            raise AirflowException(f"Error downloading PDF from URL: {str(e)}")
    
    def _download_from_s3(self) -> None:
        """Download PDF from S3"""
        if not self.s3_bucket:
            raise AirflowException("S3 bucket not specified")
        
        try:
            s3_client = boto3.client('s3')
            s3_client.download_file(
                self.s3_bucket,
                self.source,
                self.destination_path
            )
        except Exception as e:
            raise AirflowException(f"Error downloading PDF from S3: {str(e)}")


class PDFProcessingOperator(BaseOperator):
    """
    Custom operator to process PDF files using different parsing methods
    """
    
    @apply_defaults
    def __init__(
        self,
        pdf_path: str,
        parser_type: str,
        output_path: str,
        chunking_strategy: str = 'fixed_size',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pdf_path = pdf_path
        self.parser_type = parser_type
        self.output_path = output_path
        self.chunking_strategy = chunking_strategy
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the operator
        
        Args:
            context: Airflow context
            
        Returns:
            Dictionary with processing results
        """
        self.log.info(f"Processing PDF {self.pdf_path} with parser {self.parser_type}")
        
        # Import modules here to avoid dependency issues
        import fitz  # PyMuPDF
        import json
        import sys
        import importlib.util
        
        # Add API directory to path for importing modules
        sys.path.append('/app')
        
        # Import parsing and chunking modules
        from api.services.parsing import parse_pdf
        from api.services.chunking import chunk_text
        
        try:
            # Read PDF content
            with open(self.pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # Parse PDF
            extracted_text = parse_pdf(pdf_content, self.parser_type)
            
            # Create chunks
            chunks = chunk_text(extracted_text, self.chunking_strategy)
            
            # Save results
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            result = {
                'pdf_path': self.pdf_path,
                'parser_type': self.parser_type,
                'chunking_strategy': self.chunking_strategy,
                'chunks_count': len(chunks),
                'chunks': chunks
            }
            
            with open(self.output_path, 'w') as f:
                json.dump(result, f)
            
            self.log.info(f"PDF processed successfully. {len(chunks)} chunks created.")
            
            return result
            
        except Exception as e:
            self.log.error(f"Error processing PDF: {str(e)}")
            raise AirflowException(f"PDF processing failed: {str(e)}")