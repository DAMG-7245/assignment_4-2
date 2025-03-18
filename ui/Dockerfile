FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ui/ .
COPY common/ ./common/

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]