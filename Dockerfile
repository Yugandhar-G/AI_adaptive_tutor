# Multi-stage Dockerfile for Adaptive Learning System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify curl is installed (for health checks)
RUN curl --version

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model (required for knowledge graph)
RUN python -m spacy download en_core_web_sm

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Expose ports
# 8000 for FastAPI backend
# 8501 for Streamlit frontend
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["python", "streamlit/backend_api.py"]

