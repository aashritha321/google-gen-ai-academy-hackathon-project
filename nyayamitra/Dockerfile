# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set environment variables
# PYTHONUNBUFFERED keeps python from buffering stdout/stderr (essential for Cloud Run logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set the working directory
WORKDIR /app

# Install system packages required for FAISS and standard builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8080
EXPOSE 8080

# Streamlit binding configurations for Google Cloud Run
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080", "--server.headless=true"]
