# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files first for better layer caching
COPY requirements.txt requirements-lightweight.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir markdown

# Copy the rest of the application
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Create .env file if not exists
RUN if [ ! -f .env ]; then cp .env.example .env; fi

# Set environment variables for lightweight model
ENV USE_SMALL_MODEL=true
ENV MINDNEST_MODE=lightweight

# Expose port
EXPOSE 8000

# Run the application in lightweight mode
CMD ["python", "run_direct.py", "--lightweight", "--lightweight-model"] 