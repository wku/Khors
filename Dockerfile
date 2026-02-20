FROM python:3.11-slim

# Install system dependencies
# git is needed for gitpython (even if we don't commit, we might read logs or use git tools)
# curl for healthchecks or downloading
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory structure
RUN mkdir -p /app/data/state /app/data/logs /app/data/memory /app/data/archive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV REPO_DIR=/app
ENV DRIVE_ROOT=/app/data

# Default command (can be overridden in docker-compose)
CMD ["python", "launcher.py"]
