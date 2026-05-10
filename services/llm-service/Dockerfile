FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and pin setuptools for stability
RUN pip install --no-cache-dir --upgrade pip "setuptools<58.0.0"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user and switch to it
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Railway provides PORT env var
ENV PORT=8002
EXPOSE ${PORT}

# Run using the module path since main.py is in the app/ directory
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
