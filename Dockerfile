FROM ghcr.io/cheshire-cat-ai/core:latest

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt


