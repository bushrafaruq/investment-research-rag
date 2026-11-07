FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt anthropic

COPY rag ./rag
RUN useradd --create-home appuser && mkdir -p /app/index && chown appuser /app/index
USER appuser

ENV RAG_INDEX=/app/index/chunks.jsonl RAG_ANSWERER=auto RAG_EMBEDDER=lsa
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
