FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY challenge ./challenge
COPY data ./data

ENV PYTHONUNBUFFERED=1

CMD exec uvicorn challenge.api:app --host 0.0.0.0 --port ${PORT:-8080}