ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3001", "--log-level", "info"]
