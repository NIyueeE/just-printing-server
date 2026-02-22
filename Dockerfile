ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 监听端口改为 3001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001"]
