# Method B EKS: thin RabbitMQ consumer → GPU gateway (no local torch/ffmpeg/viz).
FROM python:3.12-slim

WORKDIR /app
ENV AI_CONSUMER_ONLY=1
ENV INFERENCE_MODE=gateway

COPY requirements-consumer.txt .
RUN pip install --no-cache-dir -r requirements-consumer.txt

COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
