FROM python:3.12-slim

WORKDIR /app
ENV INFERENCE_MODE=gateway

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY gpu_worker/*.py ./gpu_worker/
COPY gpu_worker/pipeline ./gpu_worker/pipeline
COPY gpu_worker/models ./gpu_worker/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
