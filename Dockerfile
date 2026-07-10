FROM python:3.12-slim

WORKDIR /app
ENV INFERENCE_MODE=gateway

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY gpu_worker/*.py ./gpu_worker/
COPY gpu_worker/pipeline ./gpu_worker/pipeline
COPY gpu_worker/models ./gpu_worker/models
COPY scripts/infer/face_crop.py ./scripts/infer/face_crop.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
