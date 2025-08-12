FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenMP for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --upgrade pip && pip install -r /code/requirements.txt

# copy app + (optionally) model artifacts
COPY serving /code/serving
COPY jobs /code/jobs
# if you bake the model into the repo/image, keep this:
COPY serving/models /code/serving/models

ENV FEATURE_BACKEND=redis
EXPOSE 7860
# Spaces sets PORT; default to 7860 locally
CMD ["sh","-c","uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
