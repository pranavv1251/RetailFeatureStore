# docker/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OpenMP runtime for LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) deps first
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# 2) code & artifacts
# (note the trailing slashes on destinations)
COPY serving/        /app/serving/
COPY serving/models/ /app/serving/models/
COPY jobs/           /app/jobs/

# 3) optional: keep a place for logs
RUN mkdir -p /app/data/preds

# default runtime env (talk to a redis service named "redis")
ENV FEATURE_BACKEND=redis \
    REDIS_URL=redis://redis:6379/0

EXPOSE 8000
CMD ["sh","-c","uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]