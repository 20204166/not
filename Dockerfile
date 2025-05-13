# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Don’t generate .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install OS deps + upgrade pip
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --user --upgrade pip

# 2) Install PyTorch CPU wheels and your Python requirements under /root/.local
COPY requirements.txt .
RUN pip install --user --no-cache-dir \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip

# 3) Copy your Flask app code
COPY app/ /app/app/
COPY run.py /app/

# 4) Download & unpack your model via Kaggle CLI
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

RUN pip install --user --no-cache-dir kaggle \
 && mkdir -p /root/.kaggle \
 && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
      > /root/.kaggle/kaggle.json \
 && chmod 600 /root/.kaggle/kaggle.json \
 && mkdir -p /app/app/models/saved_model \
 && kaggle datasets download -d bekithembancube/saved-model \
      -p /app/app/models/saved_model \
 && unzip /app/app/models/saved_model/saved-model.zip \
      -d /app/app/models/saved_model \
 && rm /app/app/models/saved_model/saved-model.zip

# ─── Stage 2: final ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# 1) Bring in Python packages from builder
COPY --from=builder /root/.local /root/.local

# 2) Bring in your app code from builder
COPY --from=builder /app /app

# 3) Set PATH and Flask-specific env vars
ENV PATH=/root/.local/bin:$PATH \
    FLASK_APP=run.py \
    APP_CONFIG=app.config.Config

EXPOSE 5000

# 4) Healthcheck against your /healthz endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl --fail http://localhost:5000/healthz || exit 1

# 5) Run under Gunicorn with 4 workers
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app", "--workers", "4"]
