# 0) build-time args for Kaggle creds
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# 1) Base image
FROM python:3.10-slim-bookworm

# 2) Prevent .pyc files, unbuffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Set working dir
WORKDIR /app

# 4) Install OS deps + CPU-only PyTorch
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# 5) Python requirements
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
 && rm -rf /root/.cache/pip

# 6) Copy your Flask app into the image
COPY . /app

# 7) Install Kaggle CLI, auth, download & unpack your model
RUN pip install --no-cache-dir kaggle \
 && mkdir -p /root/.kaggle \
 && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
      > /root/.kaggle/kaggle.json \
 && chmod 600 /root/.kaggle/kaggle.json \
 && mkdir -p /app/models/saved_model \
 && kaggle datasets download -d bekithembancube/saved-model \
      -p /app/models/saved_model \
 && unzip /app/models/saved_model/saved-model.zip \
      -d /app/models/saved_model \
 && rm /app/models/saved_model/saved-model.zip

# 8) Flask/Gunicorn configuration
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl --fail http://localhost:5000/health || exit 1

# 9) Launch
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
