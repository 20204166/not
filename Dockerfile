# Use a slim Python 3.10 base image (suitable for ML apps)
FROM python:3.10-slim-bookworm

# Don’t generate .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# ---- Build-time args for Kaggle CLI auth ----
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# ---- 1) Install OS deps (unzip for Kaggle ZIPs) and CPU PyTorch ----
RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# ---- 2) Install your Python requirements ----
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && rm -rf /root/.cache/pip

# ---- 3) Single RUN: install Kaggle CLI, auth, download & unzip dataset ----
# ---- 3) Single RUN: install Kaggle CLI, auth, create target dir, download & unzip ----
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
 

# ---- 4) Copy your application code ----
COPY . /app

# ---- 5) Flask & Gunicorn setup ----
ENV FLASK_APP=run.py \
    FLASK_ENV=production

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl --fail http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
