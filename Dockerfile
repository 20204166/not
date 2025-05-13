
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# 2) Install Python requirements (fixed path)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && rm -rf /root/.cache/pip

# 3) Copy in just your Flask app code
COPY app/ /app/app/
COPY run.py /app/

# 4) Install Kaggle CLI, authenticate, download & unpack model
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

RUN pip install --no-cache-dir kaggle \
    && mkdir -p /root/.kaggle \
    && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
         > /root/.kaggle/kaggle.json \
    && chmod 600 /root/.kaggle/kaggle.json \
    \
    # create the exact folder your Flask config points to
    && mkdir -p /app/app/models/saved_model \
    \
    # download the zip right into that folder
    && kaggle datasets download -d bekithembancube/saved-model \
         -p /app/app/models/saved_model \
    \
    && echo ">>> ABOUT TO UNZIP <<<" \
    && ls -l /app/app/models/saved_model \
    \
    # unpack in-place & clean up
    && unzip /app/app/models/saved_model/saved-model.zip \
         -d /app/app/models/saved_model \
    && rm /app/app/models/saved_model/saved-model.zip \
    \
    && echo ">>> AFTER UNZIP <<<" \
    && ls -l /app/app/models/saved_model


# 5) Flask/Gunicorn setup
ENV FLASK_APP=run.py \
APP_CONFIG=app.config.Config

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl --fail http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app", "--workers", "4"]
