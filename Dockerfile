FROM python:3.10-slim

# Don’t write .pyc and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for optional audio and TF
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libsndfile1 \
      portaudio19-dev \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy & install requirements, then gunicorn
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy code
COPY . .

EXPOSE 5000

# Use run.py’s app object as Gunicorn WSGI entrypoint
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
