# Dockerfile

FROM python:3.10-slim

# Don’t write .pyc and unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

WORKDIR /app

# Install OS deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps (including gunicorn)
COPY requirements.txt .
# make sure requirements.txt now contains a line: gunicorn
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy your app code
COPY . .

# Expose and run
EXPOSE 5000
CMD ["gunicorn", "--workers", "3", "--bind", "0.0.0.0:5000", "run:app"]
