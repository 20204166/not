FROM python:3.10-slim

# Prevent .pyc and enable unbuffered logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=run.py \
    FLASK_ENV=production

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps (including gunicorn)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy app code
COPY . /app

# Expose port
EXPOSE 5000

# Kick off via gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
