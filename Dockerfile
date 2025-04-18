# Dockerfile

FROM python:3.10-slim

# Don’t write .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc curl \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python deps (including gunicorn in requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy the rest of your code
COPY . /app

# Flask env
ENV FLASK_APP=run.py \
    FLASK_ENV=production

# Expose port
EXPOSE 5000

# Let Docker / K8s know how to check liveness
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl --fail http://localhost:5000/health || exit 1

# Start your app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
