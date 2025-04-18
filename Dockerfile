# Dockerfile

FROM python:3.10-slim

# Don’t write .pyc and force unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install gcc (for any native deps) and clean up
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy & install requirements, then gunicorn
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install gunicorn

# Copy the rest of your code
COPY . /app 

# Make sure /app is readable/executable
RUN chmod -R 755 /app

EXPOSE 5000

# Launch your factory app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
