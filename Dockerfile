# Dockerfile
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory.
WORKDIR /app

# Install OS-level dependencies (e.g., gcc).
RUN apt-get update && apt-get install -y gcc

# Copy and install Python dependencies.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire codebase.
COPY . /app

# Set Flask environment variables.
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Expose port 5000.
EXPOSE 5000

# Run Gunicorn to serve the app.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (gcc for any packages that need compiling),
# then clean up apt caches to keep the image small
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them, including gunicorn
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install gunicorn \
 && pip install -r requirements.txt

# Copy your application code
COPY . /app

# Tell Flask and Gunicorn which app to run
ENV FLASK_APP=run.py \
    FLASK_ENV=production

# Expose the port Gunicorn will listen on
EXPOSE 5000

# Launch with 4 workers (you can adjust) via Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "run:app"]
