# Dockerfile

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory.
WORKDIR /app

# Install OS‑level dependencies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies, including gunicorn.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install gunicorn

# Copy the rest of the codebase.
COPY . /app

# Expose port 
EXPOSE 5000

# Set Flask config via env (if you like you can also mount these at runtime instead)
ENV FLASK_ENV=production \
    SECRET_KEY="sImyHkLJS1y/0eWPkcZxJSmqrcR5nUCJOAAxXoMKbrs="

# Use Gunicorn to run your app factory
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
