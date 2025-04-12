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
