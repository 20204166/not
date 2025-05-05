# Use a slim Python 3.10 base image (suitable for ML apps)
#
FROM python:3.10-slim-bookworm


# Don’t generate .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install OS dependencies for audio and science libs (gcc, PortAudio, etc.)
# Install OS dependencies for audio, science libs, and C extensions (GCC, Cairo, etc.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      pkg-config \
      libcairo2-dev \
      python3-dev \
      gcc \
      curl \
      libasound2 libasound2-dev \
      libportaudio2 libportaudiocpp0 portaudio19-dev \
 && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies (incl. Flask, TensorFlow, Gunicorn, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy the application code into the image
COPY . /app

# Set Flask environment variables (can be overridden in Compose/K8s)
ENV FLASK_APP=run.py \
    FLASK_ENV=production

# Expose the Flask/Gunicorn port
EXPOSE 5000

# Add a health check to ensure the server is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \  
  CMD curl --fail http://localhost:5000/health || exit 1

# Launch the Flask app using Gunicorn (binds to 0.0.0.0 for external access)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
