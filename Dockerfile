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
# In your Dockerfile, replace the existing apt-get step with this:
# 1) Upgrade pip and install CPU-only PyTorch
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.5.1+cpu \
      torchvision==0.20.1+cpu \
      --index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest of your requirements without caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3) Clean up any remaining cache
RUN rm -rf /root/.cache/pip
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
