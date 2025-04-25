FROM python:3.8-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python libs
RUN pip install --no-cache-dir \
        flask gunicorn bitsandbytes transformers accelerate

# Create program dir and copy files
WORKDIR /opt/program
COPY predictor.py serve nginx.conf wsgi.py ./

# Make 'serve' executable
RUN chmod +x serve

# Expose port and start
EXPOSE 8080
ENTRYPOINT ["./serve"]
