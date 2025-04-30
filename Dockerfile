FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1) Install OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip git build-essential nginx ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python packages
RUN pip3 install --no-cache-dir \
     -f https://download.pytorch.org/whl/cu118/torch_stable.html \
     "torch==2.1.0+cu118" \
     "torchvision==0.16.0+cu118" \
     "torchaudio==2.1.0+cu118" \
     "numpy<2" \
     "transformers==4.51.0" \
     "bitsandbytes>=0.41.0" \
     "accelerate>=0.26.0" \
     "flask" \
     "gunicorn==23.0.0" \
     "pillow==9.5.0"

# 3) Copy your inference package
WORKDIR /opt/program/NER
COPY NER/ . 

# 4) Remove CRLF line endings and make serve executable
#    (you can also install dos2unix and run `dos2unix serve`–
#     but a simple sed works without extra packages)
RUN sed -i 's/\r$//' serve \
    && chmod +x serve

# 5) Expose SageMaker’s expected port
EXPOSE 8080

# 6) Start the serve script exactly per docs
ENTRYPOINT ["./serve"]
