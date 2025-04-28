FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip git build-essential nginx ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
     -f https://download.pytorch.org/whl/cu118/torch_stable.html \
     torch==2.1.0+cu118 \
     torchvision==0.16.0+cu118 \
     torchaudio==2.1.0+cu118 \
     numpy<2 \
     transformers==4.51.0 \
     bitsandbytes>=0.41.0 \
     accelerate>=0.26.0 \
     fastapi==0.95.1 \
     uvicorn[standard]==0.23.2 \
     pillow==9.5.0

COPY . /opt/program
WORKDIR /opt/program
RUN chmod +x serve

EXPOSE 8080
ENTRYPOINT ["./serve"]
