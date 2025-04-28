# ── Use NVIDIA CUDA base (matches SageMaker G5) ────────────────────
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# ── Non-interactive installs ────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip git build-essential && \
    rm -rf /var/lib/apt/lists/*

# ── Python libs (all pinned & quoted) ──────────────────────────
RUN pip3 install --no-cache-dir \
     -f https://download.pytorch.org/whl/cu118/torch_stable.html \
     "torch==2.1.0+cu118" \
     "torchvision==0.16.0+cu118" \
     "torchaudio==2.1.0+cu118" \
     "numpy<2" \
     "transformers==4.51.0" \
     "bitsandbytes>=0.41.0" \   
     "accelerate>=0.26.0" \         
     "fastapi==0.95.1" \
     "uvicorn[standard]==0.23.2" \
     "pillow==9.5.0"

# ── Copy your app code ──────────────────────────────────────────────
WORKDIR /opt/ml/model/code
COPY app.py       .
COPY inference.py .

# ── Expose & launch ────────────────────────────────────────────────
# <<< drop the CUDA entrypoint, run uvicorn directly >>>
ENTRYPOINT []
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]