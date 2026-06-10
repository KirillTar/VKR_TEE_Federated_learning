# Flower FL server/client — SCONE-compatible base image
# Based on python:3.11-slim-bookworm (glibc 2.36, compatible with SCONE)
# For SCONE HW mode: replace base image with registry.scontain.com/scone-production/...

FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache)
COPY requirements/scone.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY core/ core/
COPY scone/ scone/
COPY experiments/ experiments/

# SCONE-specific: limit threads (enclave memory constraints)
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Default: run server.
# Use python3.11 (not bare "python") because after sconification on debian:12-slim base
# the /usr/local/bin/python symlink is dropped — only python3.11 remains on PATH.
CMD ["python3.11", "scone/server.py", \
     "--n_clients", "20", \
     "--m_assumed", "6", \
     "--epsilon", "4.0", \
     "--n_rounds", "200"]
