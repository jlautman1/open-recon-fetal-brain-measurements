# ------------------------------------------------------------
#  Fetal-pipeline GPU image (Python 3.8 + CUDA 11.0.3)
# ------------------------------------------------------------
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
# Tell pip to prefer binaries where possible
ENV PIP_PREFER_BINARY=1

# 1) Install system deps + Python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        python3.8 \
        python3.8-venv \
        python3.8-dev \
        python3-pip \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2) Make python3.8 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 3) Upgrade pip, setuptools, wheel to latest
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4) Install SimpleITK *only* from a prebuilt wheel
RUN python -m pip install --no-cache-dir --only-binary=SimpleITK SimpleITK==2.3.1

# 5) Copy requirements, strip out any SimpleITK line, and install the rest
COPY requirements.txt .
RUN grep -v '^SimpleITK' requirements.txt > reqs.txt && \
    python -m pip install --no-cache-dir -r reqs.txt

# 6) Copy your code
COPY . .

# 7) Set PYTHONPATH for your SubSegmentation folder
ENV PYTHONPATH="${PYTHONPATH}:/workspace/Code/FetalMeasurements-master/SubSegmentation"

# 8) Default to bash so you can exec in
CMD ["/bin/bash"]
