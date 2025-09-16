# ------------------------------------------------------------
#  OpenRecon Fetal Brain Segmentation Pipeline
#  Based on working fetal-brain-measurement Dockerfile
# ------------------------------------------------------------
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# === OpenRecon UI Label ===
LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="eyJnZW5lcmFsIjogeyJuYW1lIjogeyJlbiI6ICJGZXRhbCBCcmFpbiBNZWFzdXJlbWVudHMifSwgInZlcnNpb24iOiAiMS4wLjAiLCAidmVuZG9yIjogIlNpZW1lbnNIZWFsdGhpbmVlcnNBRyIsICJpbmZvcm1hdGlvbiI6IHsiZW4iOiAiRGVtbyBvZiBmZXRhbCBicmFpbiBtZWFzdXJlbWVudHMgdjEhIn0sICJpZCI6ICJQeXRob25NUkRpMmkiLCAicmVndWxhdG9yeV9pbmZvcm1hdGlvbiI6IHsiZGV2aWNlX3RyYWRlX25hbWUiOiAiUHl0aG9uTVJEaTJpIiwgInByb2R1Y3Rpb25faWRlbnRpZmllciI6ICIxLjAuMCIsICJtYW51ZmFjdHVyZXJfYWRkcmVzcyI6ICJUZWwgQXZpdiwgSXNyYWVsIiwgIm1hZGVfaW4iOiAiSUwiLCAibWFudWZhY3R1cmVfZGF0ZSI6ICIyMDI1LzA1LzA4IiwgIm1hdGVyaWFsX251bWJlciI6ICJQeXRob25NUkRfaTJpXzEuMC4wIiwgImd0aW4iOiAiMDA4NjAwMDAxNzEyMTIiLCAidWRpIjogIigwMSkwMDg2MDAwMDE3MTIxMigyMSkxLjMuMCIsICJzYWZldHlfYWR2aWNlcyI6ICIiLCAic3BlY2lhbF9vcGVyYXRpbmdfaW5zdHJ1Y3Rpb25zIjogIlJ1biBmZXRhbCBicmFpbiBtZWFzdXJlbWVudHMgb24gcmVjb25zdHJ1Y3RlZCBpbWFnZXMgYW5kIHJldHVybiBST0kgb3ZlcmxheXMuIiwgImFkZGl0aW9uYWxfcmVsZXZhbnRfaW5mb3JtYXRpb24iOiAiIn19LCAicmVjb25zdHJ1Y3Rpb24iOiB7InRyYW5zZmVyX3Byb3RvY29sIjogeyJwcm90b2NvbCI6ICJJU01STVJEIiwgInZlcnNpb24iOiAiMS40LjEifSwgInBvcnQiOiA5MDAyLCAiZW1pdHRlciI6ICJpbWFnZSIsICJpbmplY3RvciI6ICJpbWFnZSIsICJjYW5fcHJvY2Vzc19hZGp1c3RtZW50X2RhdGEiOiBmYWxzZSwgImNhbl91c2VfZ3B1IjogdHJ1ZSwgIm1pbl9jb3VudF9yZXF1aXJlZF9ncHVzIjogMCwgIm1pbl9yZXF1aXJlZF9ncHVfbWVtb3J5IjogMjA0OCwgIm1pbl9yZXF1aXJlZF9tZW1vcnkiOiA0MDk2LCAibWluX2NvdW50X3JlcXVpcmVkX2NwdV9jb3JlcyI6IDEsICJjb250ZW50X3F1YWxpZmljYXRpb25fdHlwZSI6ICJSRVNFQVJDSCJ9LCAicGFyYW1ldGVycyI6IFt7ImlkIjogImNvbmZpZyIsICJ0eXBlIjogImNob2ljZSIsICJsYWJlbCI6IHsiZW4iOiAiY29uZmlnIn0sICJ2YWx1ZXMiOiBbeyJpZCI6ICJvcGVucmVjb24iLCAibmFtZSI6IHsiZW4iOiAiRmV0YWwgQnJhaW4gTWVhc3VyZW1lbnRzIn19LCB7ImlkIjogImludmVydGNvbnRyYXN0IiwgIm5hbWUiOiB7ImVuIjogImludmVydGNvbnRyYXN0In19XSwgImRlZmF1bHQiOiAib3BlbnJlY29uIiwgImluZm9ybWF0aW9uIjogeyJlbiI6ICJEZWZpbmUgdGhlIGNvbmZpZyB0byBiZSBleGVjdXRlZCBieSBNUkQgc2VydmVyIn19LCB7ImlkIjogImN1c3RvbWNvbmZpZyIsICJsYWJlbCI6IHsiZW4iOiAiQ3VzdG9tIENvbmZpZyJ9LCAidHlwZSI6ICJzdHJpbmciLCAiaW5mb3JtYXRpb24iOiB7ImVuIjogIkN1c3RvbSBjb25maWcgZmlsZSBub3QgbGlzdGVkIGluIGRyb3AtZG93biBtZW51In0sICJkZWZhdWx0IjogIiJ9LCB7ImlkIjogImZyZWV0ZXh0IiwgImxhYmVsIjogeyJlbiI6ICJmcmVldGV4dCJ9LCAidHlwZSI6ICJzdHJpbmciLCAiaW5mb3JtYXRpb24iOiB7ImVuIjogIkZyZWUgdGV4dCBkYXRhIn0sICJkZWZhdWx0IjogIiJ9XX0="

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
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
        build-essential \
        cmake \
        g++ \
        libhdf5-dev \
        libxml2-dev \
        libxslt1-dev \
        libboost-all-dev \
        libfftw3-dev \
        libpugixml-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Make python3.8 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 3) Upgrade pip, setuptools, wheel to latest
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4) Install ISMRMRD Python libraries
RUN python -m pip install --no-cache-dir ismrmrd-python-tools

# 5) Install SimpleITK *only* from a prebuilt wheel
RUN python -m pip install --no-cache-dir --only-binary=SimpleITK SimpleITK==2.3.1

# 6) Copy fetal brain pipeline requirements and install
COPY fetal-brain-measurement/requirements.txt .
RUN grep -v '^SimpleITK' requirements.txt > reqs.txt && \
    python -m pip install --no-cache-dir -r reqs.txt

# 7) Copy fetal brain pipeline code
COPY fetal-brain-measurement/ /workspace/fetal-brain-measurement/

# 8) Copy OpenRecon server code
COPY python-ismrmrd-server/ /opt/code/python-ismrmrd-server/

# 9) Copy OpenRecon handler files to server directory
COPY fetal-brain-measurement/openrecon.py /opt/code/python-ismrmrd-server/
COPY fetal-brain-measurement/openrecon.json /opt/code/python-ismrmrd-server/

# 10) Set PYTHONPATH for fetal brain pipeline and OpenRecon
ENV PYTHONPATH="/workspace/fetal-brain-measurement:/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation:/opt/code/python-ismrmrd-server"

# 11) Set working directory to OpenRecon server
WORKDIR /opt/code/python-ismrmrd-server

# 12) Default command to run OpenRecon server with fetal brain handler
CMD ["python", "main.py", "-d=openrecon"]
