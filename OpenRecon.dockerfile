# ------------------------------------------------------------
#  OpenRecon Fetal Brain Segmentation Pipeline
#  Multi-stage build: ISMRMRD + Fetal Brain Pipeline
# ------------------------------------------------------------

# === Stage 1: Build ISMRMRD ===
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 as mrd_converter

WORKDIR /tmp

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libhdf5-dev \
        libxml2-dev \
        libxslt1-dev \
        libboost-all-dev \
        libfftw3-dev \
        pkg-config \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Build PugiXML from source
RUN git clone https://github.com/zeux/pugixml.git && \
    cd pugixml && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

# Build ISMRMRD from source
RUN git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    mkdir build && cd build && \
    cmake -DBUILD_UTILITIES=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF .. && \
    make -j$(nproc) && \
    make install

# === Stage 2: Runtime with Fetal Brain Pipeline ===
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# === OpenRecon UI Label ===
LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="eyJnZW5lcmFsIjogeyJuYW1lIjogeyJlbiI6ICJGZXRhbCBCcmFpbiBNZWFzdXJlbWVudHMifSwgInZlcnNpb24iOiAiMS4wLjAiLCAidmVuZG9yIjogIlNpZW1lbnNIZWFsdGhpbmVlcnNBRyIsICJpbmZvcm1hdGlvbiI6IHsiZW4iOiAiRGVtbyBvZiBmZXRhbCBicmFpbiBtZWFzdXJlbWVudHMgdjEhIn0sICJpZCI6ICJQeXRob25NUkRpMmkiLCAicmVndWxhdG9yeV9pbmZvcm1hdGlvbiI6IHsiZGV2aWNlX3RyYWRlX25hbWUiOiAiUHl0aG9uTVJEaTJpIiwgInByb2R1Y3Rpb25faWRlbnRpZmllciI6ICIxLjAuMCIsICJtYW51ZmFjdHVyZXJfYWRkcmVzcyI6ICJUZWwgQXZpdiwgSXNyYWVsIiwgIm1hZGVfaW4iOiAiSUwiLCAibWFudWZhY3R1cmVfZGF0ZSI6ICIyMDI1LzA1LzA4IiwgIm1hdGVyaWFsX251bWJlciI6ICJQeXRob25NUkRfaTJpXzEuMC4wIiwgImd0aW4iOiAiMDA4NjAwMDAxNzEyMTIiLCAidWRpIjogIigwMSkwMDg2MDAwMDE3MTIxMigyMSkxLjMuMCIsICJzYWZldHlfYWR2aWNlcyI6ICIiLCAic3BlY2lhbF9vcGVyYXRpbmdfaW5zdHJ1Y3Rpb25zIjogIlJ1biBmZXRhbCBicmFpbiBtZWFzdXJlbWVudHMgb24gcmVjb25zdHJ1Y3RlZCBpbWFnZXMgYW5kIHJldHVybiBST0kgb3ZlcmxheXMuIiwgImFkZGl0aW9uYWxfcmVsZXZhbnRfaW5mb3JtYXRpb24iOiAiIn19LCAicmVjb25zdHJ1Y3Rpb24iOiB7InRyYW5zZmVyX3Byb3RvY29sIjogeyJwcm90b2NvbCI6ICJJU01STVJEIiwgInZlcnNpb24iOiAiMS40LjEifSwgInBvcnQiOiA5MDAyLCAiZW1pdHRlciI6ICJpbWFnZSIsICJpbmplY3RvciI6ICJpbWFnZSIsICJjYW5fcHJvY2Vzc19hZGp1c3RtZW50X2RhdGEiOiBmYWxzZSwgImNhbl91c2VfZ3B1IjogdHJ1ZSwgIm1pbl9jb3VudF9yZXF1aXJlZF9ncHVzIjogMCwgIm1pbl9yZXF1aXJlZF9ncHVfbWVtb3J5IjogMjA0OCwgIm1pbl9yZXF1aXJlZF9tZW1vcnkiOiA0MDk2LCAibWluX2NvdW50X3JlcXVpcmVkX2NwdV9jb3JlcyI6IDEsICJjb250ZW50X3F1YWxpZmljYXRpb25fdHlwZSI6ICJSRVNFQVJDSCJ9LCAicGFyYW1ldGVycyI6IFt7ImlkIjogImNvbmZpZyIsICJ0eXBlIjogImNob2ljZSIsICJsYWJlbCI6IHsiZW4iOiAiY29uZmlnIn0sICJ2YWx1ZXMiOiBbeyJpZCI6ICJvcGVucmVjb24iLCAibmFtZSI6IHsiZW4iOiAiRmV0YWwgQnJhaW4gTWVhc3VyZW1lbnRzIn19LCB7ImlkIjogImludmVydGNvbnRyYXN0IiwgIm5hbWUiOiB7ImVuIjogImludmVydGNvbnRyYXN0In19XSwgImRlZmF1bHQiOiAib3BlbnJlY29uIiwgImluZm9ybWF0aW9uIjogeyJlbiI6ICJEZWZpbmUgdGhlIGNvbmZpZyB0byBiZSBleGVjdXRlZCBieSBNUkQgc2VydmVyIn19LCB7ImlkIjogImN1c3RvbWNvbmZpZyIsICJsYWJlbCI6IHsiZW4iOiAiQ3VzdG9tIENvbmZpZyJ9LCAidHlwZSI6ICJzdHJpbmciLCAiaW5mb3JtYXRpb24iOiB7ImVuIjogIkN1c3RvbSBjb25maWcgZmlsZSBub3QgbGlzdGVkIGluIGRyb3AtZG93biBtZW51In0sICJkZWZhdWx0IjogIiJ9LCB7ImlkIjogImZyZWV0ZXh0IiwgImxhYmVsIjogeyJlbiI6ICJmcmVldGV4dCJ9LCAidHlwZSI6ICJzdHJpbmciLCAiaW5mb3JtYXRpb24iOiB7ImVuIjogIkZyZWUgdGV4dCBkYXRhIn0sICJkZWZhdWx0IjogIiJ9XX0="

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1

# Copy ISMRMRD libraries and development files from build stage
COPY --from=mrd_converter /usr/local/lib/libismrmrd* /usr/local/lib/
COPY --from=mrd_converter /usr/local/include/ismrmrd /usr/local/include/ismrmrd
COPY --from=mrd_converter /usr/local/lib/libpugixml* /usr/local/lib/
COPY --from=mrd_converter /usr/local/include/pugixml* /usr/local/include/
COPY --from=mrd_converter /usr/local/lib/pkgconfig/ /usr/local/lib/pkgconfig/
COPY --from=mrd_converter /usr/local/share/ismrmrd/ /usr/local/share/ismrmrd/

# Install runtime dependencies and build tools for Python packages
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
        libhdf5-100 \
        libhdf5-dev \
        libxml2 \
        libxml2-dev \
        libxslt1.1 \
        libxslt1-dev \
        libboost-program-options1.65.1 \
        libboost-system1.65.1 \
        libboost-filesystem1.65.1 \
        libboost-thread1.65.1 \
        libboost-dev \
        libfftw3-3 \
        libfftw3-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Update library cache
RUN ldconfig

# Make python3.8 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip, setuptools, wheel to latest
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (required by ismrmrd-python-tools)
RUN python -m pip install --no-cache-dir numpy

# Install ISMRMRD Python libraries (now that C++ libs are available)
RUN python -m pip install --no-cache-dir ismrmrd-python-tools

# Install SimpleITK *only* from a prebuilt wheel
RUN python -m pip install --no-cache-dir --timeout=300 --retries=3 --only-binary=SimpleITK SimpleITK==2.3.1

# Copy fetal brain pipeline requirements and install with compatibility fixes
COPY fetal-brain-measurement/requirements.txt .
RUN grep -v '^SimpleITK' requirements.txt > reqs_original.txt && \
    sed 's/numpy==1.14.5/numpy==1.21.0/g' reqs_original.txt | \
    sed 's/tensorflow-gpu==1.14.0/tensorflow==2.8.0/g' | \
    sed 's/Bottleneck==1.3.1/Bottleneck==1.3.5/g' | \
    sed 's/mkl-random==1.1.1/mkl-random==1.2.2/g' | \
    grep -v '^h5py\|^matplotlib\|^spacy\|^en_core_web_sm\|^jupyter\|^notebook\|^dataclasses\|^contextvars\|^mkl-\|^tensorflow==' > reqs.txt && \
    python -m pip install --no-cache-dir --timeout=300 -r reqs.txt

# Copy fetal brain pipeline code
COPY fetal-brain-measurement/ /workspace/fetal-brain-measurement/

# Copy OpenRecon server code
COPY python-ismrmrd-server/ /opt/code/python-ismrmrd-server/

# Copy OpenRecon handler files to server directory
COPY fetal-brain-measurement/openrecon.py /opt/code/python-ismrmrd-server/
COPY fetal-brain-measurement/openrecon.json /opt/code/python-ismrmrd-server/

# Set PYTHONPATH for fetal brain pipeline and OpenRecon
ENV PYTHONPATH="/workspace/fetal-brain-measurement:/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation:/opt/code/python-ismrmrd-server"

# Set working directory to OpenRecon server
WORKDIR /opt/code/python-ismrmrd-server

# Default command to run OpenRecon server with fetal brain handler
CMD ["python", "main.py", "-d=openrecon"]
