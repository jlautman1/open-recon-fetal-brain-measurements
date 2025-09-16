#!/usr/bin/env python3
import json
import base64

# Load the JSON metadata
with open('openrecon_json_ui.json', 'r') as f:
    jsonData = json.load(f)

# Create the encoded label
encoded = base64.b64encode(json.dumps(jsonData).encode("utf-8")).decode("utf-8")
labelLine = f'LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="{encoded}"'

# Generate the corrected Dockerfile
dockerfile_content = f'''# ------------------------------------------------------------
#  OpenRecon Fetal Brain Segmentation Pipeline
#  Based on working fetal-brain-measurement Dockerfile
# ------------------------------------------------------------
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

# === OpenRecon UI Label ===
{labelLine}

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1

# 1) Install system deps + Python 3.8
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        wget \\
        git \\
        libgl1 \\
        libglib2.0-0 \\
        python3.8 \\
        python3.8-venv \\
        python3.8-dev \\
        python3-pip \\
        build-essential \\
        cmake \\
        g++ \\
        libhdf5-dev \\
        libxml2-dev \\
        libxslt1-dev \\
        libboost-all-dev \\
        libfftw3-dev \\
        libpugixml-dev && \\
    rm -rf /var/lib/apt/lists/*

# 2) Make python3.8 the default "python"
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \\
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 3) Upgrade pip, setuptools, wheel to latest
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 4) Install ISMRMRD Python libraries
RUN python -m pip install --no-cache-dir ismrmrd-python-tools

# 5) Install SimpleITK *only* from a prebuilt wheel
RUN python -m pip install --no-cache-dir --only-binary=SimpleITK SimpleITK==2.3.1

# 6) Copy fetal brain pipeline requirements and install
COPY fetal-brain-measurement/requirements.txt .
RUN grep -v '^SimpleITK' requirements.txt > reqs.txt && \\
    python -m pip install --no-cache-dir -r reqs.txt

# 7) Copy fetal brain pipeline code
COPY fetal-brain-measurement/ /workspace/fetal-brain-measurement/

# 8) Copy OpenRecon server code
COPY python-ismrmrd-server/ /opt/code/python-ismrmrd-server/

# 9) Copy OpenRecon handler files to server directory (they are in fetal-brain-measurement)
COPY fetal-brain-measurement/openrecon.py /opt/code/python-ismrmrd-server/
COPY fetal-brain-measurement/openrecon.json /opt/code/python-ismrmrd-server/

# 10) Set PYTHONPATH for fetal brain pipeline and OpenRecon
ENV PYTHONPATH="/workspace/fetal-brain-measurement:/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation:/opt/code/python-ismrmrd-server"

# 11) Set working directory to OpenRecon server
WORKDIR /opt/code/python-ismrmrd-server

# 12) Default command to run OpenRecon server with fetal brain handler
CMD ["python", "main.py", "-d=openrecon"]
'''

# Write the Dockerfile
with open('../OpenRecon.dockerfile', 'w', encoding='utf-8') as f:
    f.write(dockerfile_content)

print("✅ Generated correct Dockerfile: ../OpenRecon.dockerfile")
print(f"🏷️  Docker image will have label: {labelLine[:100]}...")
print("📁 Ready to build!")
