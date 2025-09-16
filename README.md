# OpenRecon Fetal Brain Segmentation Pipeline

A complete integration of fetal brain measurement and segmentation AI pipeline with Siemens OpenRecon MRI systems using ISMRMRD format.

## 🎯 Overview

This repository contains a complete pipeline for automated fetal brain measurements and segmentation that integrates with Siemens OpenRecon MRI reconstruction systems. The pipeline processes ISMRMRD data directly from MRI scanners and provides real-time fetal brain analysis.

## 🏗️ Architecture

```
MRI Scanner → DICOM → ISMRMRD → OpenRecon Server → NIfTI → AI Pipeline → NIfTI Results → ISMRMRD → DICOM → MRI Scanner
     ↑                                                ↓                    ↓                           ↓                    ↑
     │                                             Convert              Process               Convert Back            │
     │                                          (ISMRMRD→NIfTI)     (Measurements)         (NIfTI→ISMRMRD)        │
     └─────────────────────────────── Results delivered back to clinician ──────────────────────────────────────┘
```

### Detailed Data Flow:

1. **🏥 MRI Scanner** → Acquires raw k-space data
2. **📄 DICOM** → Standard medical imaging format from scanner  
3. **🔄 ISMRMRD** → Standardized raw data format for reconstruction
4. **🖥️ OpenRecon Server** → Receives ISMRMRD data, manages processing
5. **🔄 NIfTI Conversion** → Convert ISMRMRD images to NIfTI format for AI processing
6. **🧠 AI Pipeline** → Fetal brain segmentation and measurement algorithms
7. **📊 NIfTI Results** → Processed images with measurements and segmentations
8. **🔄 ISMRMRD Conversion** → Convert results back to ISMRMRD format
9. **📄 DICOM Output** → Convert to DICOM with embedded measurements
10. **🏥 Back to MRI** → Results displayed on MRI console for clinician

### Key Components:

- **`fetal-brain-measurement/`** - Core fetal brain AI pipeline with measurement algorithms
- **`python-ismrmrd-server/`** - OpenRecon server framework for MRI integration  
- **`nifti_to_ismrmrd_converter.py`** - Bidirectional format conversion (NIfTI ↔ ISMRMRD)
- **`fetal-brain-measurement/openrecon.py`** - OpenRecon i2i handler with format conversions
- **`ismrmrd-python-tools/`** - ISMRMRD utilities and reconstruction tools
- **`OpenRecon.dockerfile`** - Complete container for clinical deployment

## 🚀 Features

- ✅ **Real-time processing** - Direct integration with MRI scanners
- ✅ **Automated measurements** - CBD, BBD, TCD, gestational age estimation
- ✅ **Brain volume calculation** - 3D volumetric analysis
- ✅ **Seamless format conversion** - DICOM → ISMRMRD → NIfTI → ISMRMRD → DICOM
- ✅ **ISMRMRD compatibility** - Standard medical imaging format with bidirectional conversion
- ✅ **Clinical integration** - Results embedded back into MRI workflow
- ✅ **Docker deployment** - Containerized for easy deployment
- ✅ **OpenRecon integration** - Full Siemens MRI system compatibility

## 📋 Requirements

### System Requirements:
- Docker with GPU support
- NVIDIA GPU (for AI inference)
- Python 3.8+
- ISMRMRD libraries

### Dependencies:
- TensorFlow/Keras for AI models
- ISMRMRD Python tools
- NiBabel for medical imaging
- NumPy, SciPy for numerical processing

## 🐳 Docker Images

This project uses multiple Docker images for different purposes:

### 1. **Base Fetal Brain Pipeline** (Ready on DockerHub)
```bash
# Pull the working fetal brain measurement pipeline
docker pull jlautman1/fetal-pipeline-gpu-rebuilt:latest
```
**Purpose**: Local fetal brain measurement processing  
**Contains**: Complete AI pipeline, models, dependencies  
**Use case**: Research, development, local NIfTI processing  

### 2. **OpenRecon Integration Image** (Build from source)
```bash
# Build the complete OpenRecon-enabled container
docker build -f fetal-brain-measurement/Dockerfile.openrecon \
  -t openrecon-fetal-brain:latest .

# Or use the comprehensive build
docker build -f OpenRecon.dockerfile -t openrecon-fetal-complete:latest .
```
**Purpose**: Clinical deployment on Siemens MRI systems  
**Contains**: Fetal pipeline + OpenRecon server + ISMRMRD tools  
**Use case**: Real-time MRI processing, clinical deployment  

### 3. **Development Server** (Build from source)  
```bash
# Build the python-ismrmrd-server for development
cd python-ismrmrd-server
docker build -t python-ismrmrd-server:latest .
```
**Purpose**: OpenRecon development and testing  
**Contains**: ISMRMRD server framework, development tools  
**Use case**: Development, testing OpenRecon handlers  

### Quick Start:
```bash
# For local pipeline testing:
docker pull jlautman1/fetal-pipeline-gpu-rebuilt:latest
docker run --gpus all -it jlautman1/fetal-pipeline-gpu-rebuilt:latest

# For OpenRecon deployment:
docker build -f OpenRecon.dockerfile -t openrecon-fetal:latest .
docker run -p 9002:9002 openrecon-fetal:latest
```

### Docker Image Hierarchy:
```
nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
└── jlautman1/fetal-pipeline-gpu-rebuilt:latest (Base pipeline)
    └── openrecon-fetal-brain:latest (+ OpenRecon integration)
        └── openrecon-fetal-complete:latest (+ ISMRMRD tools)
```

## 🧪 Testing

### Test the complete pipeline:
```bash
python test_complete_pipeline.py
```

### Test OpenRecon integration:
```bash
python test_openrecon_pipeline.py
```

### Test format conversion:
```bash
python nifti_to_ismrmrd_converter.py
```

## 📁 Project Structure

```
.
├── fetal-brain-measurement/          # Core AI pipeline
│   ├── Code/                         # Fetal measurement algorithms
│   ├── Models/                       # Pre-trained AI models
│   ├── Inputs/                       # Sample input data
│   ├── openrecon.py                  # OpenRecon handler
│   └── Dockerfile.openrecon          # Deployment container
├── python-ismrmrd-server/            # OpenRecon server framework
├── ismrmrd-python-tools/             # ISMRMRD utilities
├── nifti_to_ismrmrd_converter.py     # Format conversion
├── OpenRecon.dockerfile              # Main deployment container
└── test_*.py                         # Test scripts
```

## 🔧 Configuration

The pipeline can be configured through:
- `fetal-brain-measurement/openrecon.json` - OpenRecon settings
- Environment variables for Docker deployment
- Model parameters in the AI pipeline

## 📊 Output

The pipeline generates:
- **Measurements**: CBD, BBD, TCD in millimeters with confidence scores
- **Gestational age**: Estimated from measurements with normative percentiles
- **Brain volume**: Total brain volume in mm³
- **Segmentation masks**: Brain structure segmentation (NIfTI format)
- **Visualizations**: Annotated measurement images (CBD.png, BBD.png, TCD.png)
- **📄 Clinical Reports**: Comprehensive PDF reports with:
  - Measurement values and confidence intervals
  - Normative percentile graphs 
  - Annotated anatomical images
  - Quality assessment metrics
- **📄 Example Report**: [View sample output](fetal-brain-measurement/output/Pat13249_Se8_Res0.46875_0.46875_Spac4.0/report.pdf)

## 🤝 Contributing

This pipeline integrates multiple components:
- Fetal brain measurement algorithms
- OpenRecon MRI integration
- ISMRMRD format handling
- Docker containerization

## 📚 Documentation

- **[Local Pipeline Guide](fetal-brain-measurement/README.md)** - Complete local NIfTI processing documentation
- **[OpenRecon Technical Guide](fetal-brain-measurement/README.openrecon.md)** - Detailed OpenRecon integration and architecture
- **[OpenRecon Deployment Guide](fetal-brain-measurement/DEPLOYMENT_GUIDE.md)** - Step-by-step MRI deployment process
- **[Server Framework Docs](python-ismrmrd-server/readme.md)** - ISMRMRD server development documentation
- **[ISMRMRD Tools Docs](ismrmrd-python-tools/README.md)** - ISMRMRD utilities and tools reference

## 🏥 Clinical Integration

This pipeline is designed for integration with clinical MRI workflows:
- Real-time processing during MRI scans
- Immediate results available to clinicians
- ISMRMRD standard compliance
- OpenRecon system compatibility

## 🔗 Related Projects

- [ISMRMRD](https://ismrmrd.github.io/) - Medical imaging data standard
- [OpenRecon](https://openrecon.ismrmrd.org/) - Siemens reconstruction framework
- [Fetal Brain Measurement (Local Pipeline)](https://github.com/jlautman1/fetal-brain-measurement) - Original local-only fetal brain measurement pipeline

## 🧪 Running the Local Pipeline on NIfTI Input

For running just the local fetal brain measurement pipeline on NIfTI files (without OpenRecon integration):

### Quick Start with Docker:
```bash
# Pull the pre-built fetal brain pipeline image
docker pull jlautman1/fetal-pipeline-gpu-rebuilt:latest

# Run the pipeline on your NIfTI file
docker run --gpus all -it \
  -v "/path/to/your/data:/workspace/fetal-brain-measurement" \
  jlautman1/fetal-pipeline-gpu-rebuilt:latest bash

# Inside the container, run the pipeline:
python3 /workspace/fetal-brain-measurement/Code/FetalMeasurements-master/execute.py \
  -i /workspace/fetal-brain-measurement/Inputs/Fixed \
  -o /workspace/fetal-brain-measurement/output
```

### Detailed Instructions:
See the [complete local pipeline guide](fetal-brain-measurement/README.md) for:
- Input data preparation
- Parameter configuration  
- Output interpretation
- Example reports and visualizations

### Example Output:
The pipeline generates:
- **Measurements**: CBD, BBD, TCD values with confidence scores
- **Segmentation**: Brain structure masks in NIfTI format
- **Visualizations**: Annotated slice images (CBD.png, BBD.png, TCD.png)
- **Report**: Comprehensive PDF report with normative percentile graphs
- **📄 Sample Report**: [Example PDF Report](fetal-brain-measurement/output/new_data_first/WORKING_GOOD-Pat13249_Se8_Res0.46875_0.46875_Spac4.0/report.pdf)
- **📁 More Examples**: Browse `fetal-brain-measurement/output/` for multiple example outputs

## 🚀 MRI Magnet Deployment Guide

### Prerequisites:
1. Siemens OpenRecon-compatible MRI system
2. Docker support on the MRI host system
3. Network access for Docker image deployment

### Deployment Steps:

#### 1. Build the OpenRecon Container:
```bash
# Pull the required base images
docker pull jlautman1/fetal-pipeline-gpu-rebuilt:latest

# Build the OpenRecon-enabled container
docker build -f fetal-brain-measurement/Dockerfile.openrecon \
  -t openrecon-fetal-brain:latest .
```

#### 2. Package for MRI Deployment:
```bash
# Save the Docker image as a tar file
docker save openrecon-fetal-brain:latest -o openrecon-fetal-brain.tar

# Transfer to MRI system (via USB, network, etc.)
# Load on the MRI system:
docker load -i openrecon-fetal-brain.tar
```

#### 3. Configure OpenRecon:
- Copy `fetal-brain-measurement/openrecon.json` to the OpenRecon configuration directory
- Verify the image runs: `docker run -p 9002:9002 openrecon-fetal-brain:latest`
- Test with sample ISMRMRD data using the provided test scripts

#### 4. Integration Testing:
```bash
# Test the complete integration
python test_complete_pipeline.py
python test_openrecon_pipeline.py
```

### Detailed MRI Deployment:
For complete deployment instructions including:
- OpenRecon system configuration
- Network setup and security considerations  
- Troubleshooting and validation
- Production deployment checklist

See: 
- [OpenRecon Deployment Guide](fetal-brain-measurement/DEPLOYMENT_GUIDE.md) - Step-by-step deployment process
- [OpenRecon Technical Documentation](fetal-brain-measurement/README.openrecon.md) - Detailed technical integration guide

## 📄 License

[Include your license information here]

## 📞 Contact

[Include contact information for support]
