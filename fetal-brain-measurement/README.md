# 🧠 Fetal Brain Measurement Pipeline

## Overview

This project provides an AI-powered fetal brain measurement pipeline that can be used in three different modes:
1. **Local NIfTI Processing** - Process NIfTI files directly for research and development
2. **OpenRecon Integration** - Real-time processing on Siemens MRI systems
3. **DICOM Conversion** - Convert DICOM files to NIfTI and process

## 🏗️ Architecture

The system consists of:
- **AI Pipeline**: Deep learning models for brain segmentation and measurement
- **OpenRecon Handler**: Integration with Siemens OpenRecon framework
- **Docker Environment**: Unified container with all dependencies
- **DICOM Support**: Automatic conversion and format handling

## 📁 Project Structure

```
fetal-brain-measurement/
├── Code/                           # Core pipeline implementation
│   └── FetalMeasurements-master/   # Main measurement algorithms
├── Models/                         # AI models (segmentation, slice selection)
├── Inputs/                         # Test input data
├── Dockerfile                      # Local development container
├── Dockerfile.openrecon           # OpenRecon deployment container
├── openrecon.py                   # OpenRecon i2i handler
├── openrecon.json                 # OpenRecon configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Usage Modes

### Mode 1: Local NIfTI Processing

Process NIfTI files directly for research and development:

#### Prerequisites
- Docker with GPU support
- NVIDIA Container Toolkit

#### Quick Start
```bash
# Build the development Docker image
docker build -t fetal-brain-local .

# Run the pipeline on a NIfTI file
docker run --rm --gpus all -v /path/to/data:/data fetal-brain-local \\
    python Code/FetalMeasurements-master/execute.py \\
    -f /data/your_file.nii.gz \\
    -o /data/output
```

#### Expected Input Format
NIfTI files should follow this naming convention:
```
Pat[PatientID]_Se[SeriesNumber]_Res[X]_[Y]_Spac[Z].nii.gz
```
Example: `Pat12345_Se1_Res0.5_0.5_Spac3.0.nii.gz`

#### Output
- `prediction.nii.gz` - Brain segmentation mask
- `data.json` - Measurement results (CBD, BBD, TCD, brain volume)
- `report.pdf` - Clinical report with measurements and plots
- Various PNG plots showing measurement locations

**📄 Example Output**: View a [sample report.pdf](output/new_data_first/WORKING_GOOD-Pat13249_Se8_Res0.46875_0.46875_Spac4.0/report.pdf) showing the complete measurement results, normative percentile graphs, and annotated anatomical images.

### Mode 2: OpenRecon Integration (MRI Deployment)

Deploy to Siemens MRI systems using OpenRecon framework:

#### Build OpenRecon Image
```bash
# Build the OpenRecon-compatible Docker image
docker build -f Dockerfile.openrecon -t openrecon-fetal:latest .
```

#### Deploy to MRI System
```bash
# Run OpenRecon server (typically on MRI reconstruction computer)
docker run --rm --gpus all -p 9002:9002 \\
    openrecon-fetal:latest python main.py -v
```

#### Server Configuration
- **Port**: 9002 (default OpenRecon i2i port)
- **Protocol**: ISMRMRD over TCP
- **Processing**: Real-time image-to-image transformation
- **GPU**: Required for AI model inference

#### Integration Features
- ✅ **Automatic DICOM/ISMRMRD handling** - No manual conversion needed
- ✅ **Real-time processing** - Measurements during acquisition
- ✅ **Metadata preservation** - Patient info embedded in output
- ✅ **Robust error handling** - Graceful fallbacks for edge cases
- ✅ **Timeout protection** - Prevents hanging on difficult cases

### Mode 3: DICOM Conversion & Processing

Convert DICOM files and process through the pipeline:

#### DICOM to NIfTI Conversion
The OpenRecon handler automatically handles DICOM conversion:

```python
# The conversion happens automatically in openrecon.py
# - Extracts metadata (Patient ID, Series, Resolution)
# - Converts to proper NIfTI format
# - Generates correctly named files for pipeline
# - Processes through AI pipeline
# - Returns results in ISMRMRD format
```

#### Manual DICOM Processing
For research use, you can process DICOM files directly:

```bash
# Start container with DICOM folder mounted
docker run --rm -it --gpus all -v /path/to/dicoms:/dicoms \\
    openrecon-fetal:latest bash

# Inside container: convert and process
cd /workspace
python -c \"
import pydicom
import nibabel as nib
import numpy as np
import os

# Convert DICOM series to NIfTI
# ... conversion code ...

# Run pipeline
python Code/FetalMeasurements-master/execute.py -f converted.nii.gz -o output
\"
```

## 🔧 Configuration

### Pipeline Settings
Key parameters can be adjusted in the code:

- **Timeout Settings**: MSL calculation timeout (default: 4 minutes)
- **Model Paths**: AI model locations in Models/ directory
- **Resolution Limits**: Minimum/maximum supported voxel sizes
- **Memory Management**: Point subsampling for large datasets

### OpenRecon Configuration
The `openrecon.json` file configures the MRI integration:

```json
{
    \"methods\": [\"FetalBrainMeasurement\"],
    \"defaults\": {
        \"FetalBrainMeasurement\": {
            \"timeout\": 300,
            \"gpu_enabled\": true,
            \"debug_mode\": false
        }
    }
}
```

## 🧪 Testing & Validation

### Local Testing
```bash
# Test with provided sample data
docker run --rm --gpus all fetal-brain-local \\
    python Code/FetalMeasurements-master/execute.py \\
    -f Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz \\
    -o test_output
```

### OpenRecon Server Testing
```bash
# Start server
docker run --rm --gpus all -p 9002:9002 openrecon-fetal:latest python main.py -v

# Test connectivity (in another terminal)
python -c \"
import socket
s = socket.socket()
s.connect(('localhost', 9002))
print('✅ Server accessible')
s.close()
\"
```

## 📊 Performance & Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for models and processing
- **Docker**: Version 20.10+ with GPU support

### Processing Performance
- **Brain Segmentation**: ~2-3 minutes (TensorFlow models)
- **Sub-segmentation**: ~1-2 minutes (FastAI model)
- **MSL Calculation**: ~5-15 minutes (depends on data complexity)
- **Measurements & Report**: ~1-2 minutes

### Data Compatibility
- **Input Formats**: NIfTI (.nii.gz), DICOM, ISMRMRD
- **Image Types**: T2-weighted fetal brain MRI
- **Resolutions**: 0.3-1.0mm in-plane, 2-6mm slice thickness
- **Orientations**: Automatic detection and reorientation

## 🚨 Troubleshooting

### Common Issues

**Pipeline Timeout in MSL Step**
```
Solution: The MSL calculation has built-in timeout protection.
If it takes >4 minutes, it will return partial results.
This ensures OpenRecon integration doesn't hang.
```

**GPU Memory Errors**
```
Solution: Reduce batch sizes or enable point subsampling
for large datasets (>640×640×40 voxels).
```

**Model Loading Failures**
```
Solution: Ensure Models/ directory is properly copied to container.
Check file permissions and disk space.
```

**DICOM Conversion Issues**
```
Solution: The pipeline handles various DICOM formats automatically.
For unsupported formats, manual conversion may be needed.
```

### Debug Mode
Enable detailed logging:

```bash
# Run with debug output
docker run --rm --gpus all -e DEBUG=1 openrecon-fetal:latest python main.py -v
```

## 🏥 Clinical Use

### Measurements Provided
- **CBD**: Cerebellum-to-Brain Distance
- **BBD**: Bi-Biparietal Distance  
- **TCD**: Trans-Cerebellar Distance
- **Brain Volume**: Total fetal brain volume
- **Gestational Age**: Predicted GA based on measurements

### Report Generation
The system generates clinical reports including:
- Measurement values with confidence intervals
- Comparison to normative curves
- Visual plots showing measurement locations
- Quality assessment and recommendations

### Validation Status
- ✅ Tested on clinical fetal MRI datasets
- ✅ Validated against manual measurements
- ✅ Integrated with Siemens OpenRecon framework
- ✅ Performance tested in hospital environment

## 📄 Documentation

- `DEPLOYMENT_GUIDE.md` - Detailed deployment instructions
- `deployToOpenRecon.md` - OpenRecon-specific deployment
- Technical documentation in Code/FetalMeasurements-master/

## 🤝 Support

For technical support or clinical questions:
- Check troubleshooting section above
- Review log files for error details
- Contact development team for complex issues

## 📋 Version History

- **v1.0.0**: Initial release with OpenRecon integration
- Automatic DICOM conversion support
- Timeout protection and robust error handling
- Comprehensive documentation and testing