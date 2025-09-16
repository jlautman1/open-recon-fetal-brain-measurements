# OpenRecon Fetal Brain Segmentation Pipeline

A complete integration of fetal brain measurement and segmentation AI pipeline with Siemens OpenRecon MRI systems using ISMRMRD format.

## 🎯 Overview

This repository contains a complete pipeline for automated fetal brain measurements and segmentation that integrates with Siemens OpenRecon MRI reconstruction systems. The pipeline processes ISMRMRD data directly from MRI scanners and provides real-time fetal brain analysis.

## 🏗️ Architecture

```
MRI Scanner → ISMRMRD Data → OpenRecon Server → AI Pipeline → Results
```

### Key Components:

- **`fetal-brain-measurement/`** - Core fetal brain AI pipeline with measurement algorithms
- **`python-ismrmrd-server/`** - OpenRecon server framework for MRI integration
- **`nifti_to_ismrmrd_converter.py`** - Format conversion utilities
- **`OpenRecon.dockerfile`** - Container deployment configuration
- **`ismrmrd-python-tools/`** - ISMRMRD utilities and tools

## 🚀 Features

- ✅ **Real-time processing** - Direct integration with MRI scanners
- ✅ **Automated measurements** - CBD, BBD, TCD, gestational age estimation
- ✅ **Brain volume calculation** - 3D volumetric analysis
- ✅ **ISMRMRD compatibility** - Standard medical imaging format
- ✅ **Docker deployment** - Containerized for easy deployment
- ✅ **OpenRecon integration** - Works with Siemens MRI systems

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

## 🐳 Docker Deployment

### Quick Start:
```bash
# Build the OpenRecon container
docker build -f OpenRecon.dockerfile -t openrecon-fetal:latest .

# Run the server
docker run -p 9002:9002 openrecon-fetal:latest
```

### For MRI Integration:
The container is designed to be deployed on Siemens OpenRecon systems. See the deployment guides in `fetal-brain-measurement/` for detailed instructions.

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
- **Measurements**: CBD, BBD, TCD in millimeters
- **Gestational age**: Estimated from measurements
- **Brain volume**: Total brain volume in mm³
- **Segmentation masks**: Brain structure segmentation
- **Reports**: PDF reports with visualizations

## 🤝 Contributing

This pipeline integrates multiple components:
- Fetal brain measurement algorithms
- OpenRecon MRI integration
- ISMRMRD format handling
- Docker containerization

## 📚 Documentation

- `fetal-brain-measurement/README.md` - Core pipeline documentation
- `fetal-brain-measurement/DEPLOYMENT_GUIDE.md` - OpenRecon deployment
- `python-ismrmrd-server/readme.md` - Server framework documentation

## 🏥 Clinical Integration

This pipeline is designed for integration with clinical MRI workflows:
- Real-time processing during MRI scans
- Immediate results available to clinicians
- ISMRMRD standard compliance
- OpenRecon system compatibility

## 🔗 Related Projects

- [ISMRMRD](https://ismrmrd.github.io/) - Medical imaging data standard
- [OpenRecon](https://openrecon.ismrmrd.org/) - Siemens reconstruction framework

## 📄 License

[Include your license information here]

## 📞 Contact

[Include contact information for support]
