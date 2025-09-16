# OpenRecon Fetal Brain Measurement Integration

This directory contains the integration of the Fetal Brain Measurement Pipeline with Siemens OpenRecon MRI systems using the i2i (Image-to-Image) processing framework.

## 🏗️ Architecture

The integration consists of:

1. **i2i Handler**: Python OpenRecon i2i handler that processes images in real-time
2. **Fetal Brain Pipeline**: AI-powered measurement pipeline integrated within the handler
3. **Docker Container**: Unified environment containing all dependencies and models
4. **OpenRecon Interface**: Full compatibility with Siemens OpenRecon workflow

## 📁 Files Overview

### Core Integration Files
- `Dockerfile.openrecon` - Main Docker image for OpenRecon integration
- `openrecon.py` - Main i2i handler for fetal brain processing (correct OpenRecon structure)
- `openrecon.json` - Configuration file for the OpenRecon module

### Build and Deployment Scripts  
- `build-openrecon-image.bat` - Windows script to build the Docker image
- `run-openrecon-server.bat` - Windows script to run the OpenRecon server
- `test-client.py` - Test client for connectivity validation
- `validate-setup.py` - Pre-deployment validation script

### Documentation
- `README.openrecon.md` - This file

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Make scripts executable
chmod +x build-openrecon-image.sh run-openrecon-server.sh

# Build the integrated Docker image
./build-openrecon-image.sh
```

### 2. Run the Server

```bash
# Start the OpenRecon fetal brain measurement server
./run-openrecon-server.sh
```

The server will be available on port `9002` by default.

### 3. Test the Connection

```bash
# Basic connection test
python test-client.py

# Create and send test data
python test-client.py --create-test-data test_data.h5
python test-client.py --send-test-data test_data.h5
```

## 🧪 Testing the Pipeline

### Testing Server-Client Communication

Before deploying to a real MRI system, thoroughly test the Docker server-client communication:

#### 1. Test with Phantom Data (Basic Connectivity)

First, test basic ISMRMRD communication using the phantom test data:

```bash
# Build and start the Docker server
cd /path/to/OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0
docker build -t openrecon-fetal:latest -f fetal-brain-measurement/Dockerfile.openrecon .
docker run -d --name fetal-server --gpus all -p 9002:9002 openrecon-fetal:latest

# Generate phantom test data (in separate terminal)
cd python-ismrmrd-server
python generate_cartesian_shepp_logan_dataset.py phantom_raw.h5

# Test basic reconstruction (should work without fetal brain processing)
python client.py phantom_raw.h5 -c invertcontrast -p 9002 -o phantom_result.h5

# Expected output: "Session complete" with phantom_result.h5 created
```

#### 2. Test with Real Fetal Brain Data

Test the complete fetal brain AI pipeline:

```bash
# Create testing directory
mkdir testing
cd testing

# Create the fetal brain ISMRMRD converter script
cat > convert_phantom_to_images.py << 'EOF'
#!/usr/bin/env python3
"""
Convert the working phantom raw data to reconstructed images,
then replace image data with our fetal brain data.
This ensures we use the exact same ISMRMRD format that works.
"""

import os
import numpy as np
import h5py
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

def load_fetal_brain_data():
    """Load fetal brain NIfTI data"""
    if not HAS_NIBABEL:
        print("❌ nibabel not available, cannot load fetal brain data")
        return None
        
    nifti_path = "../fetal-brain-measurement/Inputs/fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    
    if not os.path.exists(nifti_path):
        print(f"❌ NIfTI file not found: {nifti_path}")
        return None
    
    print(f"📂 Loading fetal brain data: {nifti_path}")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    print(f"📊 NIfTI shape: {data.shape}")
    print(f"📊 Data range: {data.min():.2f} - {data.max():.2f}")
    
    # Take middle slices with brain content
    if len(data.shape) == 3:
        start_slice = data.shape[0] // 4
        end_slice = 3 * data.shape[0] // 4
        num_slices = min(5, end_slice - start_slice)
        
        brain_slices = []
        for i in range(num_slices):
            slice_idx = start_slice + i * (end_slice - start_slice) // num_slices
            slice_data = data[slice_idx, :, :]
            
            # Normalize to 12-bit range and convert to int16
            if slice_data.max() > 0:
                slice_data = (slice_data / slice_data.max()) * 4095
            slice_data = np.around(slice_data).astype(np.int16)
            
            brain_slices.append(slice_data)
            print(f"   📄 Processed slice {slice_idx}: {slice_data.shape}, range {slice_data.min()}-{slice_data.max()}")
        
        return brain_slices
    else:
        print(f"❌ Unexpected data shape: {data.shape}")
        return None

def run_phantom_reconstruction():
    """Run the phantom through the reconstruction to get proper image format"""
    print("🔄 Running phantom reconstruction to get proper image format...")
    
    cmd = "python ../python-ismrmrd-server/client.py ../python-ismrmrd-server/phantom_raw.h5 -c invertcontrast -p 9002 -o phantom_images.h5"
    print(f"Running: {cmd}")
    
    result = os.system(cmd)
    if result != 0:
        print("❌ Phantom reconstruction failed")
        return False
    
    if not os.path.exists("phantom_images.h5"):
        print("❌ Phantom image file not created")
        return False
    
    print("✅ Phantom reconstruction successful")
    return True

def replace_image_data():
    """Replace phantom image data with fetal brain data"""
    brain_slices = load_fetal_brain_data()
    if not brain_slices:
        print("❌ Could not load fetal brain data")
        return False
    
    print("🔄 Replacing phantom image data with fetal brain data...")
    
    try:
        with h5py.File("phantom_images.h5", "r+") as f:
            group_names = list(f.keys())
            print(f"Available groups: {group_names}")
            
            if not group_names:
                print("❌ No groups found in phantom images file")
                return False
            
            group_name = group_names[0]
            group = f[group_name]
            
            print(f"Using group: {group_name}")
            print(f"Group contents: {list(group.keys())}")
            
            image_groups = [key for key in group.keys() if key.startswith('image_')]
            print(f"Found {len(image_groups)} image groups: {image_groups}")
            
            if not image_groups:
                print("❌ No image groups found")
                return False
            
            for i, img_group_name in enumerate(image_groups[:len(brain_slices)]):
                if i >= len(brain_slices):
                    break
                    
                print(f"   📄 Replacing {img_group_name} with fetal brain slice {i}")
                
                img_group = group[img_group_name]
                orig_data = img_group['data'][:]
                print(f"   Original data shape: {orig_data.shape}, dtype: {orig_data.dtype}")
                
                brain_data = brain_slices[i]
                
                # Resize brain data to match original spatial dimensions
                orig_height = orig_data.shape[-2]
                orig_width = orig_data.shape[-1]
                
                print(f"   Resizing brain data from {brain_data.shape} to {orig_height}x{orig_width}")
                
                # Simple resize by cropping to center
                if brain_data.shape[0] > orig_height or brain_data.shape[1] > orig_width:
                    start_y = (brain_data.shape[0] - orig_height) // 2
                    start_x = (brain_data.shape[1] - orig_width) // 2
                    brain_data = brain_data[start_y:start_y+orig_height, start_x:start_x+orig_width]
                else:
                    pad_y = (orig_height - brain_data.shape[0]) // 2
                    pad_x = (orig_width - brain_data.shape[1]) // 2
                    brain_data = np.pad(brain_data, 
                                      ((pad_y, orig_height - brain_data.shape[0] - pad_y),
                                       (pad_x, orig_width - brain_data.shape[1] - pad_x)), 
                                      mode='constant', constant_values=0)
                
                # Reshape to match original format exactly
                if len(orig_data.shape) == 5:
                    new_data = brain_data.reshape(1, 1, 1, brain_data.shape[0], brain_data.shape[1])
                elif len(orig_data.shape) == 4:
                    new_data = brain_data.reshape(1, 1, brain_data.shape[0], brain_data.shape[1])
                elif len(orig_data.shape) == 3:
                    new_data = brain_data.reshape(1, brain_data.shape[0], brain_data.shape[1])
                else:
                    new_data = brain_data
                
                print(f"   New data shape: {new_data.shape}, dtype: {new_data.dtype}")
                
                # Update the data
                del img_group['data']
                img_group.create_dataset('data', data=new_data)
                
                print(f"   ✅ Updated {img_group_name}")
            
            print(f"✅ Successfully replaced {min(len(brain_slices), len(image_groups))} images")
        
        # Rename to fetal brain file
        if os.path.exists("fetal_brain_images_proper.h5"):
            os.remove("fetal_brain_images_proper.h5")
        os.rename("phantom_images.h5", "fetal_brain_images_proper.h5")
        
        print("✅ Created fetal_brain_images_proper.h5")
        return True
        
    except Exception as e:
        print(f"❌ Error replacing image data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🧠 Creating proper ISMRMRD fetal brain images")
    print("=" * 60)
    
    if not run_phantom_reconstruction():
        return False
    
    if not replace_image_data():
        return False
    
    print("\n🎯 Success! Use this file for testing:")
    print("   File: fetal_brain_images_proper.h5")
    print("   Command: python ../python-ismrmrd-server/client.py fetal_brain_images_proper.h5 -c openrecon -p 9002 -o fetal_result.h5")

if __name__ == "__main__":
    main()
EOF

# Install required dependencies for testing
pip install h5py xmltodict ismrmrd nibabel

# Create the fetal brain ISMRMRD test data
python convert_phantom_to_images.py

# Test the complete fetal brain AI pipeline
python ../python-ismrmrd-server/client.py fetal_brain_images_proper.h5 -c openrecon -p 9002 -o fetal_result.h5

# Expected output: "Session complete" with fetal_result.h5 created
# The result should contain embedded fetal brain measurements
```

#### 3. Verify Results

Check that the pipeline processed correctly:

```bash
# Examine the output file structure
python -c "
import h5py
f = h5py.File('testing/fetal_result.h5', 'r')
print('=== FETAL RESULT STRUCTURE ===')
print('Groups:', list(f.keys()))
group = f[list(f.keys())[0]]
print('Group contents:', list(group.keys()))
print('\\n=== CHECKING FOR METADATA ===')
img = group['image_0']
attrs = img['attributes'][0]
print('Attributes content (first 500 chars):')
print(str(attrs)[:500])
f.close()
"

# Check Docker server logs for processing details
docker logs fetal-server | tail -50

# Look for these success indicators in the logs:
# ✅ "🧠 === STARTING FETAL BRAIN MEASUREMENT PROCESSING ==="
# ✅ "💾 DEBUG: Saved NIfTI to ..."
# ✅ "🧠 DEBUG: Running fetal brain measurement pipeline"
# ✅ "✅ DEBUG: Successfully processed and sent image"
```

#### 4. Expected Log Output

A successful test should show logs like:
```
🎯 DEBUG: OpenRecon process function called
📋 DEBUG: Config: openrecon
📋 DEBUG: Metadata type: <class 'ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader'>
🖼️ DEBUG: Processing image: (1, 1, 256, 256)
🧠 === STARTING FETAL BRAIN MEASUREMENT PROCESSING ===
🧠 DEBUG: Starting fetal brain measurement processing
📂 DEBUG: Created temp directory: /tmp/fetal_openrecon_xxxxx
📁 DEBUG: Generated filename: Pat99999_Se1_Res0.5_0.5_Spac3.0.nii.gz
🔄 DEBUG: Converting ISMRMRD to NIfTI
📐 DEBUG: Original image shape: (1, 1, 256, 256)
💾 DEBUG: Saved NIfTI to /tmp/fetal_openrecon_xxxxx/Pat99999_Se1_Res0.5_0.5_Spac3.0.nii.gz
🧠 DEBUG: Running fetal brain measurement pipeline
✅ DEBUG: Fetal measurement processing completed
📊 DEBUG: Measurements extracted: CBD=45.2mm, BBD=52.1mm, TCD=18.4mm
🔗 DEBUG: Embedding measurements in DICOM metadata
✅ DEBUG: Successfully processed and sent image
```

#### 5. Troubleshooting Common Issues

**Error**: `'ismrmrdHeader' object has no attribute 'get'`
- **Fix**: Update `openrecon.py` with proper metadata handling for ISMRMRD objects

**Error**: `ValueError: could not broadcast input array from shape (512,) into shape (1,1,256,256)`
- **Fix**: Ensure fetal brain data is properly resized to match ISMRMRD image dimensions

**Error**: `File does not contain properly formatted MRD raw or image data`
- **Fix**: Use the `convert_phantom_to_images.py` script to create properly formatted ISMRMRD image files

**Error**: `ModuleNotFoundError: No module named 'fastai'`
- **Solution**: This is expected in local testing. The full pipeline should run inside the Docker container where all dependencies are installed.

### Deployment Validation

Before deploying to the MRI system:

1. **Performance Test**: Run multiple test images to verify processing time (should be <60 seconds)
2. **Memory Test**: Monitor Docker container memory usage during processing
3. **Error Handling**: Test with corrupted/invalid input data to verify graceful failure
4. **Network Test**: Verify the container can handle multiple concurrent connections

## 🔧 Configuration

### i2i Handler Configuration

The OpenRecon i2i handler can be configured by modifying `openrecon.json`:

```json
{
    "version": "2.0.0",
    "description": "Fetal brain measurement configuration for OpenRecon i2i handler",
    "parameters": {
        "processRawData": "true",
        "processImageData": "true",
        "enableMeasurements": "true",
        "outputDirectory": "/tmp/share/fetal_measurements"
    },
    "measurement_settings": {
        "CBD_enabled": "true",
        "BBD_enabled": "true", 
        "TCD_enabled": "true"
    }
}
```

### Environment Variables

- `PYTHONPATH` - Includes paths to fetal measurement modules
- `CUDA_VISIBLE_DEVICES` - Control GPU usage
- `MRD_SERVER_PORT` - Server port (default: 9002)

## 📊 Data Flow & Conversion Process

### 🏗️ **High-Level Architecture**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Siemens MRI   │    │  OpenRecon i2i  │    │ Fetal Brain     │    │ AI Processing   │    │ Enhanced DICOM  │
│    Scanner      │───▶│   Framework     │───▶│   Handler       │───▶│   Pipeline      │───▶│    Output       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼                       ▼
  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ T2W Fetal   │    │ process_image() │    │ ISMRMRD →       │    │ • Brain Seg     │    │ Original Image  │
  │ Brain Scans │    │ Function Call   │    │ NIfTI Convert   │    │ • Measurements  │    │ + DICOM Tags    │
  │ (Real-time) │    │                 │    │                 │    │ • Validation    │    │ + Metadata      │
  └─────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔄 **Detailed Conversion Process**
```
📡 SCANNER INPUT                🔄 i2i HANDLER PROCESSING              📤 DICOM OUTPUT
     │                                    │                                  │
     ▼                                    ▼                                  ▼
┌──────────────┐            ┌──────────────────────────────┐       ┌──────────────────┐
│ ISMRMRD      │   Step 1   │ FetalBrainI2IHandler.process │       │ Enhanced ISMRMRD │
│ Image Data   │──────────▶ │ • input_image               │       │ Image with       │
│ • 3D/4D      │            │ • output_image              │       │ • CBD: 79.8mm    │
│ • T2W Fetal  │            │ • metadata                  │       │ • BBD: 84.9mm    │
│ • Real-time  │            └──────────────────────────────┘       │ • TCD: 45.2mm    │
└──────────────┘                           │                      │ • GA: 36.9 weeks │
                                           ▼                      │ • Brain: 250k mm³│
                             ┌──────────────────────────────┐       └──────────────────┘
                    Step 2   │ ISMRMRD → NIfTI Conversion   │
                   ────────▶ │ • Extract image.data         │
                             │ • Create Nifti1Image         │
                             │ • Save temp file             │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 3   │ AI Pipeline Execution       │
                   ────────▶ │ • fetal_measure.execute()    │
                             │ • Brain segmentation         │
                             │ • Structure detection        │
                             │ • Measurement calculation    │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 4   │ Results Extraction           │
                   ────────▶ │ • data.json → measurements   │
                             │ • *.png → base64 plots       │
                             │ • *.pdf → base64 reports     │
                             │ • Clinical validation        │
                             └──────────────────────────────┘
                                           │
                                           ▼
                             ┌──────────────────────────────┐
                    Step 5   │ DICOM Metadata Embedding    │
                   ────────▶ │ • output_image.meta[tags]    │
                             │ • Clinical measurements      │
                             │ • Gestational age preds     │
                             │ • Visual data references    │
                             └──────────────────────────────┘
```

### 📋 **Key Technical Details**

#### **Input Processing:**
- **Format**: ISMRMRD Image objects from OpenRecon
- **Data Extraction**: `image.data` array (3D/4D NumPy)
- **Shape Handling**: Automatic 3D/4D detection and conversion
- **Temporary Storage**: `/tmp/fetal_openrecon_xxxxx/input.nii.gz`

#### **AI Pipeline Integration:**
- **Module**: `fetal_measure.FetalMeasure()`
- **Execution**: Direct Python import and function call
- **Processing**: Complete brain segmentation and measurement
- **Output**: JSON data + PNG plots + PDF reports

#### **DICOM Compliance:**
- **Metadata Embedding**: 15+ DICOM tags in `output_image.meta`
- **Visual Data**: Base64-encoded plots and reports
- **Clinical Tags**: CBD_mm, BBD_mm, TCD_mm, GA_*_weeks, Brain_Volume_mm3
- **Comments**: Human-readable summary in ImageComments tag

#### **Real-time Performance:**
- **Processing Time**: ~30-60 seconds per scan
- **Memory Usage**: Temporary files cleaned automatically
- **Error Handling**: Fallback to original image on failure
- **Debugging**: 30+ debug messages for full traceability

### Input Data Types
- **T2W Fetal Brain Scans**: Primary input for measurements
- **ISMRMRD Images**: Standard OpenRecon image format
- **3D/4D Volumes**: Single or multi-volume datasets

### Output Data
- **Enhanced ISMRMRD Images**: Original images with embedded DICOM metadata
- **Embedded Measurements**: CBD, BBD, TCD values in DICOM tags
- **Gestational Age**: Predictions based on measurements  
- **Clinical Metadata**: Brain volume, validity flags, comments
- **Visual Outputs**: Base64-encoded plots and reports (referenced in metadata)

## 🏥 OpenRecon Integration

### Installation on Scanner

1. **Copy Docker Image**:
   ```bash
   # Save image to file
   docker save openrecon-fetal-brain:latest > fetal-brain-openrecon.tar
   
   # Transfer to scanner and load
   docker load < fetal-brain-openrecon.tar
   ```

2. **Configure OpenRecon**:
   - Update `fire.ini` configuration
   - Set container startup parameters
   - Configure network settings

3. **Start Service**:
   ```bash
   # On the scanner system
   docker run -d --name fetal-brain-server \
     --gpus all \
     -p 9002:9002 \
     -v /scanner/data:/tmp/share \
     openrecon-fetal-brain:latest
   ```

### Scanner Configuration

Add to OpenRecon configuration:

```ini
[FIRE]
chroot_command = /path/to/start-fetal-server.sh
port = 9002
config = fetalbrainmeasure
timeout = 300
```

## 🔍 Measurements Provided

The system automatically computes:

### Primary Measurements
- **CBD (Cerebral Biparietal Diameter)**: Distance across cerebral hemispheres
- **BBD (Bone Biparietal Diameter)**: Skull-to-skull width  
- **TCD (Transcerebellar Diameter)**: Maximum width of cerebellum

### Additional Data
- Brain volume calculations
- Normative percentile comparisons
- Gestational age estimations
- Measurement quality scores

### Output Format

Results are embedded in DICOM metadata:
```
CBD_mm: 45.2
BBD_mm: 52.1
TCD_mm: 18.4
FetalMeasurements: {JSON with full results}
ImageComments: "Fetal Brain Measurements: CBD: 45.2mm, BBD: 52.1mm, TCD: 18.4mm"
```

## 🐛 Troubleshooting

### Common Issues

1. **Container Won't Start**:
   ```bash
   # Check logs
   docker logs openrecon-fetal-server
   
   # Check GPU access
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. **Connection Refused**:
   ```bash
   # Test port
   telnet localhost 9002
   
   # Check firewall
   sudo ufw status
   ```

3. **Measurement Failures**:
   ```bash
   # Check model files
   docker exec -it openrecon-fetal-server ls -la /workspace/Models/
   
   # Check Python paths
   docker exec -it openrecon-fetal-server python -c "import fetal_measure; print('OK')"
   ```

### Debug Mode

Enable detailed logging:
```bash
docker run -e LOG_LEVEL=DEBUG \
  -v $(pwd)/logs:/var/log \
  openrecon-fetal-brain:latest
```

### Data Inspection

Check intermediate results:
```bash
# Mount debug volume
docker run -v $(pwd)/debug:/tmp/share/debug openrecon-fetal-brain:latest

# Inspect saved data
ls -la debug/
```

## 📈 Performance

### System Requirements
- **GPU**: NVIDIA GPU with CUDA 11.0+ support
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for models and temporary data
- **Network**: Gigabit Ethernet for real-time processing

### Processing Times
- **Reconstruction**: ~5-10 seconds
- **Segmentation**: ~15-30 seconds  
- **Measurements**: ~5-10 seconds
- **Total**: ~30-60 seconds per scan

### Optimization Tips
- Use GPU acceleration for neural networks
- Optimize Docker resource limits
- Use SSD storage for temporary files
- Monitor memory usage during processing

## 🔒 Security Notes

- Container runs with restricted permissions
- No external network access required
- Data processed locally within container
- Temporary files automatically cleaned up
- HIPAA compliance considerations included

## 📞 Support

For technical support:
1. Check logs: `docker logs openrecon-fetal-server`
2. Run diagnostics: `python test-client.py`
3. Review configuration: `fetalbrainmeasure.json`
4. Contact: Ichlov Sagol Lab team

## 📝 License

This integration maintains the same license as the original fetal brain measurement pipeline. For research and development use only - not intended for clinical decision-making without regulatory approval.
