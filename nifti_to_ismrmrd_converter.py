#!/usr/bin/env python3
"""
NIfTI to ISMRMRD Converter
Converts NIfTI files to ISMRMRD format for testing the OpenRecon pipeline
"""

import os
import sys
import numpy as np
import nibabel as nib
import json
from pathlib import Path

# Add OpenRecon server paths
sys.path.append('./python-ismrmrd-server')
sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')

try:
    import ismrmrd
    print("✅ Successfully imported ismrmrd module")
except ImportError as e:
    print(f"❌ Failed to import ismrmrd: {e}")
    print("   Creating mock ISMRMRD classes for testing...")
    
    # Create mock ISMRMRD classes for testing
    class MockImage:
        def __init__(self, data):
            self.data = data.astype(np.complex64)  # ISMRMRD typically uses complex data
            self.meta = {}
            self.attribute_string = ""
            self.image_type = 1  # IMTYPE_MAGNITUDE
            self.image_index = 0
            self.image_series_index = 1
        
        @classmethod
        def from_array(cls, data):
            return cls(data)
    
    class MockMeta:
        def __init__(self):
            self._data = {}
        
        def __setitem__(self, key, value):
            self._data[key] = value
        
        def __getitem__(self, key):
            return self._data[key]
        
        def get(self, key, default=None):
            return self._data.get(key, default)
        
        def serialize(self):
            return json.dumps(self._data)
    
    # Create a mock ismrmrd module
    import types
    ismrmrd = types.ModuleType('ismrmrd')
    ismrmrd.Image = MockImage
    ismrmrd.Meta = MockMeta
    IMTYPE_MAGNITUDE = 1


def extract_metadata_from_filename(nifti_path):
    """Extract patient and series metadata from NIfTI filename"""
    filename = os.path.basename(nifti_path)
    print(f"🏷️ Extracting metadata from filename: {filename}")
    
    # Default metadata
    metadata = {
        'config': 'openrecon',
        'enable_measurements': True,
        'enable_reporting': True,
        'confidence_threshold': 0.5,
        'PatientName': 'TEST^PATIENT',
        'StudyDescription': 'MRI FETAL TEST',
        'SeriesDescription': 'T2W_HASTE_FETAL',
        'PixelSpacing': [0.5, 0.5],
        'SliceThickness': 3.0,
        'PatientID': 'TESTPAT001',
        'SeriesNumber': 1
    }
    
    # Try to parse filename format: Pat[PatientID]_Se[SeriesNumber]_Res[X]_[Y]_Spac[Z].nii.gz
    if filename.startswith('Pat') and '_Se' in filename:
        try:
            parts = filename.replace('.nii.gz', '').split('_')
            for part in parts:
                if part.startswith('Pat'):
                    patient_id = part[3:]  # Remove 'Pat' prefix
                    metadata['PatientID'] = patient_id
                    metadata['PatientName'] = f'PATIENT^{patient_id}'
                elif part.startswith('Se'):
                    series_num = int(part[2:])  # Remove 'Se' prefix
                    metadata['SeriesNumber'] = series_num
                elif part.startswith('Res'):
                    # Next part should be the Y resolution
                    idx = parts.index(part)
                    if idx + 1 < len(parts):
                        x_res = float(part[3:])  # Remove 'Res' prefix
                        y_res = float(parts[idx + 1])
                        metadata['PixelSpacing'] = [x_res, y_res]
                elif part.startswith('Spac'):
                    slice_thickness = float(part[4:])  # Remove 'Spac' prefix
                    metadata['SliceThickness'] = slice_thickness
            
            print(f"✅ Parsed metadata from filename:")
            print(f"   Patient ID: {metadata['PatientID']}")
            print(f"   Series: {metadata['SeriesNumber']}")
            print(f"   Resolution: {metadata['PixelSpacing']}")
            print(f"   Slice thickness: {metadata['SliceThickness']}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not parse filename completely: {e}")
            print("   Using default metadata values")
    
    return metadata


def convert_nifti_to_ismrmrd(nifti_path, output_path=None):
    """Convert NIfTI file to ISMRMRD format"""
    
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    print(f"🔄 Converting NIfTI to ISMRMRD format")
    print(f"   Input: {nifti_path}")
    
    # Load NIfTI data
    print("📖 Loading NIfTI file...")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    print(f"📐 Original data shape: {data.shape}")
    print(f"🔢 Value range: {data.min():.2f} - {data.max():.2f}")
    print(f"📊 Data type: {data.dtype}")
    
    # Normalize data to reasonable range for medical imaging
    if data.max() > 4095:  # If values are very high, normalize
        data = (data / data.max()) * 4095
        print(f"🔧 Normalized data to range: {data.min():.2f} - {data.max():.2f}")
    
    # Ensure we have 3D data
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
        print(f"📝 Expanded 2D to 3D: {data.shape}")
    elif len(data.shape) == 4:
        data = data[:, :, :, 0]  # Take first volume
        print(f"📝 Reduced 4D to 3D: {data.shape}")
    
    # Create ISMRMRD Image object
    print("🏗️ Creating ISMRMRD Image object...")
    
    # ISMRMRD data should be complex64, but for T2W images we use magnitude
    if np.iscomplexobj(data):
        ismrmrd_data = data.astype(np.complex64)
    else:
        # Convert real data to complex (magnitude only)
        ismrmrd_data = data.astype(np.float32) + 0j
        ismrmrd_data = ismrmrd_data.astype(np.complex64)
    
    print(f"📊 ISMRMRD data type: {ismrmrd_data.dtype}")
    print(f"📐 ISMRMRD data shape: {ismrmrd_data.shape}")
    
    # Create ISMRMRD image
    ismrmrd_image = ismrmrd.Image.from_array(ismrmrd_data.transpose())  # ISMRMRD expects different axis order
    
    # Set basic image properties
    if hasattr(ismrmrd_image, 'image_type'):
        ismrmrd_image.image_type = IMTYPE_MAGNITUDE if 'IMTYPE_MAGNITUDE' in globals() else 1
    if hasattr(ismrmrd_image, 'image_series_index'):
        ismrmrd_image.image_series_index = 1
    if hasattr(ismrmrd_image, 'image_index'):
        ismrmrd_image.image_index = 0
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(nifti_path)
    
    # Set image metadata
    if hasattr(ismrmrd_image, 'meta'):
        ismrmrd_image.meta = metadata
    
    # Create XML metadata string for ISMRMRD
    meta_obj = ismrmrd.Meta()
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            meta_obj[key] = list(value)
        else:
            meta_obj[key] = str(value)
    
    meta_obj['DataRole'] = 'Image'
    meta_obj['ImageProcessingHistory'] = ['NIfTI_CONVERSION']
    meta_obj['Keep_image_geometry'] = 1
    
    if hasattr(ismrmrd_image, 'attribute_string'):
        ismrmrd_image.attribute_string = meta_obj.serialize()
    
    print(f"✅ Successfully created ISMRMRD Image")
    print(f"   Data shape: {ismrmrd_image.data.shape}")
    print(f"   Data type: {ismrmrd_image.data.dtype}")
    
    # Save to file if requested
    if output_path:
        print(f"💾 Saving to: {output_path}")
        # For mock objects, we'll save as a pickle or numpy file
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump({
                'image': ismrmrd_image,
                'metadata': metadata,
                'original_nifti_path': nifti_path
            }, f)
        print(f"✅ Saved ISMRMRD data to {output_path}")
    
    return ismrmrd_image, metadata


def main():
    """Main function to test the converter"""
    print("🧪 NIfTI to ISMRMRD Converter Test")
    print("=" * 50)
    
    # Test with the fixed input file
    nifti_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    
    if not os.path.exists(nifti_file):
        print(f"❌ Test file not found: {nifti_file}")
        print("   Please check the file path")
        return False
    
    try:
        # Convert NIfTI to ISMRMRD
        ismrmrd_image, metadata = convert_nifti_to_ismrmrd(nifti_file, "test_ismrmrd_output.pkl")
        
        print("\n📋 Conversion Summary:")
        print(f"   Input file: {nifti_file}")
        print(f"   Output data shape: {ismrmrd_image.data.shape}")
        print(f"   Patient ID: {metadata.get('PatientID', 'Unknown')}")
        print(f"   Series: {metadata.get('SeriesNumber', 'Unknown')}")
        print(f"   Resolution: {metadata.get('PixelSpacing', 'Unknown')}")
        
        return ismrmrd_image, metadata
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = main()
    if result:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n💥 Conversion failed!")
