#!/usr/bin/env python3
"""
Convert NIfTI fetal brain data to ISMRMRD image format for testing
This simulates reconstructed images that would come from OpenRecon
"""

import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from datetime import datetime

def create_mrd_header():
    """Create MRD XML header for fetal brain imaging"""
    header = ET.Element("ismrmrdHeader")
    
    # Study information
    study = ET.SubElement(header, "studyInformation")
    ET.SubElement(study, "studyDate").text = datetime.now().strftime("%Y-%m-%d")
    ET.SubElement(study, "studyTime").text = datetime.now().strftime("%H:%M:%S")
    ET.SubElement(study, "studyID").text = "FETAL_BRAIN_T2W"
    ET.SubElement(study, "studyDescription").text = "Fetal Brain T2W Measurement Test"
    
    # Measurement information
    measurement = ET.SubElement(header, "measurementInformation")
    ET.SubElement(measurement, "measurementID").text = "12345"
    ET.SubElement(measurement, "seriesDate").text = datetime.now().strftime("%Y-%m-%d")
    ET.SubElement(measurement, "seriesTime").text = datetime.now().strftime("%H:%M:%S")
    ET.SubElement(measurement, "patientPosition").text = "HFS"
    ET.SubElement(measurement, "seriesDescription").text = "T2W_HASTE_Fetal_Brain"
    
    # Acquisition system information
    acq_system = ET.SubElement(header, "acquisitionSystemInformation")
    ET.SubElement(acq_system, "systemVendor").text = "Siemens_Healthineers"
    ET.SubElement(acq_system, "systemModel").text = "MAGNETOM_Skyra"
    ET.SubElement(acq_system, "systemFieldStrength_T").text = "3.0"
    ET.SubElement(acq_system, "receiverChannels").text = "1"
    
    # Experimental conditions
    exp_conditions = ET.SubElement(header, "experimentalConditions")
    ET.SubElement(exp_conditions, "H1resonanceFrequency_Hz").text = "123200000"
    
    # Encoding
    encoding = ET.SubElement(header, "encoding")
    
    # Encoded space
    encoded_space = ET.SubElement(encoding, "encodedSpace")
    matrix_size = ET.SubElement(encoded_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = "512"
    ET.SubElement(matrix_size, "y").text = "512"
    ET.SubElement(matrix_size, "z").text = "1"
    
    field_of_view = ET.SubElement(encoded_space, "fieldOfView_mm")
    ET.SubElement(field_of_view, "x").text = "256"
    ET.SubElement(field_of_view, "y").text = "256"
    ET.SubElement(field_of_view, "z").text = "4"
    
    # Recon space (same as encoded for images)
    recon_space = ET.SubElement(encoding, "reconSpace")
    matrix_size = ET.SubElement(recon_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = "512"
    ET.SubElement(matrix_size, "y").text = "512"
    ET.SubElement(matrix_size, "z").text = "1"
    
    field_of_view = ET.SubElement(recon_space, "fieldOfView_mm")
    ET.SubElement(field_of_view, "x").text = "256"
    ET.SubElement(field_of_view, "y").text = "256"
    ET.SubElement(field_of_view, "z").text = "4"
    
    # Sequence parameters for T2W
    seq_params = ET.SubElement(header, "sequenceParameters")
    ET.SubElement(seq_params, "TR").text = "1200"
    ET.SubElement(seq_params, "TE").text = "80"
    ET.SubElement(seq_params, "flipAngle_deg").text = "90"
    
    # Subject information
    subject = ET.SubElement(header, "subjectInformation")
    ET.SubElement(subject, "patientName").text = "TEST^FETAL^PATIENT"
    ET.SubElement(subject, "patientID").text = "FB_T2W_001"
    ET.SubElement(subject, "patientBirthdate").text = "1990-01-01"
    ET.SubElement(subject, "patientGender").text = "F"
    ET.SubElement(subject, "patientWeight_kg").text = "65"
    
    # Fetal-specific user parameters
    user_params = ET.SubElement(header, "userParameters")
    
    # Gestational age
    user_long = ET.SubElement(user_params, "userParameterLong")
    ET.SubElement(user_long, "name").text = "GestationalAge_weeks"
    ET.SubElement(user_long, "value").text = "32"
    
    # Fetal sequence identifier
    user_string = ET.SubElement(user_params, "userParameterString")
    ET.SubElement(user_string, "name").text = "SequenceType"
    ET.SubElement(user_string, "value").text = "T2W_HASTE_Fetal"
    
    return ET.tostring(header, encoding='unicode')

def load_nifti_data(nifti_path):
    """Load NIfTI data"""
    try:
        import nibabel as nib
        print(f"📂 Loading NIfTI file: {nifti_path}")
        
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        print(f"📊 NIfTI data shape: {data.shape}")
        print(f"📊 Data range: {data.min():.2f} - {data.max():.2f}")
        
        # Get voxel dimensions
        voxel_dims = nii.header.get_zooms()
        print(f"📊 Voxel dimensions: {voxel_dims}")
        
        return data, nii.header, voxel_dims
        
    except ImportError:
        print("❌ nibabel not available")
        return None, None, None
    except Exception as e:
        print(f"❌ Could not load NIfTI file: {e}")
        return None, None, None

def create_image_header(slice_idx, matrix_size, field_of_view, voxel_dims):
    """Create ISMRMRD image header"""
    # Create a minimal header structure as byte array
    # This is a simplified version - real ISMRMRD headers are more complex
    header_data = np.zeros(64, dtype=np.uint16)  # 128 bytes = 64 uint16 values
    
    # Basic header fields (simplified mapping)
    header_data[0] = 1  # version
    header_data[1] = 2  # data_type (magnitude image)
    header_data[2] = 0  # flags
    header_data[3] = 12345 & 0xFFFF  # measurement_uid (lower 16 bits)
    header_data[4] = matrix_size[0]  # matrix_size_x
    header_data[5] = matrix_size[1]  # matrix_size_y
    header_data[6] = matrix_size[2]  # matrix_size_z
    header_data[7] = int(field_of_view[0])  # field_of_view_x
    header_data[8] = int(field_of_view[1])  # field_of_view_y
    header_data[9] = int(field_of_view[2])  # field_of_view_z
    header_data[10] = 1  # channels
    header_data[11] = slice_idx  # slice
    header_data[12] = slice_idx  # image_index
    
    return header_data.tobytes()

def create_ismrmrd_images(output_path, nifti_path):
    """Create ISMRMRD file with image data (not raw k-space)"""
    print(f"🔄 Creating ISMRMRD image file: {output_path}")
    
    # Load NIfTI data
    image_data, nifti_header, voxel_dims = load_nifti_data(nifti_path)
    
    if image_data is None:
        print("❌ Failed to load NIfTI data")
        return False
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create dataset group
        dataset_group = f.create_group('dataset')
        
        # Create XML header
        header_xml = create_mrd_header()
        xml_dataset = dataset_group.create_dataset('xml', data=[header_xml], dtype=h5py.string_dtype(encoding='utf-8'))
        
        # Handle different data shapes
        if len(image_data.shape) == 3:
            # 3D volume: (height, width, slices) or (slices, height, width)
            if image_data.shape[0] < image_data.shape[2]:
                # Likely (slices, height, width) -> transpose to (height, width, slices)
                image_data = np.transpose(image_data, (1, 2, 0))
            
            num_slices = image_data.shape[2]
            height, width = image_data.shape[0], image_data.shape[1]
        elif len(image_data.shape) == 2:
            # 2D image
            height, width = image_data.shape
            num_slices = 1
            image_data = image_data[:, :, np.newaxis]
        else:
            print(f"❌ Unsupported data shape: {image_data.shape}")
            return False
        
        print(f"📊 Processing {num_slices} slices of size {height}x{width}")
        
        # Limit slices for testing (take middle slices with brain content)
        start_slice = max(0, num_slices // 4)
        end_slice = min(num_slices, 3 * num_slices // 4)
        test_slices = min(5, end_slice - start_slice)  # Maximum 5 slices for testing
        
        print(f"📊 Using {test_slices} slices from range [{start_slice}:{end_slice}]")
        
        # Process selected slices as separate images
        for i in range(test_slices):
            slice_idx = start_slice + i * (end_slice - start_slice) // test_slices
            
            print(f"   📄 Processing slice {slice_idx + 1}/{num_slices}")
            
            slice_data = image_data[:, :, slice_idx]
            
            # Ensure data is in reasonable range and convert to complex
            slice_data = slice_data.astype(np.float32)
            if slice_data.max() > 0:
                slice_data = (slice_data / slice_data.max()) * 4095  # Scale to 12-bit range
            
            # Convert to complex64 (magnitude only, imaginary = 0)
            complex_data = slice_data.astype(np.complex64)
            
            # Reshape for ISMRMRD: [channels, kz, ky, kx] -> [1, 1, height, width]
            ismrmrd_data = complex_data.reshape(1, 1, height, width)
            
            # Create image group
            img_group = dataset_group.create_group(f'image_{i}')
            
            # Store image data
            img_group.create_dataset('data', data=ismrmrd_data)
            
            # Create header
            matrix_size = [width, height, 1]
            field_of_view = [256.0, 256.0, 4.0]  # mm
            header_bytes = create_image_header(slice_idx, matrix_size, field_of_view, voxel_dims)
            img_group.create_dataset('header', data=np.frombuffer(header_bytes, dtype=np.uint8))
            
            # Add empty attributes
            img_group.create_dataset('attributes', data=b'')
    
    print(f"✅ ISMRMRD image file created successfully: {output_path}")
    print(f"📁 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    return True

def main():
    """Main function to create ISMRMRD image data"""
    # Input NIfTI file (our real fetal brain data)
    nifti_input = "../fetal-brain-measurement/Inputs/fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    
    # Output ISMRMRD file
    ismrmrd_output = "fetal_brain_images.h5"
    
    print("🧠 Creating ISMRMRD image data from fetal brain NIfTI")
    print("=" * 60)
    
    # Check if input exists
    if not os.path.exists(nifti_input):
        print(f"❌ Input file not found: {nifti_input}")
        return False
    
    # Create ISMRMRD file
    success = create_ismrmrd_images(ismrmrd_output, nifti_input)
    
    if success:
        print("\n🎯 Next steps:")
        print(f"   1. Server should be running: docker run --rm --gpus all -p 9002:9002 openrecon-fetal:latest")
        print(f"   2. Send image data to server: python client.py {ismrmrd_output} -c openrecon -p 9002 -o fetal_brain_result.h5")
        print(f"   3. Check results for fetal measurements!")
        
        return True
    else:
        print("\n❌ Failed to create ISMRMRD image file")
        return False

if __name__ == "__main__":
    main()



