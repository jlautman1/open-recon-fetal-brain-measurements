#!/usr/bin/env python3
"""
Create ISMRMRD test data from NIfTI file for pipeline testing
This simulates data that would come from the MRI scanner
"""

import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from datetime import datetime

# Mock ISMRMRD classes for environments without ismrmrd library
class MockImage:
    def __init__(self, data):
        self.data = data.astype(np.complex64)
        self.head = MockImageHeader()
        self.attribute_string = ""
        
    @classmethod
    def from_array(cls, data):
        return cls(data)
    
    def getHead(self):
        return self.head
        
    def setHead(self, head):
        self.head = head

class MockImageHeader:
    def __init__(self):
        self.version = 1
        self.data_type = 1
        self.flags = 0
        self.measurement_uid = 12345
        self.matrix_size = [256, 256, 1]
        self.field_of_view = [256.0, 256.0, 5.0]
        self.channels = 1
        self.position = [0.0, 0.0, 0.0]
        self.read_dir = [1.0, 0.0, 0.0]
        self.phase_dir = [0.0, 1.0, 0.0]
        self.slice_dir = [0.0, 0.0, 1.0]
        self.patient_table_position = [0.0, 0.0, 0.0]
        self.average = 0
        self.slice = 0
        self.contrast = 0
        self.phase = 0
        self.repetition = 0
        self.set = 0
        self.acquisition_time_stamp = 0
        self.physiology_time_stamp = [0, 0, 0]
        self.image_type = 1
        self.image_index = 0
        self.image_series_index = 1
        self.user_int = [0] * 8
        self.user_float = [0.0] * 8

def create_mrd_header():
    """Create basic MRD XML header"""
    header = ET.Element("ismrmrdHeader")
    
    # Study information
    study = ET.SubElement(header, "studyInformation")
    ET.SubElement(study, "studyDate").text = datetime.now().strftime("%Y-%m-%d")
    ET.SubElement(study, "studyTime").text = datetime.now().strftime("%H:%M:%S")
    ET.SubElement(study, "studyID").text = "FETAL_BRAIN_TEST"
    ET.SubElement(study, "studyDescription").text = "Fetal Brain T2W Test"
    
    # Measurement information
    measurement = ET.SubElement(header, "measurementInformation")
    ET.SubElement(measurement, "measurementID").text = "12345"
    ET.SubElement(measurement, "seriesDate").text = datetime.now().strftime("%Y-%m-%d")
    ET.SubElement(measurement, "seriesTime").text = datetime.now().strftime("%H:%M:%S")
    ET.SubElement(measurement, "patientPosition").text = "HFS"
    
    # Acquisition system information
    acq_system = ET.SubElement(header, "acquisitionSystemInformation")
    ET.SubElement(acq_system, "systemVendor").text = "Siemens_Healthineers"
    ET.SubElement(acq_system, "systemModel").text = "MAGNETOM_Skyra"
    ET.SubElement(acq_system, "systemFieldStrength_T").text = "3.0"
    
    # Experimental conditions
    exp_conditions = ET.SubElement(header, "experimentalConditions")
    ET.SubElement(exp_conditions, "H1resonanceFrequency_Hz").text = "123200000"
    
    # Encoding
    encoding = ET.SubElement(header, "encoding")
    
    # Encoded space
    encoded_space = ET.SubElement(encoding, "encodedSpace")
    matrix_size = ET.SubElement(encoded_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = "256"
    ET.SubElement(matrix_size, "y").text = "256"
    ET.SubElement(matrix_size, "z").text = "1"
    
    field_of_view = ET.SubElement(encoded_space, "fieldOfView_mm")
    ET.SubElement(field_of_view, "x").text = "256"
    ET.SubElement(field_of_view, "y").text = "256"
    ET.SubElement(field_of_view, "z").text = "5"
    
    # Recon space (same as encoded for simplicity)
    recon_space = ET.SubElement(encoding, "reconSpace")
    matrix_size = ET.SubElement(recon_space, "matrixSize")
    ET.SubElement(matrix_size, "x").text = "256"
    ET.SubElement(matrix_size, "y").text = "256"
    ET.SubElement(matrix_size, "z").text = "1"
    
    field_of_view = ET.SubElement(recon_space, "fieldOfView_mm")
    ET.SubElement(field_of_view, "x").text = "256"
    ET.SubElement(field_of_view, "y").text = "256"
    ET.SubElement(field_of_view, "z").text = "5"
    
    # Sequence parameters
    seq_params = ET.SubElement(header, "sequenceParameters")
    ET.SubElement(seq_params, "TR").text = "3000"
    ET.SubElement(seq_params, "TE").text = "80"
    
    # Subject information
    subject = ET.SubElement(header, "subjectInformation")
    ET.SubElement(subject, "patientName").text = "TEST^FETAL^BRAIN"
    ET.SubElement(subject, "patientID").text = "FB001"
    ET.SubElement(subject, "patientBirthdate").text = "1990-01-01"
    ET.SubElement(subject, "patientGender").text = "F"
    
    return ET.tostring(header, encoding='unicode')

def load_nifti_data(nifti_path):
    """Load NIfTI data and convert to format suitable for ISMRMRD"""
    if nifti_path is None or not os.path.exists(nifti_path):
        print("❌ NIfTI file not found, creating synthetic test data")
        # Create synthetic brain-like data
        data = np.random.rand(256, 256, 30) * 1000
        # Add some structure to make it brain-like
        center_y, center_x = 128, 128
        y, x = np.ogrid[:256, :256]
        mask = (x - center_x)**2 + (y - center_y)**2 < 80**2
        for z in range(30):
            data[:, :, z] = mask * (np.random.rand(256, 256) * 800 + 200)
        
        complex_data = data.astype(np.complex64)
        return complex_data, None
    
    try:
        import nibabel as nib
        print(f"📂 Loading NIfTI file: {nifti_path}")
        
        nii = nib.load(nifti_path)
        data = nii.get_fdata()
        print(f"📊 NIfTI data shape: {data.shape}")
        print(f"📊 Data range: {data.min():.2f} - {data.max():.2f}")
        
        # Convert to complex format for ISMRMRD
        # Normalize to reasonable range
        data_norm = (data - data.min()) / (data.max() - data.min()) * 1000
        
        # Convert to complex (real part only, imaginary = 0)
        complex_data = data_norm.astype(np.complex64)
        
        return complex_data, nii.header
        
    except ImportError:
        print("❌ nibabel not available, creating synthetic test data")
        # Create synthetic brain-like data
        data = np.random.rand(256, 256, 30) * 1000
        # Add some structure to make it brain-like
        center_y, center_x = 128, 128
        y, x = np.ogrid[:256, :256]
        mask = (x - center_x)**2 + (y - center_y)**2 < 80**2
        for z in range(30):
            data[:, :, z] = mask * (np.random.rand(256, 256) * 800 + 200)
        
        complex_data = data.astype(np.complex64)
        return complex_data, None

def create_ismrmrd_file(output_path, nifti_path):
    """Create ISMRMRD HDF5 file from NIfTI data"""
    print(f"🔄 Creating ISMRMRD file: {output_path}")
    
    # Load NIfTI data
    image_data, nifti_header = load_nifti_data(nifti_path)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create dataset group
        dataset_group = f.create_group('dataset')
        
        # Create XML header (store as 1D array of strings as expected by ISMRMRD)
        header_xml = create_mrd_header()
        xml_dataset = dataset_group.create_dataset('xml', data=[header_xml], dtype=h5py.string_dtype(encoding='utf-8'))
        
        # Prepare image data
        if len(image_data.shape) == 3:
            # Multiple slices
            num_slices = image_data.shape[2]
            print(f"📊 Processing {num_slices} slices")
        else:
            # Single slice
            image_data = image_data[:, :, np.newaxis]
            num_slices = 1
            print(f"📊 Processing 1 slice")
        
        # Process each slice as separate image group (as expected by client)
        for slice_idx in range(min(num_slices, 5)):  # Limit to 5 slices for testing
            print(f"   📄 Processing slice {slice_idx + 1}/{min(num_slices, 5)}")
            
            slice_data = image_data[:, :, slice_idx]
            
            # Reshape for ISMRMRD format: [channels, kz, ky, kx]
            # For reconstructed images: [1, 1, ny, nx]
            ismrmrd_data = slice_data.reshape(1, 1, slice_data.shape[0], slice_data.shape[1])
            
            # Create image header
            header = MockImageHeader()
            header.slice = slice_idx
            header.image_index = slice_idx
            header.matrix_size = [slice_data.shape[1], slice_data.shape[0], 1]  # [nx, ny, nz]
            
            # Store image data in the expected format: /group/image_X/
            img_group = dataset_group.create_group(f'image_{slice_idx}')
            img_group.create_dataset('data', data=ismrmrd_data)
            
            # Create header dataset (simplified)
            header_data = np.array([
                header.version, header.data_type, header.flags,
                header.measurement_uid, header.matrix_size[0], header.matrix_size[1], header.matrix_size[2],
                header.field_of_view[0], header.field_of_view[1], header.field_of_view[2],
                header.channels, header.slice, header.image_index
            ], dtype=np.uint32)
            
            img_group.create_dataset('header', data=header_data)
            
            # Add empty attributes
            img_group.create_dataset('attributes', data=b'')
    
    print(f"✅ ISMRMRD file created successfully: {output_path}")
    print(f"📁 File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

def main():
    """Main function to create test ISMRMRD data"""
    # Input NIfTI file (from your existing test data)
    nifti_input = "fetal-brain-measurement/Inputs/fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    
    # Output ISMRMRD file
    ismrmrd_output = "test_fetal_brain.h5"
    
    print("🧪 Creating ISMRMRD test data for fetal brain pipeline")
    print("=" * 60)
    
    # Check if input exists
    if not os.path.exists(nifti_input):
        print(f"⚠️  Input file not found: {nifti_input}")
        print("   Creating synthetic test data instead...")
        nifti_input = None
    else:
        print(f"✅ Found input file: {nifti_input}")
    
    # Create ISMRMRD file
    create_ismrmrd_file(ismrmrd_output, nifti_input)
    
    print("\n🎯 Next steps:")
    print(f"   1. Server should be running: python python-ismrmrd-server/main.py -v -d=openrecon")
    print(f"   2. Send data to server: python python-ismrmrd-server/client.py {ismrmrd_output} -c openrecon -p 9002 -o output_result.h5")
    print(f"   3. Check output file: output_result.h5")

if __name__ == "__main__":
    main()
