#!/usr/bin/env python3

import sys
import os
import tempfile
import numpy as np
import nibabel as nib

# Add the paths needed for OpenRecon
sys.path.append('/opt/code/python-ismrmrd-server')
sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master')

def create_mock_ismrmrd_image(nifti_file):
    """Create a mock ISMRMRD-like image from NIfTI data for testing"""
    
    print(f"📖 Loading NIfTI file: {nifti_file}")
    nii = nib.load(nifti_file)
    data = nii.get_fdata()
    print(f"📐 NIfTI data shape: {data.shape}")
    print(f"🔢 Value range: {data.min()} - {data.max()}")
    
    # Create a simple mock object that has the data attribute
    class MockISMRMRDImage:
        def __init__(self, data):
            self.data = data
            self.meta = {}
        
        def __str__(self):
            return f"MockISMRMRDImage(shape={self.data.shape})"
    
    # Create mock input and output images
    input_image = MockISMRMRDImage(data)
    output_image = MockISMRMRDImage(data.copy())
    
    return input_image, output_image

def test_openrecon_fetal_pipeline():
    """Test the OpenRecon fetal brain pipeline with our converted DICOM data"""
    
    print("🧪 Testing OpenRecon Fetal Brain Pipeline")
    print("=" * 60)
    
    # Use the full series NIfTI file we created
    nifti_file = "/workspace/test_series_output.nii.gz"
    
    if not os.path.exists(nifti_file):
        print(f"❌ ERROR: NIfTI file not found: {nifti_file}")
        return False
    
    try:
        # Create mock ISMRMRD images
        print("🏗️ Creating mock ISMRMRD images...")
        input_image, output_image = create_mock_ismrmrd_image(nifti_file)
        
        # Create metadata
        metadata = {
            'config': 'openrecon',
            'enable_measurements': True,
            'enable_reporting': True,
            'confidence_threshold': 0.5,
            'PatientName': 'HAREL^CARMEL',
            'StudyDescription': 'MRI FETAL',
            'SeriesDescription': 'cor_t2_haste',
            'PixelSpacing': [0.375, 0.375],
            'SliceThickness': 3.0
        }
        
        print("📋 Created metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Import and test the OpenRecon handler
        print("\n📦 Importing OpenRecon handler...")
        
        # Try to import the openrecon module
        try:
            import openrecon
            print("✅ Successfully imported openrecon module")
            
            # Test the main process function
            print("\n🚀 Running OpenRecon fetal brain pipeline...")
            result = openrecon.process_image(input_image, output_image, metadata)
            
            print(f"✅ Pipeline completed successfully!")
            print(f"📊 Result: {result}")
            
            # Check if output image has embedded metadata
            print("\n📋 Output image metadata:")
            if hasattr(output_image, 'meta') and output_image.meta:
                for key, value in output_image.meta.items():
                    print(f"   {key}: {value}")
            else:
                print("   No metadata found in output image")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR importing or running OpenRecon: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ ERROR in test setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_fetal_pipeline():
    """Test the fetal pipeline directly without OpenRecon wrapper"""
    
    print("\n🧪 Testing Direct Fetal Brain Pipeline")
    print("=" * 60)
    
    # Use the full series NIfTI file
    nifti_file = "/workspace/test_series_output.nii.gz"
    output_dir = "/workspace/test_direct_pipeline_output"
    
    if not os.path.exists(nifti_file):
        print(f"❌ ERROR: NIfTI file not found: {nifti_file}")
        return False
    
    try:
        # Import the fetal measurement module directly
        print("📦 Importing fetal measurement module...")
        sys.path.append('/workspace/fetal-brain-measurement/Code/FetalMeasurements-master')
        from fetal_measure import FetalMeasure
        
        print("✅ Successfully imported FetalMeasure")
        
        # Create FetalMeasure instance
        print("🏗️ Creating FetalMeasure instance...")
        fm = FetalMeasure()
        print("✅ FetalMeasure instance created successfully")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the pipeline
        print(f"\n🚀 Running fetal brain pipeline...")
        print(f"📁 Input: {nifti_file}")
        print(f"📁 Output: {output_dir}")
        
        result = fm.execute(nifti_file, output_dir)
        
        print("✅ Direct pipeline completed successfully!")
        
        # Check output files
        print("\n📂 Checking output files...")
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"📄 Found {len(output_files)} output files:")
            for file in sorted(output_files):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({size} bytes)")
            
            # Check specifically for report.pdf
            report_path = os.path.join(output_dir, 'report.pdf')
            if os.path.exists(report_path):
                print("🎉 PDF report was generated!")
            else:
                print("⚠️ PDF report not found")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in direct pipeline test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 OpenRecon Fetal Brain Pipeline Test Suite")
    print("=" * 80)
    
    # Test 1: OpenRecon wrapper (simulated ISMRMRD input)
    success1 = test_openrecon_fetal_pipeline()
    
    # Test 2: Direct fetal pipeline (NIfTI input)
    success2 = test_direct_fetal_pipeline()
    
    print("\n" + "=" * 80)
    print("📊 Test Results:")
    print(f"   OpenRecon wrapper test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Direct pipeline test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed!")

