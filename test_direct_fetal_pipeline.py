#!/usr/bin/env python3
"""
Direct Fetal Brain Pipeline Test
Tests the fetal brain pipeline directly using the ISMRMRD input we created
"""

import os
import sys
import numpy as np
import h5py
import json
from pathlib import Path

# Add required paths
sys.path.append('./python-ismrmrd-server')
sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')

def load_ismrmrd_as_nifti_data(ismrmrd_file):
    """Load ISMRMRD file and extract image data for the fetal brain pipeline"""
    
    print(f"📖 Loading ISMRMRD file: {ismrmrd_file}")
    
    try:
        with h5py.File(ismrmrd_file, 'r') as f:
            print("📊 ISMRMRD file structure:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"   Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"   Group: {name}")
            
            f.visititems(print_structure)
            
            # Look for image data in the dataset
            if 'dataset' not in f:
                print("❌ No dataset group found")
                return None
            
            dataset_group = f['dataset']
            
            # Look for images in different possible formats
            images_data = []
            
            # Check for image groups (image_0, image_1, etc.)
            image_groups = [key for key in dataset_group.keys() if key.startswith('image_')]
            image_groups.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically
            
            print(f"🔍 Found {len(image_groups)} image groups")
            
            for img_key in image_groups[:5]:  # Limit to first 5 images
                img_group = dataset_group[img_key]
                if 'data' in img_group:
                    data = img_group['data'][:]
                    print(f"   📄 {img_key}: {data.shape} - {data.dtype}")
                    
                    # Extract 2D slice from the 4D data [channels, kz, ky, kx]
                    if len(data.shape) == 4:
                        slice_data = data[0, 0, :, :]  # First channel, first z-slice
                    elif len(data.shape) == 3:
                        slice_data = data[0, :, :]  # First channel
                    else:
                        slice_data = data
                    
                    # Convert complex to magnitude
                    if np.iscomplexobj(slice_data):
                        slice_data = np.abs(slice_data)
                    
                    images_data.append(slice_data)
            
            if not images_data:
                print("❌ No image data found")
                return None
            
            # Stack images into 3D volume
            volume = np.stack(images_data, axis=2)  # (height, width, slices)
            
            print(f"✅ Loaded image data: {volume.shape}")
            print(f"   Data range: {volume.min():.2f} - {volume.max():.2f}")
            
            return volume.astype(np.float32)
            
    except Exception as e:
        print(f"❌ Error loading ISMRMRD file: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_as_nifti(data, output_path):
    """Save numpy data as NIfTI file for the fetal brain pipeline"""
    
    try:
        import nibabel as nib
        
        print(f"💾 Saving data as NIfTI: {output_path}")
        print(f"   Data shape: {data.shape}")
        
        # Create NIfTI image
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        
        # Save NIfTI file
        nib.save(nii, output_path)
        
        print(f"✅ Successfully saved NIfTI file")
        return True
        
    except Exception as e:
        print(f"❌ Error saving NIfTI file: {e}")
        return False

def run_fetal_brain_pipeline(nifti_input, output_dir):
    """Run the fetal brain measurement pipeline directly"""
    
    print(f"🧠 Running fetal brain measurement pipeline")
    print(f"   Input: {nifti_input}")
    print(f"   Output: {output_dir}")
    
    try:
        # Import the fetal measurement module
        from fetal_measure import FetalMeasure
        
        print("✅ Successfully imported FetalMeasure")
        
        # Create FetalMeasure instance
        fm = FetalMeasure()
        print("✅ FetalMeasure instance created")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the pipeline
        print("🚀 Executing fetal brain pipeline...")
        result = fm.execute(nifti_input, output_dir)
        
        print("✅ Fetal brain pipeline completed!")
        
        # Check output files
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            print(f"📂 Output files created: {len(output_files)}")
            for f in output_files:
                file_path = os.path.join(output_dir, f)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / 1024
                    print(f"   📄 {f} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running fetal brain pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_output_to_ismrmrd(output_dir, ismrmrd_output_file):
    """Convert pipeline output back to ISMRMRD format (simulating server response)"""
    
    print(f"🔄 Converting pipeline output to ISMRMRD format")
    print(f"   Input dir: {output_dir}")
    print(f"   Output: {ismrmrd_output_file}")
    
    try:
        # Look for measurement results (JSON files)
        results = []
        
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            
            if file.endswith('.json'):
                print(f"📄 Found results file: {file}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results.append(data)
        
        # Create a simple ISMRMRD file with the results
        with h5py.File(ismrmrd_output_file, 'w') as f:
            # Create dataset group
            dataset_group = f.create_group('dataset')
            
            # Create a simple XML header
            header_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<ismrmrdHeader>
    <studyInformation>
        <studyDescription>Fetal Brain Measurement Results</studyDescription>
    </studyInformation>
    <measurementInformation>
        <seriesDescription>T2W_HASTE_Fetal_Brain_Processed</seriesDescription>
    </measurementInformation>
</ismrmrdHeader>"""
            
            xml_dataset = dataset_group.create_dataset('xml', data=[header_xml], dtype=h5py.string_dtype(encoding='utf-8'))
            
            # Store measurement results as metadata
            if results:
                results_str = json.dumps(results, indent=2)
                dataset_group.create_dataset('measurement_results', 
                                           data=[results_str], 
                                           dtype=h5py.string_dtype(encoding='utf-8'))
                
                print(f"✅ Stored {len(results)} measurement results")
                
                # Display the measurements
                for i, result in enumerate(results):
                    print(f"📊 Results {i+1}:")
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            print(f"   {key}: {value:.2f}")
                        else:
                            print(f"   {key}: {value}")
            
            # Create a dummy image to make it a valid ISMRMRD file
            dummy_image = np.ones((1, 1, 128, 128), dtype=np.complex64)
            img_group = dataset_group.create_group('image_0')
            img_group.create_dataset('data', data=dummy_image)
            img_group.create_dataset('header', data=np.zeros(64, dtype=np.uint8))
            img_group.create_dataset('attributes', data=b'')
        
        print(f"✅ Successfully created ISMRMRD output file")
        print(f"   File: {ismrmrd_output_file}")
        print(f"   Size: {os.path.getsize(ismrmrd_output_file) / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting output to ISMRMRD: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to test the direct pipeline"""
    print("🧪 Direct Fetal Brain Pipeline Test")
    print("=" * 60)
    
    # Configuration
    ismrmrd_input = "test_fetal_brain_pipeline.h5"
    temp_nifti = "temp_input_for_pipeline.nii.gz"
    output_dir = "direct_pipeline_output"
    ismrmrd_output = "direct_pipeline_result.h5"
    
    # Step 1: Check ISMRMRD input
    print("\n📁 Step 1: Checking ISMRMRD input...")
    if not os.path.exists(ismrmrd_input):
        print(f"❌ ISMRMRD input file not found: {ismrmrd_input}")
        print("   Please run the NIfTI to ISMRMRD conversion first")
        return False
    print(f"✅ ISMRMRD input found: {ismrmrd_input}")
    
    # Step 2: Load ISMRMRD data
    print("\n📖 Step 2: Loading ISMRMRD data...")
    image_data = load_ismrmrd_as_nifti_data(ismrmrd_input)
    if image_data is None:
        print("❌ Failed to load image data from ISMRMRD file")
        return False
    
    # Step 3: Save as temporary NIfTI for pipeline
    print("\n💾 Step 3: Converting to NIfTI for pipeline...")
    if not save_as_nifti(image_data, temp_nifti):
        print("❌ Failed to save temporary NIfTI file")
        return False
    
    # Step 4: Run fetal brain pipeline
    print("\n🧠 Step 4: Running fetal brain pipeline...")
    if not run_fetal_brain_pipeline(temp_nifti, output_dir):
        print("❌ Fetal brain pipeline failed")
        return False
    
    # Step 5: Convert output back to ISMRMRD
    print("\n🔄 Step 5: Converting output to ISMRMRD...")
    if not convert_output_to_ismrmrd(output_dir, ismrmrd_output):
        print("❌ Failed to convert output to ISMRMRD")
        return False
    
    # Step 6: Cleanup
    print("\n🧹 Step 6: Cleaning up...")
    if os.path.exists(temp_nifti):
        os.remove(temp_nifti)
        print(f"   Removed temporary file: {temp_nifti}")
    
    print("\n🎉 Direct pipeline test completed successfully!")
    print(f"   Input ISMRMRD: {ismrmrd_input}")
    print(f"   Pipeline output: {output_dir}")
    print(f"   Result ISMRMRD: {ismrmrd_output}")
    print("\n✅ The complete NIfTI → ISMRMRD → Fetal Brain Pipeline → ISMRMRD flow is working!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Direct pipeline test failed!")
        sys.exit(1)


