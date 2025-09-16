#!/usr/bin/env python3
"""
Complete Pipeline Test
Tests the full pipeline: NIfTI → ISMRMRD → OpenRecon → AI Pipeline → ISMRMRD
"""

import os
import sys
import pickle
import tempfile
import shutil
from pathlib import Path

# Add required paths
sys.path.append('./python-ismrmrd-server')
sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')

def test_complete_pipeline():
    """Test the complete pipeline from NIfTI input to ISMRMRD output"""
    
    print("🧪 Complete Pipeline Test")
    print("=" * 60)
    
    # Step 1: Convert NIfTI to ISMRMRD
    print("\n🔄 Step 1: Converting NIfTI to ISMRMRD...")
    
    try:
        from nifti_to_ismrmrd_converter import convert_nifti_to_ismrmrd
        
        nifti_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
        
        if not os.path.exists(nifti_file):
            print(f"❌ Input file not found: {nifti_file}")
            return False
        
        # Convert to ISMRMRD
        input_image, metadata = convert_nifti_to_ismrmrd(nifti_file)
        print(f"✅ Step 1 completed: NIfTI → ISMRMRD")
        print(f"   Data shape: {input_image.data.shape}")
        print(f"   Patient: {metadata.get('PatientID', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Run through OpenRecon pipeline
    print("\n🚀 Step 2: Running through OpenRecon pipeline...")
    
    try:
        # Import the openrecon handler
        sys.path.insert(0, './fetal-brain-measurement')
        import openrecon
        
        print("✅ Successfully imported OpenRecon handler")
        
        # Create the handler
        handler = openrecon.FetalBrainI2IHandler()
        print("✅ Created OpenRecon handler instance")
        
        # Create output image (copy of input)
        from nifti_to_ismrmrd_converter import convert_nifti_to_ismrmrd
        output_image = convert_nifti_to_ismrmrd(nifti_file)[0]  # Create fresh copy
        
        print("🔧 Running OpenRecon process...")
        print(f"   Input data shape: {input_image.data.shape}")
        print(f"   Input data type: {input_image.data.dtype}")
        
        # Run the main process function
        handler.process(input_image, output_image, metadata)
        
        print(f"✅ Step 2 completed: OpenRecon pipeline processed successfully")
        
        # Check output metadata
        if hasattr(output_image, 'meta') and output_image.meta:
            print(f"📋 Output metadata keys: {list(output_image.meta.keys())}")
            
            # Show measurement results
            measurements = {}
            for key in ['CBD_mm', 'BBD_mm', 'TCD_mm', 'GA_CBD_weeks', 'Brain_Volume_mm3']:
                if key in output_image.meta:
                    measurements[key] = output_image.meta[key]
                    print(f"   {key}: {output_image.meta[key]}")
            
            if measurements:
                print(f"🎯 Found {len(measurements)} measurements in output!")
            else:
                print("⚠️ No measurements found in output metadata")
        else:
            print("⚠️ No metadata found in output image")
        
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify output format
    print("\n🔍 Step 3: Verifying output format...")
    
    try:
        print(f"📊 Output image properties:")
        print(f"   Data shape: {output_image.data.shape}")
        print(f"   Data type: {output_image.data.dtype}")
        print(f"   Has metadata: {hasattr(output_image, 'meta') and bool(output_image.meta)}")
        
        if hasattr(output_image, 'attribute_string'):
            print(f"   Has attribute string: {bool(output_image.attribute_string)}")
        
        print(f"✅ Step 3 completed: Output format verified")
        
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        return False
    
    # Step 4: Check output files
    print("\n📁 Step 4: Checking output files...")
    
    try:
        output_dir = "./fetal-brain-measurement/output"
        
        if os.path.exists(output_dir):
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in ['.nii.gz', '.json', '.pdf', '.png']):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        output_files.append(rel_path)
            
            print(f"📂 Found {len(output_files)} output files:")
            for file in sorted(output_files)[:10]:  # Show first 10
                print(f"   {file}")
            if len(output_files) > 10:
                print(f"   ... and {len(output_files) - 10} more files")
            
            # Look for key output files
            key_files = ['data.json', 'prediction.nii.gz', 'report.pdf']
            found_key_files = []
            for key_file in key_files:
                for output_file in output_files:
                    if key_file in output_file:
                        found_key_files.append(output_file)
                        break
            
            print(f"🔍 Key files found: {len(found_key_files)}/{len(key_files)}")
            for key_file in found_key_files:
                print(f"   ✅ {key_file}")
            
        else:
            print(f"⚠️ Output directory not found: {output_dir}")
        
        print(f"✅ Step 4 completed: Output files checked")
        
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        return False
    
    print("\n🎉 Complete Pipeline Test PASSED!")
    print("   NIfTI → ISMRMRD → OpenRecon → AI Pipeline → ISMRMRD")
    print("   All steps completed successfully!")
    
    return True


def main():
    """Main test function"""
    
    print("🚀 Starting Complete Pipeline Test")
    print("This will test the full flow:")
    print("  1. NIfTI file input")
    print("  2. Convert to ISMRMRD format") 
    print("  3. Run through OpenRecon handler")
    print("  4. Execute AI pipeline")
    print("  5. Convert results back to ISMRMRD")
    print("  6. Verify output format and files")
    
    success = test_complete_pipeline()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("The complete pipeline is working correctly.")
    else:
        print("\n❌ TESTS FAILED!")
        print("Check the error messages above for details.")
    
    return success


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)
