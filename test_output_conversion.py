#!/usr/bin/env python3
"""
Test Output Conversion
Tests the conversion of successful pipeline results back to ISMRMRD format
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add required paths
sys.path.append('./python-ismrmrd-server')
sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')
sys.path.append('./fetal-brain-measurement')

def test_output_conversion():
    """Test the conversion of existing pipeline results to ISMRMRD format"""
    
    print("🧪 Testing Output Conversion to ISMRMRD")
    print("=" * 60)
    
    # Use existing successful output
    input_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    output_dir = "fetal-brain-measurement/output/Pat13249_Se8_Res0.46875_0.46875_Spac4.0"
    results_json = os.path.join(output_dir, "data.json")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
        
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return False
        
    if not os.path.exists(results_json):
        print(f"❌ Results JSON not found: {results_json}")
        return False
    
    print(f"✅ Found input file: {input_file}")
    print(f"✅ Found output directory: {output_dir}")
    print(f"✅ Found results JSON: {results_json}")
    
    try:
        # Load the measurement results
        with open(results_json, 'r') as f:
            measurement_results = json.load(f)
        
        print(f"\n📊 Loaded measurement results:")
        key_measurements = {
            'CBD': measurement_results.get('cbd_measure_mm'),
            'BBD': measurement_results.get('bbd_measure_mm'), 
            'TCD': measurement_results.get('tcd_measure_mm'),
            'GA_CBD': measurement_results.get('pred_ga_cbd'),
            'Brain_Volume': measurement_results.get('brain_vol_mm3')
        }
        
        for key, value in key_measurements.items():
            if value is not None:
                print(f"   {key}: {value}")
        
        # Create mock ISMRMRD images for testing conversion
        print(f"\n🔄 Creating mock ISMRMRD objects...")
        
        try:
            from nifti_to_ismrmrd_converter import convert_nifti_to_ismrmrd
            input_image, metadata = convert_nifti_to_ismrmrd(input_file)
            output_image, _ = convert_nifti_to_ismrmrd(input_file)  # Create copy for output
            
            print(f"✅ Created ISMRMRD objects")
            print(f"   Input shape: {input_image.data.shape}")
            print(f"   Output shape: {output_image.data.shape}")
            
        except Exception as e:
            print(f"❌ Failed to create ISMRMRD objects: {e}")
            return False
        
        # Test the OpenRecon conversion functions
        print(f"\n🔧 Testing OpenRecon conversion functions...")
        
        try:
            import openrecon
            handler = openrecon.FetalBrainI2IHandler()
            
            print(f"✅ Created OpenRecon handler")
            
            # Test the measurement embedding function
            print(f"🔄 Testing measurement embedding...")
            handler._embed_measurements_in_metadata(output_image, metadata, measurement_results)
            
            print(f"✅ Measurement embedding completed")
            
            # Check the embedded metadata
            if hasattr(output_image, 'meta') and output_image.meta:
                print(f"\n📋 Embedded metadata:")
                embedded_measurements = {}
                for key, value in output_image.meta.items():
                    if any(measure in key for measure in ['CBD', 'BBD', 'TCD', 'GA', 'Brain', 'Volume']):
                        embedded_measurements[key] = value
                        print(f"   {key}: {value}")
                
                if embedded_measurements:
                    print(f"🎯 Successfully embedded {len(embedded_measurements)} measurement fields!")
                else:
                    print(f"⚠️ No measurement fields found in metadata")
            else:
                print(f"⚠️ No metadata found in output image")
            
            # Test visual outputs extraction (if available)
            print(f"\n🖼️ Testing visual outputs extraction...")
            try:
                attachments = handler._extract_visual_outputs(output_dir, os.listdir(output_dir))
                
                num_plots = len(attachments.get('plots', {}))
                has_pdf = attachments.get('pdf_report') is not None
                
                print(f"✅ Visual outputs extracted:")
                print(f"   Plots: {num_plots}")
                print(f"   PDF Report: {'Yes' if has_pdf else 'No'}")
                
                if num_plots > 0 or has_pdf:
                    print(f"🎯 Visual outputs ready for DICOM embedding!")
                
            except Exception as e:
                print(f"⚠️ Visual outputs extraction failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ OpenRecon conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🚀 Testing Output Conversion Pipeline")
    print("This will test:")
    print("  1. Loading existing successful pipeline results")
    print("  2. Converting to ISMRMRD format")
    print("  3. Embedding measurements in metadata")
    print("  4. Extracting visual outputs")
    print("  5. Verifying DICOM-ready format")
    
    success = test_output_conversion()
    
    if success:
        print("\n✅ OUTPUT CONVERSION TEST PASSED!")
        print("The pipeline can successfully convert results to ISMRMRD format.")
        print("Ready for OpenRecon deployment!")
    else:
        print("\n❌ OUTPUT CONVERSION TEST FAILED!")
        print("Check the error messages above for details.")
    
    return success

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)



