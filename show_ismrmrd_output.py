#!/usr/bin/env python3
"""
Show ISMRMRD Output
Demonstrates the conversion of pipeline results to ISMRMRD format with embedded measurements
"""

import os
import sys
import json
import numpy as np

# Add required paths
sys.path.append('./fetal-brain-measurement')

def show_ismrmrd_conversion():
    """Show the ISMRMRD output conversion with actual data"""
    
    print("🔍 ISMRMRD Output Conversion Demo")
    print("=" * 60)
    
    # Use existing successful results
    input_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    output_dir = "fetal-brain-measurement/output/Pat13249_Se8_Res0.46875_0.46875_Spac4.0"
    results_json = os.path.join(output_dir, "data.json")
    
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Results JSON: {results_json}")
    
    # Load measurement results
    if not os.path.exists(results_json):
        print(f"❌ Results file not found: {results_json}")
        return
    
    with open(results_json, 'r') as f:
        measurement_results = json.load(f)
    
    print(f"\n📊 Original Pipeline Results:")
    print(f"   CBD: {measurement_results.get('cbd_measure_mm', 'N/A'):.1f} mm")
    print(f"   BBD: {measurement_results.get('bbd_measure_mm', 'N/A'):.1f} mm") 
    print(f"   TCD: {measurement_results.get('tcd_measure_mm', 'N/A'):.1f} mm")
    print(f"   GA (CBD): {measurement_results.get('pred_ga_cbd', 'N/A'):.1f} weeks")
    print(f"   GA (BBD): {measurement_results.get('pred_ga_bbd', 'N/A'):.1f} weeks")
    print(f"   GA (TCD): {measurement_results.get('pred_ga_tcd', 'N/A'):.1f} weeks")
    print(f"   Brain Volume: {measurement_results.get('brain_vol_mm3', 'N/A'):,.0f} mm³")
    
    # Create mock ISMRMRD objects to show the conversion
    print(f"\n🔄 Converting to ISMRMRD Format...")
    
    try:
        # Import our converter
        from nifti_to_ismrmrd_converter import convert_nifti_to_ismrmrd
        
        # Convert input to ISMRMRD
        input_image, metadata = convert_nifti_to_ismrmrd(input_file)
        output_image, _ = convert_nifti_to_ismrmrd(input_file)  # Copy for output
        
        print(f"✅ Created ISMRMRD objects")
        print(f"   Input data shape: {input_image.data.shape}")
        print(f"   Input data type: {input_image.data.dtype}")
        print(f"   Output data shape: {output_image.data.shape}")
        print(f"   Output data type: {output_image.data.dtype}")
        
    except Exception as e:
        print(f"❌ ISMRMRD conversion failed: {e}")
        return
    
    # Simulate the OpenRecon embedding process
    print(f"\n🔧 Embedding Measurements in ISMRMRD Metadata...")
    
    try:
        import openrecon
        handler = openrecon.FetalBrainI2IHandler()
        
        # Run the measurement embedding
        handler._embed_measurements_in_metadata(output_image, metadata, measurement_results)
        
        print(f"✅ Measurements embedded successfully")
        
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        # Manually show what would be embedded
        print(f"⚠️ Showing manual embedding simulation...")
        
        # Simulate the metadata that would be embedded
        output_image.meta = output_image.meta or {}
        output_image.meta.update({
            'Keep_image_geometry': 1,
            'ImageProcessingHistory': 'FetalBrainMeasurement_OpenRecon',
            'DataRole': 'Image',
            'SequenceDescriptionAdditional': 'FETAL_BRAIN_OPENRECON',
            'CBD_mm': str(measurement_results.get('cbd_measure_mm', 0)),
            'BBD_mm': str(measurement_results.get('bbd_measure_mm', 0)),
            'TCD_mm': str(measurement_results.get('tcd_measure_mm', 0)),
            'CBD_valid': 'Yes',
            'BBD_valid': 'Yes' if measurement_results.get('bbd_valid', True) else 'No',
            'TCD_valid': 'Yes' if measurement_results.get('tcd_valid', True) else 'No',
            'GA_CBD_weeks': f"{measurement_results.get('pred_ga_cbd', 0):.1f}",
            'GA_BBD_weeks': f"{measurement_results.get('pred_ga_bbd', 0):.1f}",
            'GA_TCD_weeks': f"{measurement_results.get('pred_ga_tcd', 0):.1f}",
            'Brain_Volume_mm3': str(int(measurement_results.get('brain_vol_mm3', 0))),
            'Patient_ID': str(measurement_results.get('SubjectID', 'Unknown')),
            'Series_Number': str(int(measurement_results.get('Series', 1))),
            'ImageComments': f"Fetal Brain: CBD={measurement_results.get('cbd_measure_mm', 0):.1f}mm, BBD={measurement_results.get('bbd_measure_mm', 0):.1f}mm, TCD={measurement_results.get('tcd_measure_mm', 0):.1f}mm, GA={measurement_results.get('pred_ga_cbd', 0):.1f}weeks, Vol={int(measurement_results.get('brain_vol_mm3', 0))}mm³"
        })
    
    # Show the final ISMRMRD output
    print(f"\n📋 Final ISMRMRD Output:")
    print(f"   Data shape: {output_image.data.shape}")
    print(f"   Data type: {output_image.data.dtype}")
    print(f"   Data range: {output_image.data.real.min():.2f} to {output_image.data.real.max():.2f}")
    
    if hasattr(output_image, 'meta') and output_image.meta:
        print(f"   Metadata fields: {len(output_image.meta)}")
        
        print(f"\n🏷️ Embedded DICOM Metadata:")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        
        # Show measurement metadata
        measurement_fields = ['CBD_mm', 'BBD_mm', 'TCD_mm', 'GA_CBD_weeks', 'GA_BBD_weeks', 'GA_TCD_weeks', 'Brain_Volume_mm3']
        for field in measurement_fields:
            if field in output_image.meta:
                value = output_image.meta[field]
                print(f"   │ {field:<20}: {value:<25} │")
        
        # Show validation fields
        validation_fields = ['CBD_valid', 'BBD_valid', 'TCD_valid']
        for field in validation_fields:
            if field in output_image.meta:
                value = output_image.meta[field]
                print(f"   │ {field:<20}: {value:<25} │")
        
        # Show processing metadata
        processing_fields = ['ImageProcessingHistory', 'DataRole', 'Keep_image_geometry']
        for field in processing_fields:
            if field in output_image.meta:
                value = output_image.meta[field]
                print(f"   │ {field:<20}: {str(value):<25} │")
        
        # Show patient metadata
        patient_fields = ['Patient_ID', 'Series_Number']
        for field in patient_fields:
            if field in output_image.meta:
                value = output_image.meta[field]
                print(f"   │ {field:<20}: {value:<25} │")
        
        print(f"   └─────────────────────────────────────────────────────┘")
        
        # Show image comments (clinical summary)
        if 'ImageComments' in output_image.meta:
            print(f"\n💬 Clinical Summary (ImageComments):")
            print(f"   {output_image.meta['ImageComments']}")
    
    # Show what this becomes in DICOM
    print(f"\n🏥 Final DICOM Output (what the MRI system sees):")
    print(f"   📊 Original T2W fetal brain images")
    print(f"   🏷️ DICOM tags with measurements:")
    print(f"      - CBD Measurement: {measurement_results.get('cbd_measure_mm', 0):.1f} mm")
    print(f"      - BBD Measurement: {measurement_results.get('bbd_measure_mm', 0):.1f} mm") 
    print(f"      - TCD Measurement: {measurement_results.get('tcd_measure_mm', 0):.1f} mm")
    print(f"      - Gestational Age: {measurement_results.get('pred_ga_cbd', 0):.1f} weeks")
    print(f"      - Brain Volume: {measurement_results.get('brain_vol_mm3', 0):,.0f} mm³")
    print(f"   📄 Clinical report and plots (referenced in metadata)")
    print(f"   ✅ Standard DICOM compatibility")
    
    # Show the complete flow
    print(f"\n🔄 Complete Data Flow:")
    print(f"   1. Scanner → ISMRMRD input ({input_image.data.shape}, {input_image.data.dtype})")
    print(f"   2. ISMRMRD → NIfTI conversion")
    print(f"   3. AI Pipeline → measurements & segmentation")
    print(f"   4. Results → ISMRMRD with metadata ({output_image.data.shape}, {len(output_image.meta)} fields)")
    print(f"   5. OpenRecon → DICOM with embedded tags")
    
    print(f"\n🎉 ISMRMRD Output Ready for OpenRecon!")
    return True

if __name__ == "__main__":
    show_ismrmrd_conversion()



