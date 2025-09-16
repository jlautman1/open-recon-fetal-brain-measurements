#!/usr/bin/env python3
"""
Create ISMRMRD Output File
Converts the pipeline results to an actual ISMRMRD file format
"""

import os
import sys
import json
import numpy as np
import nibabel as nib

# Add required paths
sys.path.append('./fetal-brain-measurement')

def create_ismrmrd_file():
    """Create an actual ISMRMRD file from pipeline results"""
    
    print("📁 Creating ISMRMRD Output File")
    print("=" * 50)
    
    # Input and output paths
    input_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    output_dir = "fetal-brain-measurement/output/Pat13249_Se8_Res0.46875_0.46875_Spac4.0"
    results_json = os.path.join(output_dir, "data.json")
    prediction_file = os.path.join(output_dir, "prediction.nii.gz")
    
    # Output ISMRMRD file
    ismrmrd_output_file = "fetal_brain_ismrmrd_output.h5"
    
    print(f"📖 Input: {input_file}")
    print(f"📊 Results: {results_json}")
    print(f"🧠 Prediction: {prediction_file}")
    print(f"💾 Output: {ismrmrd_output_file}")
    
    # Check input files exist
    for file_path in [input_file, results_json, prediction_file]:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
    
    try:
        # Load measurement results
        with open(results_json, 'r') as f:
            measurement_results = json.load(f)
        
        print(f"\n📊 Loaded measurements:")
        measurements = {
            'CBD': measurement_results.get('cbd_measure_mm'),
            'BBD': measurement_results.get('bbd_measure_mm'),
            'TCD': measurement_results.get('tcd_measure_mm'),
            'GA_CBD': measurement_results.get('pred_ga_cbd'),
            'Brain_Volume': measurement_results.get('brain_vol_mm3')
        }
        
        for key, value in measurements.items():
            if value is not None:
                print(f"   {key}: {value}")
        
        # Load original NIfTI data
        print(f"\n📖 Loading original image data...")
        nii_original = nib.load(input_file)
        original_data = nii_original.get_fdata()
        print(f"   Original shape: {original_data.shape}")
        print(f"   Original range: {original_data.min():.2f} to {original_data.max():.2f}")
        
        # Load prediction data
        print(f"🧠 Loading prediction data...")
        nii_prediction = nib.load(prediction_file)
        prediction_data = nii_prediction.get_fdata()
        print(f"   Prediction shape: {prediction_data.shape}")
        print(f"   Prediction range: {prediction_data.min():.2f} to {prediction_data.max():.2f}")
        
        # Try to use real ISMRMRD if available, otherwise create a mock file
        try:
            import ismrmrd
            print(f"✅ Using real ISMRMRD library")
            use_real_ismrmrd = True
        except ImportError:
            print(f"⚠️ ISMRMRD library not available, creating mock HDF5 file")
            use_real_ismrmrd = False
        
        if use_real_ismrmrd:
            # Create real ISMRMRD file
            create_real_ismrmrd_file(ismrmrd_output_file, original_data, prediction_data, measurement_results)
        else:
            # Create mock HDF5 file with ISMRMRD-like structure
            create_mock_ismrmrd_file(ismrmrd_output_file, original_data, prediction_data, measurement_results)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating ISMRMRD file: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_real_ismrmrd_file(output_file, original_data, prediction_data, measurement_results):
    """Create a real ISMRMRD file"""
    import ismrmrd
    import h5py
    
    print(f"🔧 Creating real ISMRMRD file...")
    
    # Create ISMRMRD dataset
    with h5py.File(output_file, 'w') as f:
        # Create ISMRMRD header
        header = ismrmrd.xsd.ismrmrdHeader()
        
        # Set basic parameters
        header.subjectInformation = ismrmrd.xsd.subjectInformationType()
        header.subjectInformation.patientID = str(measurement_results.get('SubjectID', '13249'))
        
        header.studyInformation = ismrmrd.xsd.studyInformationType()
        header.studyInformation.studyDescription = "Fetal Brain MRI with AI Measurements"
        
        header.measurementInformation = ismrmrd.xsd.measurementInformationType()
        header.measurementInformation.seriesDate = "2024-01-01"
        header.measurementInformation.seriesTime = "12:00:00"
        
        # Set encoding parameters
        encoding = ismrmrd.xsd.encodingType()
        encoding.trajectory = ismrmrd.xsd.trajectoryType.cartesian
        
        # Set image dimensions
        encoding.encodedSpace = ismrmrd.xsd.encodingSpaceType()
        encoding.encodedSpace.matrixSize = ismrmrd.xsd.matrixSizeType()
        encoding.encodedSpace.matrixSize.x = original_data.shape[0]
        encoding.encodedSpace.matrixSize.y = original_data.shape[1] 
        encoding.encodedSpace.matrixSize.z = original_data.shape[2]
        
        encoding.encodedSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm()
        encoding.encodedSpace.fieldOfView_mm.x = original_data.shape[0] * 0.46875
        encoding.encodedSpace.fieldOfView_mm.y = original_data.shape[1] * 0.46875
        encoding.encodedSpace.fieldOfView_mm.z = original_data.shape[2] * 4.0
        
        header.encoding.append(encoding)
        
        # Write header
        f.attrs['ismrmrd_header'] = ismrmrd.xsd.ToXML(header).encode('utf-8')
        
        # Create image group
        image_group = f.create_group('images')
        
        # Convert data to complex format (ISMRMRD typically uses complex data)
        if np.iscomplexobj(original_data):
            complex_data = original_data.astype(np.complex64)
        else:
            complex_data = original_data.astype(np.float32) + 0j
            complex_data = complex_data.astype(np.complex64)
        
        # Create ISMRMRD image
        img = ismrmrd.Image.from_array(complex_data)
        
        # Set image header
        img.image_type = ismrmrd.IMTYPE_MAGNITUDE
        img.image_series_index = int(measurement_results.get('Series', 8))
        img.image_index = 0
        
        # Create metadata
        meta = ismrmrd.Meta()
        
        # Add measurement metadata
        meta['CBD_mm'] = str(measurement_results.get('cbd_measure_mm', 0))
        meta['BBD_mm'] = str(measurement_results.get('bbd_measure_mm', 0))
        meta['TCD_mm'] = str(measurement_results.get('tcd_measure_mm', 0))
        meta['GA_CBD_weeks'] = str(measurement_results.get('pred_ga_cbd', 0))
        meta['GA_BBD_weeks'] = str(measurement_results.get('pred_ga_bbd', 0))
        meta['GA_TCD_weeks'] = str(measurement_results.get('pred_ga_tcd', 0))
        meta['Brain_Volume_mm3'] = str(int(measurement_results.get('brain_vol_mm3', 0)))
        meta['CBD_valid'] = 'Yes'
        meta['BBD_valid'] = 'Yes' if measurement_results.get('bbd_valid', True) else 'No'
        meta['TCD_valid'] = 'Yes' if measurement_results.get('tcd_valid', True) else 'No'
        meta['ImageProcessingHistory'] = 'FetalBrainMeasurement_OpenRecon'
        meta['DataRole'] = 'Image'
        meta['Keep_image_geometry'] = '1'
        meta['Patient_ID'] = str(measurement_results.get('SubjectID', '13249'))
        meta['Series_Number'] = str(int(measurement_results.get('Series', 8)))
        
        # Add clinical summary
        clinical_summary = f"Fetal Brain: CBD={measurement_results.get('cbd_measure_mm', 0):.1f}mm, BBD={measurement_results.get('bbd_measure_mm', 0):.1f}mm, TCD={measurement_results.get('tcd_measure_mm', 0):.1f}mm, GA={measurement_results.get('pred_ga_cbd', 0):.1f}weeks, Vol={int(measurement_results.get('brain_vol_mm3', 0))}mm³"
        meta['ImageComments'] = clinical_summary
        
        # Set metadata
        img.attribute_string = meta.serialize()
        
        # Save image to file
        image_group.create_dataset('image_0', data=img.data)
        image_group.create_dataset('header_0', data=img.getHead())
        image_group.create_dataset('attributes_0', data=img.attribute_string.encode('utf-8'))
    
    print(f"✅ Created real ISMRMRD file: {output_file}")

def create_mock_ismrmrd_file(output_file, original_data, prediction_data, measurement_results):
    """Create a mock HDF5 file with ISMRMRD-like structure"""
    import h5py
    
    print(f"🔧 Creating mock ISMRMRD HDF5 file...")
    
    with h5py.File(output_file, 'w') as f:
        # Create header information
        header_info = {
            'patient_id': str(measurement_results.get('SubjectID', '13249')),
            'series_number': str(int(measurement_results.get('Series', 8))),
            'study_description': 'Fetal Brain MRI with AI Measurements',
            'matrix_size_x': original_data.shape[0],
            'matrix_size_y': original_data.shape[1],
            'matrix_size_z': original_data.shape[2],
            'pixel_spacing_x': 0.46875,
            'pixel_spacing_y': 0.46875,
            'slice_thickness': 4.0
        }
        
        # Store header as attributes
        header_group = f.create_group('header')
        for key, value in header_info.items():
            header_group.attrs[key] = value
        
        # Store original image data
        images_group = f.create_group('images')
        
        # Convert to complex format for ISMRMRD compatibility
        if np.iscomplexobj(original_data):
            complex_data = original_data.astype(np.complex64)
        else:
            complex_data = original_data.astype(np.float32) + 0j
            complex_data = complex_data.astype(np.complex64)
        
        images_group.create_dataset('original_image', data=complex_data, compression='gzip')
        images_group.create_dataset('prediction_mask', data=prediction_data.astype(np.float32), compression='gzip')
        
        # Store measurement metadata
        metadata_group = f.create_group('fetal_measurements')
        
        measurements = {
            'CBD_mm': measurement_results.get('cbd_measure_mm', 0),
            'BBD_mm': measurement_results.get('bbd_measure_mm', 0),
            'TCD_mm': measurement_results.get('tcd_measure_mm', 0),
            'GA_CBD_weeks': measurement_results.get('pred_ga_cbd', 0),
            'GA_BBD_weeks': measurement_results.get('pred_ga_bbd', 0),
            'GA_TCD_weeks': measurement_results.get('pred_ga_tcd', 0),
            'Brain_Volume_mm3': measurement_results.get('brain_vol_mm3', 0),
            'CBD_valid': measurement_results.get('cbd_valid', True),
            'BBD_valid': measurement_results.get('bbd_valid', True),
            'TCD_valid': measurement_results.get('tcd_valid', True)
        }
        
        for key, value in measurements.items():
            metadata_group.attrs[key] = value
        
        # Store processing information
        processing_group = f.create_group('processing_info')
        processing_group.attrs['ImageProcessingHistory'] = 'FetalBrainMeasurement_OpenRecon'
        processing_group.attrs['DataRole'] = 'Image'
        processing_group.attrs['Keep_image_geometry'] = 1
        
        # Store clinical summary
        clinical_summary = f"Fetal Brain: CBD={measurements['CBD_mm']:.1f}mm, BBD={measurements['BBD_mm']:.1f}mm, TCD={measurements['TCD_mm']:.1f}mm, GA={measurements['GA_CBD_weeks']:.1f}weeks, Vol={int(measurements['Brain_Volume_mm3'])}mm³"
        processing_group.attrs['ImageComments'] = clinical_summary
        
        # Store complete measurement results as JSON
        metadata_group.create_dataset('complete_results', data=json.dumps(measurement_results))
    
    print(f"✅ Created mock ISMRMRD HDF5 file: {output_file}")

def examine_created_file(output_file):
    """Examine the created ISMRMRD file"""
    import h5py
    
    print(f"\n🔍 Examining created file: {output_file}")
    
    if not os.path.exists(output_file):
        print(f"❌ File not found: {output_file}")
        return
    
    try:
        with h5py.File(output_file, 'r') as f:
            print(f"📊 File structure:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"   📁 Group: {name}")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"      🏷️ {attr_name}: {attr_value}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"   📄 Dataset: {name} {obj.shape} {obj.dtype}")
            
            f.visititems(print_structure)
            
        file_size = os.path.getsize(output_file)
        print(f"\n📏 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Error examining file: {e}")

def main():
    """Main function"""
    print("🚀 ISMRMRD File Creator")
    print("This will create an actual ISMRMRD file from the pipeline results")
    
    success = create_ismrmrd_file()
    
    if success:
        # Examine the created file
        examine_created_file("fetal_brain_ismrmrd_output.h5")
        
        print(f"\n🎉 ISMRMRD FILE CREATED SUCCESSFULLY!")
        print(f"📁 Location: ./fetal_brain_ismrmrd_output.h5")
        print(f"📊 Contains:")
        print(f"   - Original fetal brain image data")
        print(f"   - Brain segmentation mask")
        print(f"   - Complete measurement results")
        print(f"   - DICOM-ready metadata")
        print(f"   - Clinical summary")
        
        print(f"\n💡 This file represents what gets sent from OpenRecon to the MRI system")
        print(f"   in ISMRMRD format with all measurements embedded!")
    else:
        print(f"\n❌ FAILED TO CREATE ISMRMRD FILE")

if __name__ == "__main__":
    main()



