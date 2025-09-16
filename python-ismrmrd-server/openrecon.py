#!/usr/bin/env python3
"""
OpenRecon i2i handler for Fetal Brain Measurement Pipeline
Based on working i2i_fetal_handler.py structure
Processes incoming ISMRMRD images and calls the fetal measurement pipeline.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import base64
from typing import Optional, Dict, Any

# Import ISMRMRD for OpenRecon integration
try:
    import ismrmrd
    print("✅ DEBUG: Successfully imported ismrmrd module")
except ImportError as e:
    print(f"⚠️ DEBUG: Failed to import ismrmrd: {e}")
    print("   Creating mock ISMRMRD for development...")
    # Create minimal mock for development
    class MockImage:
        def __init__(self, data):
            self.data = data
        def getHead(self):
            return {}
        def setHead(self, head):
            pass
        @classmethod
        def from_array(cls, data):
            return cls(data)
    
    class MockISMRMRD:
        Image = MockImage
    
    ismrmrd = MockISMRMRD()

# Type checking imports that won't fail if the module doesn't exist
try:
    from fetal_measure import FetalMeasure as FetalMeasureType  # type: ignore
except ImportError:
    # Create a placeholder type for development environments
    FetalMeasureType = Any

# Add the fetal measurement pipeline to the path - supports both local and Docker environments
script_dir = os.path.dirname(os.path.abspath(__file__))

# Try different possible paths for the fetal measurement module
possible_paths = [
    # Docker container paths
    '/workspace/fetal-brain-measurement/Code/FetalMeasurements-master',
    '/workspace/fetal-brain-measurement/Code/FetalMeasurements-master/SubSegmentation',
    # Local development paths (relative to this script)
    os.path.join(script_dir, 'Code', 'FetalMeasurements-master'),
    os.path.join(script_dir, 'Code', 'FetalMeasurements-master', 'SubSegmentation'),
    # Alternative local paths
    os.path.join(os.path.dirname(script_dir), 'fetal-brain-measurement', 'Code', 'FetalMeasurements-master'),
    os.path.join(os.path.dirname(script_dir), 'fetal-brain-measurement', 'Code', 'FetalMeasurements-master', 'SubSegmentation'),
]

# Add existing paths to sys.path
for path in possible_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"🔧 DEBUG: Added to sys.path: {path}")

# Verify we can find the fetal_measure module
try:
    import importlib.util
    fetal_measure_paths = [
        os.path.join(path, 'fetal_measure.py') for path in possible_paths 
        if os.path.exists(os.path.join(path, 'fetal_measure.py'))
    ]
    if fetal_measure_paths:
        print(f"✅ DEBUG: Found fetal_measure.py at: {fetal_measure_paths[0]}")
    else:
        print("⚠️ DEBUG: fetal_measure.py not found in any path")
except Exception as e:
    print(f"⚠️ DEBUG: Error checking for fetal_measure.py: {e}")

def _import_fetal_measure() -> Any:
    """
    Dynamically import the FetalMeasure class with robust error handling.
    Returns the FetalMeasure class or raises ImportError if not found.
    """
    # Try multiple import methods
    import_methods = [
        # Method 1: Direct import
        lambda: __import__('fetal_measure', fromlist=['FetalMeasure']).FetalMeasure,
        # Method 2: Package import
        lambda: __import__('Code.FetalMeasurements-master.fetal_measure', fromlist=['FetalMeasure']).FetalMeasure,
        # Method 3: Manual file loading
        lambda: _manual_import_fetal_measure()
    ]
    
    for i, method in enumerate(import_methods):
        try:
            FetalMeasure = method()
            print(f"✅ DEBUG: Successfully imported FetalMeasure using method {i+1}")
            return FetalMeasure
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            print(f"⚠️ DEBUG: Import method {i+1} failed: {e}")
            continue
    
    raise ImportError("Could not import FetalMeasure using any available method")

def _manual_import_fetal_measure() -> Any:
    """Manually import fetal_measure module from file system"""
    print("🔧 DEBUG: Attempting manual import of fetal_measure...")
    
    # Try to find and import the module manually
    fetal_measure_file = None
    for path in sys.path:
        potential_file = os.path.join(path, 'fetal_measure.py')
        if os.path.exists(potential_file):
            fetal_measure_file = potential_file
            break
    
    if not fetal_measure_file:
        print("🔍 DEBUG: Current sys.path entries:")
        for i, path in enumerate(sys.path[:10]):  # Show first 10 entries
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"   {i}: {exists} {path}")
        raise ImportError("Could not find fetal_measure.py in any sys.path location")
    
    print(f"📁 DEBUG: Found fetal_measure.py at: {fetal_measure_file}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fetal_measure", fetal_measure_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {fetal_measure_file}")
    
    fetal_measure_module = importlib.util.module_from_spec(spec)
    sys.modules["fetal_measure"] = fetal_measure_module
    spec.loader.exec_module(fetal_measure_module)
    
    if not hasattr(fetal_measure_module, 'FetalMeasure'):
        raise AttributeError("FetalMeasure class not found in fetal_measure module")
    
    return fetal_measure_module.FetalMeasure

class FetalBrainI2IHandler:
    """Handler for fetal brain measurement in i2i workflow"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_dir = None
        
    def _setup_logging(self):
        """Setup logging for the handler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - FetalBrainOpenRecon - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def process(self, input_image, output_image, metadata):
        """
        Process the input image through the fetal brain measurement pipeline
        
        Args:
            input_image: ISMRMRD image data
            output_image: Output image to be filled
            metadata: Image metadata
        """
        try:
            self.logger.info("🧠 === STARTING FETAL BRAIN MEASUREMENT PROCESSING ===")
            print("🧠 DEBUG: Starting fetal brain measurement processing")
            
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp(prefix="fetal_openrecon_")
            
            # Generate filename in expected format: Pat[ID]_Se[Series]_Res[X]_[Y]_Spac[Z].nii.gz
            # Extract metadata for filename generation with robust defaults
            patient_id = getattr(metadata, 'patient_id', getattr(metadata, 'subject_id', '99999'))
            series_num = getattr(metadata, 'series_number', getattr(metadata, 'series_id', '1'))
            
            # Extract resolution from ISMRMRD headers if available
            try:
                # Try to get resolution from ISMRMRD encoding
                encoding = getattr(input_image, 'encoding', None)
                if encoding and hasattr(encoding[0], 'encodedSpace'):
                    fov = encoding[0].encodedSpace.fieldOfView_mm
                    matrix = encoding[0].encodedSpace.matrixSize
                    res_x = f"{fov.x / matrix.x:.3f}"
                    res_y = f"{fov.y / matrix.y:.3f}"
                    res_z = f"{fov.z / matrix.z:.1f}"
                else:
                    raise AttributeError("No encoding found")
            except (AttributeError, IndexError, ZeroDivisionError):
                # Fall back to metadata or defaults
                res_x = getattr(metadata, 'pixel_spacing_x', '0.5')
                res_y = getattr(metadata, 'pixel_spacing_y', '0.5') 
                res_z = getattr(metadata, 'slice_thickness', '3.0')
            
            # Clean patient ID to be numeric only
            patient_id_clean = ''.join(filter(str.isdigit, str(patient_id))) or '99999'
            
            # Create properly formatted filename
            input_filename = f"Pat{patient_id_clean}_Se{series_num}_Res{res_x}_{res_y}_Spac{res_z}.nii.gz"
            input_file = os.path.join(self.temp_dir, input_filename)
            output_dir = os.path.join(self.temp_dir, "output")
            
            print(f"📂 DEBUG: Created temp directory: {self.temp_dir}")
            print(f"📁 DEBUG: Generated filename: {input_filename}")
            print(f"📁 DEBUG: Input file: {input_file}")
            print(f"📁 DEBUG: Output directory: {output_dir}")
            
            # Convert ISMRMRD image to NIfTI
            self._ismrmrd_to_nifti(input_image, input_file, metadata)
            
            # Run the fetal measurement pipeline
            measurement_results = self._run_fetal_pipeline(input_file, output_dir, metadata)
            
            # Convert results back to ISMRMRD format with embedded measurements
            self._nifti_to_ismrmrd(output_dir, output_image, metadata, measurement_results)
            
            self.logger.info("✅ Fetal brain measurement processing completed successfully")
            print("✅ DEBUG: Fetal brain measurement processing completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Error in fetal brain processing: {str(e)}")
            print(f"❌ DEBUG: Error in fetal brain processing: {str(e)}")
            import traceback
            print("🔍 DEBUG: Full error traceback:")
            traceback.print_exc()
            # Return the original image on error
            output_image.data[:] = input_image.data[:]
            raise
        finally:
            # Clean up temporary files
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 DEBUG: Cleaned up temporary directory: {self.temp_dir}")

    def _ismrmrd_to_nifti(self, ismrmrd_image, output_file, metadata):
        """Convert ISMRMRD image to NIfTI format"""
        self.logger.info("Converting ISMRMRD to NIfTI")
        print("🔄 DEBUG: Converting ISMRMRD to NIfTI")
        
        # Extract image data
        image_data = ismrmrd_image.data
        print(f"📐 DEBUG: Original image shape: {image_data.shape}")
        
        # Handle different data types and orientations
        if len(image_data.shape) == 3:
            # Single volume
            data = image_data
            print("📊 DEBUG: Processing as single volume (3D)")
        elif len(image_data.shape) == 4:
            # Multi-volume, take first volume
            data = image_data[:, :, :, 0]
            print("📊 DEBUG: Processing as multi-volume (4D), taking first volume")
        else:
            error_msg = f"Unexpected image shape: {image_data.shape}"
            print(f"❌ DEBUG: {error_msg}")
            raise ValueError(error_msg)
        
        print(f"📐 DEBUG: Final data shape for NIfTI: {data.shape}")
        
        # Create NIfTI image with identity affine matrix
        nii_img = nib.Nifti1Image(data, np.eye(4))
        
        # Save to file
        nib.save(nii_img, output_file)
        file_size = os.path.getsize(output_file)
        self.logger.info(f"Saved NIfTI to {output_file} ({file_size} bytes)")
        print(f"💾 DEBUG: Saved NIfTI to {output_file} ({file_size} bytes)")

    def _run_fetal_pipeline(self, input_file, output_dir, metadata):
        """Run the fetal brain measurement pipeline"""
        self.logger.info("Running fetal brain measurement pipeline")
        print("🧠 DEBUG: Running fetal brain measurement pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get parameters from metadata
        config = metadata.get('config', 'openrecon')
        enable_measurements = metadata.get('enable_measurements', True)
        enable_reporting = metadata.get('enable_reporting', True)
        confidence_threshold = metadata.get('confidence_threshold', 0.5)
        
        print(f"⚙️ DEBUG: Configuration - Config: {config}, Measurements: {enable_measurements}")
        print(f"⚙️ DEBUG: Reporting: {enable_reporting}, Threshold: {confidence_threshold}")
        
        try:
            # Import and run the fetal measurement pipeline directly
            print("📦 DEBUG: Importing fetal measurement module...")
            
            FetalMeasure = _import_fetal_measure()
            print("✅ DEBUG: Successfully imported FetalMeasure")
            
            fm = FetalMeasure()
            print("✅ DEBUG: FetalMeasure instance created successfully")
            
            # Execute the pipeline
            print(f"🚀 DEBUG: Executing pipeline on {input_file}")
            fm.execute(input_file, output_dir)
            
            # Check output files
            result_files = os.listdir(output_dir)
            print(f"📁 DEBUG: Pipeline generated {len(result_files)} files: {result_files}")
            
            # Load measurement results from JSON
            measurement_results = {}
            json_files = [f for f in result_files if f.endswith('.json')]
            if json_files:
                json_path = os.path.join(output_dir, json_files[0])
                with open(json_path, 'r') as f:
                    measurement_results = json.load(f)
                    print(f"📊 DEBUG: Loaded measurement data with {len(measurement_results)} fields")
                    
                    # Print key measurements
                    if 'cbd_measure_mm' in measurement_results:
                        print(f"📏 DEBUG: CBD: {measurement_results['cbd_measure_mm']:.2f} mm")
                    if 'bbd_measure_mm' in measurement_results:
                        print(f"📏 DEBUG: BBD: {measurement_results['bbd_measure_mm']:.2f} mm")
                    if 'tcd_measure_mm' in measurement_results:
                        print(f"📏 DEBUG: TCD: {measurement_results['tcd_measure_mm']:.2f} mm")
                    if 'pred_ga_cbd' in measurement_results:
                        print(f"📅 DEBUG: Predicted GA (CBD): {measurement_results['pred_ga_cbd']:.1f} weeks")
            
            # Extract plot and PDF data for metadata embedding
            print("🔄 DEBUG: Extracting visual outputs for metadata embedding...")
            measurement_results['dicom_attachments'] = self._extract_visual_outputs(output_dir, result_files)
            
            return measurement_results
            
        except ImportError as e:
            error_msg = f"Failed to import fetal measurement module: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ DEBUG: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Fetal pipeline execution failed: {str(e)}"
            self.logger.error(error_msg)
            print(f"❌ DEBUG: {error_msg}")
            raise RuntimeError(error_msg)

    def _extract_visual_outputs(self, output_dir, result_files):
        """Extract plot and PDF data for DICOM metadata embedding"""
        print("📊 DEBUG: Extracting visual outputs...")
        
        attachments = {'plots': {}, 'pdf_report': None}
        
        # Extract plot data
        plot_files = ['cbd.png', 'bbd.png', 'tcd.png', 'cbd_norm.png', 'bbd_norm.png', 'tcd_norm.png']
        for plot_file in plot_files:
            src_path = os.path.join(output_dir, plot_file)
            if os.path.exists(src_path):
                with open(src_path, 'rb') as f:
                    plot_bytes = f.read()
                plot_key = plot_file.replace('.png', '')
                attachments['plots'][plot_key] = {
                    'data': base64.b64encode(plot_bytes).decode('utf-8'),
                    'size': len(plot_bytes),
                    'format': 'PNG'
                }
                print(f"📈 DEBUG: Extracted {plot_file} ({len(plot_bytes)} bytes)")
        
        # Extract PDF report
        pdf_files = [f for f in result_files if f.endswith('.pdf')]
        if pdf_files:
            pdf_path = os.path.join(output_dir, pdf_files[0])
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            attachments['pdf_report'] = {
                'data': base64.b64encode(pdf_bytes).decode('utf-8'),
                'size': len(pdf_bytes),
                'filename': f"fetal_brain_report.pdf"
            }
            print(f"📄 DEBUG: Extracted PDF report ({len(pdf_bytes)} bytes)")
                
        print(f"💾 DEBUG: Extracted {len(attachments['plots'])} plots and {'1' if attachments['pdf_report'] else '0'} PDF reports")
        return attachments

    def _nifti_to_ismrmrd(self, output_dir, output_image, metadata, measurement_results):
        """Convert pipeline results back to ISMRMRD format with embedded measurements"""
        self.logger.info("Converting results to ISMRMRD format with embedded measurements")
        print("🔄 DEBUG: Converting results to ISMRMRD format with embedded measurements")
        
        # Look for the main segmentation output (optional - we can return original image)
        prediction_file = os.path.join(output_dir, "prediction.nii.gz")
        
        if not os.path.exists(prediction_file):
            print("⚠️ DEBUG: prediction.nii.gz not found, looking for alternatives...")
            # Try alternative output files
            for alt_file in ["cropped.nii.gz", "data.nii.gz", "output.nii.gz"]:
                alt_path = os.path.join(output_dir, alt_file)
                if os.path.exists(alt_path):
                    prediction_file = alt_path
                    print(f"✅ DEBUG: Found alternative output: {alt_file}")
                    break
        
        if os.path.exists(prediction_file):
            print(f"📁 DEBUG: Loading prediction from: {prediction_file}")
            # Load the NIfTI file
            nii_img = nib.load(prediction_file)
            data = nii_img.get_fdata()
            
            # Ensure the output has the same shape as the input
            if data.shape != output_image.data.shape[:3]:
                print(f"⚠️ DEBUG: Shape mismatch: data {data.shape} vs output {output_image.data.shape}")
                # Resize or pad the data to match
                data = self._resize_to_match(data, output_image.data.shape[:3])
            
            # Copy data to output image
            if len(output_image.data.shape) == 3:
                output_image.data[:] = data.astype(output_image.data.dtype)
            elif len(output_image.data.shape) == 4:
                output_image.data[:, :, :, 0] = data.astype(output_image.data.dtype)
                
            print("✅ DEBUG: Successfully loaded and converted prediction data")
        else:
            print("⚠️ DEBUG: No prediction file found, returning original image with measurements")
            # Keep original image data (just add measurements to metadata)
        
        # Embed measurement results in metadata
        self._embed_measurements_in_metadata(output_image, metadata, measurement_results)
        
        print("✅ DEBUG: Successfully converted to ISMRMRD format with measurements")

    def _embed_measurements_in_metadata(self, output_image, metadata, measurement_results):
        """Embed measurement results in ISMRMRD metadata"""
        print("📋 DEBUG: Embedding measurement results in metadata...")
        
        # Preserve image geometry and set processing metadata
        if hasattr(output_image, 'meta'):
            if output_image.meta is None:
                output_image.meta = {}
        else:
            output_image.meta = {}
        
        # Basic image metadata
        output_image.meta['Keep_image_geometry'] = 1
        output_image.meta['ImageProcessingHistory'] = 'FetalBrainMeasurement_OpenRecon'
        output_image.meta['DataRole'] = 'Image'
        output_image.meta['SequenceDescriptionAdditional'] = 'FETAL_BRAIN_OPENRECON'
        
        if measurement_results and 'error' not in measurement_results:
            # Basic measurements
            if 'cbd_measure_mm' in measurement_results:
                output_image.meta['CBD_mm'] = str(measurement_results['cbd_measure_mm'])
                output_image.meta['CBD_valid'] = 'Yes'
                print(f"📏 DEBUG: Embedded CBD: {measurement_results['cbd_measure_mm']:.2f} mm")
                
            if 'bbd_measure_mm' in measurement_results:
                output_image.meta['BBD_mm'] = str(measurement_results['bbd_measure_mm'])
                output_image.meta['BBD_valid'] = 'Yes' if measurement_results.get('bbd_valid', True) else 'No'
                print(f"📏 DEBUG: Embedded BBD: {measurement_results['bbd_measure_mm']:.2f} mm")
                
            if 'tcd_measure_mm' in measurement_results:
                output_image.meta['TCD_mm'] = str(measurement_results['tcd_measure_mm'])
                output_image.meta['TCD_valid'] = 'Yes' if measurement_results.get('tcd_valid', True) else 'No'
                print(f"📏 DEBUG: Embedded TCD: {measurement_results['tcd_measure_mm']:.2f} mm")
            
            # Gestational age predictions
            if 'pred_ga_cbd' in measurement_results:
                output_image.meta['GA_CBD_weeks'] = f"{measurement_results['pred_ga_cbd']:.1f}"
                print(f"📅 DEBUG: Embedded GA (CBD): {measurement_results['pred_ga_cbd']:.1f} weeks")
            if 'pred_ga_bbd' in measurement_results:
                output_image.meta['GA_BBD_weeks'] = f"{measurement_results['pred_ga_bbd']:.1f}"
                print(f"📅 DEBUG: Embedded GA (BBD): {measurement_results['pred_ga_bbd']:.1f} weeks")
            if 'pred_ga_tcd' in measurement_results:
                output_image.meta['GA_TCD_weeks'] = f"{measurement_results['pred_ga_tcd']:.1f}"
                print(f"📅 DEBUG: Embedded GA (TCD): {measurement_results['pred_ga_tcd']:.1f} weeks")
                
            # Brain volume
            if 'brain_vol_mm3' in measurement_results:
                output_image.meta['Brain_Volume_mm3'] = f"{measurement_results['brain_vol_mm3']:.0f}"
                print(f"🧠 DEBUG: Embedded brain volume: {measurement_results['brain_vol_mm3']:.0f} mm³")
            
            # Create summary comment
            comments = []
            if 'cbd_measure_mm' in measurement_results:
                comments.append(f"CBD: {measurement_results['cbd_measure_mm']:.1f}mm")
            if 'bbd_measure_mm' in measurement_results:
                comments.append(f"BBD: {measurement_results['bbd_measure_mm']:.1f}mm")
            if 'tcd_measure_mm' in measurement_results:
                comments.append(f"TCD: {measurement_results['tcd_measure_mm']:.1f}mm")
            
            if comments:
                output_image.meta['ImageComments'] = f"Fetal Brain: {', '.join(comments)}"
                print(f"📝 DEBUG: Added summary comment: {output_image.meta['ImageComments']}")
        
        else:
            output_image.meta['ImageComments'] = 'Fetal Brain Processing: No measurements available'
            print("⚠️ DEBUG: No valid measurements to embed")
        
        print(f"✅ DEBUG: Embedded {len(output_image.meta)} metadata fields")

    def _resize_to_match(self, data, target_shape):
        """Resize data to match target shape"""
        try:
            from scipy.ndimage import zoom
        except ImportError:
            print("⚠️ DEBUG: SciPy not available, using simple cropping/padding")
            return self._simple_resize(data, target_shape)
        
        if data.shape == target_shape:
            return data
        
        # Calculate zoom factors
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
        
        # Resize the data
        resized_data = zoom(data, zoom_factors, order=1)
        
        print(f"🔄 DEBUG: Resized data from {data.shape} to {resized_data.shape}")
        return resized_data
    
    def _simple_resize(self, data, target_shape):
        """Simple resize using cropping/padding when SciPy is not available"""
        if data.shape == target_shape:
            return data
        
        # Create output array
        result = np.zeros(target_shape, dtype=data.dtype)
        
        # Calculate copy regions
        copy_shape = tuple(min(data.shape[i], target_shape[i]) for i in range(3))
        
        # Copy data
        result[:copy_shape[0], :copy_shape[1], :copy_shape[2]] = \
            data[:copy_shape[0], :copy_shape[1], :copy_shape[2]]
        
        print(f"🔄 DEBUG: Simple resized data from {data.shape} to {result.shape}")
        return result

# Create handler instance
fetal_handler = FetalBrainI2IHandler()

def process(connection, config, metadata):
    """
    Main OpenRecon entry point for fetal brain measurements.
    This is the function called by the OpenRecon server framework.
    
    Args:
        connection: OpenRecon connection object
        config: Configuration parameters  
        metadata: MRD header metadata
    """
    print("🎯 DEBUG: OpenRecon process function called")
    print(f"📋 DEBUG: Config: {config}")
    print(f"📋 DEBUG: Metadata type: {type(metadata)}")
    
    # Create handler instance
    handler = FetalBrainI2IHandler()
    
    # Process incoming images
    for item in connection:
        if isinstance(item, ismrmrd.Image):
            print(f"🖼️ DEBUG: Processing image: {item.data.shape}")
            
            # Create output image (copy of input)
            output_image = ismrmrd.Image.from_array(item.data.copy())
            output_image.setHead(item.getHead())
            
            try:
                # Process through fetal brain pipeline
                handler.process(item, output_image, metadata)
                
                # Send processed image back
                connection.send_image(output_image)
                print("✅ DEBUG: Successfully processed and sent image")
                
            except Exception as e:
                print(f"❌ DEBUG: Error processing image: {e}")
                import traceback
                traceback.print_exc()
                # Send original image on error
                connection.send_image(item)
                
        elif item is None:
            break
        else:
            print(f"⚠️ DEBUG: Skipping non-image item: {type(item)}")
    
    print("🎯 DEBUG: OpenRecon process function completed")


def process_image(input_image, output_image, metadata):
    """
    Legacy i2i entry point for fetal brain measurements.
    
    Args:
        input_image: Input ISMRMRD image
        output_image: Output ISMRMRD image to be filled
        metadata: Image metadata
    """
    print("🎯 DEBUG: OpenRecon i2i process_image function called")
    return fetal_handler.process(input_image, output_image, metadata)
