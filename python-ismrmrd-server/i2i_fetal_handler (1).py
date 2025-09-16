#!/usr/bin/env python3
"""
Custom i2i handler for Fetal Brain Measurement Pipeline
This handler processes incoming ISMRMRD images and calls the fetal measurement pipeline.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np
import nibabel as nib

# Add the pipeline code to the path
sys.path.append('/app/fetal-brain-measurement/Code/FetalMeasurements-master')

class FetalBrainI2IHandler:
    """Handler for fetal brain measurement in i2i workflow"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.temp_dir = None
        
    def _setup_logging(self):
        """Setup logging for the handler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - FetalI2I - %(levelname)s - %(message)s'
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
            self.logger.info("Starting fetal brain measurement processing")
            
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp(prefix="fetal_")
            input_file = os.path.join(self.temp_dir, "input.nii.gz")
            output_dir = os.path.join(self.temp_dir, "output")
            
            # Convert ISMRMRD image to NIfTI
            self._ismrmrd_to_nifti(input_image, input_file, metadata)
            
            # Run the fetal measurement pipeline
            self._run_fetal_pipeline(input_file, output_dir, metadata)
            
            # Convert results back to ISMRMRD format
            self._nifti_to_ismrmrd(output_dir, output_image, metadata)
            
            self.logger.info("Fetal brain measurement processing completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in fetal brain processing: {str(e)}")
            # Return a zero image on error
            output_image.data[:] = 0
            raise
        finally:
            # Clean up temporary files
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
    
    def _ismrmrd_to_nifti(self, ismrmrd_image, output_file, metadata):
        """Convert ISMRMRD image to NIfTI format"""
        self.logger.info("Converting ISMRMRD to NIfTI")
        
        # Extract image data
        image_data = ismrmrd_image.data
        
        # Handle different data types and orientations
        if len(image_data.shape) == 3:
            # Single volume
            data = image_data
        elif len(image_data.shape) == 4:
            # Multi-volume, take first volume
            data = image_data[:, :, :, 0]
        else:
            raise ValueError(f"Unexpected image shape: {image_data.shape}")
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(data, np.eye(4))
        
        # Save to file
        nib.save(nii_img, output_file)
        self.logger.info(f"Saved NIfTI to {output_file}")
    
    def _run_fetal_pipeline(self, input_file, output_dir, metadata):
        """Run the fetal brain measurement pipeline"""
        self.logger.info("Running fetal brain measurement pipeline")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get parameters from metadata
        config = metadata.get('config', 'i2i')
        model_version = metadata.get('model_version', '1')
        enable_measurements = metadata.get('enable_measurements', True)
        enable_reporting = metadata.get('enable_reporting', True)
        confidence_threshold = metadata.get('confidence_threshold', 0.5)
        customconfig = metadata.get('customconfig', '')
        
        # Build command for the pipeline
        cmd = [
            'python', '/app/segment_entrypoint.py',
            '--input', input_file,
            '--output', output_dir,
            '--config', config,
            '--model_version', str(model_version),
            '--enable_measurements', str(enable_measurements).lower(),
            '--enable_reporting', str(enable_reporting).lower(),
            '--confidence_threshold', str(confidence_threshold),
            '--customconfig', customconfig
        ]
        
        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the pipeline
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd='/app'
        )
        
        if result.returncode != 0:
            self.logger.error(f"Pipeline failed with return code {result.returncode}")
            self.logger.error(f"stdout: {result.stdout}")
            self.logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"Fetal pipeline failed: {result.stderr}")
        
        self.logger.info("Pipeline completed successfully")
        self.logger.info(f"Pipeline stdout: {result.stdout}")
    
    def _nifti_to_ismrmrd(self, output_dir, output_image, metadata):
        """Convert pipeline results back to ISMRMRD format"""
        self.logger.info("Converting results to ISMRMRD format")
        
        # Look for the main segmentation output
        prediction_file = os.path.join(output_dir, "prediction.nii.gz")
        
        if not os.path.exists(prediction_file):
            self.logger.warning("prediction.nii.gz not found, looking for alternatives")
            # Try alternative output files
            for alt_file in ["cropped.nii.gz", "data.nii.gz"]:
                if os.path.exists(os.path.join(output_dir, alt_file)):
                    prediction_file = os.path.join(output_dir, alt_file)
                    break
        
        if not os.path.exists(prediction_file):
            self.logger.error("No suitable output file found")
            # Create a zero image
            output_image.data[:] = 0
            return
        
        # Load the NIfTI file
        nii_img = nib.load(prediction_file)
        data = nii_img.get_fdata()
        
        # Ensure the output has the same shape as the input
        if data.shape != output_image.data.shape[:3]:
            self.logger.warning(f"Shape mismatch: data {data.shape} vs output {output_image.data.shape}")
            # Resize or pad the data to match
            data = self._resize_to_match(data, output_image.data.shape[:3])
        
        # Copy data to output image
        if len(output_image.data.shape) == 3:
            output_image.data[:] = data
        elif len(output_image.data.shape) == 4:
            output_image.data[:, :, :, 0] = data
        
        # Preserve image geometry
        if hasattr(output_image, 'field_of_view'):
            output_image.field_of_view = metadata.get('field_of_view', output_image.field_of_view)
        
        if hasattr(output_image, 'image_type'):
            output_image.image_type = metadata.get('image_type', output_image.image_type)
        
        # Set metadata to indicate this is a derived image
        if hasattr(output_image, 'meta'):
            if output_image.meta is None:
                output_image.meta = {}
            output_image.meta['Keep_image_geometry'] = 1
            output_image.meta['ImageProcessingHistory'] = 'FetalBrainMeasurement'
        
        self.logger.info("Successfully converted to ISMRMRD format")
    
    def _resize_to_match(self, data, target_shape):
        """Resize data to match target shape"""
        from scipy.ndimage import zoom
        
        if data.shape == target_shape:
            return data
        
        # Calculate zoom factors
        zoom_factors = [target_shape[i] / data.shape[i] for i in range(3)]
        
        # Resize the data
        resized_data = zoom(data, zoom_factors, order=1)
        
        self.logger.info(f"Resized data from {data.shape} to {resized_data.shape}")
        return resized_data

# Create handler instance
fetal_handler = FetalBrainI2IHandler()

def process_image(input_image, output_image, metadata):
    """
    Main processing function called by the ISMRMRD server
    
    Args:
        input_image: Input ISMRMRD image
        output_image: Output ISMRMRD image to be filled
        metadata: Image metadata
    """
    return fetal_handler.process(input_image, output_image, metadata) 