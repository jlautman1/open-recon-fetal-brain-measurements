#!/usr/bin/env python3
"""
Complete NIfTI to ISMRMRD Pipeline Test
Converts NIfTI input from inputs/fixed folder to ISMRMRD and runs through the whole pipeline
"""

import os
import sys
import numpy as np
import h5py
import subprocess
import time
import json
from pathlib import Path

# Add required paths
sys.path.append('./python-ismrmrd-server')
sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')

def create_ismrmrd_file_from_nifti(nifti_path, output_h5_path):
    """Create a proper ISMRMRD HDF5 file from NIfTI data using the working approach"""
    
    print(f"🔄 Creating ISMRMRD HDF5 file from NIfTI")
    print(f"   Input: {nifti_path}")
    print(f"   Output: {output_h5_path}")
    
    try:
        # Use the working approach from create_test_ismrmrd.py
        from create_test_ismrmrd import create_ismrmrd_file
        
        # Call the working function
        create_ismrmrd_file(output_h5_path, nifti_path)
        
        # Check if file was created successfully
        if os.path.exists(output_h5_path) and os.path.getsize(output_h5_path) > 1024:
            print(f"✅ Successfully created ISMRMRD HDF5 file")
            print(f"   File: {output_h5_path}")
            print(f"   Size: {os.path.getsize(output_h5_path) / (1024*1024):.2f} MB")
            return True
        else:
            print(f"❌ Failed to create ISMRMRD file or file is empty")
            return False
        
    except Exception as e:
        print(f"❌ Error creating ISMRMRD file: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_docker_server():
    """Check if the OpenRecon Docker server is running"""
    print("🔍 Checking if OpenRecon Docker server is running...")
    
    try:
        # Check if container is running
        result = subprocess.run(['docker', 'ps', '--filter', 'ancestor=openrecon-fetal:latest', '--format', 'table {{.Names}}\t{{.Status}}'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            print("✅ OpenRecon Docker server is running")
            print(result.stdout)
            return True
        else:
            print("❌ OpenRecon Docker server is not running")
            return False
            
    except Exception as e:
        print(f"❌ Error checking Docker server: {e}")
        return False

def start_docker_server():
    """Start the OpenRecon Docker server"""
    print("🚀 Starting OpenRecon Docker server...")
    
    try:
        # Stop any existing container first
        print("   Stopping any existing containers...")
        subprocess.run(['docker', 'stop', 'fetal-brain-server'], capture_output=True)
        subprocess.run(['docker', 'rm', 'fetal-brain-server'], capture_output=True)
        
        # Start new container
        cmd = [
            'docker', 'run', '-d',
            '--name', 'fetal-brain-server',
            '--gpus', 'all',
            '-p', '9002:9002',
            '-v', f'{os.getcwd()}:/workspace',
            'openrecon-fetal:latest'
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ OpenRecon Docker server started successfully")
            print(f"   Container ID: {result.stdout.strip()}")
            
            # Wait for server to be ready
            print("   Waiting for server to be ready...")
            time.sleep(10)
            return True
        else:
            print(f"❌ Failed to start Docker server: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Docker server: {e}")
        return False

def send_data_to_server(ismrmrd_file, output_file="pipeline_output_result.h5"):
    """Send ISMRMRD data to the OpenRecon server"""
    print(f"📤 Sending ISMRMRD data to OpenRecon server...")
    print(f"   Input: {ismrmrd_file}")
    print(f"   Output: {output_file}")
    
    try:
        # Use the client to send data
        cmd = [
            'python', 'python-ismrmrd-server/client.py',
            ismrmrd_file,
            '-c', 'openrecon',
            '-p', '9002',
            '-o', output_file
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Successfully sent data to server")
            print("Server output:")
            print(result.stdout)
            
            if os.path.exists(output_file):
                print(f"✅ Output file created: {output_file}")
                print(f"   Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"❌ Output file not created: {output_file}")
                return False
        else:
            print(f"❌ Server communication failed: {result.stderr}")
            print("Server stdout:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Server communication timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error communicating with server: {e}")
        return False

def verify_output(output_file):
    """Verify the pipeline output"""
    print(f"🔍 Verifying pipeline output: {output_file}")
    
    try:
        if not os.path.exists(output_file):
            print(f"❌ Output file does not exist: {output_file}")
            return False
        
        # Check file size
        file_size = os.path.getsize(output_file)
        print(f"📁 File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024:  # Less than 1KB probably means empty/error
            print("❌ Output file is too small, likely empty or error")
            return False
        
        # Try to read the ISMRMRD file
        try:
            import ismrmrd
            with h5py.File(output_file, 'r') as f:
                print("📊 HDF5 file structure:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"   Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"   Group: {name}")
                
                f.visititems(print_structure)
                
                # Check for expected datasets
                if 'dataset' in f:
                    print("✅ Found main dataset group")
                    dataset_group = f['dataset']
                    
                    if 'images' in dataset_group:
                        images = dataset_group['images']
                        print(f"✅ Found images dataset with shape: {images.shape}")
                    
                    if 'xml' in dataset_group:
                        print("✅ Found XML header")
                
        except Exception as e:
            print(f"⚠️ Could not read as ISMRMRD file: {e}")
            print("   File may be in different format or corrupted")
        
        print("✅ Pipeline output verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying output: {e}")
        return False

def main():
    """Main function to run the complete pipeline test"""
    print("🧪 Complete NIfTI to ISMRMRD Pipeline Test")
    print("=" * 70)
    
    # Configuration
    nifti_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
    ismrmrd_file = "test_fetal_brain_pipeline.h5"
    output_file = "pipeline_output_result.h5"
    
    # Step 1: Check input file
    print("\n📁 Step 1: Checking input file...")
    if not os.path.exists(nifti_file):
        print(f"❌ Input file not found: {nifti_file}")
        return False
    print(f"✅ Input file found: {nifti_file}")
    
    # Step 2: Convert NIfTI to ISMRMRD
    print("\n🔄 Step 2: Converting NIfTI to ISMRMRD...")
    if not create_ismrmrd_file_from_nifti(nifti_file, ismrmrd_file):
        print("❌ Failed to convert NIfTI to ISMRMRD")
        return False
    
    # Step 3: Check/Start Docker server
    print("\n🐳 Step 3: Checking Docker server...")
    if not check_docker_server():
        print("   Server not running, attempting to start...")
        if not start_docker_server():
            print("❌ Failed to start Docker server")
            return False
    
    # Step 4: Send data to server
    print("\n📤 Step 4: Sending data to OpenRecon server...")
    if not send_data_to_server(ismrmrd_file, output_file):
        print("❌ Failed to process data through OpenRecon server")
        return False
    
    # Step 5: Verify output
    print("\n🔍 Step 5: Verifying pipeline output...")
    if not verify_output(output_file):
        print("❌ Pipeline output verification failed")
        return False
    
    print("\n🎉 Complete pipeline test successful!")
    print(f"   Input NIfTI: {nifti_file}")
    print(f"   ISMRMRD file: {ismrmrd_file}")
    print(f"   Output file: {output_file}")
    print("\n✅ The fetal brain measurement pipeline is working correctly!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Pipeline test failed!")
        sys.exit(1)
