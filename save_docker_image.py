#!/usr/bin/env python3
"""
Save the fetal brain OpenRecon Docker image as a tar file for deployment.
Based on OpenRecon deployment requirements.
"""

import subprocess
import re
import os
from datetime import datetime

def checkDockerVersion(minVersion="20.0.0"):
    """Check if Docker version meets minimum requirement"""
    try:
        # Get Docker version
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Docker not found or not running")
        
        # Extract version number from output like "Docker version 24.0.9, build 2936816"
        version_match = re.search(r'Docker version (\d+\.\d+\.\d+)', result.stdout)
        if not version_match:
            raise Exception("Could not parse Docker version")
        
        current_version = version_match.group(1)
        
        # Simple version comparison (assumes semantic versioning)
        current_parts = [int(x) for x in current_version.split('.')]
        min_parts = [int(x) for x in minVersion.split('.')]
        
        # Pad to same length
        while len(current_parts) < len(min_parts):
            current_parts.append(0)
        while len(min_parts) < len(current_parts):
            min_parts.append(0)
        
        # Compare versions
        for i in range(len(current_parts)):
            if current_parts[i] > min_parts[i]:
                break
            elif current_parts[i] < min_parts[i]:
                print(f"⚠️ Warning: Docker version {current_version} is older than recommended {minVersion}")
                return False
        
        print(f"✅ Docker version check...")
        print(f"   Available version {current_version} is OK")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Docker version: {e}")
        return False

def list_docker_images():
    """List available Docker images to find our fetal brain image"""
    try:
        result = subprocess.run(['docker', 'images'], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception("Failed to list Docker images")
        
        print("📋 Available Docker images:")
        print(result.stdout)
        return result.stdout
        
    except Exception as e:
        print(f"❌ Error listing Docker images: {e}")
        return None

def save_docker_image():
    """Save the fetal brain Docker image as tar file"""
    
    # Step 1: Check Docker version
    print("🔍 Step 1: Checking Docker version...")
    if not checkDockerVersion("20.0.0"):
        print("❌ Docker version check failed. Please update Docker.")
        return False
    
    # Step 2: List available images
    print("\n📋 Step 2: Listing available Docker images...")
    images_output = list_docker_images()
    if not images_output:
        return False
    
    # Step 3: Identify the fetal brain image
    print("\n🔍 Step 3: Identifying fetal brain Docker image...")
    
    # Look for our fetal brain image (could be different names)
    possible_names = [
        "openrecon-fetal:latest",
        "openrecon-fetal-brain:latest", 
        "fetal-brain-openrecon:latest",
        "openrecon_fetal:latest"
    ]
    
    found_image = None
    for name in possible_names:
        if name.split(':')[0] in images_output:  # Check if repository name exists
            found_image = name
            break
    
    # Also check if we can find openrecon-fetal directly
    if not found_image and "openrecon-fetal" in images_output:
        found_image = "openrecon-fetal:latest"
    
    if not found_image:
        print("❌ Could not find fetal brain Docker image.")
        print("   Looking for one of:", possible_names)
        print("   Please check 'docker images' output above.")
        return False
    
    print(f"✅ Found fetal brain image: {found_image}")
    
    # Step 4: Create output directory
    print("\n📁 Step 4: Creating output directory...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"deployment_package_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Created: {output_dir}")
    
    # Step 5: Save Docker image as tar file
    print("\n💾 Step 5: Saving Docker image as tar file...")
    
    # OpenRecon naming convention: OpenRecon_<Vendor>_<Name>_V<version>
    base_filename = "OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0"
    tar_filename = f"{base_filename}.tar"
    tar_path = os.path.join(output_dir, tar_filename)
    
    print(f"   Saving {found_image} to {tar_path}")
    print("   This may take several minutes for large images...")
    
    try:
        # Use docker save command (not docker export!)
        result = subprocess.run([
            'docker', 'save',
            '-o', tar_path,
            found_image
        ], capture_output=True, text=True, timeout=3600)  # 60 minute timeout
        
        if result.returncode != 0:
            raise Exception(f"Docker save failed: {result.stderr}")
        
        # Check if file was created and get size
        if os.path.exists(tar_path):
            file_size = os.path.getsize(tar_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"✅ Successfully saved Docker image!")
            print(f"   File: {tar_path}")
            print(f"   Size: {file_size_mb:.1f} MB")
            
            # Step 6: Provide deployment instructions
            print(f"\n🚀 Step 6: Deployment instructions:")
            print(f"   1. Copy {tar_filename} to the MRI system")
            print(f"   2. On MRI system, run: docker load -i {tar_filename}")
            print(f"   3. Start container: docker run -d --gpus all -p 9002:9002 {found_image}")
            print(f"   4. Configure OpenRecon to use localhost:9002")
            
            return True
        else:
            raise Exception("Tar file was not created")
            
    except subprocess.TimeoutExpired:
        print("❌ Docker save timed out (>10 minutes). Image may be very large.")
        return False
    except Exception as e:
        print(f"❌ Error saving Docker image: {e}")
        return False

def main():
    """Main function"""
    print("🐳 OpenRecon Fetal Brain Docker Image Export Tool")
    print("=" * 60)
    
    success = save_docker_image()
    
    if success:
        print("\n🎉 Docker image export completed successfully!")
        print("   Your image is ready for OpenRecon deployment.")
    else:
        print("\n❌ Docker image export failed.")
        print("   Please check the errors above and try again.")

if __name__ == "__main__":
    main()
