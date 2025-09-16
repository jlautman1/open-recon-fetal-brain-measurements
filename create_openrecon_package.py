#!/usr/bin/env python3
"""
Create OpenRecon deployment package (.zip) with Docker image (.tar) and documentation (.pdf)
Following OpenRecon requirements: https://github.com/kspaceKelvin/python-ismrmrd-server
"""

import os
import shutil
import subprocess
from datetime import datetime

def create_openrecon_package():
    """Create OpenRecon deployment package"""
    
    print("📦 OpenRecon Deployment Package Creator")
    print("=" * 60)
    
    # OpenRecon naming convention
    base_name = "OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0"
    tar_file = f"{base_name}.tar"
    pdf_file = f"{base_name}.pdf"
    zip_file = f"{base_name}.zip"
    
    print(f"📋 Package files:")
    print(f"   Docker image: {tar_file}")
    print(f"   Documentation: {pdf_file}")
    print(f"   Final package: {zip_file}")
    print()
    
    # Step 1: Check if Docker tar file exists
    print("🔍 Step 1: Checking Docker tar file...")
    if os.path.exists(tar_file):
        tar_size = os.path.getsize(tar_file) / (1024**3)  # GB
        print(f"✅ Found Docker tar: {tar_file} ({tar_size:.1f} GB)")
    else:
        print(f"❌ Docker tar file not found: {tar_file}")
        print("   Please run the docker save command first.")
        return False
    
    # Step 2: Prepare PDF documentation
    print("\n📄 Step 2: Preparing PDF documentation...")
    
    # Check for existing PDF files
    pdf_sources = [
        "additional-readme/openrecon_README[1].pdf",
        "additional-readme/openrecon_README.pdf",
        "fetal-brain-measurement/README.openrecon.pdf",
        "README.pdf"
    ]
    
    source_pdf = None
    for pdf_path in pdf_sources:
        if os.path.exists(pdf_path):
            source_pdf = pdf_path
            break
    
    if source_pdf:
        print(f"✅ Found documentation: {source_pdf}")
        print(f"   Copying to: {pdf_file}")
        shutil.copy2(source_pdf, pdf_file)
    else:
        print("⚠️ No PDF documentation found. Creating placeholder...")
        print("   You should replace this with proper documentation.")
        
        # Create minimal PDF placeholder (you'll need to replace this)
        placeholder_content = """
        OpenRecon Fetal Brain Measurement Pipeline V1.0.0
        
        This package contains the AI-powered fetal brain measurement system
        for Siemens OpenRecon MRI systems.
        
        Installation:
        1. docker load -i OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0.tar
        2. docker run -d --gpus all -p 9002:9002 openrecon-fetal:latest
        3. Configure OpenRecon: localhost:9002, module: openrecon
        
        Features:
        - Automatic CBD, BBD, TCD measurements
        - Real-time processing during MRI scan
        - DICOM metadata embedding
        - Clinical reporting
        
        Contact: Ichlov Sagol Lab
        """
        
        with open("temp_readme.txt", "w") as f:
            f.write(placeholder_content)
        
        print(f"   Created placeholder text file: temp_readme.txt")
        print(f"   Please convert this to PDF and rename to: {pdf_file}")
        return False
    
    # Step 3: Check for 7-Zip
    print("\n🗜️ Step 3: Checking 7-Zip availability...")
    zip_exe = "C:/Program Files/7-Zip/7z.exe"
    
    if os.path.exists(zip_exe):
        print(f"✅ Found 7-Zip: {zip_exe}")
    else:
        print("❌ 7-Zip not found at standard location")
        print("   Please install 7-Zip from: https://www.7-zip.org/")
        print("   Or use manual zip creation (see instructions below)")
        return False
    
    # Step 4: Create ZIP package using 7-Zip (Deflate algorithm)
    print(f"\n📦 Step 4: Creating ZIP package...")
    print(f"   Using Deflate compression (required for OpenRecon)")
    print(f"   This may take several minutes for large files...")
    
    try:
        # Remove existing zip file if present
        if os.path.exists(zip_file):
            os.remove(zip_file)
        
        # Use 7-Zip with Deflate compression
        result = subprocess.run([
            zip_exe, "a",           # Add to archive
            "-tzip",                # ZIP format
            "-mm=Deflate",          # Use Deflate compression (required!)
            "-mx=5",                # Medium compression level
            zip_file,               # Output zip file
            tar_file,               # Add tar file
            pdf_file                # Add pdf file
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            if os.path.exists(zip_file):
                zip_size = os.path.getsize(zip_file) / (1024**3)  # GB
                print(f"✅ Successfully created ZIP package!")
                print(f"   File: {zip_file}")
                print(f"   Size: {zip_size:.1f} GB")
                
                # Step 5: Verify package contents
                print(f"\n🔍 Step 5: Verifying package contents...")
                verify_result = subprocess.run([
                    zip_exe, "l", zip_file  # List contents
                ], capture_output=True, text=True)
                
                if verify_result.returncode == 0:
                    print("✅ Package contents verified:")
                    print(verify_result.stdout)
                
                # Step 6: Deployment instructions
                print(f"\n🚀 Step 6: Deployment Instructions")
                print("=" * 40)
                print(f"1. 📁 Copy {zip_file} to MRI system")
                print(f"2. 📂 Extract the ZIP file on MRI system")
                print(f"3. 🐳 Load Docker image:")
                print(f"   docker load -i {tar_file}")
                print(f"4. ▶️ Start container:")
                print(f"   docker run -d --name fetal-brain-server --gpus all -p 9002:9002 openrecon-fetal:latest")
                print(f"5. ⚙️ Configure OpenRecon:")
                print(f"   - Server: localhost:9002")
                print(f"   - Module: openrecon")
                print(f"   - Type: Image-to-Image (i2i)")
                
                return True
                
            else:
                print("❌ ZIP file was not created")
                return False
        else:
            print(f"❌ 7-Zip failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ZIP creation timed out (>30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error creating ZIP: {e}")
        return False

def manual_instructions():
    """Provide manual ZIP creation instructions"""
    print("\n📋 Manual ZIP Creation Instructions")
    print("=" * 40)
    print("If the automated script fails, create the ZIP manually:")
    print()
    print("1. Install 7-Zip from: https://www.7-zip.org/")
    print("2. Right-click in Windows Explorer")
    print("3. Select '7-Zip' → 'Add to archive...'")
    print("4. Set Archive format: ZIP")
    print("5. Set Compression method: Deflate")
    print("6. Add both files:")
    print("   - OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0.tar")
    print("   - OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0.pdf")
    print("7. Name: OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0.zip")
    print()
    print("⚠️ IMPORTANT: Must use Deflate, not Deflate64!")

def main():
    """Main function"""
    success = create_openrecon_package()
    
    if not success:
        manual_instructions()
    else:
        print(f"\n🎉 OpenRecon deployment package ready!")
        print(f"   Your fetal brain AI system is ready for MRI deployment.")

if __name__ == "__main__":
    main()



