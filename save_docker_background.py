#!/usr/bin/env python3
"""
Save Docker image in background with progress monitoring
"""

import subprocess
import os
import time
import threading
from datetime import datetime

def monitor_file_size(filepath, interval=30):
    """Monitor file size growth during save"""
    while True:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {size_mb:.1f} MB written...")
        time.sleep(interval)

def save_docker_image_background():
    """Save Docker image with background monitoring"""
    
    # Setup
    output_dir = "deployment_package"
    os.makedirs(output_dir, exist_ok=True)
    
    tar_path = os.path.join(output_dir, "OpenRecon_IchilovSagolLab_FetalBrainSegmentation_V1.0.0.tar")
    
    print("🐳 Starting Docker save in background...")
    print(f"📁 Output: {tar_path}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 Expected size: ~12GB (this will take 15-30 minutes)")
    print("=" * 60)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_file_size, args=(tar_path, 30), daemon=True)
    monitor_thread.start()
    
    try:
        # Start docker save process (no timeout)
        process = subprocess.Popen([
            'docker', 'save',
            '-o', tar_path,
            'openrecon-fetal:latest'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("🔄 Docker save process started...")
        print("💡 You can press Ctrl+C to check status (won't stop the process)")
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            if os.path.exists(tar_path):
                final_size = os.path.getsize(tar_path) / (1024 * 1024)
                print(f"\n🎉 SUCCESS!")
                print(f"📁 File: {tar_path}")
                print(f"📊 Final size: {final_size:.1f} MB")
                print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"\n🚀 Next steps:")
                print(f"1. Copy {os.path.basename(tar_path)} to MRI system")
                print(f"2. Run: docker load -i {os.path.basename(tar_path)}")
                print(f"3. Start: docker run -d --gpus all -p 9002:9002 openrecon-fetal:latest")
                
            else:
                print("❌ File was not created despite successful process")
        else:
            print(f"❌ Docker save failed: {stderr}")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Interrupted by user at {datetime.now().strftime('%H:%M:%S')}")
        print("💡 Docker save is likely still running in background")
        print(f"💡 Check file size: dir {tar_path}")
        print("💡 Check processes: docker ps")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    save_docker_image_background()



