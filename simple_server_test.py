#!/usr/bin/env python3
"""
Simple test to check if our OpenRecon server can process a basic request
"""

import socket
import time

def test_server_connection():
    """Test if we can connect to the server"""
    try:
        print("🔌 Testing connection to server at localhost:9002...")
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        
        # Try to connect
        sock.connect(('localhost', 9002))
        print("✅ Successfully connected to server!")
        
        # Close connection
        sock.close()
        print("🔌 Connection closed")
        
        return True
        
    except ConnectionRefusedError:
        print("❌ Connection refused - is the server running?")
        return False
    except socket.timeout:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def check_server_logs():
    """Instructions for checking server logs"""
    print("\n📋 To check server processing:")
    print("1. Look at the Docker container logs")
    print("2. The server should show:")
    print("   - 'Accepting connection from: ...'")
    print("   - Processing messages from openrecon.py")
    print("   - Any errors in the fetal brain pipeline")
    
    print("\n🎯 Expected workflow:")
    print("   Client connects → Server loads openrecon.py → Processes images → Returns results")

def alternative_test():
    """Suggest alternative testing approaches"""
    print("\n🔄 Alternative testing approaches:")
    print("1. Test the openrecon.py handler directly:")
    print("   - Create simple test images")
    print("   - Call FetalBrainI2IHandler.process() directly")
    print("   - Check if the AI pipeline runs")
    
    print("\n2. Test individual components:")
    print("   - ISMRMRD to NIfTI conversion")
    print("   - Fetal brain AI pipeline")
    print("   - NIfTI to ISMRMRD conversion")
    
    print("\n3. Use Docker exec to test inside container:")
    print("   - docker exec -it <container_id> bash")
    print("   - Run components individually")

def main():
    print("🧪 Simple OpenRecon Server Test")
    print("=" * 50)
    
    # Test basic connection
    if test_server_connection():
        print("\n✅ Server is reachable!")
        print("   The server is running and accepting connections.")
        print("   The ISMRMRD client connectivity issue might be data format related.")
    else:
        print("\n❌ Cannot reach server")
        print("   Make sure the Docker container is running:")
        print("   docker run --rm --gpus all -p 9002:9002 openrecon-fetal:latest")
    
    check_server_logs()
    alternative_test()

if __name__ == "__main__":
    main()



