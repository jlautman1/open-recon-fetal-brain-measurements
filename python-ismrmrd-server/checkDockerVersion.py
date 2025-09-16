import subprocess
import re

def checkDockerVersion(minVersion):
    """
    Check if Docker version meets minimum requirement
    """
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
                print(f"Warning: Docker version {current_version} is older than recommended {minVersion}")
                return False
        
        print(f"### Check docker version...")
        print(f"#-> Available version {current_version} ok. ")
        return True
        
    except Exception as e:
        print(f"Error checking Docker version: {e}")
        return False

