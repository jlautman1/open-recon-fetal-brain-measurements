#!/usr/bin/env python3
"""
Simple Pipeline Test
Tests the basic pipeline components individually
"""

import os
import sys

def test_converter():
    """Test just the NIfTI to ISMRMRD converter"""
    print("🔧 Testing NIfTI → ISMRMRD Converter...")
    
    try:
        from nifti_to_ismrmrd_converter import convert_nifti_to_ismrmrd
        
        nifti_file = "fetal-brain-measurement/Inputs/Fixed/Pat13249_Se8_Res0.46875_0.46875_Spac4.0.nii.gz"
        
        if not os.path.exists(nifti_file):
            print(f"❌ Input file not found: {nifti_file}")
            return False
        
        # Convert to ISMRMRD
        input_image, metadata = convert_nifti_to_ismrmrd(nifti_file)
        
        print(f"✅ Converter works!")
        print(f"   Data shape: {input_image.data.shape}")
        print(f"   Patient: {metadata.get('PatientID', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Converter failed: {e}")
        return False

def test_openrecon_import():
    """Test OpenRecon handler import"""
    print("\n🔧 Testing OpenRecon Handler Import...")
    
    try:
        sys.path.insert(0, './fetal-brain-measurement')
        import openrecon
        
        print("✅ OpenRecon imported successfully")
        
        # Try to create handler
        handler = openrecon.FetalBrainI2IHandler()
        print("✅ Handler created successfully")
        return True
        
    except Exception as e:
        print(f"❌ OpenRecon import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fetal_measure_import():
    """Test fetal measure pipeline import"""
    print("\n🔧 Testing Fetal Measure Import...")
    
    try:
        sys.path.append('./fetal-brain-measurement/Code/FetalMeasurements-master')
        
        from fetal_measure import FetalMeasure
        
        print("✅ FetalMeasure imported successfully")
        
        # Try to create instance
        fm = FetalMeasure()
        print("✅ FetalMeasure instance created successfully")
        return True
        
    except Exception as e:
        print(f"❌ FetalMeasure import failed: {e}")
        return False

def main():
    """Run simple tests"""
    print("🧪 Simple Pipeline Component Tests")
    print("=" * 50)
    
    tests = [
        ("NIfTI Converter", test_converter),
        ("OpenRecon Import", test_openrecon_import), 
        ("FetalMeasure Import", test_fetal_measure_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n📊 Test Results:")
    print("-" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    return passed == len(results)

if __name__ == "__main__":
    result = main()
    sys.exit(0 if result else 1)



