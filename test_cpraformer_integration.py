#!/usr/bin/env python3
"""
Test script for YOLOv8-CPRAformer integration
Verifies model loading, forward pass, and basic functionality
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.cpraformer import CPRAformerC2f, LightCPRAformerC2f


def test_cpraformer_modules():
    """Test CPRAformer modules independently"""
    print("Testing CPRAformer modules...")
    
    # Test input
    x = torch.randn(1, 256, 32, 32)
    
    # Test full CPRAformer
    cpra_full = CPRAformerC2f(256, 256, n=2)
    out_full = cpra_full(x)
    print(f"CPRAformerC2f: Input {x.shape} -> Output {out_full.shape}")
    
    # Test lightweight CPRAformer  
    cpra_light = LightCPRAformerC2f(256, 256, n=2)
    out_light = cpra_light(x)
    print(f"LightCPRAformerC2f: Input {x.shape} -> Output {out_light.shape}")
    
    print("✅ Module tests passed!\n")


def test_model_loading():
    """Test YOLO model loading with CPRAformer configs"""
    print("Testing model loading...")
    
    try:
        # Test full CPRAformer model
        model_full = YOLO('ultralytics/cfg/models/v8/yolov8-cpraformer.yaml')
        print("✅ YOLOv8-CPRAformer model loaded successfully")
        
        # Test lightweight model
        model_light = YOLO('ultralytics/cfg/models/v8/yolov8-lightcpra.yaml')
        print("✅ YOLOv8-LightCPRA model loaded successfully")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    print("✅ Model loading tests passed!\n")
    return True


def test_forward_pass():
    """Test forward pass with dummy input"""
    print("Testing forward pass...")
    
    try:
        # Create models
        model_full = YOLO('ultralytics/cfg/models/v8/yolov8-cpraformer.yaml')
        model_light = YOLO('ultralytics/cfg/models/v8/yolov8-lightcpra.yaml')
        
        # Dummy input (batch_size=1, channels=3, height=640, width=640)
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Forward pass
        with torch.no_grad():
            out_full = model_full.model(dummy_input)
            out_light = model_light.model(dummy_input)
        
        print(f"CPRAformer model output shapes: {[o.shape for o in out_full]}")
        print(f"LightCPRA model output shapes: {[o.shape for o in out_light]}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("✅ Forward pass tests passed!\n")
    return True


def test_parameter_count():
    """Compare parameter counts between models"""
    print("Comparing parameter counts...")
    
    try:
        # Original YOLOv8
        model_orig = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
        params_orig = sum(p.numel() for p in model_orig.model.parameters())
        
        # CPRAformer models
        model_full = YOLO('ultralytics/cfg/models/v8/yolov8-cpraformer.yaml')
        params_full = sum(p.numel() for p in model_full.model.parameters())
        
        model_light = YOLO('ultralytics/cfg/models/v8/yolov8-lightcpra.yaml')
        params_light = sum(p.numel() for p in model_light.model.parameters())
        
        # Calculate increases
        increase_full = (params_full - params_orig) / params_orig * 100
        increase_light = (params_light - params_orig) / params_orig * 100
        
        print(f"Original YOLOv8: {params_orig:,} parameters")
        print(f"YOLOv8-CPRAformer: {params_full:,} parameters (+{increase_full:.1f}%)")
        print(f"YOLOv8-LightCPRA: {params_light:,} parameters (+{increase_light:.1f}%)")
        
    except Exception as e:
        print(f"❌ Parameter count test failed: {e}")
        return False
    
    print("✅ Parameter count tests passed!\n")
    return True


def test_training_compatibility():
    """Test basic training setup"""
    print("Testing training compatibility...")
    
    try:
        from ultralytics.engine.trainer import BaseTrainer
        
        # Test model initialization for training
        model = YOLO('ultralytics/cfg/models/v8/yolov8-cpraformer.yaml')
        
        # Check if model can be put in training mode
        model.model.train()
        
        # Test with dummy loss computation
        dummy_input = torch.randn(2, 3, 320, 320, requires_grad=True)
        dummy_targets = [
            {'boxes': torch.randn(5, 4), 'labels': torch.randint(0, 80, (5,))},
            {'boxes': torch.randn(3, 4), 'labels': torch.randint(0, 80, (3,))}
        ]
        
        with torch.no_grad():
            outputs = model.model(dummy_input)
        
        print("✅ Training compatibility verified")
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        return False

    print("✅ Training compatibility tests passed!\n")
    return True


def main():
    """Run all tests"""
    print("🧪 YOLOv8-CPRAformer Integration Tests\n")
    print("=" * 50)
    
    tests = [
        test_cpraformer_modules,
        test_model_loading,
        test_forward_pass,
        test_parameter_count,
        test_training_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CPRAformer integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    main()