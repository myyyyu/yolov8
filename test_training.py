#!/usr/bin/env python3
"""
Test YOLOv8-CPRAformer training setup
"""

from ultralytics import YOLO
import os

def test_training_setup():
    """Test if YOLOv8-CPRAformer can be loaded and trained"""
    
    print("🧪 Testing YOLOv8-CPRAformer training setup...")
    
    # Use absolute paths
    model_path = r"C:\Users\15268\Desktop\ultralytics-main\ultralytics-main\ultralytics\cfg\models\v8\yolov8-cpraformer.yaml"
    data_path = r"C:\Users\15268\Desktop\ultralytics-main\ultralytics-main\underwater_plastics\data.yaml"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model config not found: {model_path}")
        return False
        
    if not os.path.exists(data_path):
        print(f"❌ Data config not found: {data_path}")
        return False
        
    print("✅ Configuration files exist")
    
    try:
        # Test model loading
        print("Loading model...")
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        
        # Test a very short training (1 epoch, small batch)
        print("Testing training setup...")
        results = model.train(
            data=data_path,
            epochs=1,
            imgsz=640,
            batch=1,
            device='0',
            workers=0,
            cache=False,
            save=False,
            plots=False,
            verbose=True
        )
        
        print("✅ Training test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_setup()
    if success:
        print("\n🎉 YOLOv8-CPRAformer is ready for training!")
        print("\nTo start full training, use:")
        print("yolo task=detect mode=train model=ultralytics/cfg/models/v8/yolov8-cpraformer.yaml data=underwater_plastics/data.yaml epochs=50 imgsz=640 device='0' workers=0")
    else:
        print("\n⚠️ Please fix the issues above before training.")