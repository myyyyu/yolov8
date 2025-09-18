#!/usr/bin/env python3
"""
YOLOv8-CPRAformer Training Script
直接使用本地修改的ultralytics代码
"""

import sys
import os

# 确保使用本地修改的ultralytics
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def train_cpraformer():
    print("🚀 Starting YOLOv8-CPRAformer Training...")
    
    # 使用相对路径
    model_path = "ultralytics/cfg/models/v8/yolov8-cpraformer.yaml"
    data_path = "underwater_plastics/data.yaml"
    
    try:
        # 加载模型
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # 开始训练
        print("Starting training...")
        results = model.train(
            data=data_path,
            epochs=150,
            imgsz=640,
            device=0,
            workers=0,
            batch=16,
            name='cpraformer-underwater-plastics',
            project='runs/detect',
            save=True,
            plots=True,
            verbose=True,

        )
        
        print("🎉 Training completed successfully!")
        print(f"Results saved in: runs/detect/cpraformer-underwater-plastics")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 切换到正确的工作目录
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    print(f"Working directory: {os.getcwd()}")
    
    # 开始训练
    train_cpraformer()