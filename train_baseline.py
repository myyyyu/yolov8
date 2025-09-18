#!/usr/bin/env python3
"""
测试基准YOLOv8n性能
"""

import sys
import os
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def train_baseline():
    print("🎯 训练基准YOLOv8n...")
    
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    
    # 加载基准模型
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-baseline.yaml')
    print(f"✅ 基准模型加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    try:
        # 开始训练
        results = model.train(
            data="underwater_plastics/data.yaml",
            epochs=50,
            imgsz=640,
            device=0,
            workers=0,
            batch=16,  # 确保和之前一致
            name='baseline-yolov8n-test',
            project='runs/detect',
            save=True,
            plots=True,
            verbose=True,
            patience=20,
            lr0=0.01,
            warmup_epochs=5,
        )
        
        print(f"🎉 基准训练完成!")
        print(f"最终mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ 基准训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_baseline()