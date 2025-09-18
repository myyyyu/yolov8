#!/usr/bin/env python3
"""
比较不同版本的CPRAformer性能
"""

import sys
import os
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def compare_models():
    print("🔍 比较不同CPRAformer版本的性能...")
    
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    
    models = {
        "原版YOLOv8n": "yolov8n.yaml",
        "复杂CPRAformer": "ultralytics/cfg/models/v8/yolov8-cpraformer.yaml", 
        "简化CPRAformer": "ultralytics/cfg/models/v8/yolov8n-simple-cpra.yaml"
    }
    
    for name, config in models.items():
        try:
            print(f"\n--- 测试 {name} ---")
            model = YOLO(config)
            
            # 短期训练测试
            results = model.train(
                data="underwater_plastics/data.yaml",
                epochs=5,  # 只训练5轮看收敛情况
                imgsz=640,
                device=0,
                workers=0,
                batch=16,
                name=f'test-{name.replace(" ", "-")}',
                project='runs/compare',
                save=False,
                plots=False,
                verbose=False
            )
            
            print(f"✅ {name}: mAP50 = {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
            
        except Exception as e:
            print(f"❌ {name} 失败: {e}")

if __name__ == "__main__":
    compare_models()