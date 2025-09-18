#!/usr/bin/env python3
"""
测试简化版CPRAformer性能
"""

import sys
import os
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def train_simple_cpra():
    print("🚀 训练简化版CPRAformer...")
    
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    
    # 加载简化版模型
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-simple-cpra.yaml')
    print(f"✅ 模型加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    try:
        # 开始训练
        results = model.train(
            data="underwater_plastics/data.yaml",
            epochs=50,  # 训练50轮对比
            imgsz=640,
            device=0,
            workers=0,
            batch=16,
            name='simple-cpraformer-test',
            project='runs/detect',
            save=True,
            plots=True,
            verbose=True,
            # 添加一些稳定训练的参数
            patience=20,  # 早停
            lr0=0.01,     # 学习率
            warmup_epochs=5,
        )
        
        print(f"🎉 训练完成!")
        print(f"最终mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"训练结果保存在: runs/detect/simple-cpraformer-test")
        
        return results
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    train_simple_cpra()