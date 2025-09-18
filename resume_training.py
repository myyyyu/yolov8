#!/usr/bin/env python3
"""
从现有的CPRAformer checkpoint继续训练
"""

import sys
import os
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def resume_training():
    print("🔄 从现有checkpoint继续训练CPRAformer...")
    
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    
    # 查找最新的权重文件
    checkpoint_path = "\kaggle\working\cpraformer-underwater-plastics-continued3\weights\last.pt"
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到checkpoint文件: {checkpoint_path}")
        print("可用的runs目录:")
        if os.path.exists("runs/detect"):
            for dir_name in os.listdir("runs/detect"):
                if "cpraformer" in dir_name:
                    weights_path = f"runs/detect/{dir_name}/weights"
                    if os.path.exists(weights_path):
                        print(f"  - {dir_name}: {os.listdir(weights_path)}")
        return None
    
    try:
        # 加载现有模型
        print(f"📁 加载checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        
        # 继续训练，调整一些参数来提升性能
        print("🚀 开始继续训练...")
        results = model.train(
            data="underwater_plastics/data.yaml",
            epochs=150,  # 再训练100轮
            imgsz=640,
            device=0,
            workers=0,
            batch=16,
            name='cpraformer-underwater-plastics-continued',
            project='runs/detect',
            save=True,
            plots=True,
            verbose=True,
            
            # 调整超参数来提升性能
            lr0=0.005,  # 降低学习率
            lrf=0.001,  # 最终学习率
            warmup_epochs=3,
            patience=30,  # 增加耐心
            
            # 数据增强优化
            hsv_h=0.02,  # 色调变化
            hsv_s=0.8,   # 饱和度变化  
            hsv_v=0.5,   # 亮度变化
            mixup=0.15,  # mixup增强
            copy_paste=0.3,  # copy-paste增强
            
            # 优化器
            optimizer='AdamW',
            cos_lr=True,  # 余弦学习率衰减
            
            # 其他优化
            close_mosaic=20,  # 提前关闭mosaic
            amp=True,  # 混合精度训练
        )
        
        print(f"🎉 继续训练完成!")
        print(f"最终mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"训练结果保存在: runs/detect/cpraformer-underwater-plastics-continued")
        
        return results
        
    except Exception as e:
        print(f"❌ 继续训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resume_training()