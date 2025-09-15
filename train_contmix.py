#!/usr/bin/env python3
"""
训练ContMix-YOLO8模型的脚本
"""

import os
from ultralytics import YOLO

def main():
    print("=== ContMix-YOLO8 训练脚本 ===")
    
    # 设置路径
    model_config = "ultralytics/cfg/models/v8/yolov8-contmix.yaml"
    data_config = "underwater_plastics/data.yaml"
    pretrained = "yolov8n.pt"
    
    print(f"模型配置: {model_config}")
    print(f"数据配置: {data_config}")
    print(f"预训练权重: {pretrained}")
    
    try:
        # 创建模型
        print("\n1. 加载模型...")
        model = YOLO(model_config)
        print("✓ 模型加载成功")
        
        # 开始训练
        print("\n2. 开始训练...")
        results = model.train(
            data=data_config,
            epochs=100,
            imgsz=640,
            device='0',
            workers=0,
            pretrained=pretrained,
            save_period=10,  # 每10个epoch保存一次
            project='runs/train',
            name='yolov8-contmix',
            exist_ok=True
        )
        
        print("✓ 训练完成!")
        return results
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()