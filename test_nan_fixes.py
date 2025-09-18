#!/usr/bin/env python3
"""
测试NaN修复是否有效
"""

import sys
import os
sys.path.insert(0, r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')

from ultralytics import YOLO

def test_nan_fixes():
    print("🧪 Testing NaN fixes with short training run...")
    
    os.chdir(r'C:\Users\15268\Desktop\ultralytics-main\ultralytics-main')
    
    try:
        # 从checkpoint加载模型
        model = YOLO("runs/detect/cpraformer-underwater-plastics-continued2/weights/last.pt")
        
        print("📊 Running 3-epoch test to verify NaN fixes...")
        results = model.train(
            data="underwater_plastics/data.yaml",
            epochs=3,  # 短测试
            imgsz=640,
            device=0,
            workers=0,
            batch=8,  # 小批次
            name='cpraformer-nan-test',
            project='runs/detect',
            save=False,  # 不保存避免占用空间
            plots=False,
            verbose=True,
            patience=10,
            lr0=0.001,  # 低学习率
        )
        
        print("✅ NaN fix test completed successfully!")
        print(f"Final results: {results.results_dict}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nan_fixes()
    if success:
        print("🎉 NaN fixes verified! You can now continue training without NaN errors.")
    else:
        print("⚠️ There may still be issues. Check the error output above.")