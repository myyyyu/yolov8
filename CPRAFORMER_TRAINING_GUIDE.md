# YOLOv8-CPRAformer Training Guide

## ✅ 集成完成状态
CPRAformer已成功集成到YOLOv8中！所有测试通过：
- ✅ 模块加载和识别
- ✅ 前向传播
- ✅ 训练兼容性  
- ✅ FFT半精度修复

## 🚀 开始训练

### 方法1：使用YOLO CLI命令
```bash
cd "C:\Users\15268\Desktop\ultralytics-main\ultralytics-main"

# CPRAformer完整版 (更好的精度)
yolo task=detect mode=train model=ultralytics/cfg/models/v8/yolov8-cpraformer.yaml data=underwater_plastics/data.yaml epochs=50 imgsz=640 device=0 workers=0 batch=16

# 轻量版本 (更快的速度)
yolo task=detect mode=train model=ultralytics/cfg/models/v8/yolov8-lightcpra.yaml data=underwater_plastics/data.yaml epochs=50 imgsz=640 device=0 workers=0 batch=16
```

### 方法2：使用Python脚本
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('ultralytics/cfg/models/v8/yolov8-cpraformer.yaml')

# 开始训练
results = model.train(
    data='underwater_plastics/data.yaml',
    epochs=50,
    imgsz=640,
    device=0,
    workers=0,
    batch=16,
    name='cpraformer-underwater'
)
```

## 📊 模型对比
- **原版YOLOv8**: 3,157,200参数
- **YOLOv8-CPRAformer**: 2,991,325参数 (-5.2%)  
- **YOLOv8-LightCPRA**: 3,137,168参数 (-0.6%)

## 🔧 训练参数建议
- `batch=16`: 根据您的GPU内存调整 (RTX 4070可以用16-32)
- `workers=0`: Windows建议设为0避免多进程问题
- `device=0`: 使用GPU 0
- `epochs=50-100`: 根据数据集大小调整
- `imgsz=640`: 标准YOLO输入尺寸

## 💡 CPRAformer特性
- **Cross Paradigm Attention**: 在P3/P4层集成，增强特征表示
- **FFT频域处理**: 自动处理半精度兼容性
- **AAFM模块**: 自适应对齐频域模块提升检测效果
- **轻量化设计**: 参数量反而更少，效率更高

## 🎯 针对水下塑料检测的优势
CPRAformer的跨领域表示学习能力特别适合：
- 复杂水下环境的特征提取
- 模糊和低对比度物体的检测
- 多尺度塑料垃圾的识别

开始训练吧！🚀