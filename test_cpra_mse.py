"""
CPRA-MSEC2f模型测试脚本
测试新的CPRAformer多尺度增强C2f模块性能
"""
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.cpraformer import CPRA_MSEC2f, MultiScaleEnhancement
import time

def test_cpra_mse_modules():
    """测试CPRA-MSEC2f模块的基本功能"""
    print("=" * 60)
    print("测试CPRA-MSEC2f模块基本功能")
    print("=" * 60)
    
    # 测试MultiScaleEnhancement模块
    print("\n1. 测试MultiScaleEnhancement模块:")
    channels = 256
    mse = MultiScaleEnhancement(channels)
    x = torch.randn(2, channels, 32, 32)
    
    start_time = time.time()
    output = mse(x)
    end_time = time.time()
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    print(f"   参数数量: {sum(p.numel() for p in mse.parameters()):,}")
    
    # 测试CPRA_MSEC2f模块  
    print("\n2. 测试CPRA_MSEC2f模块:")
    c1, c2 = 128, 256
    cpra_mse = CPRA_MSEC2f(c1, c2, n=2)
    x = torch.randn(2, c1, 64, 64)
    
    start_time = time.time()
    output = cpra_mse(x)
    end_time = time.time()
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    print(f"   参数数量: {sum(p.numel() for p in cpra_mse.parameters()):,}")

def test_model_loading():
    """测试模型加载和基本推理"""
    print("\n" + "=" * 60)
    print("测试CPRA-MSEC2f模型加载")
    print("=" * 60)
    
    try:
        # 尝试加载CPRA-MSE配置
        print("\n1. 加载yolov8n-cpra-mse.yaml配置:")
        model = YOLO('yolov8n-cpra-mse.yaml')
        print(f"   模型加载成功!")
        
        # 模型信息
        print(f"   模型类型: {type(model.model)}")
        
        # 测试推理
        print("\n2. 测试推理:")
        x = torch.randn(1, 3, 640, 640)
        
        model.model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model.model(x)
            end_time = time.time()
        
        print(f"   输入形状: {x.shape}")
        if isinstance(output, (list, tuple)):
            print(f"   输出数量: {len(output)}")
            for i, o in enumerate(output):
                if hasattr(o, 'shape'):
                    print(f"   输出{i}形状: {o.shape}")
                else:
                    print(f"   输出{i}类型: {type(o)}")
        else:
            print(f"   输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")
        print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
        
        # 显示模型参数统计
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print("\n3. 模型统计:")
        print(f"   总参数数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        return True
        
    except Exception as e:
        print(f"   模型加载失败: {e}")
        return False

def compare_with_baseline():
    """与基础模型进行对比"""
    print("\n" + "=" * 60)
    print("性能对比测试")
    print("=" * 60)
    
    models_to_compare = [
        ("YOLOv8n原版", "yolov8n.yaml"),
        ("CPRA-MSE增强版", "yolov8n-cpra-mse.yaml")
    ]
    
    results = {}
    
    for name, config in models_to_compare:
        print(f"\n测试 {name}:")
        try:
            model = YOLO(config) if config.endswith('.yaml') else YOLO(config)
            
            # 计算参数数量
            total_params = sum(p.numel() for p in model.model.parameters())
            
            # 测试推理速度
            x = torch.randn(1, 3, 640, 640)
            model.model.eval()
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model.model(x)
            
            # 正式测试
            times = []
            with torch.no_grad():
                for _ in range(20):
                    start_time = time.time()
                    _ = model.model(x)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
            
            avg_time = sum(times) / len(times)
            
            results[name] = {
                'params': total_params,
                'inference_time': avg_time
            }
            
            print(f"   ✓ 参数数量: {total_params:,}")
            print(f"   ✓ 平均推理时间: {avg_time:.2f}ms")
            
        except Exception as e:
            print(f"   ✗ 测试失败: {e}")
            results[name] = None
    
    # 显示对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    baseline_name = "YOLOv8n原版"
    enhanced_name = "CPRA-MSE增强版"
    
    if results.get(baseline_name) and results.get(enhanced_name):
        baseline = results[baseline_name]
        enhanced = results[enhanced_name]
        
        param_ratio = enhanced['params'] / baseline['params']
        time_ratio = enhanced['inference_time'] / baseline['inference_time']
        
        print(f"\n参数数量对比:")
        print(f"   基础版本: {baseline['params']:,}")
        print(f"   增强版本: {enhanced['params']:,}")
        print(f"   参数增长: {param_ratio:.2f}x ({(param_ratio-1)*100:+.1f}%)")
        
        print(f"\n推理速度对比:")
        print(f"   基础版本: {baseline['inference_time']:.2f}ms")
        print(f"   增强版本: {enhanced['inference_time']:.2f}ms") 
        print(f"   速度变化: {time_ratio:.2f}x ({(time_ratio-1)*100:+.1f}%)")

if __name__ == "__main__":
    print("CPRA-MSEC2f模型测试")
    print("开始测试新的CPRAformer多尺度增强C2f模块...")
    
    # 1. 测试基础模块
    test_cpra_mse_modules()
    
    # 2. 测试模型加载
    model_loaded = test_model_loading()
    
    # 3. 性能对比
    if model_loaded:
        compare_with_baseline()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)