#!/usr/bin/env python3
"""
演示脚本：展示基于扩散模型的文本引导图像超分辨率

这个脚本展示了如何使用项目进行图像超分辨率处理，
包括单张图像处理、批量处理和参数调优。
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from pipeline.super_resolution_pipeline import SuperResolutionPipeline

def demo_single_image_processing():
    """演示单张图像处理"""
    print("=" * 60)
    print("演示1: 单张图像超分辨率处理")
    print("=" * 60)
    
    # 创建测试图像（如果没有的话）
    test_image_path = "examples/test3.png"
    if not os.path.exists(test_image_path):
        print("创建测试图像...")
        # 这里可以创建一个简单的测试图像
        # 或者提示用户提供图像
        print(f"请将测试图像放在: {test_image_path}")
        return
    
    # 创建输出目录
    output_dir = "examples/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建处理流程
    print("初始化超分辨率处理流程...")
    pipeline = SuperResolutionPipeline(device="cuda")
    
    # 处理图像
    output_path = os.path.join(output_dir, "sr_demo3.png")
    
    print(f"开始处理图像: {test_image_path}")
    start_time = time.time()
    
    hr_final, process_info = pipeline.process(
        lr_image=test_image_path,
        strength=0.15,
        save_intermediate=True,
        output_path=output_path
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"处理完成！耗时: {processing_time:.2f}秒")
    print(f"结果保存到: {output_path}")
    
    # 打印处理信息
    print("\n处理信息:")
    for key, value in process_info.items():
        print(f"  {key}: {value}")
    
    return output_path

def demo_parameter_tuning():
    """演示参数调优"""
    print("\n" + "=" * 60)
    print("演示2: 参数调优 - 不同strength值的效果")
    print("=" * 60)
    
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("跳过参数调优演示（需要测试图像）")
        return
    
    output_dir = "parameter_tuning"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建处理流程
    pipeline = SuperResolutionPipeline(device="cuda")
    
    # 测试不同的strength值
    strength_values = [0.1, 0.15, 0.2, 0.25]
    
    results = []
    for strength in strength_values:
        print(f"\n测试 strength = {strength}")
        
        output_path = os.path.join(output_dir, f"strength_{strength:.2f}.png")
        
        start_time = time.time()
        hr_final, process_info = pipeline.process(
            lr_image=test_image_path,
            strength=strength,
            output_path=output_path
        )
        end_time = time.time()
        
        results.append({
            'strength': strength,
            'output_path': output_path,
            'processing_time': end_time - start_time,
            'info': process_info
        })
        
        print(f"  处理时间: {end_time - start_time:.2f}秒")
        print(f"  输出路径: {output_path}")
    
    # 总结结果
    print("\n参数调优结果总结:")
    print("-" * 40)
    for result in results:
        print(f"Strength {result['strength']:.2f}: {result['processing_time']:.2f}秒")
    
    return results


def main():
    """主演示函数"""
    print("基于扩散模型的文本引导图像超分辨率 - 演示程序")
    print("=" * 80)
    
    # 创建必要的目录
    Config.create_directories()
    
    # 演示1: 单张图像处理
    demo_single_image_processing()
    
    # 演示2: 参数调优
    """try:
        demo_parameter_tuning()
    except Exception as e:
        print(f"演示2失败: {e}")
    
    # 演示3: 批量处理
    try:
        demo_batch_processing()
    except Exception as e:
        print(f"演示3失败: {e}")
    
    # 演示4: 评估
    try:
        demo_evaluation()
    except Exception as e:
        print(f"演示4失败: {e}")
    
    print("\n" + "=" * 80)
    print("演示程序完成！")
    print("=" * 80)
    
    print("\n使用说明:")
    print("1. 将您的测试图像放在 examples/test_image.jpg")
    print("2. 运行 python examples/demo.py")
    print("3. 查看 examples/outputs/ 目录中的结果")
    print("4. 调整 strength 参数以获得最佳效果")"""

if __name__ == "__main__":
    main() 