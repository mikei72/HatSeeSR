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
from utils.image_utils import load_image, save_image, create_low_resolution_image

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

def demo_batch_processing():
    """演示批量处理"""
    print("\n" + "=" * 60)
    print("演示3: 批量图像处理")
    print("=" * 60)
    
    # 创建测试数据目录
    test_data_dir = "examples/test_data"
    output_dir = "examples/batch_outputs"
    
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有测试数据
    test_images = [f for f in os.listdir(test_data_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_images:
        print("创建测试数据...")
        # 创建一些测试图像
        demo_create_test_data(test_data_dir)
        test_images = [f for f in os.listdir(test_data_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 创建处理流程
    pipeline = SuperResolutionPipeline(device="cuda")
    
    # 批量处理
    print("开始批量处理...")
    start_time = time.time()
    
    results = []
    for i, filename in enumerate(test_images):
        print(f"处理第 {i+1}/{len(test_images)} 张图像: {filename}")
        
        input_path = os.path.join(test_data_dir, filename)
        output_path = os.path.join(output_dir, f"sr_{filename}")
        
        try:
            hr_final, process_info = pipeline.process(
                lr_image=input_path,
                strength=0.15,
                output_path=output_path
            )
            
            results.append({
                'filename': filename,
                'success': True,
                'info': process_info
            })
            
            print(f"  ✓ 处理成功")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            results.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 总结结果
    successful = sum(1 for r in results if r['success'])
    print(f"\n批量处理完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"成功处理: {successful}/{len(test_images)} 张图像")
    print(f"平均每张图像: {total_time/len(test_images):.2f}秒")
    
    return results

def demo_evaluation():
    """演示评估功能"""
    print("\n" + "=" * 60)
    print("演示4: 结果评估")
    print("=" * 60)
    
    # 检查是否有评估数据
    gt_dir = "examples/ground_truth"
    pred_dir = "examples/batch_outputs"
    
    if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
        print("跳过评估演示（需要真实图像和预测图像）")
        return
    
    # 创建评估器
    calculator = MetricsCalculator(device="cuda")
    
    # 获取图像文件
    gt_files = [f for f in os.listdir(gt_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pred_files = [f for f in os.listdir(pred_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not gt_files or not pred_files:
        print("未找到评估图像")
        return
    
    print(f"找到 {len(gt_files)} 张真实图像和 {len(pred_files)} 张预测图像")
    
    # 加载图像并计算指标
    gt_images = []
    pred_images = []
    
    for gt_file in gt_files[:5]:  # 限制数量以节省时间
        gt_path = os.path.join(gt_dir, gt_file)
        gt_image = load_image(gt_path)
        gt_images.append(gt_image)
        
        # 查找对应的预测图像
        pred_file = f"sr_{gt_file}"
        pred_path = os.path.join(pred_dir, pred_file)
        
        if os.path.exists(pred_path):
            pred_image = load_image(pred_path)
            pred_images.append(pred_image)
        else:
            print(f"警告: 未找到对应的预测图像 {pred_file}")
    
    if len(gt_images) != len(pred_images):
        print("真实图像和预测图像数量不匹配")
        return
    
    # 计算指标
    print("计算评估指标...")
    metrics = calculator.calculate_average_metrics(gt_images, pred_images)
    
    # 显示结果
    print("\n评估结果:")
    print("-" * 30)
    for metric_name, value in metrics.items():
        if not metric_name.endswith('_std'):
            std_key = f"{metric_name}_std"
            if std_key in metrics:
                print(f"{metric_name.upper()}: {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"{metric_name.upper()}: {value:.4f}")
    
    return metrics

def demo_create_test_data(output_dir: str):
    """创建测试数据"""
    print("创建测试数据...")
    
    # 这里可以创建一些简单的测试图像
    # 或者从网络下载一些测试图像
    
    # 示例：创建一个简单的测试图像
    import numpy as np
    from PIL import Image
    
    # 创建一个简单的测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 添加一些简单的图案
    test_image[50:150, 50:150] = [255, 0, 0]  # 红色方块
    test_image[100:200, 100:200] = [0, 255, 0]  # 绿色方块
    
    # 保存测试图像
    test_image_pil = Image.fromarray(test_image)
    test_image_path = os.path.join(output_dir, "test_image_1.jpg")
    test_image_pil.save(test_image_path)
    
    # 创建更多测试图像
    for i in range(2, 4):
        # 创建不同颜色的测试图像
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        color = colors[i % len(colors)]
        
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_image[50:150, 50:150] = color
        
        test_image_pil = Image.fromarray(test_image)
        test_image_path = os.path.join(output_dir, f"test_image_{i}.jpg")
        test_image_pil.save(test_image_path)
    
    print(f"创建了 {3} 张测试图像")

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