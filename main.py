#!/usr/bin/env python3
"""
基于扩散模型的文本引导图像超分辨率主程序

使用方法:
    python main.py --input input_image.jpg --output output_image.png
    python main.py --input_dir input_folder --output_dir output_folder
    python main.py --train_lora --train_data train_folder
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
from typing import Any

from utils.metrics_utils import calculate_metrics

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config.config import Config
from pipeline.super_resolution_pipeline import SuperResolutionPipeline

def process_single_image(input_path: str, 
                        output_path: str,
                        strength: float = Config.DEFAULT_STRENGTH,
                        device: str = "cuda",
                        lora_path: Optional[str] = None) -> dict:
    """
    处理单张图像
    
    Args:
        input_path: 输入图像路径
        output_path: 输出图像路径
        strength: 重绘强度
        device: 计算设备
        lora_path: LoRA权重路径
        
    Returns:
        处理信息
    """
    print(f"正在处理图像: {input_path}")
    
    # 创建处理流程
    pipeline = SuperResolutionPipeline(
        device=device
    )
    
    # 执行超分辨率处理
    hr_final, process_info = pipeline.process(
        lr_image=input_path,
        save_intermediate=True,
        output_path=output_path
    )
    
    print(f"处理完成，结果保存到: {output_path}")
    return process_info

def process_batch_images(input_dir: str,
                        output_dir: str,
                        pipeline: Any) -> List[dict]:
    """
    批量处理图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        strength: 重绘强度
        device: 计算设备
        lora_path: LoRA权重路径
        
    Returns:
        处理信息列表
    """
    print(f"正在批量处理目录: {input_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入图像列表
    input_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not input_files:
        print("未找到图像文件")
        return []
    
    print(f"找到 {len(input_files)} 张图像")
    
    # 批量处理
    results = []
    for i, filename in enumerate(input_files):
        print(f"处理第 {i+1}/{len(input_files)} 张图像: {filename}")
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"sr_{filename}")
        
        try:
            hr_final, process_info = pipeline.process(
                lr_image=input_path,
                save_intermediate=True,
                output_path=output_path
            )
            
            results.append(process_info)
            print(f"✓ {filename} 处理完成\n")
            
        except Exception as e:
            print(f"✗ {filename} 处理失败: {e}")
            continue
    
    print(f"批量处理完成，成功处理 {len(results)} 张图像")
    return results


def evaluate_results(gt_dir: str,
                     pred_dir: str,
                     output_dir: str = "evaluation_results",
                     sr_prefix: str = "sr_"):
    """
    评估超分辨率结果，同时对比基础SR图像和最终SR图像。

    Args:
        gt_dir: 真实高分辨率图像目录。
        pred_dir: 包含基础SR和最终SR图像的预测目录。
        output_dir: 评估结果的输出目录。
        sr_prefix: 最终SR图像的文件名前缀。
    """
    print("开始评估超分辨率结果 (对比 Base vs Final SR)...")

    os.makedirs(output_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not gt_files:
        print(f"错误：在目录‘{gt_dir}’中未找到图像文件")
        return

    all_results = []

    print("-" * 70)
    for gt_file in gt_files:
        print(f"处理文件: {gt_file}")
        gt_path = os.path.join(gt_dir, gt_file)

        # --- 查找并评估第一阶段的基础HR图像 (hr_base) ---
        base_file = f"hr_base_{gt_file}"
        base_path = os.path.join(pred_dir, base_file)

        if not os.path.exists(base_path):
            print(f"  [警告] 跳过！ 未找到对应的基础图像： {base_file}")
            print("-" * 70)
            continue

        metrics_base = calculate_metrics(
            sr_path=base_path,
            gt_path=gt_path,
            crop_border=4,
            test_y_channel=True
        )

        # --- 查找并评估第二阶段的最终SR图像 ---
        pred_file = f"{sr_prefix}{gt_file}"
        pred_path = os.path.join(pred_dir, pred_file)

        if not os.path.exists(pred_path):
            print(f"  [警告] 跳过！ 未找到对应的最终SR图像： {pred_file}")
            print("-" * 70)
            continue

        metrics_final = calculate_metrics(
            sr_path=pred_path,
            gt_path=gt_path,
            crop_border=4,
            test_y_channel=True
        )

        # --- 打印并记录结果 ---
        if metrics_base and metrics_final:
            print(f"  - Base  -> PSNR: {metrics_base['psnr']:.4f} dB, SSIM: {metrics_base['ssim']:.4f}")
            print(f"  - Final -> PSNR: {metrics_final['psnr']:.4f} dB, SSIM: {metrics_final['ssim']:.4f}")

            result_row = {
                'filename': gt_file,
                'psnr_base': metrics_base['psnr'],
                'ssim_base': metrics_base['ssim'],
                'psnr_final_sr': metrics_final['psnr'],
                'ssim_final_sr': metrics_final['ssim'],
            }
            all_results.append(result_row)
        print("-" * 70)

    if not all_results:
        print("评估完成，但没有成功处理任何图像对。")
        return

    # --- 结果汇总与保存 ---
    df = pd.DataFrame(all_results)

    # 计算所有指标的平均值
    avg_psnr_base = df['psnr_base'].mean()
    avg_ssim_base = df['ssim_base'].mean()
    avg_psnr_final = df['psnr_final_sr'].mean()
    avg_ssim_final = df['ssim_final_sr'].mean()

    # 创建一个平均值行
    avg_row = pd.DataFrame([{
        'filename': 'Average',
        'psnr_base': avg_psnr_base,
        'ssim_base': avg_ssim_base,
        'psnr_final_sr': avg_psnr_final,
        'ssim_final_sr': avg_ssim_final
    }])

    # 将平均值行添加到 DataFrame 的末尾
    df_final = pd.concat([df, avg_row], ignore_index=True)

    # 保存到 CSV 文件
    pred_dir_name = os.path.basename(os.path.normpath(pred_dir))
    output_csv_path = os.path.join(output_dir, f'{pred_dir_name}.csv')
    df_final.to_csv(output_csv_path, index=False, float_format='%.4f')

    print("\n--- 全局评估摘要 ---")
    print(f"成功处理图像对数量: {len(all_results)}")
    print(f"平均 PSNR (Base): {avg_psnr_base:.4f} dB")
    print(f"平均 SSIM (Base): {avg_ssim_base:.4f}")
    print(f"平均 PSNR (Final SR): {avg_psnr_final:.4f} dB")
    print(f"平均 SSIM (Final SR): {avg_ssim_final:.4f}")
    print(f"\n详细对比结果已保存至: {output_csv_path}")
    print("评估完成。")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于扩散模型的文本引导图像超分辨率")
    
    # 基本参数
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")

    # 单张图像处理
    parser.add_argument("--input", type=str, help="输入图像路径")
    parser.add_argument("--output", type=str, help="输出图像路径")
    
    # 批量处理
    parser.add_argument(
        "--input_dir",
        type=str,
        nargs='+',  # 支持多个目录
        help="输入图像目录，可以同时指定多个"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs='+',  # 支持多个目录
        help="输出图像目录，顺序与输入目录一一对应"
    )
    
    # 评估
    parser.add_argument("--evaluate", action="store_true", help="评估结果")
    parser.add_argument(
        "--gt_dir",
        type=str,
        nargs='+',
        help="真实高分辨率图像目录")
    parser.add_argument("--pred_dir", type=str, help="预测高分辨率图像目录")
    
    args = parser.parse_args()
    
    # 创建必要的目录
    Config.create_directories()
    
    if args.evaluate:
        if not args.gt_dir or not args.pred_dir:
            print("错误: 评估需要指定真实图像目录 (--gt_dir) 和预测图像目录 (--pred_dir)")
            return
        
        evaluate_results(args.gt_dir, args.pred_dir)
    
    elif args.input and args.output:
        # 单张图像处理
        process_info = process_single_image(
            input_path=args.input,
            output_path=args.output,
            strength=args.strength,
            device=args.device,
            lora_path=args.lora_path
        )
        
        print("处理信息:")
        for key, value in process_info.items():
            print(f"  {key}: {value}")

    elif args.input_dir and args.output_dir and args.gt_dir:
        if len(args.input_dir) != len(args.output_dir):
            raise ValueError("输入目录与输出目录数量必须相同！")

        # 创建处理流程
        pipeline = SuperResolutionPipeline(
            device=args.device
        )

        for in_dir, out_dir, gt_dir in zip(args.input_dir, args.output_dir, args.gt_dir):
            print(f"处理输入目录: {in_dir} -> 输出目录: {out_dir}")
            results = process_batch_images(
                input_dir=in_dir,
                output_dir=out_dir,
                pipeline=pipeline
            )

            print(f"批量处理完成，共处理 {len(results)} 张图像")

            evaluate_results(gt_dir, out_dir)
    
    else:
        print("请指定操作类型:")
        print("  单张图像处理: --input <输入图像> --output <输出图像>")
        print("  批量处理: --input_dir <输入目录> --output_dir <输出目录> --gt_dir <真实图像目录>")
        print("  评估结果: --evaluate --gt_dir <真实图像目录> --pred_dir <预测图像目录>")

if __name__ == "__main__":
    main()
