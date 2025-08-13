import torch
from PIL import Image
import numpy as np
from typing import Union, Optional, Tuple
import os
from pathlib import Path

from torchaudio.functional import resample

from models.hat_model import HATModel
from models.seesr_model import initialize_models_and_pipeline, enhance_single_image
from config.config import Config

class SuperResolutionPipeline:
    """文本引导图像超分辨率处理流程"""
    
    def __init__(self, 
                 device: str = "cuda",
                 hat_model_path: Optional[str] = None,
                 ram_ft_path: Optional[str] = None,
                 seesr_model_path: Optional[str] = None,
                 pretrained_model_path: Optional[str] = None):
        """
        初始化超分辨率处理流程
        
        Args:
            device: 计算设备
            hat_model_path: HAT模型路径
            ram_model_path: RAM模型路径
            sd_model_path: SD模型路径
            lora_path: LoRA权重路径
        """
        self.device = device
        
        # 使用默认路径或提供的路径
        hat_model_path = hat_model_path or Config.HAT_MODEL_PATH
        ram_ft_path = ram_ft_path or Config.RAM_FT_PATH
        seesr_model_path = seesr_model_path or Config.SEESR_MODEL_PATH
        pretrained_model_path = pretrained_model_path or Config.PRETRAINED_MODEL_PATH
        
        # 初始化模型
        print("正在初始化超分辨率处理流程...")
        
        # 步骤1: 保真度超分模型
        print("1. 加载HAT模型...")
        self.hat_model = HATModel(hat_model_path, device)
        
        # 步骤2: SeeSR模型
        print("2. 加载SeeSR模型...")
        self.pipeline, self.model, self.generator, self.accelerator, self.args = (
            initialize_models_and_pipeline(
                seesr_model_path=seesr_model_path,
                ram_ft_path=ram_ft_path,
                pretrained_model_path=pretrained_model_path,
            ))
    
    def step1_fidelity_upscaling(self, lr_image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        步骤1: 保真度超分
        
        Args:
            lr_image: 低分辨率输入图像
            
        Returns:
            高保真度的基础图像
        """
        print("执行步骤1: 保真度超分...")
        
        # 使用HAT模型进行4倍超分
        hr_base = self.hat_model.upscale(
            lr_image, 
            target_size=Config.TARGET_SIZE
        )
        
        print(f"保真度超分完成，输出尺寸: {hr_base.shape}")
        return hr_base
    
    def step2_seesr_process(self, image: Union[str, Image.Image, np.ndarray]) -> tuple[str, Image.Image]:
        """
        步骤2: 语义标签生成
        
        Args:
            image: 输入图像（可以是LR或HR_base）
            
        Returns:
            逗号分隔的关键词标签
        """
        print("执行步骤2: SeeSR模型处理...")
        
        prompt, hr_final = enhance_single_image(
            image_array=image,
            pipeline=self.pipeline,
            model=self.model,
            generator=self.generator,
            accelerator=self.accelerator,
            args=self.args
        )

        w, h = hr_final.size
        hr_final = hr_final.resize((w // 4, h // 4), resample=Image.Resampling.LANCZOS)

        return prompt, hr_final
    
    def process(self, 
               lr_image: Union[str, Image.Image, np.ndarray],
               strength: float = Config.DEFAULT_STRENGTH,
               save_intermediate: bool = False,
               output_path: Optional[str] = None) -> Tuple[Image.Image, dict]:
        """
        完整的超分辨率处理流程
        
        Args:
            lr_image: 低分辨率输入图像
            strength: 重绘强度
            save_intermediate: 是否保存中间结果
            output_path: 输出路径
            
        Returns:
            最终图像和处理信息
        """
        print("开始文本引导图像超分辨率处理...")
        
        # 步骤1: 保真度超分
        hr_base = self.step1_fidelity_upscaling(lr_image)

        # 步骤2: SeeSR模型处理
        text_prompt, hr_final = self.step2_seesr_process(hr_base)

        # 保存中间结果（可选）
        if save_intermediate:
            self._save_intermediate_results(hr_base, text_prompt, output_path)
        
        # 保存最终结果
        if output_path:
            self._save_final_result(hr_final, output_path)

        """from utils.metrics_utils import calculate_metrics
        metrics1 = calculate_metrics('examples/hr_base_demo3.png', 'examples/gt3.png', 10, True)
        metrics2 = calculate_metrics('examples/sr_demo3.png', 'examples/gt3.png', 10, True)
        print(f'PSNR: {metrics1["psnr"]:.4f} dB, SSIM: {metrics1["ssim"]:.4f}')
        print(f'PSNR: {metrics2["psnr"]:.4f} dB, SSIM: {metrics2["ssim"]:.4f}')"""
        
        # 返回结果和处理信息
        process_info = {
            "input_size": self._get_image_size(lr_image),
            "hr_base_size": hr_base.shape[:2],
            "final_size": hr_final.size,
            "text_prompt": text_prompt,
            "strength": strength,
            "upscale_factor": Config.UPSCALE_FACTOR
        }
        
        print("文本引导图像超分辨率处理完成")
        return hr_final, process_info
    
    def _save_intermediate_results(self, hr_base: np.ndarray, text_prompt: str, output_path: str):
        """保存中间结果"""
        if output_path:
            # 保存HR_base
            hr_base_pil = Image.fromarray(hr_base)
            hr_base_path = output_path.replace("sr", "hr_base")
            hr_base_pil.save(hr_base_path)
            
            # 保存文本提示词
            prompt_path = output_path.replace("sr", "prompt").replace("png", "txt")
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(text_prompt)
    
    def _save_final_result(self, hr_final: Image.Image, output_path: str):
        """保存最终结果"""
        hr_final.save(output_path)
        print(f"最终结果已保存到: {output_path}")
    
    def _get_image_size(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[int, int]:
        """获取图像尺寸"""
        if isinstance(image, str):
            with Image.open(image) as img:
                return img.size
        elif isinstance(image, Image.Image):
            return image.size
        elif isinstance(image, np.ndarray):
            return (image.shape[1], image.shape[0])
        else:
            return (0, 0)
    
    def batch_process(self, 
                     lr_images: list,
                     strength: float = Config.DEFAULT_STRENGTH,
                     output_dir: Optional[str] = None) -> list:
        """
        批量处理图像
        
        Args:
            lr_images: 低分辨率图像列表
            strength: 重绘强度
            output_dir: 输出目录
            
        Returns:
            处理结果列表
        """
        results = []
        
        for i, lr_image in enumerate(lr_images):
            print(f"处理第 {i+1}/{len(lr_images)} 张图像...")
            
            # 生成输出路径
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"result_{i:04d}.png")
            
            # 处理单张图像
            hr_final, process_info = self.process(
                lr_image, 
                strength, 
                save_intermediate=True,
                output_path=output_path
            )
            
            results.append({
                "image": hr_final,
                "info": process_info
            })

        return results
