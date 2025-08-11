import cv2
import numpy as np
from PIL import Image
import torch
from typing import Union, Tuple, Optional
import os

def load_image(image_path: str) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        图像数组
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 使用OpenCV加载
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def save_image(image: Union[np.ndarray, Image.Image], 
               output_path: str, 
               quality: int = 95) -> None:
    """
    保存图像
    
    Args:
        image: 图像数据
        output_path: 输出路径
        quality: 保存质量
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if isinstance(image, np.ndarray):
        # numpy数组转PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    
    # 保存图像
    image.save(output_path, quality=quality)

def resize_image(image: Union[np.ndarray, Image.Image], 
                target_size: Tuple[int, int],
                interpolation: str = "bilinear") -> Union[np.ndarray, Image.Image]:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        interpolation: 插值方法
        
    Returns:
        调整后的图像
    """
    if isinstance(image, np.ndarray):
        # OpenCV插值方法映射
        cv2_interp = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }
        
        resized = cv2.resize(image, target_size, interpolation=cv2_interp[interpolation])
        return resized
    
    elif isinstance(image, Image.Image):
        # PIL插值方法映射
        pil_interp = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS
        }
        
        resized = image.resize(target_size, resample=pil_interp[interpolation])
        return resized
    
    else:
        raise TypeError("不支持的图像类型")

def normalize_image(image: np.ndarray, 
                   mean: Optional[Tuple[float, float, float]] = None,
                   std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    图像标准化
    
    Args:
        image: 输入图像
        mean: 均值
        std: 标准差
        
    Returns:
        标准化后的图像
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)  # ImageNet标准
    if std is None:
        std = (0.229, 0.224, 0.225)   # ImageNet标准
    
    # 转换为float32
    image = image.astype(np.float32) / 255.0
    
    # 标准化
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
    
    return image

def denormalize_image(image: np.ndarray,
                     mean: Optional[Tuple[float, float, float]] = None,
                     std: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """
    图像反标准化
    
    Args:
        image: 输入图像
        mean: 均值
        std: 标准差
        
    Returns:
        反标准化后的图像
    """
    if mean is None:
        mean = (0.485, 0.456, 0.406)  # ImageNet标准
    if std is None:
        std = (0.229, 0.224, 0.225)   # ImageNet标准
    
    # 反标准化
    for i in range(3):
        image[:, :, i] = image[:, :, i] * std[i] + mean[i]
    
    # 裁剪到[0, 1]范围
    image = np.clip(image, 0, 1)
    
    return image

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Tensor转图像数组
    
    Args:
        tensor: 输入tensor
        
    Returns:
        图像数组
    """
    # 确保tensor在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 转换为numpy
    image = tensor.detach().numpy()
    
    # 处理维度
    if image.ndim == 4:
        image = image.squeeze(0)  # 移除batch维度
    
    if image.ndim == 3:
        # CHW -> HWC
        if image.shape[0] in [1, 3, 4]:
            image = np.transpose(image, (1, 2, 0))
    
    # 确保值在[0, 1]范围内
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    return image

def image_to_tensor(image: Union[np.ndarray, Image.Image], 
                   normalize: bool = True) -> torch.Tensor:
    """
    图像转Tensor
    
    Args:
        image: 输入图像
        normalize: 是否标准化
        
    Returns:
        Tensor
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 转换为float32
    image = image.astype(np.float32) / 255.0
    
    # HWC -> CHW
    if image.ndim == 3:
        image = np.transpose(image, (2, 0, 1))
    
    # 添加batch维度
    image = np.expand_dims(image, axis=0)
    
    # 转换为tensor
    tensor = torch.from_numpy(image)
    
    # 标准化
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    
    return tensor

def create_low_resolution_image(hr_image: Union[np.ndarray, Image.Image],
                               scale_factor: int = 4,
                               interpolation: str = "bilinear") -> np.ndarray:
    """
    创建低分辨率图像（用于测试）
    
    Args:
        hr_image: 高分辨率图像
        scale_factor: 缩放因子
        interpolation: 插值方法
        
    Returns:
        低分辨率图像
    """
    if isinstance(hr_image, Image.Image):
        hr_image = np.array(hr_image)
    
    # 获取原始尺寸
    h, w = hr_image.shape[:2]
    
    # 计算低分辨率尺寸
    lr_h, lr_w = h // scale_factor, w // scale_factor
    
    # 下采样
    lr_image = resize_image(hr_image, (lr_w, lr_h), interpolation)
    
    return lr_image

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算PSNR
    
    Args:
        img1: 图像1
        img2: 图像2
        
    Returns:
        PSNR值
    """
    # 确保图像类型一致
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    计算SSIM
    
    Args:
        img1: 图像1
        img2: 图像2
        
    Returns:
        SSIM值
    """
    # 使用skimage计算SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, multichannel=True, data_range=255)
    except ImportError:
        print("警告: 未安装skimage，无法计算SSIM")
        return 0.0 