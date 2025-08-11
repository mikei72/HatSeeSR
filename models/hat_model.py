from PIL import Image, ImageOps
import numpy as np
from typing import Union
import os

# 从我们创建的模块中导入 HATInference 类
from models.HAT.hat_inference import HATInference

# 在这里定义模型的固定路径，或者从全局配置中读取
# 这使得主流程代码更加干净
CONFIG_FILE_PATH = "models/HAT/HAT-L_SRx4_ImageNet-pretrain.yml"


class HATModel:
    """
    一个高层封装，用于调用 HAT 超分模型。
    它在内部实例化并持有一个 HATInference 对象。
    """

    def __init__(self, hat_model_path: str, device: str = 'cuda'):
        """
        初始化模型加载器。

        Args:
            hat_model_path (str): 要加载的 HAT 模型权重 (.pth) 文件路径。
            device (str): 使用的设备, e.g., 'cuda' or 'cpu'
        """
        print("正在初始化 HAT 模型...")

        if not os.path.exists(hat_model_path):
            raise FileNotFoundError(f"指定的 HAT 模型权重文件未找到: {hat_model_path}")

        try:
            # 在内部实例化底层的 HATInference，并传入动态的模型路径
            self.predictor = HATInference(
                config_path=CONFIG_FILE_PATH,
                model_path=hat_model_path,  # 使用传入的参数
                device=device
            )
            print(f"HAT 模型已从 '{hat_model_path}' 成功加载。")
        except Exception as e:
            print(f"HAT 模型初始化失败。请检查路径配置和依赖。")
            raise e

    def upscale(self, lr_image: Union[str, Image.Image, np.ndarray], target_size: int = None) -> np.ndarray:
        """
        对输入的低分辨率图像进行超分。

        Args:
            lr_image: 低分辨率输入，可以是文件路径(str), PIL.Image, 或 NumPy 数组。
            target_size: (可选) 如果提供，会将超分后的图像缩放到这个尺寸。

        Returns:
            超分辨率后的图像，以 NumPy 数组 (H, W, C) 格式返回。
        """
        # 1. 统一输入格式为 PIL.Image
        if isinstance(lr_image, str):
            try:
                image_pil = Image.open(lr_image)
            except FileNotFoundError:
                raise FileNotFoundError(f"输入图片路径未找到: {lr_image}")
        elif isinstance(lr_image, np.ndarray):
            image_pil = Image.fromarray(lr_image)
        elif isinstance(lr_image, Image.Image):
            image_pil = lr_image
        else:
            raise TypeError(f"不支持的输入类型: {type(lr_image)}")

        # 2. 校正 EXIF 方向
        image_pil = ImageOps.exif_transpose(image_pil)

        # 3. 调用底层推理接口
        # predictor.infer 接收并返回 PIL.Image
        sr_image_pil = self.predictor.infer(image_pil)

        # 4. 转换为 NumPy 数组
        sr_image_np = np.array(sr_image_pil)

        return sr_image_np