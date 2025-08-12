import os
from pathlib import Path

class Config:
    """项目配置文件"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 模型路径配置
    HAT_MODEL_PATH = os.getenv("HAT_MODEL_PATH", "/inspire/hdd/project/embodied-multimodality/public/xyzii/Diff/HatSeeSR/models/weights/HAT-L_SRx4_ImageNet-pretrain.pth")
    RAM_FT_PATH = os.getenv("RAM_MODEL_PATH", "/inspire/hdd/project/embodied-multimodality/public/xyzii/Diff/HatSeeSR/models/weights/DAPE.pth")
    SEESR_MODEL_PATH = os.getenv("SD_MODEL_PATH", "/inspire/hdd/project/embodied-multimodality/public/xyzii/Diff/HatSeeSR/models/weights/seesr")
    PRETRAINED_MODEL_PATH = os.getenv("PRETRAINED_MODEL_PATH", "/inspire/hdd/project/embodied-multimodality/public/xyzii/Diff/HatSeeSR/models/weights/sd-turbo")
    
    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    TEMP_DIR = PROJECT_ROOT / "temp"
    
    # 超分辨率参数
    UPSCALE_FACTOR = 4
    TARGET_SIZE = (1024, 1024)  # 目标分辨率
    
    # 扩散模型参数
    STRENGTH_RANGE = (0.1, 0.25)  # 重绘强度范围
    DEFAULT_STRENGTH = 0.15  # 默认重绘强度
    GUIDANCE_SCALE = 7.5  # 分类器自由引导尺度
    NUM_INFERENCE_STEPS = 50  # 推理步数
    
    # RAM模型参数
    RAM_CONFIDENCE_THRESHOLD = 0.5  # 标签置信度阈值
    MAX_TAGS = 10  # 最大标签数量
    
    # LoRA训练参数
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # 训练参数
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    SAVE_STEPS = 500
    
    # 评估参数
    METRICS = ["psnr", "ssim", "lpips"]
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.OUTPUT_DIR,
            cls.TEMP_DIR,
            # cls.PROJECT_ROOT / "models" / "hat",
            # cls.PROJECT_ROOT / "models" / "ram",
            # cls.PROJECT_ROOT / "models" / "stable-diffusion-v1-5",
            cls.PROJECT_ROOT / "logs",
            cls.PROJECT_ROOT / "checkpoints"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        return directories 