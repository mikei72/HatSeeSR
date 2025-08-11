import torch
import torch.nn.functional as F
import yaml
import os
from PIL import Image, ImageOps
import numpy as np

# 我们只需要 basicsr 的底层工具，不再需要 build_model
from models.HAT.hat.archs.hat_arch import HAT  # 直接导入模型架构
from basicsr.utils import img2tensor, tensor2img

from models.HAT.hat.models import hat_model


# --- 辅助类 (无需改动) ---
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")


# --------------------

class HATInference:
    """
    一个封装了 HAT 模型的推理接口。
    采用直接调用网络的方式，以获得对数据流的完全控制。
    """

    def __init__(self, config_path: str, model_path: str, device: str = 'cuda'):
        # 1. 加载配置
        try:
            with open(config_path, 'r') as f:
                opt = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")

        self.opt = AttrDict(opt)

        # 2. 准备设备
        if device == 'cuda' and not torch.cuda.is_available():
            print("警告: CUDA 不可用，自动切换到 CPU 模式。")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # --- 修正部分：直接构建和加载网络，绕过 Model 控制器 ---
        # 提取网络配置
        network_opt = self.opt.network_g
        # 直接实例化 HAT 网络
        self.net_g = HAT(
            upscale=network_opt.upscale,
            in_chans=network_opt.in_chans,
            img_size=network_opt.img_size,
            window_size=network_opt.window_size,
            compress_ratio=network_opt.compress_ratio,
            squeeze_factor=network_opt.squeeze_factor,
            conv_scale=network_opt.conv_scale,
            overlap_ratio=network_opt.overlap_ratio,
            img_range=network_opt.img_range,
            depths=network_opt.depths,
            embed_dim=network_opt.embed_dim,
            num_heads=network_opt.num_heads,
            mlp_ratio=network_opt.mlp_ratio,
            upsampler=network_opt.upsampler,
            resi_connection=network_opt.resi_connection
        ).to(self.device)

        print(f"正在从 '{model_path}' 加载模型权重...")
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
        param_key = self.opt.path.get('param_key_g', 'params')
        params = load_net[param_key] if param_key in load_net else load_net

        self.net_g.load_state_dict(params, strict=True)
        print("模型权重加载成功！")
        # --- 修正结束 ---

        # 3. 设置为评估模式并获取关键参数
        self.net_g.eval()
        self.scale = self.opt.scale
        self.window_size = self.opt.network_g.window_size

        print(f"HAT 模型已准备就绪。放大倍数: x{self.scale}, 窗口大小: {self.window_size}")

    def infer(self, input_image: Image.Image) -> Image.Image:
        # 1. 预处理
        img_lq = input_image.convert('RGB')
        # 将 uint8[0,255] 的图像转为 float32[0,1] 的张量
        img_lq = np.array(img_lq) / 255.
        img_lq = torch.from_numpy(img_lq).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 2. 自动填充
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        img_lq = F.pad(img_lq, (0, w_pad, 0, h_pad), 'replicate')

        # 3. 直接通过网络进行推理
        with torch.no_grad():
            output_tensor = self.net_g(img_lq)

        # 4. 裁剪掉填充区域
        h_new, w_new = h_old * self.scale, w_old * self.scale
        output_tensor = output_tensor[:, :, :h_new, :w_new]

        # 5. 后处理
        # 将 float32[0,1] 的张量转为 uint8[0,255] 的图像
        output_img = output_tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = (output_img * 255.0).round().astype(np.uint8)
        output_img = np.transpose(output_img, (1, 2, 0))  # C,H,W -> H,W,C

        return Image.fromarray(output_img)


# --- 主程序入口 ---
if __name__ == '__main__':
    MODEL_PATH = "experiments/pretrained_models/HAT-L_SRx4_ImageNet-pretrain.pth"
    CONFIG_FILE_PATH = "options/test/HAT-L_SRx4_ImageNet-pretrain.yml"  # 你的 .yml 文件
    INPUT_IMAGE_PATH = "datasets/Urban100/LR_bicubic/X4/img012.png"  # 你的低分辨率图片
    OUTPUT_IMAGE_PATH = "results/demo_output_x4_final.png"

    print("正在初始化 HAT 模型...")
    try:
        hat_predictor = HATInference(
            config_path=CONFIG_FILE_PATH,
            model_path=MODEL_PATH,
            device='cuda'
        )
    except Exception as e:
        print(f"模型初始化失败。错误: {e}")
        import traceback; traceback.print_exc(); exit()

    print(f"正在加载输入图片: {INPUT_IMAGE_PATH}")
    try:
        low_res_image = Image.open(INPUT_IMAGE_PATH)
        low_res_image = ImageOps.exif_transpose(low_res_image)

        print("开始进行超分辨率处理...")
        high_res_image = hat_predictor.infer(low_res_image)
        os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
        high_res_image.save(OUTPUT_IMAGE_PATH)
        print(f"处理完成！结果已保存到: {OUTPUT_IMAGE_PATH}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback; traceback.print_exc()