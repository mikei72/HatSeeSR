# 基于扩散模型的文本引导图像超分辨率

本项目实现了一个基于扩散模型的文本引导图像超分辨率系统，采用"高保真先行，生成式精修"的核心理念，通过三个关键步骤实现高质量的图像超分辨率。

## 核心理念

**高保真先行，生成式精修**

- 首先使用HAT模型进行保真度超分，确保基础图像质量
- 然后使用经过优化的SeeSR模型（基于RAM和Stable Diffusion）自动生成文本引导并进行生成式精修
- SeeSR模型结合了RAM的语义理解和Stable Diffusion的强大生成能力，提供高质量的超分辨率结果

## 核心工作流程

### 步骤1：保真度超分 (Fidelity Upscaling)
- **任务**: 将低分辨率图像（LR）进行4倍放大
- **工具**: 预训练的HAT-L模型
- **输出**: 高保真度的基础图像 HR_base

### 步骤2+3：生成式精修 (Generative Refinement with Text Guidance)
- **任务**: 在HR_base基础上，通过SeeSR模型自动产生语义标签作为文本提示，并据此增加真实感细节
- **工具**: 经过优化的SeeSR模型，该模型预先训练了RAM以增强语义理解能力，并在SD2-Base框架上进一步训练得到
- **输出**: 最终的高分辨率图像 HR_final

## 关键参数

- **num_inference_steps (推理步数)**: 生成式模型的推理次数，平衡PSNR与视觉效果
- **建议取值**: 2-8范围内

## 项目结构

```
HatSeeSR/
├── config/
│   └── config.py              # 配置文件
├── examples/                  # 测试输出
├── models/
│   ├── HAT                    # HAT模型
│   ├── SeeSR                  # SeeSR模型
│   ├── weights                # 预训练权重
│   ├── hat_model.py           # HAT模型包装器
│   └── seesr_model.py         # SeeSR模型包装器
├── pipeline/
│   └── super_resolution_pipeline.py  # 核心处理流程
├── training/
│   └── lora_trainer.py        # LoRA训练模块
├── utils/
│   ├── image_utils.py         # 图像处理工具
│   └── metrics_utils          # 评估指标
├── demo.py                    # 测试demo
├── download.py                # 下载权重脚本
├── main.py                    # 主程序入口
├── README.md                  # 项目说明
├── run.txt                    # 运行命令行示例
└── requirements.txt           # 依赖包列表
```


## 安装指南

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- 至少8GB显存

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/mikei72/HatSeeSR
cd HatSeeSR

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 下载预训练模型

通过网盘分享的文件：数据集
链接: https://pan.baidu.com/s/1_uwMaxgZ3QYKXvbHp8D2og?pwd=3ryp 提取码: 3ryp 
--来自百度网盘超级会员v3的分享

下载其中的HAT、RAM、DAPE、seesr和sd-turbo权重
放在 models/weights 目录下


## 使用方法

### 1. 单张图像处理

```bash
python main.py --input <输入图像> --output <输出图像>
```

### 2. 批量处理

```bash
python --input_dir <输入目录> --output_dir <输出目录> --gt_dir <真实图像目录>
```

### 3. 评估结果

```bash
python main.py --evaluate --gt_dir <真实图像目录> --pred_dir <预测图像目录>
```


## 参数说明

### 基本参数
- `--device`: 计算设备 (默认: cuda)

### 模型路径（config）
- HAT_MODEL_PATH: HAT模型路径
- RAM_FT_PATH: 基于RAM训练的DAPE模型路径
- SEESR_MODEL_PATH: SeeSR模型路径
- PRETRAINED_MODEL_PATH: 基础Stable Diffusion模型权重路径


## 性能优化

### 1. 内存优化
- 使用`--device cpu`在CPU上运行（较慢但内存需求低）
- 调整batch_size减少内存使用

### 2. 速度优化
- 使用xformers加速注意力计算
- 启用混合精度训练
- 使用更小的图像尺寸进行测试

### 3. 质量优化
- 调整strength参数平衡保真度和生成质量
- 使用训练好的LoRA提升特定场景效果
- 优化RAM生成的提示词质量


## 评估指标

项目支持以下评估指标：
- **PSNR**: 峰值信噪比，衡量图像保真度
- **SSIM**: 结构相似性指数，衡量视觉质量


## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用CPU模式
   - 降低图像分辨率

2. **模型加载失败**
   - 检查模型文件路径
   - 确保模型文件完整
   - 检查依赖包版本

3. **处理速度慢**
   - 使用GPU加速
   - 启用xformers
   - 减少推理步数


## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- [HAT](https://github.com/XPixelGroup/HAT) - 超分辨率模型
- [RAM](https://github.com/xdecoder/RAM) - 图像识别模型
- [SeeSR](https://github.com/cswry/SeeSR) - SeeSR模型
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - 扩散模型
- [Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{hatseesr2025,
  title={基于扩散模型的文本引导图像超分辨率},
  author={mikei72},
  year={2025},
  url={https://github.com/mikei72/HatSeeSR}
}
``` 