'''
 * SeeSR: Towards Semantics-Aware Real-World Image Super-Resolution
 * Modified from diffusers by Rongyuan Wu
 * 24/12/2023
'''
import os
import sys

sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from models.SeeSR.pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from models.SeeSR.utils.misc import load_dreambooth_lora
from models.SeeSR.utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from models.SeeSR.ram.models.ram_lora import ram
from models.SeeSR.ram import inference_ram as inference
from models.SeeSR.ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool = False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)

    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")

    if (
            is_model_key_starts_with_module and
            (not is_state_dict_key_starts_with_module)
    ):
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if (
            (not is_model_key_starts_with_module) and
            is_state_dict_key_starts_with_module
    ):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)


def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    from models.SeeSR.models.controlnet import ControlNetModel
    from models.SeeSR.models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer and models.

    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained_orig(args.pretrained_model_path, args.seesr_model_path,
                                                     subfolder="unet", use_image_cross_attention=True)
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor,
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )

    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size,
                                        decoder_tile_size=args.vae_decoder_tiled_size)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline


def load_tag_model(args, device='cuda'):
    model = ram(pretrained='models/weights/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)

    return model


def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""

    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)

    validation_prompt = f"{res[0]}, {args.prompt},"

    return validation_prompt, ram_encoder_hidden_states


def initialize_models_and_pipeline(
        seesr_model_path: str,
        ram_ft_path: str,
        pretrained_model_path: str,
        mixed_precision: str = 'fp16',
        seed: int = 42
):
    print("正在初始化模型和 pipeline...")

    args = argparse.Namespace()

    args.seesr_model_path = seesr_model_path
    args.ram_ft_path = ram_ft_path
    args.pretrained_model_path = pretrained_model_path
    args.mixed_precision = mixed_precision
    args.seed = seed

    args.prompt = ""
    args.added_prompt = "clean, high-resolution, 8k"
    args.negative_prompt = "dotted, noise, blur, lowres, smooth"
    args.guidance_scale = 1.0
    args.conditioning_scale = 1.0
    args.blending_alpha = 1.0
    args.num_inference_steps = 2
    args.process_size = 512
    args.vae_decoder_tiled_size = 224
    args.vae_encoder_tiled_size = 1024
    args.latent_tiled_size = 96
    args.latent_tiled_overlap = 32
    args.upscale = 4
    args.sample_times = 1
    args.align_method = "adain"
    args.start_steps = 999
    args.start_point = "lr"
    args.save_prompts = False

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention=True)
    model = load_tag_model(args, accelerator.device)

    generator = None
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

    print("初始化完成")
    return pipeline, model, generator, accelerator, args


def enhance_single_image(
        image_array: np.ndarray,
        pipeline,
        model,
        generator,
        accelerator,
        args: argparse.Namespace
):
    if not accelerator.is_main_process:
        return

    try:
        validation_image = Image.fromarray(image_array).convert("RGB")
    except Exception as e:
        print(f"错误: 无法从 NumPy 数组创建图像。请确保数组格式正确 (H, W, C) 且数据类型为 uint8。错误信息: {e}")
        return

    validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
    validation_prompt += args.added_prompt

    ori_width, ori_height = validation_image.size
    rscale = args.upscale

    resize_flag = False
    if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        scale = (args.process_size // rscale) / min(ori_width, ori_height)
        validation_image = validation_image.resize((int(scale * ori_width), int(scale * ori_height)))
        resize_flag = True

    validation_image = validation_image.resize((validation_image.size[0] * rscale, validation_image.size[1] * rscale))
    validation_image = validation_image.resize((validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8))
    width, height = validation_image.size
    resize_flag = True

    for sample_idx in range(args.sample_times):
        with torch.autocast("cuda"):
            image = pipeline(
                validation_prompt, validation_image,
                num_inference_steps=args.num_inference_steps,
                generator=generator, height=height, width=width,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                conditioning_scale=args.conditioning_scale,
                start_point=args.start_point,
                ram_encoder_hidden_states=ram_encoder_hidden_states,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
                args=args,  # 将完整的 args 对象传递给 pipeline
            ).images[0]

    if args.align_method == 'wavelet':
        image = wavelet_color_fix(image, validation_image)
    elif args.align_method == 'adain':
        image = adain_color_fix(image, validation_image)

    if resize_flag:
        image = image.resize((ori_width * rscale, ori_height * rscale))

    return validation_prompt, image
