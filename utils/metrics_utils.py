import argparse
import os
import cv2
import numpy as np
from PIL import Image
from basicsr.metrics import calculate_psnr, calculate_ssim

# ==============================================================================
# 核心计算函数 (无需修改)
# ==============================================================================

def center_crop(img, target_h, target_w):
    h, w = img.shape[:2]
    y_start = (h - target_h) // 2
    x_start = (w - target_w) // 2
    return img[y_start:y_start+target_h, x_start:x_start+target_w, :]

def calculate_metrics(sr_path: str, gt_path: str, crop_border: int, test_y_channel: bool):
    """
    计算 SR 图像和 GT 图像之间的 PSNR 和 SSIM 指标。
    脚本内置了对旋转和尺寸不匹配问题的鲁棒自动校正。
    """
    ROTATION_TOLERANCE = 5

    try:
        sr_img = cv2.cvtColor(cv2.imread(sr_path), cv2.COLOR_BGR2RGB)
        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return

    h_sr, w_sr, _ = sr_img.shape
    h_gt, w_gt, _ = gt_img.shape

    if h_sr != h_gt or w_sr != w_gt:
        print("提示: SR 和 GT 尺寸不匹配，开始自动校正...")
        is_rotated = (abs(h_sr - w_gt) <= ROTATION_TOLERANCE and
                      abs(w_sr - h_gt) <= ROTATION_TOLERANCE)

        if is_rotated:
            print(f"  - 检测到旋转 (SR: {h_sr}x{w_sr}, GT: {h_gt}x{w_gt})，正在校正...")
            sr_img = np.rot90(sr_img, k=3)
            h_sr, w_sr, _ = sr_img.shape

        if h_sr != h_gt or w_sr != w_gt:
            print(f"  - 裁剪到共同最小尺寸...")
            target_h = min(h_sr, h_gt)
            target_w = min(w_sr, w_gt)
            sr_img = center_crop(sr_img, target_h, target_w)
            gt_img = center_crop(gt_img, target_h, target_w)
            print(f"  - 校正后尺寸: {target_h}x{target_w}")

    assert sr_img.shape == gt_img.shape

    psnr = calculate_psnr(gt_img, sr_img, crop_border=crop_border, test_y_channel=test_y_channel )
    ssim = calculate_ssim(gt_img, sr_img, crop_border=crop_border, test_y_channel=test_y_channel )

    return {'psnr': psnr, 'ssim': ssim}
