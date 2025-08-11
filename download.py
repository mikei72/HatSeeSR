from huggingface_hub import snapshot_download
import os

repo_id = "stabilityai/sd-turbo"
exclude_files = ["sd_turbo.safetensors"]

local_dir = "D:/TJU/3.2/IntelliAnalyze/SeeSR-main/preset/models"

# 下载除了指定文件以外的所有文件
snapshot_download(
    repo_id,
    local_dir=local_dir,
    ignore_patterns=exclude_files,
    resume_download=True
)