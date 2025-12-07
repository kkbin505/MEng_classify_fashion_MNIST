
import os

# 推荐使用 Coqui TTS (Text-to-Speech) 库，它集成了许多先进的 TTS 和 VCS 模型。
# 注意：xtts-v2 模型需要大量算力和 VRAM，请确保你的 GPU 性能足够。

print("--- 步骤 1: 安装 Coqui TTS (推荐) ---")
# 运行以下命令安装 Coqui TTS (如果尚未安装)
# !pip install -U coqui-tts
# 或者安装更通用的 VITS/RVC 库

print("--- 步骤 2: 导入必要库 ---")
import torch
from TTS.api import TTS
import torchaudio
import numpy as np
import time

# 检查 GPU 可用性
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA 可用。正在使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA 不可用。将使用 CPU 进行推理 (速度会非常慢)。")

print("环境检查完毕。")