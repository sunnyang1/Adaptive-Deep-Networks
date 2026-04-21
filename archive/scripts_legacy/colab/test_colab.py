"""
Colab 测试脚本 - 自动检测 CUDA/CPU
"""

import sys
import os

# 将 'src' 目录添加到 Python 路径
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.models.configs import AttnResSmallConfig
from src.models.adaptive_transformer import AdaptiveTransformer, get_device
import torch

# 自动检测设备
device = get_device()
print(f"Using device: {device}")

# 创建模型
config = AttnResSmallConfig()
model = AdaptiveTransformer(config).to(device)
print(f"Model: {model.count_parameters()/1e6:.1f}M params")

# 测试短上下文
x = torch.randint(0, 1000, (1, 1024)).to(device)
with torch.no_grad():
    out = model(x)
print(f"✓ Forward pass OK: {out.shape}")

# 测试生成（模拟）
print("\n--- 测试生成 ---")
test_prompt = torch.randint(0, 1000, (1, 10)).to(device)
with torch.no_grad():
    for i in range(5):
        logits = model(test_prompt)
        next_token = logits[:, -1:].argmax(dim=-1)
        test_prompt = torch.cat([test_prompt, next_token], dim=1)
print(f"✓ Generation OK: generated {test_prompt.shape[1]} tokens")

# 打印模型结构信息
print(f"\n--- 模型信息 ---")
print(f"Layers: {config.num_layers}")
print(f"Hidden dim: {config.hidden_dim}")
print(f"Num heads: {config.num_heads}")
print(f"Num blocks: {config.num_blocks}")
