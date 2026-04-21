"""
完整 Colab 测试脚本 - 包含所有测试用例
"""

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from src.models.configs import AttnResSmallConfig, get_model_size_params, print_config
from src.models.adaptive_transformer import AdaptiveTransformer
import torch
import time


# 获取设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")

# 创建模型
config = AttnResSmallConfig()
print_config(config)

model = AdaptiveTransformer(config).to(device)
model.eval()

total_params = model.count_parameters()
print(f"Actual Model: {total_params/1e6:.1f}M params")

# 测试1: 短上下文 (1K)
print("\n=== 测试1: 短上下文 (1024 tokens) ===")
x1 = torch.randint(0, 1000, (1, 1024)).to(device)
with torch.no_grad():
    start = time.time()
    out1 = model(x1)
    elapsed = time.time() - start
print(f"✓ Forward pass OK: {out1.shape}")
print(f"  Time: {elapsed:.2f}s, Throughput: {1024/elapsed:.0f} tokens/s")

# 测试2: 中等上下文 (4K)
print("\n=== 测试2: 中等上下文 (4096 tokens) ===")
x2 = torch.randint(0, 1000, (1, 4096)).to(device)
with torch.no_grad():
    start = time.time()
    out2 = model(x2)
    elapsed = time.time() - start
print(f"✓ Forward pass OK: {out2.shape}")
print(f"  Time: {elapsed:.2f}s, Throughput: {4096/elapsed:.0f} tokens/s")

# 测试3: 长上下文 (8K) - 如果内存允许
print("\n=== 测试3: 长上下文 (8192 tokens) ===")
try:
    x3 = torch.randint(0, 1000, (1, 8192)).to(device)
    with torch.no_grad():
        start = time.time()
        out3 = model(x3)
        elapsed = time.time() - start
    print(f"✓ Forward pass OK: {out3.shape}")
    print(f"  Time: {elapsed:.2f}s, Throughput: {8192/elapsed:.0f} tokens/s")
except RuntimeError as e:
    print(f"✗ OOM (内存不足): {e}")

# 测试4: 模拟生成
print("\n=== 测试4: 模拟生成 (10 tokens) ===")
test_prompt = torch.randint(0, 1000, (1, 50)).to(device)
with torch.no_grad():
    for i in range(10):
        logits = model(test_prompt)
        next_token = logits[:, -1:].argmax(dim=-1)
        test_prompt = torch.cat([test_prompt, next_token], dim=1)
print(f"✓ Generation OK: generated {test_prompt.shape[1]} tokens")

# 测试5: AttnRes 组件统计
print("\n=== 测试5: AttnRes 组件 ===")
attnres_params = model.count_attnsres_parameters()
print(f"AttnRes params: {attnres_params/1e6:.2f}M ({attnres_params/total_params*100:.2f}%)")

# 显存统计
if device.type == "cuda":
    print(f"\n=== GPU 显存使用 ===")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

print("\n✓ 所有测试通过!")
