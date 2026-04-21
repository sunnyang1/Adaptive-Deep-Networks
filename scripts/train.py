#!/usr/bin/env python3
"""ADN统一训练入口"""
import argparse
import sys
from pathlib import Path

# 添加项目根到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adn.core.config import ModelConfig, ADNConfig
from adn.models.adaptive_transformer import AdaptiveTransformer


def main():
    parser = argparse.ArgumentParser(description="ADN Training")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--output-dir", type=str, default="results/default")
    parser.add_argument("--use-qttt", action="store_true")
    parser.add_argument("--use-engram", action="store_true")
    parser.add_argument("--use-attnres", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # 模型尺寸配置
    size_configs = {
        "small": {"hidden_dim": 384, "num_heads": 6, "num_layers": 8, "num_blocks": 2},
        "medium": {"hidden_dim": 768, "num_heads": 12, "num_layers": 12, "num_blocks": 4},
        "large": {"hidden_dim": 1024, "num_heads": 16, "num_layers": 16, "num_blocks": 4},
    }
    
    config = ModelConfig(**size_configs[args.model_size])
    config.use_qttt = args.use_qttt
    config.use_engram = args.use_engram
    config.use_attnres = args.use_attnres
    
    print(f"Training ADN model: {args.model_size}")
    print(f"Config: {config}")
    
    # 创建模型
    model = AdaptiveTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # TODO: 添加实际训练循环（从 train_model.py 中提取）
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
