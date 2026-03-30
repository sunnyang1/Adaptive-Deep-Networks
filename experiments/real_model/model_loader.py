#!/usr/bin/env python3
"""
Model Loader for Real Model Validation

支持加载 Adaptive Deep Networks 预训练模型。
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.adaptive_transformer import AdaptiveTransformer
from src.models.configs import ModelConfig


class ModelLoader:
    """
    加载和管理预训练模型。
    
    Example:
        >>> loader = ModelLoader()
        >>> model, config = loader.load_from_checkpoint("checkpoints/adb_medium.pt")
        >>> print(f"Loaded model with {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
    
    def load_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        strict: bool = True
    ) -> tuple:
        """
        从检查点加载模型。
        
        Args:
            checkpoint_path: 模型检查点路径 (.pt 或 .pth)
            config_path: 配置文件路径 (.yaml 或 .json)，可选
            strict: 是否严格匹配状态字典
        
        Returns:
            (model, config) 元组
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 加载配置
        if config_path is None:
            # 尝试自动查找配置
            config_path = self._find_config(checkpoint_path)
        
        config = self._load_config(config_path)
        
        # 创建模型
        print(f"Creating model with config: {config}")
        model = self._create_model(config)
        
        # 加载权重
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 处理不同格式的检查点
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        model.to(self.device)
        model.eval()
        
        self.model = model
        self.config = config
        
        # 打印模型信息
        self._print_model_info(model)
        
        return model, config
    
    def load_from_pretrained(
        self,
        model_size: str = 'medium',
        use_turboquant: bool = True,
        use_polar_qttt: bool = True
    ) -> tuple:
        """
        加载预定义的预训练模型配置。
        
        Args:
            model_size: 'small' (2.2B), 'medium' (8.7B), or 'large' (27B)
            use_turboquant: 是否启用 TurboQuant
            use_polar_qttt: 是否启用 Polar qTTT
        
        Returns:
            (model, config) 元组
        """
        configs = {
            'small': {
                'vocab_size': 32000,
                'hidden_dim': 2048,
                'num_layers': 32,
                'num_heads': 32,
                'mlp_ratio': 4,
                'num_blocks': 8,
                'max_qttt_steps': 16,
            },
            'medium': {
                'vocab_size': 32000,
                'hidden_dim': 4096,
                'num_layers': 32,
                'num_heads': 32,
                'mlp_ratio': 4,
                'num_blocks': 8,
                'max_qttt_steps': 32,
            },
            'large': {
                'vocab_size': 32000,
                'hidden_dim': 5120,
                'num_layers': 64,
                'num_heads': 40,
                'mlp_ratio': 4,
                'num_blocks': 16,
                'max_qttt_steps': 32,
            }
        }
        
        if model_size not in configs:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
        
        config_dict = configs[model_size]
        # Note: use_turboquant and use_polar_qttt are runtime options,
        # not model config parameters. They should be handled at model initialization.
        
        config = ModelConfig(**config_dict)
        model = self._create_model(config)
        model.to(self.device)
        model.eval()
        
        self.model = model
        self.config = config
        
        print(f"\nInitialized {model_size} model ({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)")
        print(f"  - TurboQuant: {use_turboquant}")
        print(f"  - Polar qTTT: {use_polar_qttt}")
        
        return model, config
    
    def _find_config(self, checkpoint_path: Path) -> Optional[Path]:
        """自动查找配置文件"""
        # 尝试同目录下的 .yaml 或 .json 文件
        for ext in ['.yaml', '.yml', '.json']:
            config_path = checkpoint_path.with_suffix(ext)
            if config_path.exists():
                return config_path
        
        # 尝试 configs/ 目录
        config_dir = Path(__file__).parent.parent.parent / 'configs'
        if config_dir.exists():
            for ext in ['.yaml', '.yml', '.json']:
                config_path = config_dir / f"{checkpoint_path.stem}{ext}"
                if config_path.exists():
                    return config_path
        
        return None
    
    def _load_config(self, config_path: Optional[Path]) -> ModelConfig:
        """加载配置文件"""
        if config_path is None:
            print("Warning: No config file found, using default config")
            return ModelConfig()
        
        config_path = Path(config_path)
        print(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return ModelConfig(**config_dict)
    
    def _create_model(self, config: ModelConfig) -> nn.Module:
        """创建模型实例"""
        model = AdaptiveTransformer(config)
        return model
    
    def _print_model_info(self, model: nn.Module):
        """打印模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel loaded successfully!")
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")


def load_adb_model(
    checkpoint_path: Optional[str] = None,
    model_size: str = 'medium',
    device: Optional[str] = None,
    **kwargs
) -> tuple:
    """
    便捷函数：加载 Adaptive Deep Networks 模型。
    
    Args:
        checkpoint_path: 检查点路径，为None则初始化随机权重
        model_size: 模型大小 ('small', 'medium', 'large')
        device: 设备
        **kwargs: 传递给 ModelLoader 的参数
    
    Returns:
        (model, config) 元组
    
    Example:
        >>> model, config = load_adb_model("checkpoints/adb_medium.pt")
        >>> # Or initialize without checkpoint for testing
        >>> model, config = load_adb_model(model_size="small")
    """
    loader = ModelLoader(device=device)
    
    if checkpoint_path is not None:
        return loader.load_from_checkpoint(checkpoint_path, **kwargs)
    else:
        return loader.load_from_pretrained(model_size=model_size, **kwargs)


if __name__ == '__main__':
    # 测试模型加载
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--config', type=str, help='Config path')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    args = parser.parse_args()
    
    loader = ModelLoader()
    
    if args.checkpoint:
        model, config = loader.load_from_checkpoint(args.checkpoint, args.config)
    else:
        model, config = loader.load_from_pretrained(args.size)
    
    # 测试前向传播
    print("\nTesting forward pass...")
    batch_size = 1
    seq_len = 512
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.logits.shape if hasattr(output, 'logits') else output.shape}")
    print("Model test passed!")
