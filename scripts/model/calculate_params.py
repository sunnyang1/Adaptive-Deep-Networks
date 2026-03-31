#!/usr/bin/env python3
"""
精确计算 Adaptive Deep Networks 各模型的参数量
基于实际的模型结构实现
"""

import sys
sys.path.insert(0, 'src')

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 32000
    num_layers: int = 32
    hidden_dim: int = 4096
    num_heads: int = 32
    mlp_ratio: int = 4
    num_blocks: int = 8
    tie_weights: bool = True  # Embedding 和 LM Head 共享权重


def calculate_params(config: ModelConfig, verbose: bool = True):
    """
    计算模型参数量
    
    模型结构:
    1. Embedding: vocab_size * hidden_dim
    2. LM Head: hidden_dim * vocab_size (与 Embedding 共享，不计入)
    3. 每层包含:
       - Attention: Q, K, V, O 投影 (各 hidden_dim * hidden_dim)
       - MLP: SwiGLU (gate_proj, up_proj, down_proj)
       - LayerNorm: 2 * hidden_dim (RMSNorm，只有 weight)
       - BlockAttnRes: 2 * hidden_dim (pseudo_query) + 2 * hidden_dim (norm weights)
    """
    
    D = config.hidden_dim
    V = config.vocab_size
    L = config.num_layers
    R = config.mlp_ratio
    
    # 1. Embedding (与 LM Head 共享权重)
    embedding_params = V * D
    lm_head_params = 0 if config.tie_weights else V * D
    
    # 2. 每层参数
    # Attention: 4个投影层
    attn_params_per_layer = 4 * D * D  # Q, K, V, O
    
    # MLP: SwiGLU
    # gate_proj: D -> D*R
    # up_proj: D -> D*R  
    # down_proj: D*R -> D
    mlp_params_per_layer = 3 * D * D * R
    
    # Layer Norms: 2个 RMSNorm，每个只有 weight 参数
    layernorm_params_per_layer = 2 * D
    
    # BlockAttnRes: 
    # - pseudo_query_attn: D
    # - pseudo_query_mlp: D
    # - norm_attn: D (RMSNorm weight)
    # - norm_mlp: D (RMSNorm weight)
    attnres_params_per_layer = 4 * D
    
    total_per_layer = (attn_params_per_layer + 
                      mlp_params_per_layer + 
                      layernorm_params_per_layer + 
                      attnres_params_per_layer)
    
    all_layers_params = L * total_per_layer
    
    # 3. 总参数量
    total_params = embedding_params + lm_head_params + all_layers_params
    
    # 4. 最终 RMSNorm
    final_norm_params = D  # 最后的 RMSNorm
    total_params += final_norm_params
    
    if verbose:
        print("=" * 60)
        print(f"模型配置: {config.num_layers}层, hidden_dim={D}, vocab_size={V}")
        print("=" * 60)
        print(f"\n1. Embedding: {embedding_params:,} ({embedding_params/1e9:.3f}B)")
        if not config.tie_weights:
            print(f"   LM Head: {lm_head_params:,} ({lm_head_params/1e9:.3f}B)")
        
        print(f"\n2. 每层参数 ({L}层):")
        print(f"   - Attention (4线性层): {attn_params_per_layer:,}")
        print(f"   - MLP (SwiGLU): {mlp_params_per_layer:,}")
        print(f"   - LayerNorms (2个): {layernorm_params_per_layer:,}")
        print(f"   - BlockAttnRes: {attnres_params_per_layer:,}")
        print(f"   每层小计: {total_per_layer:,}")
        print(f"   所有层总计: {all_layers_params:,} ({all_layers_params/1e9:.3f}B)")
        
        print(f"\n3. 最终 LayerNorm: {final_norm_params:,}")
        
        print(f"\n" + "=" * 60)
        print(f"总参数量: {total_params:,}")
        print(f"         ≈ {total_params/1e9:.2f}B")
        print(f"         ≈ {total_params/1e6:.1f}M")
        print("=" * 60)
        
        # 计算 AttnRes 占比
        attnres_total = L * attnres_params_per_layer
        print(f"\nAttnRes 参数: {attnres_total:,} ({attnres_total/1e6:.2f}M)")
        print(f"AttnRes 占比: {attnres_total/total_params*100:.3f}%")
    
    return total_params


def main():
    print("\n" + "=" * 70)
    print("Adaptive Deep Networks - 参数量精确计算")
    print("=" * 70)
    
    # Small Model (1.1B) - Optimized for AttnRes: 32L/1408H/8Hd
    print("\n" + "=" * 70)
    print("【Small Model】")
    small_config = ModelConfig(
        vocab_size=32000,
        num_layers=32,
        hidden_dim=1408,
        num_heads=8,
        mlp_ratio=4,
        num_blocks=8
    )
    small_params = calculate_params(small_config)
    
    # Medium Model (5.7B) - Optimized for AttnRes: 56L/2496H/16Hd
    print("\n" + "=" * 70)
    print("【Medium Model】")
    medium_config = ModelConfig(
        vocab_size=32000,
        num_layers=56,
        hidden_dim=2496,
        num_heads=16,
        mlp_ratio=4,
        num_blocks=8
    )
    medium_params = calculate_params(medium_config)
    
    # Large Model (23.0B) - Optimized for AttnRes: 88L/4032H/18Hd
    print("\n" + "=" * 70)
    print("【Large Model】")
    large_config = ModelConfig(
        vocab_size=32000,
        num_layers=88,
        hidden_dim=4032,
        num_heads=18,
        mlp_ratio=4,
        num_blocks=11
    )
    large_params = calculate_params(large_config)
    
    # 总结
    print("\n" + "=" * 70)
    print("【总结对比】")
    print("=" * 70)
    print(f"{'Model':<10} {'配置':<30} {'参数量':<12} {'d_model/L_b':<12} {'H/L_b'}")
    print("-" * 70)
    print(f"{'Small':<10} {f'32L, 1408H, 8Hd':<30} {f'{small_params/1e9:.2f}B':<12} {f'{1408/32:.1f}':<12} {f'{8/32:.3f}'}")
    print(f"{'Medium':<10} {f'56L, 2496H, 16Hd':<30} {f'{medium_params/1e9:.2f}B':<12} {f'{2496/56:.1f}':<12} {f'{16/56:.3f}'}")
    print(f"{'Large':<10} {f'88L, 4032H, 18Hd':<30} {f'{large_params/1e9:.2f}B':<12} {f'{4032/88:.1f}':<12} {f'{18/88:.3f}'}")
    print("-" * 70)
    print("Architecture optimized for AttnRes (Paper §5.4.1): d_model/L_b ≈ 45, H/L_b ≈ 0.3")
    print("=" * 70)
    
    # 显存需求估算
    print("\n" + "=" * 70)
    print("【显存需求估算 (BF16/FP16)】")
    print("=" * 70)
    print(f"{'Model':<10} {'模型权重':<12} {'Adam状态':<12} {'激活值*':<12} {'总计':<12}")
    print("-" * 70)
    
    for name, params in [("Small", small_params), ("Medium", medium_params), ("Large", large_params)]:
        model_mem = params * 2 / 1e9  # BF16 = 2 bytes
        adam_mem = params * 8 / 1e9   # Adam: 2 states * 4 bytes
        activation_mem = params * 0.3 * 2 / 1e9  # 估算: batch=2, seq=1024
        total_mem = model_mem + adam_mem + activation_mem
        print(f"{name:<10} {model_mem:.1f}GB{'':<6} {adam_mem:.1f}GB{'':<5} {activation_mem:.1f}GB{'':<5} {total_mem:.1f}GB")
    
    print("\n* 激活值估算基于 batch_size=2, seq_len=1024")
    print("=" * 70)


if __name__ == "__main__":
    main()
