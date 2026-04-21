#!/usr/bin/env python3
"""
构建 Large 模型 (AttnRes-L) - 结构分析
======================================

根据论文配置:
- 50B 参数
- 64 层
- 5120 隐藏维度
- 40 注意力头
- 16 个 AttnRes 块
"""

import sys

sys.path.insert(0, "src")

from models.configs import get_config, get_model_size_params


def analyze_large_model():
    """分析 Large 模型结构"""

    print("=" * 70)
    print("Adaptive Deep Networks - Large Model (AttnRes-L) 结构分析")
    print("=" * 70)

    # 获取配置
    config = get_config("large")

    print("\n📋 模型配置:")
    print(f"  目标参数: 50B")
    print(f"  层数 (num_layers): {config.num_layers}")
    print(f"  隐藏维度 (hidden_dim): {config.hidden_dim}")
    print(f"  注意力头数 (num_heads): {config.num_heads}")
    print(f"  头维度 (head_dim): {config.hidden_dim // config.num_heads}")
    print(f"  MLP Ratio: {config.mlp_ratio}")
    print(f"  MLP 维度: {config.hidden_dim * config.mlp_ratio}")
    print(f"  AttnRes 块数 (num_blocks): {config.num_blocks}")
    print(f"  每层块数: {config.num_layers // config.num_blocks}")
    print(f"  词表大小 (vocab_size): {config.vocab_size}")
    print(f"  qTTT 最大步数: {config.max_qttt_steps}")
    print(f"  qTTT span 长度: {config.qttt_span_length}")

    # 详细参数计算
    print("\n" + "=" * 70)
    print("📊 参数计算详情")
    print("=" * 70)

    V = config.vocab_size  # 32000
    D = config.hidden_dim  # 5120
    H = config.num_heads  # 40
    L = config.num_layers  # 64
    N = config.num_blocks  # 16
    r = config.mlp_ratio  # 4

    # 1. Embedding
    embedding_params = V * D
    print(f"\n1. Embedding 层:")
    print(f"   Token Embedding: {V} × {D} = {embedding_params / 1e9:.2f}B")

    # 2. 每层参数
    print(f"\n2. 每层参数 (共 {L} 层):")

    # Attention: Q, K, V, O projections
    # Each: D × D
    attn_params = 4 * D * D
    print(f"   Attention:")
    print(f"     Q_proj: {D} × {D} = {D * D / 1e6:.1f}M")
    print(f"     K_proj: {D} × {D} = {D * D / 1e6:.1f}M")
    print(f"     V_proj: {D} × {D} = {D * D / 1e6:.1f}M")
    print(f"     O_proj: {D} × {D} = {D * D / 1e6:.1f}M")
    print(f"     Attention 小计: {attn_params / 1e6:.1f}M")

    # MLP: gate, up, down
    mlp_dim = D * r
    mlp_params = 3 * D * mlp_dim
    print(f"\n   MLP (SwiGLU):")
    print(f"     Gate_proj: {D} × {mlp_dim} = {D * mlp_dim / 1e6:.1f}M")
    print(f"     Up_proj:   {D} × {mlp_dim} = {D * mlp_dim / 1e6:.1f}M")
    print(f"     Down_proj: {mlp_dim} × {D} = {mlp_dim * D / 1e6:.1f}M")
    print(f"     MLP 小计: {mlp_params / 1e6:.1f}M")

    # AttnRes: pseudo-queries for attention and mlp
    attnres_params = 2 * D  # Two pseudo-query vectors per layer
    print(f"\n   AttnRes:")
    print(f"     Pseudo-query (attn): {D} = {D / 1e6:.2f}M")
    print(f"     Pseudo-query (mlp):  {D} = {D / 1e6:.2f}M")
    print(f"     AttnRes 小计: {attnres_params / 1e6:.2f}M")

    # 每层总计
    layer_params = attn_params + mlp_params + attnres_params
    print(f"\n   每层总计: {layer_params / 1e6:.2f}M")
    print(f"   {L} 层总计: {layer_params * L / 1e9:.2f}B")

    # 3. Output
    print(f"\n3. 输出层:")
    print(f"   LM Head: 与 Embedding 共享权重")

    # 4. 总参数
    print(f"\n" + "=" * 70)
    print("📈 总参数统计")
    print("=" * 70)

    # 注意：LM Head 与 Embedding 共享，所以只算一次
    total_params = embedding_params + layer_params * L

    print(f"\n   Embedding:           {embedding_params / 1e9:>8.2f}B")
    print(f"   Transformer Layers:  {layer_params * L / 1e9:>8.2f}B")
    print(f"   - Attention:        {(attn_params * L) / 1e9:>8.2f}B")
    print(f"   - MLP:              {(mlp_params * L) / 1e9:>8.2f}B")
    print(f"   - AttnRes:          {(attnres_params * L) / 1e9:>8.2f}B")
    print(f"   " + "-" * 40)
    print(f"   总计:                {total_params / 1e9:>8.2f}B")
    print(f"\n   目标参数: 50B")
    print(f"   计算参数: {total_params / 1e9:.2f}B")
    print(f"   差异: {(total_params / 1e9 - 50):.2f}B ({(total_params / 1e9 / 50 - 1) * 100:.1f}%)")

    # 5. 内存需求
    print(f"\n" + "=" * 70)
    print("💾 内存需求估算")
    print("=" * 70)

    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2
    bf16_bytes = total_params * 2
    int8_bytes = total_params * 1
    int4_bytes = total_params * 0.5

    print(f"\n   推理内存需求:")
    print(f"   FP32:  {fp32_bytes / 1e9:>6.1f} GB")
    print(f"   FP16:  {fp16_bytes / 1e9:>6.1f} GB")
    print(f"   BF16:  {bf16_bytes / 1e9:>6.1f} GB")
    print(f"   INT8:  {int8_bytes / 1e9:>6.1f} GB")
    print(f"   INT4:  {int4_bytes / 1e9:>6.1f} GB")

    # 训练内存 (AdamW: param + grad + momentum + variance = 4x)
    print(f"\n   训练内存需求 (AdamW + FP32):")
    train_memory = fp32_bytes * 4  # 简化估算
    print(f"   约 {train_memory / 1e9:.1f} GB")

    # 6. FLOPs 估算
    print(f"\n" + "=" * 70)
    print("🧮 FLOPs 估算 (per token)")
    print("=" * 70)

    # 每层的 FLOPs
    # Attention: 2 * seq_len * D^2 (approx for one token)
    attn_flops = 2 * D * D * 2  # Q, K, V, O projections
    mlp_flops = 2 * D * mlp_dim * 3  # gate, up, down
    layer_flops = attn_flops + mlp_flops

    print(f"\n   每层 FLOPs:")
    print(f"   Attention: {attn_flops / 1e9:.2f} GFLOPs")
    print(f"   MLP:       {mlp_flops / 1e9:.2f} GFLOPs")
    print(f"   总计:      {layer_flops / 1e9:.2f} GFLOPs")
    print(f"\n   {L} 层总计: {layer_flops * L / 1e3:.2f} TFLOPs/token")

    return config, total_params


def compare_models():
    """对比 Small, Medium, Large"""
    print("\n" + "=" * 70)
    print("模型大小对比")
    print("=" * 70)

    sizes = ["small", "medium", "large"]

    print(
        f"\n{'Model':<12} {'Params':<12} {'Layers':<10} {'Hidden':<10} {'Heads':<10} {'Blocks':<10}"
    )
    print("-" * 70)

    for size in sizes:
        config = get_config(size)
        V = config.vocab_size
        D = config.hidden_dim
        L = config.num_layers
        r = config.mlp_ratio

        # 计算参数
        embedding = V * D
        layer_params = 4 * D * D + 3 * D * D * r + 2 * D
        total = embedding + layer_params * L

        param_str = f"{total / 1e9:.1f}B" if total > 1e9 else f"{total / 1e6:.1f}M"

        print(
            f"{size.capitalize():<12} {param_str:<12} {L:<10} "
            f"{D:<10} {config.num_heads:<10} {config.num_blocks:<10}"
        )


def generate_config_file():
    """生成 Large 模型配置文件"""
    print("\n" + "=" * 70)
    print("📄 生成配置文件")
    print("=" * 70)

    config_content = """{
  "model_type": "adaptive_transformer",
  "model_size": "large",
  "architectures": ["AdaptiveTransformer"],
  
  "vocab_size": 32000,
  "num_layers": 64,
  "hidden_dim": 5120,
  "num_heads": 40,
  "mlp_ratio": 4,
  "max_seq_len": 32768,
  
  "attnres": {
    "num_blocks": 16,
    "pseudo_query_init": "zero"
  },
  
  "qttt": {
    "max_qttt_steps": 32,
    "qttt_span_length": 256,
    "qttt_learning_rate": 0.002
  },
  
  "gating": {
    "gating_target_rate": 0.25
  }
}
"""

    with open("results/large_model_config.json", "w") as f:
        f.write(config_content)

    print("\n配置文件已保存: results/large_model_config.json")


def main():
    """主函数"""
    # 分析 Large 模型
    config, total_params = analyze_large_model()

    # 对比模型
    compare_models()

    # 生成配置文件
    generate_config_file()

    print("\n" + "=" * 70)
    print("✅ Large 模型 (AttnRes-L) 结构分析完成")
    print("=" * 70)
    print(f"\n模型参数: {total_params / 1e9:.2f}B")
    print(f"配置: 64 层, 5120 隐藏维度, 40 注意力头, 16 AttnRes 块")
    print(f"\n如需实际创建模型，建议:")
    print(f"  - 推理: 使用 BF16/FP16 (约 100GB 显存)")
    print(f"  - 训练: 使用分布式训练 + ZeRO-3")
    print(f"  - 量化: INT8/INT4 可减少 50-75% 内存")


if __name__ == "__main__":
    main()
