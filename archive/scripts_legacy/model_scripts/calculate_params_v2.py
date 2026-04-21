#!/usr/bin/env python3
"""
尝试匹配论文标注的参数量
通过调整 vocab_size 和 mlp_ratio 来匹配论文的 1.5B/7B/50B
"""

import sys

sys.path.insert(0, "src")


def calc_params(V, L, D, R):
    """
    计算参数量
    V: vocab_size
    L: num_layers
    D: hidden_dim
    R: mlp_ratio
    """
    # Embedding
    embedding = V * D

    # Per layer
    attn_per_layer = 4 * D * D  # Q, K, V, O
    mlp_per_layer = 3 * D * D * R  # SwiGLU
    norm_per_layer = 2 * D  # 2 RMSNorms
    attnres_per_layer = 4 * D  # 2 pseudo_query + 2 norm weights

    layer_total = attn_per_layer + mlp_per_layer + norm_per_layer + attnres_per_layer
    all_layers = L * layer_total

    # Final norm
    final_norm = D

    total = embedding + all_layers + final_norm
    return total


def find_matching_config(target_params, L, D):
    """找到匹配目标参数量的配置"""
    print(f"\n目标: {target_params/1e9:.1f}B, Layers={L}, Hidden={D}")
    print("-" * 50)

    best_diff = float("inf")
    best_config = None

    # 尝试不同的 vocab_size 和 mlp_ratio
    for V in [32000, 50000, 64000, 100000]:
        for R in [2.5, 3, 3.5, 4]:
            params = calc_params(V, L, D, R)
            diff = abs(params - target_params)

            if diff < best_diff:
                best_diff = diff
                best_config = (V, R, params)

    V, R, params = best_config
    print(f"最接近的配置:")
    print(f"  vocab_size: {V}")
    print(f"  mlp_ratio: {R}")
    print(f"  计算参数量: {params/1e9:.2f}B")
    print(f"  与目标差异: {best_diff/1e6:.1f}M ({best_diff/target_params*100:.1f}%)")

    return best_config


def main():
    print("=" * 60)
    print("尝试匹配论文标注的参数量")
    print("=" * 60)

    # 论文标注 vs 当前代码计算 (AttnRes优化后配置)
    configs = [
        ("Small", 3.3e9, 48, 2048),  # 48L/2048H/16Hd = 3.3B
        ("Medium", 6.6e9, 56, 2688),  # 56L/2688H/16Hd = 6.6B
        ("Large", 27.5e9, 96, 4224),  # 96L/4224H/32Hd = 27.5B
    ]

    print("\n【默认配置计算结果 (vocab=32000, mlp_ratio=4)】")
    print("-" * 60)
    for name, target, L, D in configs:
        default_params = calc_params(32000, L, D, 4)
        print(
            f"{name}: 计算={default_params/1e9:.2f}B, 论文={target/1e9:.1f}B, 差异={(default_params-target)/1e9:+.2f}B"
        )

    # 尝试找到匹配的配置
    print("\n" + "=" * 60)
    print("【尝试匹配论文配置】")
    print("=" * 60)

    for name, target, L, D in configs:
        find_matching_config(target, L, D)

    # 另一种可能：论文使用了不同的 hidden_dim
    print("\n" + "=" * 60)
    print("【假设 vocab=32000, mlp_ratio=4，反推 hidden_dim】")
    print("=" * 60)

    for name, target, L, _ in configs:
        print(f"\n{name} (目标 {target/1e9:.1f}B, {L}层):")
        for D in [1536, 1792, 2048, 2304, 2560, 4096, 4608, 5120, 5632, 6144]:
            params = calc_params(32000, L, D, 4)
            diff = params - target
            marker = " <--" if abs(diff) < target * 0.05 else ""
            print(f"  hidden_dim={D:4d}: {params/1e9:.2f}B ({diff/1e9:+.2f}B){marker}")


if __name__ == "__main__":
    main()
