#!/usr/bin/env python3
"""
本地验证 Small 和 Medium 模型
"""

import sys
import json
from datetime import datetime

sys.path.insert(0, "src")

from models.configs import get_config, get_model_size_params
from benchmarks.flop_analysis import EfficiencyAnalyzer


def validate_model(model_size):
    """验证指定大小的模型"""
    print("\n" + "=" * 70)
    print(f"验证模型: {model_size.upper()}")
    print("=" * 70)

    # 获取配置
    config = get_config(model_size)
    params = get_model_size_params(config)
    param_str = f"{params / 1e9:.1f}B" if params > 1e9 else f"{params / 1e6:.1f}M"

    print(f"\n模型配置:")
    print(f"  参数量: {param_str}")
    print(f"  层数: {config.num_layers}")
    print(f"  隐藏维度: {config.hidden_dim}")
    print(f"  注意力头数: {config.num_heads}")
    print(f"  AttnRes 块数: {config.num_blocks}")
    print(f"  qTTT 最大步数: {config.max_qttt_steps}")
    print(f"  qTTT span 长度: {config.qttt_span_length}")

    # 创建 FLOP 分析器
    analyzer = EfficiencyAnalyzer(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )

    # 测试 1: FLOP 等价性验证
    print(f"\n{'─' * 70}")
    print("测试 1: FLOP 等价性验证 (T_think ≈ 2 * N_qTTT * k)")
    print("─" * 70)

    # 针对不同模型大小使用不同测试参数
    if model_size == "small":
        num_thinking_tokens = 4096
        num_qttt_steps = 16
        qttt_span = 128
        context_len = 32768
    else:  # medium
        num_thinking_tokens = 8192
        num_qttt_steps = 32
        qttt_span = 128
        context_len = 65536

    equivalence_results = analyzer.verify_flop_equivalence(
        batch=1,
        context_len=context_len,
        num_thinking_tokens=num_thinking_tokens,
        num_qttt_steps=num_qttt_steps,
        qttt_span=qttt_span,
    )

    analyzer.print_analysis(equivalence_results)

    # 测试 2: 不同 FLOP 预算下的策略对比
    print(f"\n{'─' * 70}")
    print("测试 2: FLOP 分配策略对比")
    print("─" * 70)

    # 根据模型大小调整预算
    budget_flops = 1e14 if model_size == "small" else 5e14

    strategy_results = analyzer.compare_allocation_strategies(
        budget_flops=budget_flops, context_len=context_len
    )

    print(f"\n预算: {budget_flops:.0e} FLOPs, 上下文长度: {context_len}")
    print("-" * 60)

    for name, strategy in strategy_results["strategies"].items():
        print(f"\n{name}:")
        print(f"  描述: {strategy['description']}")
        if strategy["thinking_tokens"] > 0:
            print(f"  Thinking tokens: {strategy['thinking_tokens']}")
        if strategy["qttt_steps"] > 0:
            print(f"  qTTT steps: {strategy['qttt_steps']}")

    # 测试 3: 不同上下文长度的效率分析
    print(f"\n{'─' * 70}")
    print("测试 3: 不同上下文长度的 FLOP 效率")
    print("─" * 70)

    context_lengths = (
        [1024, 4096, 16384, 32768] if model_size == "small" else [4096, 16384, 32768, 65536]
    )

    print(f"\n{'Context Length':<15} {'Per-Token FLOPs':<20} {'qTTT Step FLOPs':<20} {'Ratio':<10}")
    print("-" * 65)

    for ctx_len in context_lengths:
        per_token = analyzer.compute_thinking_token_flops(1, ctx_len, 1)
        qttt_step = analyzer.compute_qttt_step_flops(1, ctx_len, qttt_span)
        ratio = qttt_step / per_token if per_token > 0 else 0
        print(f"{ctx_len:<15} {per_token:<20.2e} {qttt_step:<20.2e} {ratio:<10.2f}")

    # 汇总结果
    results = {
        "model_size": model_size,
        "config": {
            "params": params,
            "num_layers": config.num_layers,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_blocks": config.num_blocks,
            "max_qttt_steps": config.max_qttt_steps,
            "qttt_span_length": config.qttt_span_length,
        },
        "flop_equivalence": equivalence_results,
        "strategies": strategy_results,
        "timestamp": datetime.now().isoformat(),
    }

    # 保存结果
    output_file = f"./results/validation_{model_size}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ 验证完成！结果已保存到: {output_file}")

    return results


def main():
    print("=" * 70)
    print("Adaptive Deep Networks 本地验证")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # 验证 Small 模型 (1.5B)
    all_results["small"] = validate_model("small")

    # 验证 Medium 模型 (7B)
    all_results["medium"] = validate_model("medium")

    # 最终汇总
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)

    for model_size, results in all_results.items():
        eq = results["flop_equivalence"]["equivalence"]
        config = results["config"]
        param_str = (
            f"{config['params'] / 1e9:.1f}B"
            if config["params"] > 1e9
            else f"{config['params'] / 1e6:.1f}M"
        )

        print(f"\n{model_size.upper()} ({param_str}):")
        print(f"  FLOP 等价性验证: {'✓ 通过' if eq['is_equivalent'] else '✗ 未通过'}")
        print(f"  理论 T_think: {eq['theoretical_t_think']}, 实际: {eq['actual_t_think']}")
        print(f"  比率: {eq['ratio']:.3f} (目标范围: 0.8 - 1.2)")

    # 保存汇总
    with open("./results/validation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ 全部验证完成！")
    print(f"汇总结果: ./results/validation_summary.json")


if __name__ == "__main__":
    main()
