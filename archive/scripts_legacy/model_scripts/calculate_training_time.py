#!/usr/bin/env python3
"""
训练时长估算器 - 针对 H20-NVLink 4卡配置
"""


def calculate_training_time(
    model_size: str,
    num_gpus: int = 4,
    dataset_size: int = 100_000_000,  # 100B tokens (论文配置)
    seq_len: int = 2048,
    batch_size_per_gpu: int = None,
    epochs: int = 3,
    h20_tflops: float = 148.0,  # H20 FP16 Tensor Core peak
):
    """
    估算训练时长

    参数:
        model_size: 'small', 'medium', 'large'
        num_gpus: GPU 数量
        dataset_size: 数据集token数
        seq_len: 序列长度
        batch_size_per_gpu: 每卡batch size
        epochs: 训练轮数
        h20_tflops: H20 峰值算力
    """

    # 模型配置
    configs = {
        "small": {
            "params": 2.2e9,
            "layers": 32,
            "hidden": 2048,
            "default_batch": 8,
        },
        "medium": {
            "params": 8.7e9,
            "layers": 32,
            "hidden": 4096,
            "default_batch": 2,
        },
        "large": {
            "params": 27e9,
            "layers": 64,
            "hidden": 5120,
            "default_batch": 1,  # 需要 DeepSpeed
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]
    params = config["params"]
    layers = config["layers"]
    hidden = config["hidden"]

    if batch_size_per_gpu is None:
        batch_size_per_gpu = config["default_batch"]

    # 计算每步处理的token数
    global_batch_size = batch_size_per_gpu * num_gpus
    tokens_per_step = global_batch_size * seq_len

    # 计算总步数
    total_tokens = dataset_size * epochs
    total_steps = total_tokens // tokens_per_step

    # 估算每步 FLOPs
    # Transformer FLOPs ≈ 6 * params * tokens (forward + backward)
    flops_per_token = 6 * params * 3  # forward=1, backward=2
    flops_per_step = flops_per_token * tokens_per_step

    # 考虑实际利用率 (H20 在典型训练中的利用率)
    # - Small/Medium 模型: 30-40% (受限于模型并行度)
    # - Large 模型: 25-35% (DeepSpeed overhead)
    utilization = {
        "small": 0.35,
        "medium": 0.30,
        "large": 0.25,
    }[model_size]

    effective_tflops = h20_tflops * num_gpus * utilization

    # 计算每步时间
    seconds_per_step = flops_per_step / (effective_tflops * 1e12)

    # 加上通信开销 (NVLink 很快，但仍有一些 overhead)
    communication_overhead = 1.1  # 10% overhead
    seconds_per_step *= communication_overhead

    # 加上数据加载等 overhead
    overhead_factor = 1.2  # 20% overhead
    seconds_per_step *= overhead_factor

    # 总训练时间
    total_seconds = total_steps * seconds_per_step
    total_hours = total_seconds / 3600
    total_days = total_hours / 24

    return {
        "model_size": model_size,
        "params_b": params / 1e9,
        "num_gpus": num_gpus,
        "global_batch": global_batch_size,
        "tokens_per_step": tokens_per_step,
        "total_steps": total_steps,
        "flops_per_step": flops_per_step,
        "effective_tflops": effective_tflops,
        "seconds_per_step": seconds_per_step,
        "total_hours": total_hours,
        "total_days": total_days,
        "config": config,
    }


def format_time(hours):
    """格式化时间显示"""
    if hours < 1:
        return f"{hours*60:.0f} 分钟"
    elif hours < 24:
        return f"{hours:.1f} 小时"
    else:
        days = int(hours // 24)
        remaining_hours = hours % 24
        return f"{days} 天 {remaining_hours:.1f} 小时"


def print_training_estimate(result, dataset_desc=""):
    """打印训练估算结果"""
    print("\n" + "=" * 70)
    print(f"【{result['model_size'].upper()} 模型训练估算】{dataset_desc}")
    print("=" * 70)

    print(f"\n模型配置:")
    print(f"  参数量: {result['params_b']:.1f}B")
    print(f"  层数: {result['config']['layers']}")
    print(f"  Hidden: {result['config']['hidden']}")

    print(f"\n训练配置:")
    print(f"  GPU 数量: {result['num_gpus']} x H20 96GB")
    print(f"  全局 Batch Size: {result['global_batch']}")
    print(f"  每步 Tokens: {result['tokens_per_step']:,}")
    print(f"  总步数: {result['total_steps']:,}")

    print(f"\n性能估算:")
    print(f"  有效算力: {result['effective_tflops']:.0f} TFLOPS")
    print(f"  每步耗时: {result['seconds_per_step']:.2f} 秒")

    print(f"\n" + "=" * 70)
    print(f"【预估总训练时间: {format_time(result['total_hours'])}】")
    print("=" * 70)


def main():
    print("=" * 70)
    print("Adaptive Deep Networks - H20 4卡训练时长估算")
    print("=" * 70)
    print("\n硬件配置: 4x H20 96GB NVLink")
    print("H20 FP16 Tensor Core 峰值: 148 TFLOPS/GPU")
    print("估算有效利用率: Small 35%, Medium 30%, Large 25%")

    # 场景 1: 论文配置 (100B tokens, 3 epochs)
    print("\n" + "=" * 70)
    print("场景 1: 论文标准训练 (100B tokens × 3 epochs)")
    print("=" * 70)

    for model in ["small", "medium", "large"]:
        result = calculate_training_time(
            model_size=model,
            num_gpus=4,
            dataset_size=100_000_000_000,  # 100B
            seq_len=2048,
            epochs=3,
        )
        print_training_estimate(result, "- 论文配置")

    # 场景 2: 快速验证 (1B tokens, 1 epoch)
    print("\n" + "=" * 70)
    print("场景 2: 快速验证 (1B tokens × 1 epoch)")
    print("=" * 70)

    for model in ["small", "medium", "large"]:
        result = calculate_training_time(
            model_size=model,
            num_gpus=4,
            dataset_size=1_000_000_000,  # 1B
            seq_len=2048,
            epochs=1,
        )
        print_training_estimate(result, "- 快速验证")

    # 场景 3: 中等规模 (10B tokens, 2 epochs)
    print("\n" + "=" * 70)
    print("场景 3: 中等规模 (10B tokens × 2 epochs)")
    print("=" * 70)

    for model in ["small", "medium", "large"]:
        result = calculate_training_time(
            model_size=model,
            num_gpus=4,
            dataset_size=10_000_000_000,  # 10B
            seq_len=2048,
            epochs=2,
        )
        print_training_estimate(result, "- 中等规模")

    # AutoDL 费用估算
    print("\n" + "=" * 70)
    print("AutoDL 费用估算 (假设 H20 4卡 ≈ ¥20/小时)")
    print("=" * 70)

    h20_price_per_hour = 20  # 估算价格

    for model in ["small", "medium", "large"]:
        result = calculate_training_time(
            model_size=model,
            num_gpus=4,
            dataset_size=100_000_000_000,
            epochs=3,
        )
        cost = result["total_hours"] * h20_price_per_hour
        print(f"{model.upper():<10} {format_time(result['total_hours']):<20} ≈ ¥{cost:.0f}")

    print("=" * 70)
    print("\n注意: 以上时间为纯训练时间，不包含数据加载、checkpoint保存等开销")
    print("实际时间可能增加 10-20%")


if __name__ == "__main__":
    main()
