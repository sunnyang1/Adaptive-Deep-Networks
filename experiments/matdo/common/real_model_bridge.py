"""
真实模型桥接层：供 MATDO 实验调用真实 AdaptiveTransformer 模型。

当前实现（MVP）：
- 复用 experiments/real_model.model_loader 加载模型
- 支持通过 forward 参数开关 AttnRes / qTTT
- RaBitQ 由于尚未深度集成到 AdaptiveTransformer forward，暂通过概念配置控制
- 评估任务优先支持 needle-in-haystack（已具备真实数据集）
"""

import sys
import torch
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.real_model.model_loader import load_adb_model
from experiments.real_model.datasets.needle_dataset import NeedleDataset


def load_matdo_model(
    checkpoint_path: Optional[str] = None,
    model_size: str = "small",
    device: str = "cuda",
    enable_rabitq: bool = True,
    enable_attnres: bool = True,
    enable_qttt: bool = True,
) -> Tuple[torch.nn.Module, Any]:
    """
    加载 MATDO 真实模型，支持组件开关。

    Args:
        checkpoint_path: 检查点路径，None 则随机初始化
        model_size: small / medium / large
        device: cuda / cpu
        enable_rabitq: 是否启用 RaBitQ（当前为配置层标记）
        enable_attnres: 是否启用 AttnRes
        enable_qttt: 是否启用 qTTT

    Returns:
        (model, config)
    """
    model, config = load_adb_model(
        checkpoint_path=checkpoint_path,
        model_size=model_size,
        device=device,
    )

    # 在 config 上记录开关状态，供后续评估使用
    config.enable_rabitq = enable_rabitq
    config.enable_attnres = enable_attnres
    config.enable_qttt = enable_qttt

    return model, config


def evaluate_needle_haystack(
    model: torch.nn.Module,
    config: Any,
    context_lengths: Tuple[int, ...] = (4096, 16384, 65536),
    num_samples: int = 5,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    在 Needle-in-Haystack 任务上评估模型。

    简化版实现：直接复用 NeedleDataset 生成数据，然后调用 model.generate()
    评估准确率。不使用 experiments/real_model.needle_haystack_real 中的复杂
    tokenization（因为它使用简化的基于 hash 的 tokenizer），而是直接用
    数字 token 模拟 prompt 格式，让模型生成答案。

    Args:
        model: 已加载的模型
        config: 模型配置
        context_lengths: 测试的上下文长度列表
        num_samples: 每个长度采样数
        device: 计算设备

    Returns:
        results 字典，包含 average_accuracy 和各长度准确率
    """
    model.eval()
    dataset = NeedleDataset(seed=42)

    results = {"task": "needle_haystack", "context_lengths": {}}
    all_accuracies = []

    use_attnres = getattr(config, "enable_attnres", True)
    use_qttt = getattr(config, "enable_qttt", False)

    for ctx_len in context_lengths:
        samples = dataset.create_dataset(
            context_tokens=ctx_len,
            num_samples=num_samples,
            depth_distribution="uniform",
        )

        correct = 0
        for sample in samples:
            # 构造 prompt 文本
            prompt_text = sample.format_prompt()
            # 简单 tokenization：每个字符映射到一个 vocab id
            # 为了保持一定稳定性，这里用 ord(c) % vocab_size
            vocab_size = getattr(config, "vocab_size", 32000)
            input_ids = [ord(c) % vocab_size for c in prompt_text]
            input_tensor = torch.tensor([input_ids], device=device)

            # 截断到模型最大长度
            max_len = getattr(config, "max_seq_len", 32768)
            if input_tensor.shape[1] > max_len:
                input_tensor = input_tensor[:, -max_len:]

            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=20,
                    use_attnres=use_attnres,
                    use_qttt=use_qttt,
                )

            generated_text = "".join(
                [chr(min(t % 1112064, 1112063)) for t in output[0, input_tensor.shape[1] :].tolist()]
            )

            evaluation = sample.evaluate(generated_text)
            if evaluation["correct"]:
                correct += 1

        accuracy = correct / num_samples * 100
        all_accuracies.append(accuracy)
        results["context_lengths"][ctx_len] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": num_samples,
        }

    results["average_accuracy"] = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    results["error"] = max(0.0, 1.0 - results["average_accuracy"] / 100.0)
    return results


def evaluate_on_task(
    model: torch.nn.Module,
    task: str,
    config: Any,
    device: str = "cuda",
    **task_kwargs,
) -> Dict[str, Any]:
    """
    统一评估接口。

    Args:
        model: 已加载模型
        task: "needle" 等
        config: 模型配置
        device: 设备
        **task_kwargs: 任务特定参数

    Returns:
        评估结果字典
    """
    if task == "needle" or task == "needle_haystack":
        return evaluate_needle_haystack(model, config, device=device, **task_kwargs)
    else:
        raise ValueError(f"Unsupported real-model task: {task}. Currently only 'needle' is supported.")
