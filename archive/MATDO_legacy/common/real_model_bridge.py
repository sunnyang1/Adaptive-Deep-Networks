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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATDO_NEW_ROOT = _REPO_ROOT / "MATDO-new"

sys.path.insert(0, str(_REPO_ROOT))

from experiments.real_model.model_loader import load_adb_model
from experiments.real_model.datasets.needle_dataset import NeedleDataset


def _ensure_matdo_new_on_path() -> None:
    if _MATDO_NEW_ROOT.is_dir() and str(_MATDO_NEW_ROOT) not in sys.path:
        sys.path.insert(0, str(_MATDO_NEW_ROOT))


def needle_paper_runtime_kwargs(
    legacy: Any,
    *,
    rho_hbm: float,
    use_paper_runtime: bool,
) -> Dict[str, Any]:
    """Keyword args for :func:`evaluate_needle_haystack` / US4–US6 paper alignment."""
    if not use_paper_runtime:
        return {}
    rho_dram = float(getattr(legacy, "us4_paper_rho_dram", 0.30))
    return {
        "use_paper_runtime": True,
        "rho_hbm": float(rho_hbm),
        "rho_dram": rho_dram,
    }


def load_matdo_online_rls_estimator_class() -> Any:
    """Return :class:`matdo_new.core.online_estimation.OnlineRLSEstimator` (lazy import)."""
    _ensure_matdo_new_on_path()
    from matdo_new.core.online_estimation import OnlineRLSEstimator

    return OnlineRLSEstimator


def _greedy_argmax_sampler(logits: object | None) -> int:
    if logits is None:
        raise RuntimeError("backend returned no logits for sampling")
    tensor = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    return int(torch.argmax(tensor, dim=-1).item())


def _build_matdo_paper_backend(
    model: torch.nn.Module,
    model_flags: Any,
    legacy: Any,
    materialized: Any,
    device: str,
) -> Any:
    """Construct ``AdaptiveTransformerRuntimeBackend`` + ``MATDOModel`` for paper policy."""
    _ensure_matdo_new_on_path()
    from matdo_new.modeling.config import (
        ExternalMemoryConfig,
        KVQuantizationConfig,
        MATDOModelConfig,
        QueryAdaptationConfig,
    )
    from matdo_new.modeling.matdo_model import MATDOModel
    from matdo_new.runtime.backend import AdaptiveTransformerRuntimeBackend

    mcfg = model.config
    head_dim = int(mcfg.hidden_dim // mcfg.num_heads)
    use_attnres = bool(getattr(model_flags, "enable_attnres", True))
    use_qttt = bool(getattr(model_flags, "enable_qttt", True))
    model_has_engram = bool(getattr(mcfg, "use_engram", False))
    use_engram = (
        bool(materialized.use_engram) and int(materialized.engram_entries) > 0 and model_has_engram
    )
    use_qttt_effective = use_qttt and int(materialized.t_steps) > 0
    qa_steps = max(1, int(materialized.t_steps)) if use_qttt_effective else 4

    mmc = MATDOModelConfig(
        model_size=str(legacy.model_size),
        use_attnres=use_attnres,
        use_qttt=use_qttt_effective,
        use_engram=use_engram,
        quantization=KVQuantizationConfig(
            total_bits=int(materialized.quantization_bits),
            head_dim=head_dim,
            device=str(device),
        ),
        query_adaptation=QueryAdaptationConfig(num_steps=qa_steps),
        external_memory=ExternalMemoryConfig(
            enabled=use_engram,
            max_entries=int(materialized.engram_entries) if use_engram else 0,
        ),
    )
    runtime_model = MATDOModel(config=mmc, backend=model)
    return AdaptiveTransformerRuntimeBackend(
        model,
        runtime_model=runtime_model,
        device=device,
        use_attnres=use_attnres,
        use_engram=use_engram if model_has_engram else False,
    )


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
    *,
    use_paper_runtime: bool = False,
    rho_hbm: float = 0.92,
    rho_dram: Optional[float] = None,
    materialized_policy: Any = None,
    legacy_matdo_config: Any = None,
) -> Dict[str, Any]:
    """
    在 Needle-in-Haystack 任务上评估模型。

    默认路径调用 ``model.generate()``。当 ``use_paper_runtime=True`` 时，改为通过
    MATDO-new 的 ``generate_tokens`` + ``AdaptiveTransformerRuntimeBackend``，并把
    :class:`matdo_new.runtime.materialize.MaterializedPolicy`（由 ``solve_policy`` 得到）
    传入 prefill/decode，使 US4 真实模型路径与论文策略一致。

    Args:
        model: 已加载的模型
        config: 模型配置（含 ``enable_attnres`` / ``enable_qttt`` 等）
        context_lengths: 测试的上下文长度列表
        num_samples: 每个长度采样数
        device: 计算设备
        use_paper_runtime: 是否走 MATDO-new runtime
        rho_hbm: 观测 HBM 利用率，用于 ``solve_policy``（应与 US4 试验 ρ 对齐）
        rho_dram: DRAM 利用率；``None`` 时用 ``legacy_matdo_config.us4_paper_rho_dram``
        materialized_policy: 若已计算策略可传入；否则在内部调用 ``solve_policy_from_legacy``
        legacy_matdo_config: ``experiments.matdo.common.config.MATDOConfig``；默认 import 全局 ``config``

    Returns:
        results 字典，包含 average_accuracy 和各长度准确率；若 ``use_paper_runtime`` 则含 ``paper_runtime: True``
    """
    from experiments.matdo.common.config import config as default_legacy_matdo_config

    legacy = legacy_matdo_config if legacy_matdo_config is not None else default_legacy_matdo_config
    rho_dram_eff = float(legacy.us4_paper_rho_dram) if rho_dram is None else float(rho_dram)

    model.eval()
    dataset = NeedleDataset(seed=42)

    results: Dict[str, Any] = {
        "task": "needle_haystack",
        "context_lengths": {},
        "paper_runtime": bool(use_paper_runtime),
    }
    if use_paper_runtime:
        results["paper_rho_hbm"] = float(rho_hbm)
        results["paper_rho_dram"] = rho_dram_eff

    all_accuracies = []

    use_attnres = getattr(config, "enable_attnres", True)
    use_qttt = getattr(config, "enable_qttt", False)

    paper_backend = None
    paper_materialized = None
    if use_paper_runtime:
        from experiments.matdo.paper_policy_bridge import solve_policy_from_legacy

        paper_materialized = materialized_policy
        if paper_materialized is None:
            _, paper_materialized = solve_policy_from_legacy(
                legacy,
                rho_hbm=float(rho_hbm),
                rho_dram=rho_dram_eff,
            )
        paper_backend = _build_matdo_paper_backend(
            model, config, legacy, paper_materialized, device
        )
        _ensure_matdo_new_on_path()
        from matdo_new.runtime.generation import generate_tokens

    for ctx_len in context_lengths:
        samples = dataset.create_dataset(
            context_tokens=ctx_len,
            num_samples=num_samples,
            depth_distribution="uniform",
        )

        correct = 0
        for sample in samples:
            prompt_text = sample.format_prompt()
            vocab_size = getattr(config, "vocab_size", 32000)
            input_ids = [ord(c) % vocab_size for c in prompt_text]
            input_tensor = torch.tensor([input_ids], device=device)

            max_len = getattr(config, "max_seq_len", 32768)
            if input_tensor.shape[1] > max_len:
                input_tensor = input_tensor[:, -max_len:]
            prompt_len = int(input_tensor.shape[1])
            prompt_list = [int(x) for x in input_tensor[0].tolist()]

            with torch.no_grad():
                if (
                    use_paper_runtime
                    and paper_backend is not None
                    and paper_materialized is not None
                ):
                    gen = generate_tokens(
                        prompt_list,
                        backend=paper_backend,
                        policy=paper_materialized,
                        max_new_tokens=20,
                        sampler=_greedy_argmax_sampler,
                    )
                    new_tokens = list(gen.generated_token_ids)
                    generated_text = "".join([chr(min(t % 1112064, 1112063)) for t in new_tokens])
                else:
                    output = model.generate(
                        input_tensor,
                        max_new_tokens=20,
                        use_attnres=use_attnres,
                        use_qttt=use_qttt,
                    )
                    generated_text = "".join(
                        [chr(min(t % 1112064, 1112063)) for t in output[0, prompt_len:].tolist()]
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

    results["average_accuracy"] = (
        sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    )
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
        raise ValueError(
            f"Unsupported real-model task: {task}. Currently only 'needle' is supported."
        )
