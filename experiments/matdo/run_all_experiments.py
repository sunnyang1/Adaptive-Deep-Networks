"""
MATDO统一实验运行脚本

按照Superpowers框架执行所有实验用户故事(US1-US6)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入各实验模块
from experiments.matdo.singularity.measure_t_opt import run_singularity_experiment
from experiments.matdo.dual_hierarchy.find_rho_oom import verify_dual_hierarchy
from experiments.matdo.shadow_price.calculate_lambda2 import verify_lambda2_explosion
from experiments.matdo.sota_comparison.compare_baselines import run_sota_comparison
from experiments.matdo.ablation.run_ablation import run_ablation_study
from experiments.matdo.online_identification.rls_estimator import run_online_identification


def print_banner(text: str):
    """打印分隔横幅"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")


def run_all_matdo_experiments(
    skip_us1: bool = False,
    skip_us2: bool = False,
    skip_us3: bool = False,
    skip_us4: bool = False,
    skip_us5: bool = False,
    skip_us6: bool = False,
    output_dir: Path = None,
    use_real_model: bool = False,
    checkpoint_path: str = None,
    model_size: str = "small",
    device: str = "cuda",
    *,
    us4_num_trials: Optional[int] = None,
    us4_enable_qttt: Optional[bool] = None,
    rls_ctx_lengths_override: Optional[Tuple[int, ...]] = None,
    use_paper_policy: bool = False,
    paper_rho_hbm: float = 0.92,
    paper_rho_dram: float = 0.30,
    us4_use_paper_runtime: Optional[bool] = None,
    us4_paper_rho_dram: Optional[float] = None,
    us5_use_paper_runtime: Optional[bool] = None,
    us6_use_paper_runtime: Optional[bool] = None,
    us6_paper_rho_hbm: Optional[float] = None,
) -> Dict:
    """
    运行所有MATDO实验

    按照依赖顺序执行:
    US1 → US2, US3 (依赖US1)
    US1-US3并行 → US4, US5 (依赖基础配置)
    US6 独立

    Args:
        skip_usX: 是否跳过特定实验
        output_dir: 输出目录
        us4_num_trials: 若给定，写入 ``config.us4_num_trials``（US4 试验次数）
        us4_enable_qttt: 若给定，写入 ``config.us4_enable_qttt``
        rls_ctx_lengths_override: 若给定，写入 ``config.rls_ctx_lengths_override``（US6）
        use_paper_policy: 若为 True，用 ``paper_policy_bridge`` 调用 MATDO-new 的
            ``solve_policy``，打印并写入 ``paper_policy_bridge.json``（不改变 US4–US6 内部求解器）
        paper_rho_hbm / paper_rho_dram: 传给 MATDO-new 的运行时观测
        us4_use_paper_runtime: 若为 True/False，写入 ``config.us4_use_paper_runtime``（US4 真实模型走 MATDO-new runtime）
        us4_paper_rho_dram: 若给定，写入 ``config.us4_paper_rho_dram``
        us5_use_paper_runtime / us6_use_paper_runtime: 若给定，写入对应 ``config`` 开关（US5/US6 Needle 与 US4 一致）
        us6_paper_rho_hbm: 若给定，写入 ``config.us6_paper_rho_hbm``

    Returns:
        all_results: 所有实验结果汇总
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置真实模型全局配置
    from experiments.matdo.common.config import config as matdo_config

    matdo_config.use_real_model = use_real_model
    if use_real_model:
        matdo_config.checkpoint_path = checkpoint_path
        matdo_config.model_size = model_size
        matdo_config.device = device
        print_banner("MATDO真实模型实验套件启动")
        print(f"模型大小: {model_size} | 设备: {device}")
        if checkpoint_path:
            print(f"检查点: {checkpoint_path}")
        else:
            print("注意: 未提供检查点，将使用随机初始化权重")
    else:
        print_banner("MATDO实验套件启动")

    # Optional US4 / US6 knobs (also used from CLI; see experiments/matdo/README.md)
    if us4_num_trials is not None:
        matdo_config.us4_num_trials = us4_num_trials
    if us4_enable_qttt is not None:
        matdo_config.us4_enable_qttt = us4_enable_qttt
    if rls_ctx_lengths_override is not None:
        matdo_config.rls_ctx_lengths_override = rls_ctx_lengths_override
    if us4_use_paper_runtime is not None:
        matdo_config.us4_use_paper_runtime = bool(us4_use_paper_runtime)
    if us4_paper_rho_dram is not None:
        matdo_config.us4_paper_rho_dram = float(us4_paper_rho_dram)
    if us5_use_paper_runtime is not None:
        matdo_config.us5_use_paper_runtime = bool(us5_use_paper_runtime)
    if us6_use_paper_runtime is not None:
        matdo_config.us6_use_paper_runtime = bool(us6_use_paper_runtime)
    if us6_paper_rho_hbm is not None:
        matdo_config.us6_paper_rho_hbm = float(us6_paper_rho_hbm)

    paper_policy_payload: Optional[Dict] = None
    if use_paper_policy:
        from experiments.matdo.paper_policy_bridge import (
            dump_policy_payload,
            policy_payload_for_experiments,
        )

        print_banner("MATDO-new paper policy (bridge)")
        paper_policy_payload = policy_payload_for_experiments(
            matdo_config,
            rho_hbm=float(paper_rho_hbm),
            rho_dram=float(paper_rho_dram),
        )
        print(json.dumps(paper_policy_payload, indent=2, ensure_ascii=False))
        dump_policy_payload(output_dir / "paper_policy_bridge.json", paper_policy_payload)
        print(f"Paper policy snapshot: {output_dir / 'paper_policy_bridge.json'}\n")

    # 设置随机种子确保可复现
    np.random.seed(42)
    print(f"输出目录: {output_dir}")
    print(f"开始时间: {datetime.now().isoformat()}")
    print()

    all_results: Dict = {}
    if paper_policy_payload is not None:
        all_results["paper_policy_bridge"] = paper_policy_payload
    singularity_results_file = None

    # ==================== US1: 二阶奇点标度律 ====================
    if not skip_us1:
        print_banner("运行 US1: 二阶奇点标度律验证")
        try:
            if use_real_model:
                print("真实模型模式: 减少采样点以控制耗时")
                us1_rhos = [0.80, 0.90, 0.94]
            else:
                us1_rhos = [0.70, 0.80, 0.88, 0.92, 0.94, 0.945]
            us1_results = run_singularity_experiment(
                rhos=us1_rhos, output_dir=output_dir / "singularity"
            )
            all_results["US1"] = us1_results

            if us1_results["acceptance"]["overall_pass"]:
                singularity_results_file = output_dir / "singularity" / "singularity_results.json"
                print("✅ US1 完成，结果将用于US2和US3")
            else:
                print("⚠️ US1 未通过，但继续其他实验")
        except Exception as e:
            print(f"❌ US1 失败: {e}")
            all_results["US1"] = {"error": str(e)}
    else:
        print("跳过 US1")
        # 尝试加载已有结果
        existing = output_dir / "singularity" / "singularity_results.json"
        if existing.exists():
            singularity_results_file = existing

    # ==================== US2: 双奇点层次 ====================
    if not skip_us2:
        print_banner("运行 US2: 双奇点层次验证")
        try:
            if singularity_results_file and singularity_results_file.exists():
                us2_results = verify_dual_hierarchy(
                    singularity_results_file=singularity_results_file,
                    output_dir=output_dir / "dual_hierarchy",
                )
            else:
                print("⚠️ 未找到US1结果，使用默认参数")
                us2_results = verify_dual_hierarchy(output_dir=output_dir / "dual_hierarchy")
            all_results["US2"] = us2_results
        except Exception as e:
            print(f"❌ US2 失败: {e}")
            all_results["US2"] = {"error": str(e)}
    else:
        print("跳过 US2")

    # ==================== US3: 存储密度溢价 ====================
    if not skip_us3:
        if use_real_model:
            print_banner("跳过 US3: 真实模型模式下暂不做存储密度溢价解析验证")
            all_results["US3"] = {"skipped": True, "reason": "Not supported in real-model mode"}
        else:
            print_banner("运行 US3: 存储密度溢价爆炸验证")
            try:
                if singularity_results_file and singularity_results_file.exists():
                    us3_results = verify_lambda2_explosion(
                        singularity_results_file=singularity_results_file,
                        output_dir=output_dir / "shadow_price",
                    )
                else:
                    print("⚠️ 未找到US1结果，使用默认参数")
                    us3_results = verify_lambda2_explosion(output_dir=output_dir / "shadow_price")
                all_results["US3"] = us3_results
            except Exception as e:
                print(f"❌ US3 失败: {e}")
                all_results["US3"] = {"error": str(e)}
    else:
        print("跳过 US3")

    # ==================== US4: SOTA对比 ====================
    if not skip_us4:
        print_banner("运行 US4: SOTA对比实验")
        try:
            # ``us4_num_trials`` lets CPU smoke runs shrink the 10× trial
            # loop down to something that finishes; the paper defaults
            # stay at 10 when the knob is left as ``None``.
            us4_trials = getattr(matdo_config, "us4_num_trials", None)
            if us4_trials is None:
                us4_trials = 10
            us4_results = run_sota_comparison(
                rho_test=0.9, num_trials=int(us4_trials), output_dir=output_dir / "sota_comparison"
            )
            all_results["US4"] = us4_results
        except Exception as e:
            print(f"❌ US4 失败: {e}")
            all_results["US4"] = {"error": str(e)}
    else:
        print("跳过 US4")

    # ==================== US5: 消融实验 ====================
    if not skip_us5:
        print_banner("运行 US5: 消融实验")
        try:
            us5_results = run_ablation_study(
                rho=0.9, num_trials=10, output_dir=output_dir / "ablation"
            )
            all_results["US5"] = us5_results
        except Exception as e:
            print(f"❌ US5 失败: {e}")
            all_results["US5"] = {"error": str(e)}
    else:
        print("跳过 US5")

    # ==================== US6: 在线辨识 ====================
    if not skip_us6:
        print_banner("运行 US6: 在线系统辨识")
        try:
            us6_results = run_online_identification(
                num_queries=200, lambda_=0.98, output_dir=output_dir / "online_identification"
            )
            all_results["US6"] = us6_results
        except Exception as e:
            print(f"❌ US6 失败: {e}")
            all_results["US6"] = {"error": str(e)}
    else:
        print("跳过 US6")

    # ==================== 汇总报告 ====================
    print_banner("MATDO实验汇总报告")

    passed = []
    failed = []
    skipped = []

    for us_name in ["US1", "US2", "US3", "US4", "US5", "US6"]:
        if us_name in all_results:
            result = all_results[us_name]
            if "error" in result:
                failed.append(us_name)
            elif result.get("acceptance", {}).get("overall_pass", False):
                passed.append(us_name)
            else:
                failed.append(us_name)
        else:
            skipped.append(us_name)

    print(f"通过: {len(passed)}/6 {passed}")
    print(f"失败: {len(failed)}/6 {failed}")
    print(f"跳过: {len(skipped)}/6 {skipped}")
    print()

    # 关键发现
    print("关键发现:")
    if "US1" in all_results and "acceptance" in all_results["US1"]:
        fit = all_results["US1"].get("fit", {})
        print(f"  • US1: 二阶奇点标度律 R²={fit.get('r_squared', 'N/A'):.4f}")
        print(f"         ρ_collapse = {fit.get('rho_collapse', 'N/A'):.4f}")

    if "US2" in all_results and "acceptance" in all_results["US2"]:
        print(f"  • US2: ρ_OOM = {all_results['US2'].get('rho_oom', 'N/A'):.4f}")
        print(f"         gap = {all_results['US2'].get('gap', 'N/A'):.4f}")

    if "US4" in all_results and "acceptance" in all_results["US4"]:
        # ``compare_baselines.run_sota_comparison`` was refactored to emit a
        # ``stats`` dict keyed by method instead of the older
        # ``improvements`` block. Compute the percentages on the fly for the
        # summary, falling back to 'N/A' when the new schema is absent.
        us4_stats = all_results["US4"].get("stats", {})
        matdo_e = us4_stats.get("MATDO-E (4D)", {}).get("mean_accuracy")

        def _pct_delta(baseline_name: str) -> str:
            base = us4_stats.get(baseline_name, {}).get("mean_accuracy")
            if matdo_e is None or base in (None, 0):
                return "N/A"
            return f"{(matdo_e - base) / base * 100:+.1f}%"

        print(f"  • US4: vs SnapKV {_pct_delta('SnapKV')}")
        print(f"         vs H2O    {_pct_delta('H2O')}")

    # 保存总结果
    summary = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total_passed": len(passed),
            "total_failed": len(failed),
        },
        "results": all_results,
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n完整结果保存至: {summary_file}")

    print_banner("MATDO实验套件完成")

    return summary


def _parse_rls_ctx_lengths(s: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse ``--rls-ctx-lengths`` like ``128,256,512`` into a tuple of ints."""

    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        return None
    return tuple(int(p) for p in parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="运行MATDO实验套件",
        epilog="US4/US6 调参说明见 experiments/matdo/README.md",
    )
    parser.add_argument("--skip-us1", action="store_true", help="跳过US1")
    parser.add_argument("--skip-us2", action="store_true", help="跳过US2")
    parser.add_argument("--skip-us3", action="store_true", help="跳过US3")
    parser.add_argument("--skip-us4", action="store_true", help="跳过US4")
    parser.add_argument("--skip-us5", action="store_true", help="跳过US5")
    parser.add_argument("--skip-us6", action="store_true", help="跳过US6")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--use-real-model", action="store_true", help="使用真实模型进行实验")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument(
        "--size", type=str, default="small", choices=["small", "medium", "large"], help="模型大小"
    )
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument(
        "--us4-num-trials",
        type=int,
        default=None,
        metavar="N",
        help="US4 SOTA 对比重复试验次数；缺省使用配置（未在配置中设置时为 10）",
    )
    parser.add_argument(
        "--us4-no-qttt",
        action="store_true",
        help="US4 真实模型路径中关闭 qTTT（CPU 上显著缩短单次 generate 时间）",
    )
    parser.add_argument(
        "--rls-ctx-lengths",
        type=str,
        default=None,
        metavar="L1,L2,...",
        help=("US6 RLS：逗号分隔的上下文长度，覆盖默认的 ctx_len=M*N_block；" "例: 128,256,512"),
    )
    parser.add_argument(
        "--paper-policy",
        action="store_true",
        help="调用 MATDO-new solve_policy 并写入 paper_policy_bridge.json（见 MATDO_NEW_BRIDGE.md）",
    )
    parser.add_argument(
        "--paper-rho-hbm",
        type=float,
        default=0.92,
        metavar="RHO",
        help="与 --paper-policy 联用：观测到的 HBM 利用率",
    )
    parser.add_argument(
        "--paper-rho-dram",
        type=float,
        default=0.30,
        metavar="RHO",
        help="与 --paper-policy 联用：观测到的 DRAM 利用率",
    )
    parser.add_argument(
        "--us4-paper-runtime",
        action="store_true",
        help="US4 真实模型 Needle 评估走 MATDO-new generate_tokens + MaterializedPolicy（需 --use-real-model）",
    )
    parser.add_argument(
        "--us4-paper-rho-dram",
        type=float,
        default=None,
        metavar="RHO",
        help="传给 evaluate_needle_haystack 的 rho_dram（默认用 config.us4_paper_rho_dram）",
    )
    parser.add_argument(
        "--us5-paper-runtime",
        action="store_true",
        help="US5 真实模型 Needle 走 MATDO-new runtime（与 --us4-paper-runtime 一致；需 --use-real-model）",
    )
    parser.add_argument(
        "--us6-paper-runtime",
        action="store_true",
        help="US6 真实模型 Needle 走 MATDO-new runtime（需 --use-real-model）",
    )
    parser.add_argument(
        "--us6-paper-rho-hbm",
        type=float,
        default=None,
        metavar="RHO",
        help="US6 paper runtime 的 rho_hbm（默认 config.us6_paper_rho_hbm）",
    )

    args = parser.parse_args()

    run_all_matdo_experiments(
        skip_us1=args.skip_us1,
        skip_us2=args.skip_us2,
        skip_us3=args.skip_us3,
        skip_us4=args.skip_us4,
        skip_us5=args.skip_us5,
        skip_us6=args.skip_us6,
        output_dir=args.output_dir,
        use_real_model=args.use_real_model,
        checkpoint_path=args.checkpoint,
        model_size=args.size,
        device=args.device,
        us4_num_trials=args.us4_num_trials,
        us4_enable_qttt=False if args.us4_no_qttt else None,
        rls_ctx_lengths_override=_parse_rls_ctx_lengths(args.rls_ctx_lengths),
        use_paper_policy=args.paper_policy,
        paper_rho_hbm=args.paper_rho_hbm,
        paper_rho_dram=args.paper_rho_dram,
        us4_use_paper_runtime=True if args.us4_paper_runtime else None,
        us4_paper_rho_dram=args.us4_paper_rho_dram,
        us5_use_paper_runtime=True if args.us5_paper_runtime else None,
        us6_use_paper_runtime=True if args.us6_paper_runtime else None,
        us6_paper_rho_hbm=args.us6_paper_rho_hbm,
    )
