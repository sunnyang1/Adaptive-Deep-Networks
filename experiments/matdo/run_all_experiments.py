"""
MATDO统一实验运行脚本

按照Superpowers框架执行所有实验用户故事(US1-US6)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

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
    
    # 设置随机种子确保可复现
    np.random.seed(42)
    print(f"输出目录: {output_dir}")
    print(f"开始时间: {datetime.now().isoformat()}")
    print()
    
    all_results = {}
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
                rhos=us1_rhos,
                output_dir=output_dir / "singularity"
            )
            all_results['US1'] = us1_results
            
            if us1_results['acceptance']['overall_pass']:
                singularity_results_file = output_dir / "singularity" / "singularity_results.json"
                print("✅ US1 完成，结果将用于US2和US3")
            else:
                print("⚠️ US1 未通过，但继续其他实验")
        except Exception as e:
            print(f"❌ US1 失败: {e}")
            all_results['US1'] = {'error': str(e)}
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
                    output_dir=output_dir / "dual_hierarchy"
                )
            else:
                print("⚠️ 未找到US1结果，使用默认参数")
                us2_results = verify_dual_hierarchy(
                    output_dir=output_dir / "dual_hierarchy"
                )
            all_results['US2'] = us2_results
        except Exception as e:
            print(f"❌ US2 失败: {e}")
            all_results['US2'] = {'error': str(e)}
    else:
        print("跳过 US2")
    
    # ==================== US3: 存储密度溢价 ====================
    if not skip_us3:
        if use_real_model:
            print_banner("跳过 US3: 真实模型模式下暂不做存储密度溢价解析验证")
            all_results['US3'] = {'skipped': True, 'reason': 'Not supported in real-model mode'}
        else:
            print_banner("运行 US3: 存储密度溢价爆炸验证")
            try:
                if singularity_results_file and singularity_results_file.exists():
                    us3_results = verify_lambda2_explosion(
                        singularity_results_file=singularity_results_file,
                        output_dir=output_dir / "shadow_price"
                    )
                else:
                    print("⚠️ 未找到US1结果，使用默认参数")
                    us3_results = verify_lambda2_explosion(
                        output_dir=output_dir / "shadow_price"
                    )
                all_results['US3'] = us3_results
            except Exception as e:
                print(f"❌ US3 失败: {e}")
                all_results['US3'] = {'error': str(e)}
    else:
        print("跳过 US3")
    
    # ==================== US4: SOTA对比 ====================
    if not skip_us4:
        print_banner("运行 US4: SOTA对比实验")
        try:
            us4_results = run_sota_comparison(
                rho_test=0.9,
                num_trials=10,
                output_dir=output_dir / "sota_comparison"
            )
            all_results['US4'] = us4_results
        except Exception as e:
            print(f"❌ US4 失败: {e}")
            all_results['US4'] = {'error': str(e)}
    else:
        print("跳过 US4")
    
    # ==================== US5: 消融实验 ====================
    if not skip_us5:
        print_banner("运行 US5: 消融实验")
        try:
            us5_results = run_ablation_study(
                rho=0.9,
                num_trials=10,
                output_dir=output_dir / "ablation"
            )
            all_results['US5'] = us5_results
        except Exception as e:
            print(f"❌ US5 失败: {e}")
            all_results['US5'] = {'error': str(e)}
    else:
        print("跳过 US5")
    
    # ==================== US6: 在线辨识 ====================
    if not skip_us6:
        print_banner("运行 US6: 在线系统辨识")
        try:
            us6_results = run_online_identification(
                num_queries=200,
                lambda_=0.98,
                output_dir=output_dir / "online_identification"
            )
            all_results['US6'] = us6_results
        except Exception as e:
            print(f"❌ US6 失败: {e}")
            all_results['US6'] = {'error': str(e)}
    else:
        print("跳过 US6")
    
    # ==================== 汇总报告 ====================
    print_banner("MATDO实验汇总报告")
    
    passed = []
    failed = []
    skipped = []
    
    for us_name in ['US1', 'US2', 'US3', 'US4', 'US5', 'US6']:
        if us_name in all_results:
            result = all_results[us_name]
            if 'error' in result:
                failed.append(us_name)
            elif result.get('acceptance', {}).get('overall_pass', False):
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
    if 'US1' in all_results and 'acceptance' in all_results['US1']:
        fit = all_results['US1'].get('fit', {})
        print(f"  • US1: 二阶奇点标度律 R²={fit.get('r_squared', 'N/A'):.4f}")
        print(f"         ρ_collapse = {fit.get('rho_collapse', 'N/A'):.4f}")
    
    if 'US2' in all_results and 'acceptance' in all_results['US2']:
        print(f"  • US2: ρ_OOM = {all_results['US2'].get('rho_oom', 'N/A'):.4f}")
        print(f"         gap = {all_results['US2'].get('gap', 'N/A'):.4f}")
    
    if 'US4' in all_results and 'acceptance' in all_results['US4']:
        imp = all_results['US4'].get('improvements', {})
        print(f"  • US4: vs SnapKV +{imp.get('vs_snapkv_pct', 'N/A'):.1f}%")
        print(f"         vs H2O +{imp.get('vs_h2o_pct', 'N/A'):.1f}%")
    
    # 保存总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'total_passed': len(passed),
            'total_failed': len(failed)
        },
        'results': all_results
    }
    
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n完整结果保存至: {summary_file}")
    
    print_banner("MATDO实验套件完成")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行MATDO实验套件")
    parser.add_argument("--skip-us1", action="store_true", help="跳过US1")
    parser.add_argument("--skip-us2", action="store_true", help="跳过US2")
    parser.add_argument("--skip-us3", action="store_true", help="跳过US3")
    parser.add_argument("--skip-us4", action="store_true", help="跳过US4")
    parser.add_argument("--skip-us5", action="store_true", help="跳过US5")
    parser.add_argument("--skip-us6", action="store_true", help="跳过US6")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录")
    parser.add_argument("--use-real-model", action="store_true", help="使用真实模型进行实验")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型检查点路径")
    parser.add_argument("--size", type=str, default="small", choices=["small", "medium", "large"], help="模型大小")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
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
    )
