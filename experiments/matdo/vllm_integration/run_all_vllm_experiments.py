"""
MATDO-E vLLM集成实验统一运行脚本

运行所有vLLM集成实验：
1. 吞吐量可持续性测试 (Fig A)
2. 延迟分解测试 (Fig B)
3. 准确率恢复测试 (Fig C)
4. vLLM消融实验

对应论文实验：
- §5.2 Cross-Model Generalization
- §5.4 Main Results
- §5.6 Ablation
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.matdo.vllm_integration.throughput_test import run_throughput_sustainability_test
from experiments.matdo.vllm_integration.latency_profiler import run_latency_breakdown_test
from experiments.matdo.vllm_integration.accuracy_recovery import run_accuracy_recovery_test
from experiments.matdo.vllm_integration.ablation_vllm import run_vllm_ablation_study


def print_banner(text: str):
    """打印分隔横幅"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")


def run_all_vllm_experiments(
    skip_throughput: bool = False,
    skip_latency: bool = False,
    skip_accuracy: bool = False,
    skip_ablation: bool = False,
    output_dir: Path = None,
) -> Dict:
    """
    运行所有vLLM集成实验
    
    Args:
        skip_*: 是否跳过特定实验
        output_dir: 输出目录
        
    Returns:
        所有实验结果汇总
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_banner("MATDO-E vLLM Integration Experiment Suite")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().isoformat()}")
    print()
    
    all_results = {}
    
    # ==================== 实验1: 吞吐量可持续性 ====================
    if not skip_throughput:
        print_banner("Experiment 1: Throughput Sustainability (Fig A)")
        try:
            results = run_throughput_sustainability_test(
                concurrency_levels=[10, 20, 30, 40, 50, 60, 70, 80],
                num_gpu_blocks=512,
                output_dir=output_dir
            )
            all_results['throughput_sustainability'] = results
        except Exception as e:
            print(f"❌ Throughput test failed: {e}")
            all_results['throughput_sustainability'] = {'error': str(e)}
    else:
        print("Skipping throughput test")
        
    # ==================== 实验2: 延迟分解 ====================
    if not skip_latency:
        print_banner("Experiment 2: Latency Breakdown (Fig B)")
        try:
            results = run_latency_breakdown_test(
                E_values=[0, 16000, 32000, 64000, 128000],
                num_requests=10,
                output_dir=output_dir
            )
            all_results['latency_breakdown'] = results
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            all_results['latency_breakdown'] = {'error': str(e)}
    else:
        print("Skipping latency test")
        
    # ==================== 实验3: 准确率恢复 ====================
    if not skip_accuracy:
        print_banner("Experiment 3: Accuracy Recovery (Fig C)")
        try:
            results = run_accuracy_recovery_test(
                M_reduction_factors=[0.5, 0.3, 0.2],
                E_values=[0, 8000, 16000, 32000, 64000, 128000],
                output_dir=output_dir
            )
            all_results['accuracy_recovery'] = results
        except Exception as e:
            print(f"❌ Accuracy test failed: {e}")
            all_results['accuracy_recovery'] = {'error': str(e)}
    else:
        print("Skipping accuracy test")
        
    # ==================== 实验4: vLLM消融实验 ====================
    if not skip_ablation:
        print_banner("Experiment 4: vLLM Ablation Study")
        try:
            results = run_vllm_ablation_study(
                rho=0.95,
                num_trials=10,
                output_dir=output_dir
            )
            all_results['vllm_ablation'] = results
        except Exception as e:
            print(f"❌ Ablation test failed: {e}")
            all_results['vllm_ablation'] = {'error': str(e)}
    else:
        print("Skipping ablation test")
    
    # ==================== 汇总报告 ====================
    print_banner("MATDO-E vLLM Integration Summary Report")
    
    # 统计通过/失败
    passed = []
    failed = []
    
    for exp_name, result in all_results.items():
        if 'error' in result:
            failed.append(exp_name)
        elif result.get('acceptance', {}).get('overall_pass', False):
            passed.append(exp_name)
        else:
            failed.append(exp_name)
    
    print(f"Passed: {len(passed)}/4")
    if passed:
        print(f"  ✓ {', '.join(passed)}")
    print(f"Failed: {len(failed)}/4")
    if failed:
        print(f"  ✗ {', '.join(failed)}")
    print()
    
    # 关键发现
    print("Key Findings:")
    
    if 'throughput_sustainability' in all_results and 'error' not in all_results['throughput_sustainability']:
        imp = all_results['throughput_sustainability'].get('acceptance', {}).get('improvement_at_max_concurrency_pct', 0)
        print(f"  • Throughput: MATDO-E achieves {imp:.1f}% improvement at max concurrency")
        
    if 'latency_breakdown' in all_results and 'error' not in all_results['latency_breakdown']:
        eff = all_results['latency_breakdown'].get('detailed_analysis', {}).get('masking_efficiency', 0)
        print(f"  • Latency: DRAM retrieval masking efficiency {eff*100:.1f}%")
        
    if 'accuracy_recovery' in all_results and 'error' not in all_results['accuracy_recovery']:
        rec = all_results['accuracy_recovery'].get('analysis', {}).get('M_0.5', {}).get('recovery', 0)
        print(f"  • Accuracy: {rec*100:.1f} percentage points recovery with Engram (M at 50%)")
        
    if 'vllm_ablation' in all_results and 'error' not in all_results['vllm_ablation']:
        best = all_results['vllm_ablation'].get('analysis', {}).get('best_overall', 'N/A')
        print(f"  • Ablation: Best overall configuration is {best}")
    
    # 保存汇总
    summary = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'passed': passed,
            'failed': failed,
            'total_passed': len(passed),
            'total_failed': len(failed),
        },
        'results': all_results,
    }
    
    summary_file = output_dir / "vllm_integration_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {summary_file}")
    
    print_banner("MATDO-E vLLM Integration Complete")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MATDO-E vLLM integration experiments")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip throughput test")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency test")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy test")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation test")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    run_all_vllm_experiments(
        skip_throughput=args.skip_throughput,
        skip_latency=args.skip_latency,
        skip_accuracy=args.skip_accuracy,
        skip_ablation=args.skip_ablation,
        output_dir=args.output_dir,
    )
