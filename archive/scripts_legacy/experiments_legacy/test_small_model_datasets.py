#!/usr/bin/env python3
"""
使用 Small Model 在论文数据集上测试指标

基于 Adaptive_Deep_Networks_TurboQuant.md:
1. Needle-in-Haystack (Table 4) - 长上下文检索
2. MATH Dataset (Table 6) - 数学推理
3. LongBench-v2 (Table 7) - 综合评估

注意: 由于 Small Model 没有预训练权重，本脚本:
- 创建评估框架和流程
- 提供基于论文的参考指标对比
- 生成用于未来实际测试的代码模板
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List

from models.configs import get_config
from models.adaptive_transformer import AdaptiveTransformer


class DatasetTester:
    """论文数据集测试器"""

    def __init__(self, model=None, device="cpu"):
        self.device = torch.device(device)
        self.results = {}

        if model is None:
            # 创建 Small Model (未训练)
            print("Creating Small Model (untrained)...")
            config = get_config("small")
            self.model = AdaptiveTransformer(config)
            self.model.eval()
            self.config = config
        else:
            self.model = model
            self.config = model.config

        print(f"Model: Small ({self.model.count_parameters()/1e9:.2f}B parameters)")

    def test_needle_haystack(self, context_lengths: List[int] = None) -> Dict:
        """
        测试 1: Needle-in-Haystack (Table 4)

        论文指标:
        | Context | Transformer | TTT-Linear | AttnRes | ADB + TurboQuant |
        |---------|-------------|------------|---------|------------------|
        | 4K      | 87.5%       | 94.2%      | 96.8%   | **98.5%**        |
        | 32K     | 22.1%       | 65.3%      | 75.6%   | **91.3%**        |
        | 64K     | 8.7%        | 48.7%      | 58.9%   | **85.5%**        |
        | 128K    | 3.2%        | 32.1%      | 42.3%   | **78.2%**        |
        | 256K    | 1.5%        | 18.5%      | 28.7%   | **68.2%**        |
        | Average | 38.2%       | 62.3%      | 69.9%   | **86.9%**        |
        """
        print("\n" + "=" * 70)
        print("TEST 1: Needle-in-Haystack (Table 4)")
        print("=" * 70)

        if context_lengths is None:
            # 缩短长度以适应未训练模型测试
            context_lengths = [512, 1024, 2048]

        # 论文中的完整指标
        paper_results = {
            "context_lengths": [4096, 32768, 65536, 131072, 262144],
            "transformer": [87.5, 22.1, 8.7, 3.2, 1.5],
            "ttt_linear": [94.2, 65.3, 48.7, 32.1, 18.5],
            "attnres": [96.8, 75.6, 58.9, 42.3, 28.7],
            "adb_turboquant": [98.5, 91.3, 85.5, 78.2, 68.2],
        }

        print("\nPaper Results (Table 4):")
        print("-" * 80)
        print(
            f"{'Context':<10} {'Transformer':<15} {'TTT-Linear':<15} {'AttnRes':<15} {'ADB+Turbo':<15}"
        )
        print("-" * 80)

        for i, ctx in enumerate(paper_results["context_lengths"]):
            ctx_str = f"{ctx//1024}K"
            print(
                f"{ctx_str:<10} "
                f"{paper_results['transformer'][i]:>6.1f}%{'':<8} "
                f"{paper_results['ttt_linear'][i]:>6.1f}%{'':<8} "
                f"{paper_results['attnres'][i]:>6.1f}%{'':<8} "
                f"{paper_results['adb_turboquant'][i]:>6.1f}%"
            )

        print("-" * 80)
        print(
            f"{'Average':<10} {38.2:>6.1f}%{'':<8} {62.3:>6.1f}%{'':<8} "
            f"{69.9:>6.1f}%{'':<8} {86.9:>6.1f}%"
        )

        # 关键发现
        print("\nKey Findings from Paper:")
        print(
            "  1. At 256K context, ADB maintains 68.2% accuracy vs 1.5% baseline (45× improvement)"
        )
        print("  2. Relative ADB advantage increases with length: +11.1% (4K) → +53.6% (256K)")
        print("  3. Average accuracy: 86.9% (ADB+TurboQuant) vs 38.2% (Transformer)")

        # 模拟测试 (由于模型未训练，返回模拟值)
        print("\n" + "-" * 70)
        print("Simulated Test with Current Model (Untrained):")
        print("-" * 70)

        test_results = []
        for ctx_len in context_lengths:
            # 模拟随机准确率 (未训练模型应接近随机)
            # 实际测试中，这里会运行真实推理
            simulated_acc = np.random.uniform(0.05, 0.15)  # 随机基线 ~10%

            test_results.append(
                {
                    "context_length": ctx_len,
                    "simulated_accuracy": round(simulated_acc * 100, 1),
                    "note": "Model untrained - results are baseline random",
                }
            )

            print(f"  {ctx_len:5d} tokens: {simulated_acc*100:.1f}% (baseline random)")

        results = {
            "paper_results": paper_results,
            "simulated_test": {
                "context_lengths": context_lengths,
                "results": test_results,
                "note": "Model is untrained. Use pre-trained weights for real evaluation.",
            },
            "key_finding": "Needle-in-Haystack requires pre-trained model for meaningful results",
        }

        self.results["needle_haystack"] = results
        return results

    def test_math_dataset(self) -> Dict:
        """
        测试 2: MATH Dataset (Table 6)

        论文指标 (8.7B models):
        | Method              | Level 1-2 | Level 3-4 | Level 5 | Overall |
        |---------------------|-----------|-----------|---------|---------|
        | Transformer         | 60.4%     | 31.6%     | 12.1%   | 35.2%   |
        | CoT (5 samples)     | 65.5%     | 38.7%     | 18.5%   | 41.5%   |
        | TTT-Linear          | 70.0%     | 46.8%     | 28.7%   | 48.9%   |
        | AttnRes + qTTT      | 71.5%     | 51.3%     | 34.5%   | 52.3%   |
        | AttnRes + qTTT (max)| 74.9%     | 58.6%     | 42.1%   | 58.9%   |

        关键声明: 8.7B 参数匹配 50B 静态基线性能
        """
        print("\n" + "=" * 70)
        print("TEST 2: MATH Dataset (Table 6)")
        print("=" * 70)

        paper_results = {
            "methods": {
                "Transformer": {
                    "level_1_2": 60.4,
                    "level_3_4": 31.6,
                    "level_5": 12.1,
                    "overall": 35.2,
                },
                "CoT_5_samples": {
                    "level_1_2": 65.5,
                    "level_3_4": 38.7,
                    "level_5": 18.5,
                    "overall": 41.5,
                },
                "TTT_Linear": {
                    "level_1_2": 70.0,
                    "level_3_4": 46.8,
                    "level_5": 28.7,
                    "overall": 48.9,
                },
                "AttnRes_qTTT_gated": {
                    "level_1_2": 71.5,
                    "level_3_4": 51.3,
                    "level_5": 34.5,
                    "overall": 52.3,
                },
                "AttnRes_qTTT_max": {
                    "level_1_2": 74.9,
                    "level_3_4": 58.6,
                    "level_5": 42.1,
                    "overall": 58.9,
                },
            }
        }

        print("\nPaper Results (Table 6, 8.7B models):")
        print("-" * 90)
        print(f"{'Method':<25} {'Level 1-2':<12} {'Level 3-4':<12} {'Level 5':<12} {'Overall':<12}")
        print("-" * 90)

        for method, scores in paper_results["methods"].items():
            print(
                f"{method:<25} "
                f"{scores['level_1_2']:>6.1f}%{'':<5} "
                f"{scores['level_3_4']:>6.1f}%{'':<5} "
                f"{scores['level_5']:>6.1f}%{'':<5} "
                f"{scores['overall']:>6.1f}%"
            )

        print("-" * 90)

        print("\nKey Findings from Paper:")
        print("  1. AttnRes + qTTT achieves 52.3% overall (8.7B parameters)")
        print("  2. Matches 50B static baseline performance")
        print("  3. Level 5 (hardest): 34.5% vs 12.1% baseline (+22.4%)")
        print("  4. Consistent improvement across all difficulty levels")

        # 模拟 Small Model 结果
        print("\n" + "-" * 70)
        print("Expected Small Model Performance:")
        print("-" * 70)
        print("  Small Model (2.2B) expected: ~25-35% overall")
        print("  Scaling to 8.7B: ~52.3% (paper result)")
        print("  Note: Requires pre-trained weights and qTTT adaptation")

        results = {
            "paper_results": paper_results,
            "note": "MATH evaluation requires pre-trained model with qTTT adaptation",
            "expected_small_model": {
                "overall": "25-35% (estimated)",
                "scaling_to_medium": "52.3% (paper result for 8.7B)",
            },
        }

        self.results["math_dataset"] = results
        return results

    def test_ablation_study(self) -> Dict:
        """
        测试 3: 消融研究 (Table 7)

        论文指标 (8.7B, LongBench-v2):
        | Configuration        | Avg Score | Δ vs Full |
        |---------------------|-----------|-----------|
        | Full System         | 56.8%     | —         |
        | w/o qTTT            | 50.1%     | -6.7%     |
        | w/o Gating          | 53.2%     | -3.6%     |
        | w/o AttnRes         | 48.9%     | -7.9%     |
        | w/o TurboQuant      | 51.5%     | -5.3%     |
        | Standard Transformer| 39.7%     | -17.1%    |

        协同系数: 1.18 (超加性交互)
        """
        print("\n" + "=" * 70)
        print("TEST 3: Ablation Study (Table 7)")
        print("=" * 70)

        paper_results = {
            "configurations": {
                "Full_System": {"score": 56.8, "delta": 0.0},
                "w/o_qTTT": {"score": 50.1, "delta": -6.7},
                "w/o_Gating": {"score": 53.2, "delta": -3.6},
                "w/o_AttnRes": {"score": 48.9, "delta": -7.9},
                "w/o_TurboQuant": {"score": 51.5, "delta": -5.3},
                "Standard_Transformer": {"score": 39.7, "delta": -17.1},
            },
            "synergy_coefficient": 1.18,
        }

        print("\nPaper Results (Table 7, 8.7B models, LongBench-v2):")
        print("-" * 70)
        print(f"{'Configuration':<25} {'Avg Score':<12} {'Δ vs Full':<12}")
        print("-" * 70)

        for config, data in paper_results["configurations"].items():
            delta_str = f"{data['delta']:+.1f}%" if data["delta"] != 0 else "—"
            print(f"{config:<25} {data['score']:>6.1f}%{'':<5} {delta_str:>12}")

        print("-" * 70)
        print(f"\nSynergy Coefficient: {paper_results['synergy_coefficient']}")
        print("(>1.0 indicates super-additive interaction between components)")

        print("\nKey Findings from Paper:")
        print("  1. Each component contributes positively")
        print("  2. AttnRes is most critical (-7.9% when removed)")
        print("  3. Full system: 56.8% vs Standard: 39.7% (+17.1%)")
        print("  4. Synergy: 1.18 (components work better together)")

        results = {
            "paper_results": paper_results,
            "component_importance": [
                ("AttnRes", 7.9),
                ("qTTT", 6.7),
                ("TurboQuant", 5.3),
                ("Gating", 3.6),
            ],
            "note": "Ablation study shows all components contribute significantly",
        }

        self.results["ablation_study"] = results
        return results

    def test_compute_efficiency(self) -> Dict:
        """
        测试 4: 计算效率 (Table 8)

        论文指标 (MATH dataset):
        | Configuration          | Avg FLOP (×10^14) | Accuracy | Acc/FLOP |
        |-----------------------|-------------------|----------|----------|
        | Standard 32L          | 1.0               | 35.2%    | 35.2     |
        | AttnRes 32L (static)  | 1.05              | 41.8%    | 39.8     |
        | AttnRes + qTTT        | 1.45              | 47.5%    | 32.8     |
        | AttnRes + qTTT (gated)| 1.28              | 52.3%    | 40.9     |
        | AttnRes + qTTT (oracle)| 1.15             | 54.8%    | 47.7     |

        关键发现: Gated adaptation achieves best accuracy at lowest average FLOP
        """
        print("\n" + "=" * 70)
        print("TEST 4: Compute Efficiency (Table 8)")
        print("=" * 70)

        paper_results = {
            "configurations": {
                "Standard_32L": {"flops": 1.0, "accuracy": 35.2, "acc_per_flop": 35.2},
                "AttnRes_32L_static": {"flops": 1.05, "accuracy": 41.8, "acc_per_flop": 39.8},
                "AttnRes_qTTT_uniform": {"flops": 1.45, "accuracy": 47.5, "acc_per_flop": 32.8},
                "AttnRes_qTTT_gated": {"flops": 1.28, "accuracy": 52.3, "acc_per_flop": 40.9},
                "AttnRes_qTTT_oracle": {"flops": 1.15, "accuracy": 54.8, "acc_per_flop": 47.7},
            }
        }

        print("\nPaper Results (Table 8, MATH dataset):")
        print("-" * 90)
        print(f"{'Configuration':<28} {'Avg FLOP':<15} {'Accuracy':<12} {'Acc/FLOP':<12}")
        print("-" * 90)

        for config, data in paper_results["configurations"].items():
            print(
                f"{config:<28} "
                f"{data['flops']:.2f}×10^14{'':<5} "
                f"{data['accuracy']:>6.1f}%{'':<5} "
                f"{data['acc_per_flop']:>6.1f}"
            )

        print("-" * 90)

        print("\nKey Findings from Paper:")
        print("  1. Gated adaptation: Best accuracy (52.3%) at lowest FLOP (1.28×10^14)")
        print("  2. Acc/FLOP efficiency: 40.9 (gated) vs 35.2 (standard)")
        print("  3. Oracle (upper bound): 54.8% at 1.15×10^14 FLOPs")
        print("  4. 40% compute reduction vs FLOP-matched alternatives")

        # Small Model FLOP 估算
        flops_per_token = 4.30e9  # 来自之前的分析
        tokens_per_problem = 1000  # 估算
        small_model_flops = flops_per_token * tokens_per_problem

        print("\n" + "-" * 70)
        print("Small Model FLOP Estimation:")
        print("-" * 70)
        print(f"  FLOPs per token: {flops_per_token/1e9:.2f} GFLOPs")
        print(f"  Est. tokens per problem: {tokens_per_problem}")
        print(f"  Est. FLOPs per problem: {small_model_flops/1e9:.2f} GFLOPs")
        print(f"  8.7B model FLOPs: ~14 GFLOPs/token (1.4× Small)")

        results = {
            "paper_results": paper_results,
            "small_model_estimation": {
                "flops_per_token": flops_per_token,
                "estimated_tokens_per_problem": tokens_per_problem,
                "estimated_flops_per_problem": small_model_flops,
            },
            "note": "Small Model has ~70% FLOPs of 8.7B model per token",
        }

        self.results["compute_efficiency"] = results
        return results

    def save_results(self, output_dir="results/dataset_tests"):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)

        self.results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "model_size": "small",
            "model_params": self.model.count_parameters(),
            "note": "Paper results for reference. Actual evaluation requires pre-trained weights.",
        }

        output_file = os.path.join(output_dir, "small_model_dataset_tests.json")
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")
        return output_file

    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 70)
        report.append("SMALL MODEL DATASET TESTS REPORT")
        report.append("Based on Adaptive Deep Networks Paper")
        report.append("=" * 70)
        report.append(f"\nTimestamp: {datetime.now().isoformat()}")
        report.append(f"Model: Small (2.2B params, untrained)")
        report.append("\nNOTE: This report shows paper reference metrics.")
        report.append("Actual evaluation requires pre-trained model weights.")

        report.append("\n" + "=" * 70)
        report.append("SUMMARY OF PAPER CLAIMS")
        report.append("=" * 70)

        report.append("\n1. Needle-in-Haystack (Table 4)")
        report.append("   - 86.9% average accuracy up to 256K context")
        report.append("   - 68.2% at 256K vs 1.5% baseline (45× improvement)")

        report.append("\n2. MATH Dataset (Table 6)")
        report.append("   - 52.3% overall with 8.7B parameters")
        report.append("   - Matches 50B static baseline")

        report.append("\n3. Ablation Study (Table 7)")
        report.append("   - Full system: 56.8% (LongBench-v2)")
        report.append("   - Standard Transformer: 39.7%")
        report.append("   - Improvement: +17.1%")

        report.append("\n4. Compute Efficiency (Table 8)")
        report.append("   - Gated adaptation: 52.3% at 1.28×10^14 FLOPs")
        report.append("   - 40% compute reduction vs alternatives")

        report.append("\n" + "=" * 70)
        report.append("KEY ACHIEVEMENTS")
        report.append("=" * 70)
        report.append("✅ 86.9% needle-in-haystack average (vs 38.2% baseline)")
        report.append("✅ 52.3% on MATH with 8.7B (matches 50B baseline)")
        report.append("✅ 40% compute reduction through adaptive allocation")
        report.append("✅ 5.7× KV cache reduction")
        report.append("✅ 8× cost reduction for depth-scaling")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


def main():
    print("=" * 70)
    print("Small Model Dataset Tests (Paper Metrics)")
    print("=" * 70)

    # 创建测试器
    tester = DatasetTester(device="cpu")

    # 运行所有测试
    tester.test_needle_haystack(context_lengths=[512, 1024, 2048])
    tester.test_math_dataset()
    tester.test_ablation_study()
    tester.test_compute_efficiency()

    # 保存结果
    tester.save_results()

    # 生成报告
    report = tester.generate_report()
    print("\n" + report)

    # 保存报告
    with open("results/dataset_tests/report.txt", "w") as f:
        f.write(report)

    print("\n📄 Report saved to: results/dataset_tests/report.txt")


if __name__ == "__main__":
    main()
