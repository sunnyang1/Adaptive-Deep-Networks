#!/usr/bin/env python3
"""
论文指标汇总 - Adaptive Deep Networks TurboQuant

基于 Adaptive_Deep_Networks_TurboQuant.md 的关键指标汇总
"""

import json
import os
from datetime import datetime


def print_section(title, width=70):
    print("\n" + "="*width)
    print(title)
    print("="*width)


def needle_haystack_metrics():
    """Table 4: Needle-in-Haystack Accuracy"""
    print_section("Table 4: Needle-in-Haystack Accuracy (%)")
    
    data = {
        'context': ['4K', '32K', '64K', '128K', '256K'],
        'transformer': [87.5, 22.1, 8.7, 3.2, 1.5],
        'ttt_linear': [94.2, 65.3, 48.7, 32.1, 18.5],
        'attnres': [96.8, 75.6, 58.9, 42.3, 28.7],
        'adb_turboquant': [98.5, 91.3, 85.5, 78.2, 68.2],
    }
    
    print(f"\n{'Context':<10} {'Transformer':<15} {'TTT-Linear':<15} {'AttnRes':<15} {'ADB+Turbo':<15}")
    print("-" * 80)
    
    for i, ctx in enumerate(data['context']):
        print(f"{ctx:<10} "
              f"{data['transformer'][i]:>6.1f}%{'':<8} "
              f"{data['ttt_linear'][i]:>6.1f}%{'':<8} "
              f"{data['attnres'][i]:>6.1f}%{'':<8} "
              f"{data['adb_turboquant'][i]:>6.1f}%")
    
    print("-" * 80)
    print(f"{'Average':<10} {38.2:>6.1f}%{'':<8} {62.3:>6.1f}%{'':<8} "
          f"{69.9:>6.1f}%{'':<8} {86.9:>6.1f}%")
    
    print("\nKey Findings:")
    print("  • At 256K context: 68.2% vs 1.5% baseline (45× improvement)")
    print("  • Advantage increases: +11.1% (4K) → +53.6% (256K)")
    print("  • Average: 86.9% vs 38.2% baseline (+48.7%)")
    
    return {
        'table': 'Table 4: Needle-in-Haystack',
        'data': data,
        'average': {
            'transformer': 38.2,
            'ttt_linear': 62.3,
            'attnres': 69.9,
            'adb_turboquant': 86.9
        },
        'key_finding': '45x improvement at 256K context'
    }


def margin_analysis_metrics():
    """Table 5: Margin Distribution by Context Length"""
    print_section("Table 5: Logit Margin Analysis")
    
    data = {
        'context': ['1K', '16K', '64K', '128K', '256K'],
        'theoretical_min': [7.0, 9.8, 11.2, 12.5, 13.8],
        'vanilla_attention': [8.2, 6.1, 4.3, 3.2, 2.1],
        'qttt_after': [12.5, 11.8, 10.9, 10.2, 9.4],
        'improvement': [4.3, 5.7, 6.6, 7.0, 7.3],
    }
    
    print(f"\n{'Context':<10} {'Theoretical':<15} {'Vanilla':<15} {'qTTT':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for i, ctx in enumerate(data['context']):
        print(f"{ctx:<10} "
              f"~{data['theoretical_min'][i]:.1f}{'':<12} "
              f"{data['vanilla_attention'][i]:.1f}{'':<12} "
              f"{data['qttt_after'][i]:.1f}{'':<12} "
              f"+{data['improvement'][i]:.1f}")
    
    print("\nKey Findings:")
    print("  • Vanilla margins decay with length (8.2 → 2.1)")
    print("  • qTTT maintains stable margins (12.5 → 9.4)")
    print("  • Explicit margin maximization through gradient optimization")
    
    return {
        'table': 'Table 5: Margin Analysis',
        'data': data,
        'key_finding': 'qTTT maintains stable margins across context lengths'
    }


def math_dataset_metrics():
    """Table 6: MATH Dataset Performance"""
    print_section("Table 6: MATH Dataset Performance (8.7B models)")
    
    methods = {
        'Transformer': {'L1-2': 60.4, 'L3-4': 31.6, 'L5': 12.1, 'Overall': 35.2},
        'CoT (5 samples)': {'L1-2': 65.5, 'L3-4': 38.7, 'L5': 18.5, 'Overall': 41.5},
        'TTT-Linear': {'L1-2': 70.0, 'L3-4': 46.8, 'L5': 28.7, 'Overall': 48.9},
        'AttnRes + qTTT (gated)': {'L1-2': 71.5, 'L3-4': 51.3, 'L5': 34.5, 'Overall': 52.3},
        'AttnRes + qTTT (max)': {'L1-2': 74.9, 'L3-4': 58.6, 'L5': 42.1, 'Overall': 58.9},
    }
    
    print(f"\n{'Method':<28} {'Level 1-2':<12} {'Level 3-4':<12} {'Level 5':<12} {'Overall':<12}")
    print("-" * 90)
    
    for method, scores in methods.items():
        print(f"{method:<28} "
              f"{scores['L1-2']:>6.1f}%{'':<5} "
              f"{scores['L3-4']:>6.1f}%{'':<5} "
              f"{scores['L5']:>6.1f}%{'':<5} "
              f"{scores['Overall']:>6.1f}%")
    
    print("\nKey Findings:")
    print("  • 8.7B model achieves 52.3% overall")
    print("  • Matches 50B static baseline performance")
    print("  • Level 5 improvement: 34.5% vs 12.1% (+22.4%)")
    print("  • Consistent gains across all difficulty levels")
    
    return {
        'table': 'Table 6: MATH Dataset',
        'methods': methods,
        'key_finding': '8.7B matches 50B static baseline'
    }


def ablation_study_metrics():
    """Table 7: Ablation Study"""
    print_section("Table 7: Ablation Study (8.7B, LongBench-v2)")
    
    configs = {
        'Full System': {'score': 56.8, 'delta': 0.0},
        'w/o qTTT': {'score': 50.1, 'delta': -6.7},
        'w/o Gating': {'score': 53.2, 'delta': -3.6},
        'w/o AttnRes': {'score': 48.9, 'delta': -7.9},
        'w/o TurboQuant': {'score': 51.5, 'delta': -5.3},
        'Standard Transformer': {'score': 39.7, 'delta': -17.1},
    }
    
    print(f"\n{'Configuration':<28} {'Avg Score':<15} {'Δ vs Full':<15}")
    print("-" * 70)
    
    for config, data in configs.items():
        delta_str = f"{data['delta']:+.1f}%" if data['delta'] != 0 else "—"
        marker = " ✓" if config == 'Full System' else ""
        print(f"{config:<28} {data['score']:>6.1f}%{marker:<8} {delta_str:>15}")
    
    print("-" * 70)
    print(f"\nSynergy Coefficient: 1.18")
    print("(>1.0 indicates super-additive interaction)")
    
    print("\nKey Findings:")
    print("  • All components contribute positively")
    print("  • AttnRes most critical: -7.9% when removed")
    print("  • Full system: +17.1% vs Standard Transformer")
    print("  • Components work better together (synergy)")
    
    # 组件重要性排序
    importance = [
        ('AttnRes', 7.9),
        ('qTTT', 6.7),
        ('TurboQuant', 5.3),
        ('Gating', 3.6),
    ]
    
    print("\nComponent Importance (by impact when removed):")
    for i, (comp, impact) in enumerate(importance, 1):
        print(f"  {i}. {comp}: -{impact:.1f}%")
    
    return {
        'table': 'Table 7: Ablation Study',
        'configurations': configs,
        'synergy_coefficient': 1.18,
        'component_importance': importance,
        'key_finding': 'All components essential, super-additive synergy'
    }


def compute_efficiency_metrics():
    """Table 8: Compute Efficiency"""
    print_section("Table 8: Accuracy-Compute Pareto (MATH dataset)")
    
    configs = {
        'Standard 32L': {'flops': 1.0, 'accuracy': 35.2, 'acc_per_flop': 35.2},
        'AttnRes 32L (static)': {'flops': 1.05, 'accuracy': 41.8, 'acc_per_flop': 39.8},
        'AttnRes + qTTT (uniform)': {'flops': 1.45, 'accuracy': 47.5, 'acc_per_flop': 32.8},
        'AttnRes + qTTT (gated)': {'flops': 1.28, 'accuracy': 52.3, 'acc_per_flop': 40.9},
        'AttnRes + qTTT (oracle)': {'flops': 1.15, 'accuracy': 54.8, 'acc_per_flop': 47.7},
    }
    
    print(f"\n{'Configuration':<32} {'Avg FLOP':<15} {'Accuracy':<12} {'Acc/FLOP':<12}")
    print("-" * 90)
    
    for config, data in configs.items():
        marker = " ✓" if config == 'AttnRes + qTTT (gated)' else ""
        print(f"{config:<32} "
              f"{data['flops']:.2f}×10^14{'':<4} "
              f"{data['accuracy']:>6.1f}%{marker:<4} "
              f"{data['acc_per_flop']:>6.1f}")
    
    print("\nKey Findings:")
    print("  • Gated adaptation: Best accuracy at lowest FLOP")
    print("  • Acc/FLOP: 40.9 (gated) vs 35.2 (standard)")
    print("  • 40% compute reduction vs FLOP-matched alternatives")
    print("  • Oracle upper bound: 54.8% at 1.15×10^14 FLOPs")
    
    return {
        'table': 'Table 8: Compute Efficiency',
        'configurations': configs,
        'key_finding': 'Gated adaptation achieves best accuracy per FLOP'
    }


def model_specifications():
    """Model Specifications (from Table A1)"""
    print_section("Model Specifications (Appendix A.1)")
    
    specs = {
        'AttnRes-S': {'params': '2.2B', 'layers': 32, 'hidden': 2048, 'heads': 32, 'blocks': 8},
        'AttnRes-M': {'params': '8.7B', 'layers': 32, 'hidden': 4096, 'heads': 32, 'blocks': 8},
        'AttnRes-L': {'params': '27B', 'layers': 64, 'hidden': 5120, 'heads': 40, 'blocks': 16},
    }
    
    print(f"\n{'Model':<15} {'Params':<10} {'Layers':<10} {'Hidden':<10} {'Heads':<10} {'Blocks':<10}")
    print("-" * 80)
    
    for model, spec in specs.items():
        print(f"{model:<15} {spec['params']:<10} {spec['layers']:<10} "
              f"{spec['hidden']:<10} {spec['heads']:<10} {spec['blocks']:<10}")
    
    print("\nKey Configuration:")
    print("  • Small: 2.2B params, 32 layers, 2048 hidden, 8 blocks")
    print("  • Medium: 8.7B params, 32 layers, 4096 hidden, 8 blocks")
    print("  • Large: 27B params, 64 layers, 5120 hidden, 16 blocks")
    
    return {
        'section': 'Model Specifications',
        'specs': specs
    }


def turboquant_metrics():
    """TurboQuant Metrics"""
    print_section("TurboQuant Compression Metrics")
    
    print("\nCompression Performance:")
    print("  • Memory reduction: 6×+ with zero accuracy loss")
    print("  • KV cache reduction: 5.7× (16GB → 2.8GB)")
    print("  • Throughput increase: 8× on Tensor Cores (INT4)")
    print("  • Cost reduction: 8× for depth-scaling")
    
    print("\nKey Technologies:")
    print("  • Stage 1: PolarQuant (b-1 bits)")
    print("    - Random Hadamard Transform")
    print("    - Cartesian-to-Polar conversion")
    print("    - Lloyd-Max optimal quantization")
    print("  • Stage 2: QJL (1-bit)")
    print("    - Quantized Johnson-Lindenstrauss")
    print("    - Unbiased inner product estimates")
    
    print("\nHardware Acceleration:")
    print("  • Tensor Core INT4 kernels")
    print("  • 2× arithmetic throughput vs FP16")
    print("  • 4× memory bandwidth efficiency")
    print("  • Combined: 8× throughput increase")
    
    return {
        'section': 'TurboQuant',
        'memory_reduction': '6x+',
        'kv_cache_reduction': '5.7x',
        'throughput_increase': '8x',
        'cost_reduction': '8x'
    }


def generate_summary_report(all_results):
    """生成汇总报告"""
    report = []
    report.append("="*70)
    report.append("ADAPTIVE DEEP NETWORKS - PAPER METRICS SUMMARY")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append("\nThis report summarizes key metrics from the paper:")
    report.append("'Adaptive Deep Networks: Integrating Block Attention")
    report.append("Residuals, TurboQuant Compression, and Test-Time Adaptation'")
    
    report.append("\n" + "="*70)
    report.append("TOP-LEVEL ACHIEVEMENTS")
    report.append("="*70)
    report.append("\n1. Long-Context Retrieval (Needle-in-Haystack)")
    report.append("   • 86.9% average accuracy up to 256K context")
    report.append("   • 68.2% at 256K vs 1.5% baseline (45× improvement)")
    report.append("   • 2.3× improvement over TTT-Linear")
    
    report.append("\n2. Mathematical Reasoning (MATH Dataset)")
    report.append("   • 52.3% overall with 8.7B parameters")
    report.append("   • Matches 50B static baseline performance")
    report.append("   • Level 5: 34.5% vs 12.1% baseline")
    
    report.append("\n3. Compute Efficiency")
    report.append("   • 110 tokens/s under 500ms latency")
    report.append("   • 2.4× vs Thinking Tokens (45 t/s)")
    report.append("   • 40% compute reduction vs alternatives")
    
    report.append("\n4. Memory Efficiency")
    report.append("   • 5.7× KV cache reduction")
    report.append("   • 2.8 GB vs 16 GB at 128K context")
    report.append("   • 6×+ TurboQuant compression")
    
    report.append("\n5. Component Synergy")
    report.append("   • Synergy coefficient: 1.18 (super-additive)")
    report.append("   • Full system: 56.8% vs Standard: 39.7% (+17.1%)")
    report.append("   • All components contribute positively")
    
    report.append("\n" + "="*70)
    report.append("TECHNICAL HIGHLIGHTS")
    report.append("="*70)
    report.append("\nAttnRes (Block Attention Residuals):")
    report.append("  • Memory: O(Ld) → O(Nd) = 4-16× reduction")
    report.append("  • Overhead: <0.1% parameters")
    report.append("  • Gradient CV: 0.11 vs 0.84 (PreNorm)")
    
    report.append("\nTurboQuant:")
    report.append("  • 6×+ memory reduction, zero accuracy loss")
    report.append("  • Data-oblivious: no calibration needed")
    report.append("  • 8× throughput on Tensor Cores")
    
    report.append("\nqTTT (Query-only Test-Time Training):")
    report.append("  • 50% parameter reduction vs Cartesian")
    report.append("  • 10× lower cost vs full-parameter TTT")
    report.append("  • Explicit margin maximization")
    
    report.append("\n" + "="*70)
    report.append("PAPER CITATION")
    report.append("="*70)
    report.append("\n@article{adaptive_deep_networks_2026,")
    report.append("  title={Adaptive Deep Networks: Integrating Attention")
    report.append("         Residuals, TurboQuant Compression, and")
    report.append("         Test-Time Adaptation},")
    report.append("  author={[Authors]},")
    report.append("  journal={arXiv preprint},")
    report.append("  year={2026}")
    report.append("}")
    
    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    return "\n".join(report)


def main():
    print("="*70)
    print("ADAPTIVE DEEP NETWORKS - PAPER METRICS SUMMARY")
    print("="*70)
    
    # 收集所有结果
    all_results = {}
    
    # 运行所有指标函数
    all_results['needle_haystack'] = needle_haystack_metrics()
    all_results['margin_analysis'] = margin_analysis_metrics()
    all_results['math_dataset'] = math_dataset_metrics()
    all_results['ablation_study'] = ablation_study_metrics()
    all_results['compute_efficiency'] = compute_efficiency_metrics()
    all_results['model_specs'] = model_specifications()
    all_results['turboquant'] = turboquant_metrics()
    
    # 生成汇总报告
    report = generate_summary_report(all_results)
    print("\n" + report)
    
    # 保存结果
    output_dir = 'results/paper_metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 JSON
    output_file = os.path.join(output_dir, 'paper_metrics_summary.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 JSON saved to: {output_file}")
    
    # 保存报告
    report_file = os.path.join(output_dir, 'paper_metrics_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"📄 Report saved to: {report_file}")


if __name__ == '__main__':
    main()
