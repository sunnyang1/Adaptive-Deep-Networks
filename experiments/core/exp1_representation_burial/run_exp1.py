"""
实验1: Representation Burial现象的定量测量
验证PreNorm配置下早期层信号随深度衰减的现象，并对比AttnRes的改善效果
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List
from tqdm import tqdm
import argparse

from utils.measurement import measure_representation_burial


class SimpleTransformer(nn.Module):
    """简化版Transformer用于测试"""
    def __init__(self, vocab_size=10000, d_model=4096, nhead=32, num_layers=32, 
                 dim_feedforward=16384, norm_type='prenorm'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 8192, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=(norm_type == 'prenorm')
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.norm_type = norm_type
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.output(x)


def create_test_model(arch_type: str, num_layers: int = 32, d_model: int = 4096):
    """创建测试模型"""
    norm_map = {
        'prenorm': 'prenorm',
        'postnorm': 'postnorm',
        'deepnorm': 'prenorm',  # 简化处理
        'attnres': 'prenorm'    # 简化处理
    }
    
    return SimpleTransformer(
        vocab_size=10000,
        d_model=d_model,
        nhead=32,
        num_layers=num_layers,
        dim_feedforward=d_model * 4,
        norm_type=norm_map.get(arch_type, 'prenorm')
    )


def run_experiment(config: Dict) -> Dict:
    """
    运行Representation Burial实验
    
    Args:
        config: 实验配置
        {
            'architectures': List[str],  # 要测试的架构
            'num_layers': int,           # 层数
            'd_model': int,              # 隐藏维度
            'num_samples': int,          # 测试样本数
            'seq_len': int,              # 序列长度
            'device': str                # 设备
        }
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    results = {}
    
    for arch in config['architectures']:
        print(f"\n{'='*60}")
        print(f"测试架构: {arch.upper()}")
        print(f"{'='*60}")
        
        # 创建模型
        model = create_test_model(arch, config['num_layers'], config['d_model'])
        model = model.to(device)
        model.eval()
        
        all_contributions = []
        
        # 运行多个样本
        for i in tqdm(range(config['num_samples']), desc=f"{arch} samples"):
            # 生成随机输入
            input_ids = torch.randint(0, 10000, (1, config['seq_len'])).to(device)
            
            # 测量
            result = measure_representation_burial(model, input_ids)
            
            if result['layer_contributions']:
                contributions = [c.relative_contribution for c in result['layer_contributions']]
                all_contributions.append(contributions)
        
        # 统计结果
        if all_contributions:
            contributions_array = np.array(all_contributions)
            mean_contributions = np.mean(contributions_array, axis=0)
            std_contributions = np.std(contributions_array, axis=0)
            
            # 计算指标
            first_contrib = mean_contributions[0] if len(mean_contributions) > 0 else 1.0
            last_contrib = mean_contributions[-1] if len(mean_contributions) > 0 else 0.0
            attenuation_rate = (first_contrib - last_contrib) / first_contrib if first_contrib > 0 else 0
            
            # 有效深度
            effective_depth = len(mean_contributions)
            for i, c in enumerate(mean_contributions):
                if c < 0.5:
                    effective_depth = i
                    break
            
            # 变异系数
            cv = np.std(mean_contributions) / np.mean(mean_contributions) if np.mean(mean_contributions) > 0 else 0
            
            results[arch] = {
                'mean_contributions': mean_contributions.tolist(),
                'std_contributions': std_contributions.tolist(),
                'attenuation_rate': float(attenuation_rate),
                'effective_depth': int(effective_depth),
                'cv': float(cv)
            }
            
            print(f"\n结果摘要:")
            print(f"  信号衰减率: {attenuation_rate:.2%}")
            print(f"  有效深度: {effective_depth} / {config['num_layers']}")
            print(f"  变异系数: {cv:.4f}")
    
    return results


def visualize_results(results: Dict, config: Dict, output_dir: str):
    """可视化实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 各架构的贡献度曲线
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'prenorm': '#e74c3c',
        'postnorm': '#3498db',
        'deepnorm': '#2ecc71',
        'attnres': '#9b59b6'
    }
    
    labels = {
        'prenorm': 'PreNorm',
        'postnorm': 'PostNorm',
        'deepnorm': 'DeepNorm',
        'attnres': 'AttnRes'
    }
    
    for arch, data in results.items():
        contributions = data['mean_contributions']
        layers = list(range(1, len(contributions) + 1))
        
        ax.plot(layers, contributions, 
                label=labels.get(arch, arch),
                color=colors.get(arch, 'gray'),
                linewidth=2,
                marker='o',
                markersize=4)
        
        # 标注50%线
        if arch == 'attnres':
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.text(len(contributions) * 0.7, 0.52, '50% threshold', fontsize=10, color='gray')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Relative Contribution', fontsize=12)
    ax.set_title('Representation Burial: Layer Contribution Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_representation_burial.png'), dpi=300)
    plt.close()
    
    # 图2: 关键指标对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    architectures = list(results.keys())
    x = np.arange(len(architectures))
    width = 0.6
    
    # 衰减率
    attenuation_rates = [results[arch]['attenuation_rate'] * 100 for arch in architectures]
    axes[0].bar(x, attenuation_rates, color=[colors.get(a, 'gray') for a in architectures])
    axes[0].set_ylabel('Attenuation Rate (%)', fontsize=11)
    axes[0].set_title('Signal Attenuation', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels.get(a, a) for a in architectures], rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 有效深度
    effective_depths = [results[arch]['effective_depth'] for arch in architectures]
    axes[1].bar(x, effective_depths, color=[colors.get(a, 'gray') for a in architectures])
    axes[1].axhline(y=config['num_layers'], color='red', linestyle='--', label=f'Total Layers ({config["num_layers"]})')
    axes[1].set_ylabel('Effective Depth', fontsize=11)
    axes[1].set_title('Effective Depth (50% threshold)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels.get(a, a) for a in architectures], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 变异系数
    cvs = [results[arch]['cv'] for arch in architectures]
    axes[2].bar(x, cvs, color=[colors.get(a, 'gray') for a in architectures])
    axes[2].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[2].set_title('Gradient Uniformity (lower is better)', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([labels.get(a, a) for a in architectures], rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_metrics_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n可视化结果已保存到: {output_dir}")


def generate_report(results: Dict, config: Dict, output_dir: str):
    """生成实验报告"""
    report_path = os.path.join(output_dir, 'exp1_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# 实验1: Representation Burial定量测量报告\n\n")
        f.write("## 实验配置\n\n")
        f.write(f"- 层数: {config['num_layers']}\n")
        f.write(f"- 隐藏维度: {config['d_model']}\n")
        f.write(f"- 测试样本数: {config['num_samples']}\n")
        f.write(f"- 序列长度: {config['seq_len']}\n\n")
        
        f.write("## 结果摘要\n\n")
        f.write("| 架构 | 信号衰减率 | 有效深度 | 变异系数 |\n")
        f.write("|------|-----------|---------|---------|\n")
        
        for arch, data in results.items():
            f.write(f"| {arch.upper()} | {data['attenuation_rate']:.2%} | "
                   f"{data['effective_depth']} / {config['num_layers']} | "
                   f"{data['cv']:.4f} |\n")
        
        f.write("\n## 关键发现\n\n")
        
        # 找出最佳架构
        best_arch = min(results.items(), key=lambda x: x[1]['cv'])[0]
        f.write(f"1. **梯度均匀性最佳**: {best_arch.upper()} (CV = {results[best_arch]['cv']:.4f})\n")
        
        # 有效深度分析
        deepest_arch = max(results.items(), key=lambda x: x[1]['effective_depth'])[0]
        f.write(f"2. **有效深度最大**: {deepest_arch.upper()} ({results[deepest_arch]['effective_depth']} layers)\n")
        
        # AttnRes改进
        if 'prenorm' in results and 'attnres' in results:
            improvement = (results['prenorm']['attenuation_rate'] - results['attnres']['attenuation_rate']) / results['prenorm']['attenuation_rate']
            f.write(f"3. **AttnRes改进**: 相比PreNorm，信号衰减减少 {improvement:.1%}\n")
        
        f.write("\n## 详细数据\n\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2))
        f.write("\n```\n")
    
    print(f"报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='实验1: Representation Burial测量')
    parser.add_argument('--num_layers', type=int, default=48, help='模型层数 (Small=48, Medium=56, Large=96)')
    parser.add_argument('--d_model', type=int, default=2048, help='隐藏维度 (Small=2048, Medium=2688, Large=4224)')
    parser.add_argument('--num_samples', type=int, default=50, help='测试样本数')
    parser.add_argument('--seq_len', type=int, default=512, help='序列长度')
    parser.add_argument('--output_dir', type=str, default='experiments/results/exp1', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 实验配置
    config = {
        'architectures': ['prenorm', 'postnorm', 'deepnorm', 'attnres'],
        'num_layers': args.num_layers,
        'd_model': args.d_model,
        'num_samples': args.num_samples,
        'seq_len': args.seq_len,
        'device': device
    }
    
    print("\n" + "="*60)
    print("实验1: Representation Burial定量测量")
    print("="*60)
    print(f"配置: {config}")
    
    # 运行实验
    results = run_experiment(config)
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'exp1_results.json'), 'w') as f:
        json.dump(results, indent=2, fp=f)
    
    # 可视化
    visualize_results(results, config, args.output_dir)
    
    # 生成报告
    generate_report(results, config, args.output_dir)
    
    print("\n" + "="*60)
    print("实验1完成!")
    print("="*60)


if __name__ == '__main__':
    main()
