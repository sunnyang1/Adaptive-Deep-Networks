#!/usr/bin/env python3
"""
统一实验运行脚本

运行所有实验：核心实验、TurboQuant实验和基准测试
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import subprocess


# 实验定义
EXPERIMENTS = {
    'core': [
        {
            'id': 'exp1',
            'name': 'Representation Burial测量',
            'script': 'core/exp1_representation_burial/run_exp1.py',
            'quick_args': ['--num_samples', '10'],
            'full_args': ['--num_samples', '50'],
        },
        {
            'id': 'exp2',
            'name': 'Logit Margin分析',
            'script': 'core/exp2_margin_analysis/run_exp2.py',
            'quick_args': ['--context_lengths', '1024', '4096'],
            'full_args': ['--context_lengths', '1024', '4096', '16384'],
        },
        {
            'id': 'exp3',
            'name': '梯度流测量',
            'script': 'core/exp3_gradient_flow/run_exp3.py',
            'quick_args': ['--num_steps', '100'],
            'full_args': ['--num_steps', '1000'],
        },
        {
            'id': 'exp4',
            'name': 'FLOP等价验证',
            'script': 'core/exp4_flop_equivalence/run_exp4.py',
            'quick_args': ['--total_flops', '1e13'],
            'full_args': ['--total_flops', '5e14'],
        },
        {
            'id': 'exp5',
            'name': '组件协同效应',
            'script': 'core/exp5_synergy/run_exp5.py',
            'quick_args': [],
            'full_args': [],
        },
        {
            'id': 'exp6',
            'name': '辅助验证实验',
            'script': 'core/exp6_auxiliary/run_exp6.py',
            'quick_args': [],
            'full_args': [],
        },
    ],
    'turboquant': [
        {
            'id': 'tq_validation',
            'name': 'TurboQuant验证',
            'script': '../scripts/validate_turboquant_setup.py',
            'quick_args': [],
            'full_args': [],
        },
    ],
}


def run_experiment(exp, args, results_dir):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"运行: {exp['name']} ({exp['id']})")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / exp['script']
    if not script_path.exists():
        print(f"⚠️  脚本不存在: {script_path}")
        return {'status': 'skipped', 'reason': 'script_not_found'}
    
    # 选择参数
    exp_args = exp['quick_args'] if args.quick else exp['full_args']
    
    # 构建命令
    cmd = [sys.executable, str(script_path)] + exp_args
    if args.device:
        cmd.extend(['--device', args.device])
    
    print(f"命令: {' '.join(cmd)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout * 60 if args.timeout else None,
        )
        elapsed = time.time() - start_time
        
        # 保存输出
        output_file = results_dir / f"{exp['id']}_output.log"
        with open(output_file, 'w') as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print(f"✅ 成功 ({elapsed:.1f}s)")
            return {
                'status': 'success',
                'elapsed': elapsed,
                'output_file': str(output_file),
            }
        else:
            print(f"❌ 失败 (返回码: {result.returncode})")
            return {
                'status': 'failed',
                'returncode': result.returncode,
                'elapsed': elapsed,
                'output_file': str(output_file),
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  超时 (> {args.timeout}分钟)")
        return {'status': 'timeout', 'timeout': args.timeout}
    except Exception as e:
        print(f"❌ 异常: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description='运行所有实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行所有实验（完整模式）
  python run_all.py

  # 快速模式（减少计算量）
  python run_all.py --quick

  # 只运行核心实验
  python run_all.py --category core

  # 指定设备
  python run_all.py --device cpu

  # 设置超时（分钟）
  python run_all.py --timeout 30
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='快速模式：使用减少的参数')
    parser.add_argument('--category', choices=['core', 'turboquant', 'all'],
                       default='all', help='实验类别')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='运行设备')
    parser.add_argument('--timeout', type=int, default=60,
                       help='单个实验超时时间（分钟），默认60')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='结果输出目录')
    parser.add_argument('--list', action='store_true',
                       help='列出所有实验')
    
    args = parser.parse_args()
    
    # 列出实验
    if args.list:
        print("\n可用实验:\n")
        for cat, exps in EXPERIMENTS.items():
            print(f"【{cat.upper()}】")
            for exp in exps:
                print(f"  - {exp['id']}: {exp['name']}")
        print()
        return
    
    # 创建结果目录
    results_dir = Path(__file__).parent / args.output_dir / 'unified'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Adaptive Deep Networks - 统一实验运行")
    print(f"{'='*60}")
    print(f"模式: {'快速' if args.quick else '完整'}")
    print(f"类别: {args.category}")
    print(f"设备: {args.device or 'auto'}")
    print(f"结果目录: {results_dir}")
    print(f"{'='*60}\n")
    
    # 选择实验
    if args.category == 'all':
        categories = ['core', 'turboquant']
    else:
        categories = [args.category]
    
    # 运行实验
    all_results = {}
    total_start = time.time()
    
    for cat in categories:
        if cat not in EXPERIMENTS:
            continue
        
        print(f"\n{'#'*60}")
        print(f"# {cat.upper()} 实验")
        print(f"{'#'*60}")
        
        cat_results = {}
        for exp in EXPERIMENTS[cat]:
            result = run_experiment(exp, args, results_dir)
            cat_results[exp['id']] = result
        
        all_results[cat] = cat_results
    
    total_elapsed = time.time() - total_start
    
    # 生成摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick' if args.quick else 'full',
        'device': args.device or 'auto',
        'total_time': total_elapsed,
        'categories': {}
    }
    
    for cat, results in all_results.items():
        success = sum(1 for r in results.values() if r['status'] == 'success')
        failed = sum(1 for r in results.values() if r['status'] == 'failed')
        summary['categories'][cat] = {
            'total': len(results),
            'success': success,
            'failed': failed,
            'experiments': results
        }
    
    # 保存结果
    summary_file = results_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("实验完成摘要")
    print(f"{'='*60}")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print()
    
    for cat, stats in summary['categories'].items():
        print(f"【{cat.upper()}】")
        print(f"  成功: {stats['success']}/{stats['total']}")
        if stats['failed'] > 0:
            print(f"  失败: {stats['failed']}")
        for exp_id, result in stats['experiments'].items():
            status_icon = '✅' if result['status'] == 'success' else '❌'
            elapsed = result.get('elapsed', 0)
            print(f"    {status_icon} {exp_id}: {result['status']} ({elapsed:.1f}s)")
        print()
    
    print(f"详细结果: {summary_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
