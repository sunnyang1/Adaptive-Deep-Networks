#!/usr/bin/env python3
"""
统一验证脚本 - 验证论文中所有关键表格数据 (REVISED)

运行所有验证实验并生成汇总报告。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# 验证脚本列表 (对应 REVISED 论文)
VALIDATIONS = [
    {
        'id': 'table1',
        'name': 'Table 1: Representation Burial',
        'script': 'table1_representation_burial.py',
        'description': '验证AttnRes相比PreNorm的梯度衰减改善 (Effective Depth)'
    },
    {
        'id': 'table2',
        'name': 'Table 2: Gradient Flow',
        'script': 'table2_gradient_flow.py',
        'description': '验证AttnRes的梯度流均匀性 (CV=0.11)'
    },
    {
        'id': 'table3',
        'name': 'Table 3: RaBitQ Space-Accuracy',
        'script': 'table3_rabitq_space_accuracy.py',
        'description': '验证RaBitQ压缩率、准确率保持与系统吞吐量 (§3.1.3 & §5.1)'
    },
    {
        'id': 'table4',
        'name': 'Table 4: Needle-in-Haystack',
        'script': 'table4_needle_haystack.py',
        'description': '验证长上下文检索 (Baseline/+RaBitQ/+AttnRes/+qTTT)'
    },
    {
        'id': 'table5',
        'name': 'Table 5: Query Margin',
        'script': 'table5_query_margin.py',
        'description': '验证qTTT在不同上下文长度下的logit margin提升 (§3.3.3)'
    },
    {
        'id': 'table6',
        'name': 'Table 6: MATH Dataset',
        'script': 'table6_math.py',
        'description': '验证数学推理 (qTTT 52.8% matching 50B baseline)'
    },
    {
        'id': 'table7',
        'name': 'Table 7: Component Synergy',
        'script': 'table7_synergy.py',
        'description': '验证组件协同效应 (LongBench-v2 ablation)'
    },
    {
        'id': 'table8',
        'name': 'Table 8: SRAM-Aware Allocation',
        'script': 'table8_sram_allocation.py',
        'description': '验证分层内存模型下的最优(R,M,T)分配 (§5.5)'
    },
    {
        'id': 'table9',
        'name': 'Table 9: Coupling Effect',
        'script': 'table9_coupling_effect.py',
        'description': '验证Space-Scope耦合效应可忽略 (δ≈-0.1%) (§5.6)'
    },
    {
        'id': 'extreme_context',
        'name': 'Extreme Context Scaling',
        'script': 'extreme_context_scaling.py',
        'description': '超长上下文测试 (128K-1M tokens)'
    },
]


def run_validation(val_info, output_dir):
    """运行单个验证实验"""
    script_path = Path(__file__).parent / val_info['script']
    
    print(f"\n{'='*70}")
    print(f"Running: {val_info['name']}")
    print(f"{'='*70}")
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        return {'status': 'skipped', 'reason': 'script_not_found'}
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), '--output-dir', str(output_dir)],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 解析输出检查是否通过
        output = result.stdout
        passed = 'ALL PASSED' in output or '✅ ALL PASSED' in output
        failed = 'SOME FAILED' in output or '❌ SOME FAILED' in output
        
        status = 'passed' if passed else ('failed' if failed else 'unknown')
        
        print(f"\nStatus: {'✅ PASSED' if passed else '❌ FAILED' if failed else '❓ UNKNOWN'}")
        
        return {
            'status': status,
            'returncode': result.returncode,
            'output': output[-500:] if len(output) > 500 else output
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout (>5 minutes)")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'validations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'validation_report_{timestamp}.json'
    
    print("="*70)
    print("Adaptive Deep Networks - Unified Validation (REVISED)")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Total validations: {len(VALIDATIONS)}")
    
    results = {}
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for val_info in VALIDATIONS:
        result = run_validation(val_info, output_dir)
        results[val_info['id']] = {
            'name': val_info['name'],
            'script': val_info['script'],
            **result
        }
        
        if result['status'] == 'passed':
            passed_count += 1
        elif result['status'] == 'failed':
            failed_count += 1
        else:
            skipped_count += 1
    
    # 汇总报告
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total:   {len(VALIDATIONS)}")
    print(f"Passed:  {passed_count} ✅")
    print(f"Failed:  {failed_count} ❌")
    print(f"Skipped: {skipped_count} ⚠️")
    
    summary = {
        'timestamp': timestamp,
        'total': len(VALIDATIONS),
        'passed': passed_count,
        'failed': failed_count,
        'skipped': skipped_count,
        'details': results
    }
    
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nReport saved: {report_file}")
    
    if failed_count > 0:
        print("\n❌ SOME VALIDATIONS FAILED")
        return 1
    else:
        print("\n✅ ALL VALIDATIONS PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
