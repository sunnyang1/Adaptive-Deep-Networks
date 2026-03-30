#!/usr/bin/env python3
"""
统一验证脚本 - 验证论文中所有关键表格数据

运行所有验证实验并生成汇总报告。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# 验证脚本列表
VALIDATIONS = [
    {
        'id': 'table1',
        'name': 'Table 1: Representation Burial',
        'script': 'table1_representation_burial.py',
        'description': '验证AttnRes相比PreNorm的梯度衰减改善'
    },
    {
        'id': 'table2',
        'name': 'Table 2: Gradient Flow',
        'script': 'table2_gradient_flow.py',
        'description': '验证AttnRes的梯度流均匀性 (CV改善7.6x)'
    },
    {
        'id': 'turboquant',
        'name': 'TurboQuant Compression',
        'script': 'turboquant_compression.py',
        'description': '验证6x+压缩比和零精度损失'
    },
    {
        'id': 'table4',
        'name': 'Table 4: Needle-in-Haystack',
        'script': 'table4_needle_haystack.py',
        'description': '验证长上下文检索 (平均86.9%, 256K达68.2%)'
    },
    {
        'id': 'table6',
        'name': 'Table 6: MATH Dataset',
        'script': 'table6_math.py',
        'description': '验证数学推理 (52.3%匹配50B baseline)'
    },
    {
        'id': 'table7',
        'name': 'Table 7: Component Synergy',
        'script': 'table7_synergy.py',
        'description': '验证组件协同效应 (系数1.18)'
    },
    {
        'id': 'extreme_context',
        'name': 'Extreme Context Scaling (up to 1M)',
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
            'output': output[-500:] if len(output) > 500 else output  # 最后500字符
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout (>5 minutes)")
        return {'status': 'timeout'}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    print("="*70)
    print("Adaptive Deep Networks - Paper Validation Suite")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total validations: {len(VALIDATIONS)}")
    
    output_dir = Path(__file__).parent.parent / 'results' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for val_info in VALIDATIONS:
        result = run_validation(val_info, output_dir)
        results[val_info['id']] = {
            'name': val_info['name'],
            'description': val_info['description'],
            'result': result
        }
    
    # 生成汇总
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if r['result']['status'] == 'passed')
    failed = sum(1 for r in results.values() if r['result']['status'] == 'failed')
    skipped = sum(1 for r in results.values() if r['result']['status'] == 'skipped')
    
    print(f"\nTotal: {len(VALIDATIONS)} validations")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⚠️  Skipped: {skipped}")
    
    print(f"\n{'ID':<12} {'Status':<10} {'Name':<40}")
    print("-" * 70)
    
    for val_id, info in results.items():
        status = info['result']['status']
        icon = {'passed': '✅', 'failed': '❌', 'skipped': '⚠️', 'timeout': '⏱️', 'error': '💥'}.get(status, '❓')
        print(f"{val_id:<12} {icon} {status:<8} {info['name']:<40}")
    
    # 保存汇总
    summary = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total': len(VALIDATIONS),
            'passed': passed,
            'failed': failed,
            'skipped': skipped
        },
        'results': results,
        'overall_passed': passed == len(VALIDATIONS)
    }
    
    summary_file = output_dir / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Detailed report: {summary_file}")
    
    # 最终状态
    print("\n" + "="*70)
    if passed == len(VALIDATIONS):
        print("✅ ALL VALIDATIONS PASSED - Paper claims verified!")
    elif passed >= len(VALIDATIONS) * 0.8:
        print(f"⚠️  MOSTLY PASSED ({passed}/{len(VALIDATIONS)}) - Minor discrepancies")
    else:
        print(f"❌ VALIDATIONS FAILED ({failed}/{len(VALIDATIONS)}) - Review needed")
    print("="*70)
    
    return 0 if passed == len(VALIDATIONS) else 1


if __name__ == '__main__':
    sys.exit(main())
