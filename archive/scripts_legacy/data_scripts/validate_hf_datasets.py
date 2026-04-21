#!/usr/bin/env python3
"""
HuggingFace 数据集验证 - 绕过 _lzma 限制
直接使用 HTTP API 检查数据集可用性
"""

import json
import urllib.request
import urllib.error
from datetime import datetime

HF_API_BASE = "https://huggingface.co/api/datasets"


def check_dataset(repo_id, config=None, split=None):
    """检查 HuggingFace 数据集可用性"""
    try:
        url = f"{HF_API_BASE}/{repo_id}"
        req = urllib.request.Request(url, method='GET')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            result = {
                'status': 'AVAILABLE',
                'repo_id': repo_id,
                'id': data.get('id'),
                'downloads': data.get('downloads', 0),
                'likes': data.get('likes', 0),
                'tags': data.get('tags', [])
            }
            
            # 检查 splits
            if 'splits' in data:
                result['splits'] = data['splits']
            
            return result
            
    except urllib.error.HTTPError as e:
        return {'status': 'ERROR', 'repo_id': repo_id, 'error': f'HTTP {e.code}'}
    except Exception as e:
        return {'status': 'ERROR', 'repo_id': repo_id, 'error': str(e)}


def validate_all_datasets():
    """验证所有论文 5.2 节数据集"""
    
    print("=" * 70)
    print("HuggingFace 数据集可用性验证 (HTTP API)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 定义要验证的数据集
    datasets = {
        'Long-Context Retrieval': [
            ('THUDM/LongBench-v2', 'LongBench-v2'),
        ],
        'Mathematical Reasoning': [
            ('hendrycks/competition_math', 'MATH'),
            ('openai/gsm8k', 'GSM8K'),
        ],
        'General Tasks': [
            ('Rowan/hellaswag', 'HellaSwag'),
            ('allenai/ai2_arc', 'ARC-Challenge'),
            ('openai/openai_humaneval', 'HumanEval'),
            ('lukaemon/bbh', 'BBH (Big-Bench Hard)'),
        ]
    }
    
    results = {}
    
    for category, ds_list in datasets.items():
        print(f"\n{'=' * 70}")
        print(f"📂 {category}")
        print("=" * 70)
        
        results[category] = []
        
        for repo_id, name in ds_list:
            print(f"\n🔍 检查: {name}")
            print(f"   Repo: {repo_id}")
            
            result = check_dataset(repo_id)
            results[category].append(result)
            
            if result['status'] == 'AVAILABLE':
                print(f"   ✅ 可用")
                print(f"   📊 Downloads: {result.get('downloads', 'N/A')}")
                print(f"   👍 Likes: {result.get('likes', 'N/A')}")
                if 'splits' in result:
                    print(f"   📁 Splits: {result['splits']}")
            else:
                print(f"   ❌ 错误: {result.get('error', 'Unknown')}")
    
    # 保存结果
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': generate_summary(results)
    }
    
    with open('results/hf_validation_api.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\n{'=' * 70}")
    print("验证报告已保存: results/hf_validation_api.json")
    print("=" * 70)
    
    return results


def generate_summary(results):
    """生成验证摘要"""
    total = 0
    available = 0
    
    for category, ds_list in results.items():
        for result in ds_list:
            total += 1
            if result['status'] == 'AVAILABLE':
                available += 1
    
    return {
        'total': total,
        'available': available,
        'success_rate': f"{available/total*100:.1f}%"
    }


if __name__ == '__main__':
    validate_all_datasets()
