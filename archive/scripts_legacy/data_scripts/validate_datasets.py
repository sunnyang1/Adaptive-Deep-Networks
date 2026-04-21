#!/usr/bin/env python3
"""
数据集验证框架 - 根据论文 5.2 节
===============================

验证所有评估数据集的完整性、格式正确性和可用性。
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, 'src')
sys.path.insert(0, 'data/zero_scrolls')

class DatasetValidator:
    """数据集验证器"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def log(self, message, level='INFO'):
        """记录日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")
        
    def validate_zero_scrolls(self):
        """验证 ZeroScrolls 数据集"""
        self.log("=" * 70)
        self.log("验证 ZeroScrolls 数据集")
        self.log("=" * 70)
        
        try:
            from zero_scrolls_loader import ZeroScrollsDataset
        except ImportError:
            self.log("zero_scrolls_loader 未找到，尝试直接导入...", 'WARN')
            return None
            
        dataset = ZeroScrollsDataset('data/zero_scrolls')
        available_tasks = dataset.list_available_tasks()
        
        results = {
            'total_tasks': 10,
            'available_tasks': len(available_tasks),
            'tasks': {},
            'issues': []
        }
        
        self.log(f"可用任务: {len(available_tasks)}/10")
        
        for task in available_tasks:
            self.log(f"\n验证任务: {task}")
            task_result = {'status': 'OK'}
            
            try:
                # 检查是否能加载数据
                test_data = dataset.load_task(task, 'test')
                val_data = dataset.load_task(task, 'validation')
                
                task_result['test_samples'] = len(test_data)
                task_result['val_samples'] = len(val_data)
                
                # 验证数据格式
                if test_data:
                    sample = test_data[0]
                    required_fields = ['input', 'output', 'id']
                    missing_fields = [f for f in required_fields if f not in sample]
                    
                    if missing_fields:
                        task_result['status'] = 'ERROR'
                        task_result['missing_fields'] = missing_fields
                        results['issues'].append(f"{task}: 缺失字段 {missing_fields}")
                    else:
                        # 检查文本长度
                        input_len = len(sample['input'].split())
                        task_result['sample_input_len'] = input_len
                        
                        # 特别检查 narrative_qa 的长上下文
                        if task == 'narrative_qa' and input_len < 10000:
                            results['issues'].append(
                                f"{task}: 输入长度异常短 ({input_len} tokens)"
                            )
                            
                self.log(f"  ✓ Test: {len(test_data)} 样本")
                self.log(f"  ✓ Validation: {len(val_data)} 样本")
                
            except Exception as e:
                task_result['status'] = 'ERROR'
                task_result['error'] = str(e)
                results['issues'].append(f"{task}: {str(e)}")
                self.log(f"  ✗ 错误: {str(e)[:100]}", 'ERROR')
                
            results['tasks'][task] = task_result
        
        # 检查缺失的任务
        all_tasks = set(ZeroScrollsDataset.TASKS)
        available = set(available_tasks)
        missing = all_tasks - available
        if missing:
            results['missing_tasks'] = list(missing)
            self.log(f"\n缺失任务: {missing}", 'WARN')
        
        self.results['zero_scrolls'] = results
        return results
    
    def validate_math_datasets(self):
        """验证 MATH 和 GSM8K 数据集"""
        self.log("\n" + "=" * 70)
        self.log("验证数学推理数据集 (MATH & GSM8K)")
        self.log("=" * 70)
        
        results = {'math': {}, 'gsm8k': {}}
        
        # 尝试通过 HuggingFace 验证
        try:
            from datasets import load_dataset
            
            # 验证 MATH
            self.log("\n验证 MATH...")
            try:
                math_train = load_dataset('hendrycks/competition_math', 
                                          split='train', 
                                          trust_remote_code=True)
                math_test = load_dataset('hendrycks/competition_math', 
                                         split='test', 
                                         trust_remote_code=True)
                
                results['math'] = {
                    'status': 'AVAILABLE_HF',
                    'train_samples': len(math_train),
                    'test_samples': len(math_test),
                    'fields': list(math_train[0].keys()) if len(math_train) > 0 else []
                }
                
                # 验证难度级别
                levels = set()
                for item in math_test:
                    levels.add(item.get('level', 'unknown'))
                results['math']['difficulty_levels'] = sorted(levels)
                
                self.log(f"  ✓ Train: {len(math_train)}, Test: {len(math_test)}")
                self.log(f"  ✓ 难度级别: {sorted(levels)}")
                
            except Exception as e:
                results['math'] = {'status': 'ERROR', 'error': str(e)}
                self.log(f"  ✗ MATH 加载失败: {str(e)[:100]}", 'ERROR')
            
            # 验证 GSM8K
            self.log("\n验证 GSM8K...")
            try:
                gsm8k_train = load_dataset('openai/gsm8k', 'main', 
                                           split='train', 
                                           trust_remote_code=True)
                gsm8k_test = load_dataset('openai/gsm8k', 'main', 
                                          split='test', 
                                          trust_remote_code=True)
                
                results['gsm8k'] = {
                    'status': 'AVAILABLE_HF',
                    'train_samples': len(gsm8k_train),
                    'test_samples': len(gsm8k_test),
                    'fields': list(gsm8k_train[0].keys()) if len(gsm8k_train) > 0 else []
                }
                
                self.log(f"  ✓ Train: {len(gsm8k_train)}, Test: {len(gsm8k_test)}")
                
            except Exception as e:
                results['gsm8k'] = {'status': 'ERROR', 'error': str(e)}
                self.log(f"  ✗ GSM8K 加载失败: {str(e)[:100]}", 'ERROR')
                
        except ImportError:
            self.log("datasets 库未安装，跳过 HuggingFace 验证", 'WARN')
            results['note'] = 'datasets library not installed'
        
        self.results['math_datasets'] = results
        return results
    
    def validate_general_tasks(self):
        """验证通用任务数据集"""
        self.log("\n" + "=" * 70)
        self.log("验证通用任务数据集")
        self.log("=" * 70)
        
        results = {}
        
        try:
            from datasets import load_dataset
            
            datasets_to_check = [
                ('HellaSwag', 'Rowan/hellaswag', 'validation'),
                ('ARC-Challenge', 'allenai/ai2_arc', 'test', 'ARC-Challenge'),
                ('HumanEval', 'openai/openai_humaneval', 'test'),
            ]
            
            for name, repo, split, *config in datasets_to_check:
                self.log(f"\n验证 {name}...")
                try:
                    if config:
                        ds = load_dataset(repo, config[0], split=split, 
                                        trust_remote_code=True)
                    else:
                        ds = load_dataset(repo, split=split, 
                                        trust_remote_code=True)
                    
                    results[name] = {
                        'status': 'AVAILABLE_HF',
                        'samples': len(ds),
                        'fields': list(ds[0].keys()) if len(ds) > 0 else []
                    }
                    self.log(f"  ✓ {len(ds)} 样本")
                    
                except Exception as e:
                    results[name] = {'status': 'ERROR', 'error': str(e)}
                    self.log(f"  ✗ 错误: {str(e)[:100]}", 'ERROR')
            
            # BBH 特殊处理（多配置）
            self.log("\n验证 BBH...")
            try:
                from datasets import get_dataset_config_names
                configs = get_dataset_config_names('lukaemon/bbh')
                results['BBH'] = {
                    'status': 'AVAILABLE_HF',
                    'tasks': len(configs),
                    'task_list': configs[:5] + ['...'] if len(configs) > 5 else configs
                }
                self.log(f"  ✓ {len(configs)} 个任务")
            except Exception as e:
                results['BBH'] = {'status': 'ERROR', 'error': str(e)}
                self.log(f"  ✗ 错误: {str(e)[:100]}", 'ERROR')
                
        except ImportError:
            self.log("datasets 库未安装，跳过 HuggingFace 验证", 'WARN')
            results['note'] = 'datasets library not installed'
        
        self.results['general_tasks'] = results
        return results
    
    def validate_longbench(self):
        """验证 LongBench-v2"""
        self.log("\n" + "=" * 70)
        self.log("验证 LongBench-v2")
        self.log("=" * 70)
        
        results = {}
        
        try:
            from datasets import load_dataset
            
            self.log("\n尝试加载 LongBench-v2...")
            ds = load_dataset('THUDM/LongBench-v2', split='train', 
                            trust_remote_code=True)
            
            results = {
                'status': 'AVAILABLE_HF',
                'total_samples': len(ds),
                'fields': list(ds[0].keys()) if len(ds) > 0 else []
            }
            
            # 分析数据分布
            difficulties = defaultdict(int)
            lengths = defaultdict(int)
            domains = defaultdict(int)
            
            for item in ds:
                difficulties[item.get('difficulty', 'unknown')] += 1
                lengths[item.get('length', 'unknown')] += 1
                domains[item.get('domain', 'unknown')] += 1
            
            results['difficulty_distribution'] = dict(difficulties)
            results['length_distribution'] = dict(lengths)
            results['domain_distribution'] = dict(domains)
            
            self.log(f"  ✓ 总样本数: {len(ds)}")
            self.log(f"  ✓ 难度分布: {dict(difficulties)}")
            self.log(f"  ✓ 长度分布: {dict(lengths)}")
            
        except Exception as e:
            results = {'status': 'ERROR', 'error': str(e)}
            self.log(f"  ✗ 错误: {str(e)[:100]}", 'ERROR')
        
        self.results['longbench'] = results
        return results
    
    def generate_needle_haystack_sample(self):
        """生成 Needle-in-Haystack 示例"""
        self.log("\n" + "=" * 70)
        self.log("生成 Needle-in-Haystack 示例")
        self.log("=" * 70)
        
        import random
        random.seed(42)
        
        needle = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
        query = "What is the best thing to do in San Francisco?"
        
        # 生成示例 haystack
        filler_sentences = [
            "The weather today is quite pleasant.",
            "I enjoy reading books in my free time.",
            "Technology has changed the way we live.",
            "The city is known for its diverse culture.",
            "Many people visit this place for vacation.",
        ]
        
        # 创建 4K tokens 级别的示例
        target_tokens = 4096
        tokens_per_sentence = 10  # 估算
        num_sentences = target_tokens // tokens_per_sentence
        
        haystack_parts = []
        for _ in range(num_sentences):
            haystack_parts.append(random.choice(filler_sentences))
        
        # 在中间位置插入 needle
        insert_pos = len(haystack_parts) // 2
        haystack_parts.insert(insert_pos, needle)
        
        haystack = " ".join(haystack_parts)
        
        results = {
            'needle': needle,
            'query': query,
            'context_lengths': [1024, 4096, 16384, 32768, 65536, 131072, 262144],
            'sample_context': haystack[:500] + "...",
            'sample_tokens_estimate': len(haystack.split()),
            'needle_insert_position': f"{insert_pos}/{len(haystack_parts)} (50%)"
        }
        
        self.log(f"  ✓ 示例生成完成")
        self.log(f"  ✓ 预估 tokens: {len(haystack.split())}")
        self.log(f"  ✓ Needle 位置: 50%")
        
        self.results['needle_haystack'] = results
        return results
    
    def save_report(self):
        """保存验证报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.results,
            'summary': self.generate_summary()
        }
        
        report_path = self.output_dir / 'dataset_validation_complete.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.log(f"\n报告已保存: {report_path}")
        return report
    
    def generate_summary(self):
        """生成验证摘要"""
        summary = {
            'total_validations': len(self.results),
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
        
        # ZeroScrolls
        if 'zero_scrolls' in self.results:
            zs = self.results['zero_scrolls']
            if zs.get('available_tasks', 0) == 10:
                summary['passed'] += 1
            else:
                summary['warnings'] += 1
        
        # Math datasets
        if 'math_datasets' in self.results:
            md = self.results['math_datasets']
            if md.get('math', {}).get('status') == 'AVAILABLE_HF':
                summary['passed'] += 1
            if md.get('gsm8k', {}).get('status') == 'AVAILABLE_HF':
                summary['passed'] += 1
        
        # General tasks
        if 'general_tasks' in self.results:
            gt = self.results['general_tasks']
            available = sum(1 for v in gt.values() 
                          if isinstance(v, dict) and v.get('status') == 'AVAILABLE_HF')
            if available >= 3:
                summary['passed'] += 1
        
        # LongBench
        if 'longbench' in self.results:
            lb = self.results['longbench']
            if lb.get('status') == 'AVAILABLE_HF':
                summary['passed'] += 1
            else:
                summary['failed'] += 1
        
        return summary
    
    def print_final_summary(self):
        """打印最终摘要"""
        self.log("\n" + "=" * 70)
        self.log("数据集验证摘要")
        self.log("=" * 70)
        
        # ZeroScrolls
        if 'zero_scrolls' in self.results:
            zs = self.results['zero_scrolls']
            self.log(f"\n📚 ZeroScrolls:")
            self.log(f"  任务数: {zs.get('available_tasks', 0)}/10")
            if zs.get('issues'):
                self.log(f"  ⚠️  问题: {len(zs['issues'])}")
                for issue in zs['issues'][:3]:
                    self.log(f"    - {issue}")
        
        # Math datasets
        if 'math_datasets' in self.results:
            md = self.results['math_datasets']
            self.log(f"\n📐 数学数据集:")
            math_ok = md.get('math', {}).get('status') == 'AVAILABLE_HF'
            gsm8k_ok = md.get('gsm8k', {}).get('status') == 'AVAILABLE_HF'
            self.log(f"  MATH: {'✓' if math_ok else '✗'}")
            self.log(f"  GSM8K: {'✓' if gsm8k_ok else '✗'}")
        
        # General tasks
        if 'general_tasks' in self.results:
            gt = self.results['general_tasks']
            self.log(f"\n🎯 通用任务:")
            for name, info in gt.items():
                if isinstance(info, dict):
                    status = '✓' if info.get('status') == 'AVAILABLE_HF' else '✗'
                    self.log(f"  {name}: {status}")
        
        # LongBench
        if 'longbench' in self.results:
            lb = self.results['longbench']
            self.log(f"\n📏 LongBench-v2:")
            status = '✓' if lb.get('status') == 'AVAILABLE_HF' else '✗'
            self.log(f"  状态: {status}")
            if 'total_samples' in lb:
                self.log(f"  样本数: {lb['total_samples']}")
        
        # Needle-in-Haystack
        if 'needle_haystack' in self.results:
            nh = self.results['needle_haystack']
            self.log(f"\n🔍 Needle-in-Haystack:")
            self.log(f"  状态: ✓ (本地生成)")
            self.log(f"  支持长度: {nh.get('context_lengths', [])}")
        
        self.log("\n" + "=" * 70)
        self.log("✓ 验证完成")
        self.log("=" * 70)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("数据集验证框架 - 论文 5.2 节")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validator = DatasetValidator()
    
    # 执行各项验证
    validator.validate_zero_scrolls()
    validator.validate_math_datasets()
    validator.validate_general_tasks()
    validator.validate_longbench()
    validator.generate_needle_haystack_sample()
    
    # 保存报告
    validator.save_report()
    
    # 打印摘要
    validator.print_final_summary()
    
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
