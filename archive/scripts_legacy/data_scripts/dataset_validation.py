#!/usr/bin/env python3
"""
数据集验证脚本 - 根据论文 5.2 节验证所需数据集

需要验证的数据集:
1. Long-Context Retrieval:
   - Needle-in-Haystack (本地生成)
   - LongBench-v2 (THUDM/LongBench-v2)
   - ZeroScrolls (本地实现)

2. Mathematical Reasoning:
   - MATH (hendrycks/competition_math)
   - GSM8K (openai/gsm8k)

3. Language Modeling and General Tasks:
   - HellaSwag (Rowan/hellaswag)
   - ARC-Challenge (allenai/ai2_arc)
   - HumanEval (openai/openai_humaneval)
   - BBH (lukaemon/bbh)
"""

import os
import sys
import json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, 'src')

def check_dataset_availability():
    """检查数据集可用性"""
    print("=" * 70)
    print("数据集可用性验证")
    print("=" * 70)
    
    results = {}
    
    # 尝试导入 datasets 库
    try:
        from datasets import load_dataset, get_dataset_config_names
        print("✓ datasets 库已安装")
    except ImportError:
        print("✗ datasets 库未安装，请先安装: pip install datasets")
        return None
    
    # 定义需要验证的数据集
    datasets_to_check = [
        ("LongBench-v2", "THUDM/LongBench-v2", "long_context"),
        ("MATH", "hendrycks/competition_math", "math"),
        ("GSM8K", "openai/gsm8k", "math"),
        ("HellaSwag", "Rowan/hellaswag", "general"),
        ("ARC-Challenge", "allenai/ai2_arc", "general"),
        ("HumanEval", "openai/openai_humaneval", "code"),
        ("BBH", "lukaemon/bbh", "reasoning"),
    ]
    
    print("\n验证 HuggingFace 数据集...")
    print("-" * 70)
    
    for name, repo, category in datasets_to_check:
        print(f"\n[{category}] {name}")
        print(f"  Repository: {repo}")
        
        try:
            # 尝试加载数据集信息
            if name == "BBH":
                # BBH 有多个子集
                configs = get_dataset_config_names(repo)
                print(f"  可用配置: {len(configs)} 个任务")
                print(f"  示例任务: {', '.join(configs[:5])}...")
                
                # 加载一个示例任务
                ds = load_dataset(repo, configs[0], split='test', trust_remote_code=True)
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "configs": len(configs),
                    "example_samples": len(ds)
                }
                
            elif name == "ARC-Challenge":
                ds = load_dataset(repo, "ARC-Challenge", split='test', trust_remote_code=True)
                print(f"  测试集样本数: {len(ds)}")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "test_samples": len(ds)
                }
                
            elif name == "MATH":
                ds_train = load_dataset(repo, split='train', trust_remote_code=True)
                ds_test = load_dataset(repo, split='test', trust_remote_code=True)
                print(f"  训练集: {len(ds_train)} 样本")
                print(f"  测试集: {len(ds_test)} 样本")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "train_samples": len(ds_train),
                    "test_samples": len(ds_test)
                }
                
            elif name == "GSM8K":
                ds_train = load_dataset(repo, "main", split='train', trust_remote_code=True)
                ds_test = load_dataset(repo, "main", split='test', trust_remote_code=True)
                print(f"  训练集: {len(ds_train)} 样本")
                print(f"  测试集: {len(ds_test)} 样本")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "train_samples": len(ds_train),
                    "test_samples": len(ds_test)
                }
                
            elif name == "HumanEval":
                ds = load_dataset(repo, split='test', trust_remote_code=True)
                print(f"  测试集: {len(ds)} 问题")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "problems": len(ds)
                }
                
            elif name == "HellaSwag":
                ds_val = load_dataset(repo, split='validation', trust_remote_code=True)
                print(f"  验证集: {len(ds_val)} 样本")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "val_samples": len(ds_val)
                }
                
            elif name == "LongBench-v2":
                ds = load_dataset(repo, split='train', trust_remote_code=True)
                print(f"  总样本数: {len(ds)}")
                # 查看数据结构
                example = ds[0]
                print(f"  数据字段: {list(example.keys())}")
                results[name] = {
                    "status": "✓ 可用",
                    "repo": repo,
                    "total_samples": len(ds),
                    "fields": list(example.keys())
                }
                
        except Exception as e:
            print(f"  ✗ 加载失败: {str(e)[:100]}")
            results[name] = {
                "status": "✗ 失败",
                "error": str(e)[:100]
            }
    
    return results


def generate_needle_in_haystack_example():
    """生成 Needle-in-Haystack 示例数据"""
    print("\n" + "=" * 70)
    print("Needle-in-Haystack 示例生成")
    print("=" * 70)
    
    needle = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    query = "What is the best thing to do in San Francisco?"
    
    # 生成不同长度的上下文
    context_lengths = [1024, 4096, 16384, 32768, 65536, 131072, 262144]
    
    print(f"\nNeedle: {needle}")
    print(f"Query: {query}")
    print(f"\n支持上下文长度: {context_lengths}")
    
    # 模拟生成 haystack
    import random
    random.seed(42)
    
    filler_text = [
        "The weather today is quite pleasant.",
        "I enjoy reading books in my free time.",
        "Technology has changed the way we live.",
        "The city is known for its diverse culture.",
        "Many people visit this place for vacation.",
        "The local cuisine is delicious and unique.",
        "Public transportation is efficient here.",
        "There are many parks and recreational areas.",
        "The architecture is a blend of modern and historic.",
        "Local artists display their work in galleries.",
    ]
    
    for length in [1024, 4096]:
        # 生成指定 token 长度的 haystack (简化估计: 1 token ≈ 4 字符)
        target_chars = length * 4
        haystack_parts = []
        current_chars = 0
        
        while current_chars < target_chars:
            sentence = random.choice(filler_text) + " "
            haystack_parts.append(sentence)
            current_chars += len(sentence)
        
        # 在随机位置插入 needle
        insert_pos = random.randint(0, len(haystack_parts))
        haystack_parts.insert(insert_pos, needle + " ")
        
        haystack = "".join(haystack_parts)
        
        print(f"\nContext length {length}:")
        print(f"  总字符数: {len(haystack)}")
        print(f"  Needle 位置: {insert_pos}/{len(haystack_parts)} (约 {(insert_pos/len(haystack_parts)*100):.1f}%)")
        print(f"  包含 Needle: {needle in haystack}")
    
    return {
        "needle": needle,
        "query": query,
        "context_lengths": context_lengths,
        "description": "Generated synthetic long-context data with embedded facts"
    }


def verify_dataset_structure():
    """验证数据集结构是否符合论文描述"""
    print("\n" + "=" * 70)
    print("数据集结构验证")
    print("=" * 70)
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets 库未安装")
        return
    
    # 验证 MATH 数据集结构
    print("\n[MATH Dataset]")
    print("-" * 70)
    ds = load_dataset("hendrycks/competition_math", split='test', trust_remote_code=True)
    example = ds[0]
    print(f"样本字段: {list(example.keys())}")
    print(f"问题示例: {example['problem'][:100]}...")
    print(f"难度级别: {example.get('level', 'N/A')}")
    print(f"问题类型: {example.get('type', 'N/A')}")
    
    # 检查难度级别分布
    levels = {}
    for item in ds:
        level = item.get('level', 'unknown')
        levels[level] = levels.get(level, 0) + 1
    print(f"\n难度级别分布:")
    for level in sorted(levels.keys()):
        print(f"  {level}: {levels[level]} 题")
    
    # 验证 GSM8K 结构
    print("\n[GSM8K Dataset]")
    print("-" * 70)
    ds = load_dataset("openai/gsm8k", "main", split='test', trust_remote_code=True)
    example = ds[0]
    print(f"样本字段: {list(example.keys())}")
    print(f"问题示例: {example['question'][:100]}...")
    print(f"答案格式: {example['answer'][:100]}...")
    
    # 验证 HumanEval 结构
    print("\n[HumanEval Dataset]")
    print("-" * 70)
    ds = load_dataset("openai/openai_humaneval", split='test', trust_remote_code=True)
    example = ds[0]
    print(f"样本字段: {list(example.keys())}")
    print(f"任务ID: {example['task_id']}")
    print(f"提示词: {example['prompt'][:150]}...")
    print(f"入口函数: {example['entry_point']}")
    
    # 验证 BBH 结构
    print("\n[BBH Dataset - boolean_expressions 任务]")
    print("-" * 70)
    ds = load_dataset("lukaemon/bbh", "boolean_expressions", split='test', trust_remote_code=True)
    example = ds[0]
    print(f"样本字段: {list(example.keys())}")
    print(f"输入示例: {example['input'][:100]}...")
    print(f"目标输出: {example['target']}")


def generate_dataset_report():
    """生成数据集验证报告"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {}
    }
    
    # HuggingFace 数据集
    hf_results = check_dataset_availability()
    if hf_results:
        report["datasets"].update(hf_results)
    
    # Needle-in-Haystack
    nih_result = generate_needle_in_haystack_example()
    report["datasets"]["Needle-in-Haystack"] = {
        "status": "✓ 本地生成",
        **nih_result
    }
    
    # ZeroScrolls (标记为需要额外实现)
    report["datasets"]["ZeroScrolls"] = {
        "status": "⚠ 需本地实现",
        "description": "Long-document understanding tasks (up to 100K tokens)",
        "reference": "https://www.zero.scrolls-benchmark.com/",
        "tasks": [
            "GovReport", "SummScreen", "QMSum", "SQuALITY",
            "SpaceDigest", "BookSumSort", "MuSiQue", "Qasper"
        ]
    }
    
    # 保存报告
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "dataset_validation_report.json")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n\n报告已保存: {report_path}")
    
    return report


def print_summary(report):
    """打印验证摘要"""
    print("\n" + "=" * 70)
    print("数据集验证摘要")
    print("=" * 70)
    
    available = 0
    failed = 0
    
    for name, info in report["datasets"].items():
        status = info.get("status", "未知")
        if "✓" in status:
            available += 1
        elif "✗" in status:
            failed += 1
        
        print(f"{name:<25} {status}")
    
    print("-" * 70)
    print(f"可用: {available} | 失败: {failed} | 总计: {len(report['datasets'])}")
    
    # 按类别分组
    print("\n按类别:")
    print("  Long-Context Retrieval: LongBench-v2, Needle-in-Haystack, ZeroScrolls")
    print("  Mathematical Reasoning: MATH, GSM8K")
    print("  General Tasks: HellaSwag, ARC-Challenge, HumanEval, BBH")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("Adaptive Deep Networks 数据集验证")
    print("根据论文 5.2 节验证所需数据集")
    print("=" * 70)
    
    # 生成报告
    report = generate_dataset_report()
    
    # 验证数据结构
    try:
        verify_dataset_structure()
    except Exception as e:
        print(f"\n数据结构验证部分失败: {e}")
    
    # 打印摘要
    print_summary(report)
    
    print("\n" + "=" * 70)
    print("✓ 数据集验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
