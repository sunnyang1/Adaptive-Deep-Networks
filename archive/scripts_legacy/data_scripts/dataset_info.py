#!/usr/bin/env python3
"""
数据集信息整理 - 根据论文 5.2 节整理所需数据集信息
"""

import json
from datetime import datetime

# 数据集信息
DATASETS = {
    "Long-Context Retrieval Benchmarks": {
        "Needle-in-Haystack": {
            "description": "Long-context retrieval benchmark with embedded facts",
            "context_lengths": [1024, 4096, 16384, 32768, 65536, 131072, 262144],
            "needle_example": "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
            "query_example": "What is the best thing to do in San Francisco?",
            "generation": "Local synthetic generation",
            "evaluation": "Exact match accuracy across 10 depths per length",
            "source": "Generated locally following Kamradt (2023)",
            "status": "✓ Local generation"
        },
        "LongBench-v2": {
            "description": "Multi-task long-context benchmark for deep understanding and reasoning",
            "average_context": "35K tokens (up to 200K)",
            "task_categories": [
                "Single-document QA",
                "Multi-document QA", 
                "Summarization",
                "Few-shot learning",
                "Synthetic tasks",
                "Code completion"
            ],
            "huggingface": "THUDM/LongBench-v2",
            "data_format": {
                "_id": "Unique identifier",
                "domain": "Primary domain category",
                "sub_domain": "Specific sub-domain",
                "difficulty": "easy or hard",
                "length": "short, medium, or long",
                "question": "Input/command for the task",
                "choice_A/B/C/D": "Multiple choice options",
                "answer": "Groundtruth answer (A/B/C/D)",
                "context": "Long context (documents, books, code)"
            },
            "paper": "LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks",
            "github": "https://github.com/THUDM/LongBench",
            "status": "✓ HuggingFace available"
        },
        "ZeroScrolls": {
            "description": "Zero-shot benchmark for long text understanding",
            "max_context": "Up to 100K tokens",
            "tasks": [
                "GovReport - Government report summarization",
                "SummScreen - TV show summarization",
                "QMSum - Meeting summarization",
                "SQuALITY - Quality-focused QA",
                "SpaceDigest - Scientific article aggregation",
                "BookSumSort - Book summary sorting",
                "MuSiQue - Multi-hop question answering",
                "Qasper - Scientific paper QA"
            ],
            "paper": "ZeroSCROLLS: A Zero-Shot Benchmark for Long Text Understanding (EMNLP 2023)",
            "website": "https://www.zero.scrolls-benchmark.com/",
            "arxiv": "https://arxiv.org/abs/2305.14196",
            "status": "⚠ Requires local implementation"
        }
    },
    
    "Mathematical Reasoning Benchmarks": {
        "MATH": {
            "description": "Competition mathematics problems with step-by-step solutions",
            "total_problems": 12500,
            "split": {"train": 7500, "test": 5000},
            "difficulty_levels": [1, 2, 3, 4, 5],
            "categories": [
                "Algebra", "Geometry", "Counting & Probability",
                "Number Theory", "Precalculus", "Calculus"
            ],
            "huggingface": "hendrycks/competition_math",
            "format": {
                "problem": "Mathematical problem statement",
                "level": "Difficulty level (1-5)",
                "type": "Problem category",
                "solution": "Step-by-step solution with boxed answer"
            },
            "paper": "Measuring Mathematical Problem Solving With the MATH Dataset (NeurIPS 2021)",
            "github": "https://github.com/hendrycks/math",
            "status": "✓ HuggingFace available"
        },
        "GSM8K": {
            "description": "Grade school math word problems",
            "total_problems": 8500,
            "split": {"train": 7473, "test": 1319},
            "steps_per_problem": "2-8 steps",
            "operations": ["Addition", "Subtraction", "Multiplication", "Division"],
            "huggingface": "openai/gsm8k",
            "format": {
                "question": "Word problem",
                "answer": "Step-by-step solution and final answer"
            },
            "paper": "Training Verifiers to Solve Math Word Problems",
            "github": "https://github.com/openai/grade-school-math",
            "status": "✓ HuggingFace available"
        }
    },
    
    "Language Modeling and General Tasks": {
        "HellaSwag": {
            "description": "Commonsense natural language inference through sentence completion",
            "validation_samples": 10042,
            "format": "Multiple choice (4 options)",
            "human_accuracy": "~95%",
            "task": "Select most plausible continuation",
            "huggingface": "Rowan/hellaswag",
            "paper": "HellaSwag: Can a Machine Really Finish Your Sentence?",
            "status": "✓ HuggingFace available"
        },
        "ARC-Challenge": {
            "description": "AI2 Reasoning Challenge - Grade-school science questions",
            "total_questions": 7787,
            "split": {
                "ARC-Easy": 5197,
                "ARC-Challenge": 2590
            },
            "format": "Multiple choice QA",
            "requirements": "Reasoning beyond surface-level facts",
            "huggingface": "allenai/ai2_arc",
            "paper": "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge",
            "status": "✓ HuggingFace available"
        },
        "HumanEval": {
            "description": "Hand-crafted programming problems for code generation",
            "problems": 164,
            "language": "Python",
            "format": {
                "task_id": "Problem identifier",
                "prompt": "Function signature and docstring",
                "entry_point": "Function name",
                "canonical_solution": "Reference solution",
                "test": "Unit tests"
            },
            "evaluation": "Pass@k metric",
            "huggingface": "openai/openai_humaneval",
            "paper": "Evaluating Large Language Models Trained on Code",
            "github": "https://github.com/openai/human-eval",
            "status": "✓ HuggingFace available"
        },
        "BBH": {
            "description": "Big-Bench Hard - 23 challenging tasks from BIG-Bench",
            "tasks": [
                "boolean_expressions", "causal_judgement", "date_understanding",
                "disambiguation_qa", "dyck_languages", "formal_fallacies",
                "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                "logical_deduction_seven_objects", "logical_deduction_three_objects",
                "movie_recommendation", "multistep_arithmetic_two", "navigate",
                "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
                "ruin_names", "salient_translation_error_detection", "snarks",
                "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects"
            ],
            "huggingface": "lukaemon/bbh",
            "paper": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
            "status": "✓ HuggingFace available"
        }
    }
}

# 预训练语料
PRETRAINING_CORPORA = {
    "C4": {
        "description": "Colossal Clean Crawled Corpus",
        "size": "365B tokens",
        "source": "Cleaned Common Crawl",
        "purpose": "Pre-training corpus",
        "metric": "Perplexity"
    },
    "The Pile": {
        "description": "Diverse domain validation corpus",
        "size": "300B tokens",
        "source": "Academic, web, books, code",
        "purpose": "Pre-training corpus",
        "metric": "Perplexity"
    }
}


def print_dataset_info():
    """打印数据集信息"""
    print("=" * 80)
    print("Adaptive Deep Networks - 数据集信息整理 (论文 5.2 节)")
    print("=" * 80)
    
    for category, datasets in DATASETS.items():
        print(f"\n\n{'=' * 80}")
        print(f"{category}")
        print("=" * 80)
        
        for name, info in datasets.items():
            print(f"\n{'─' * 60}")
            print(f"📊 {name}")
            print("─" * 60)
            print(f"描述: {info['description']}")
            print(f"状态: {info['status']}")
            
            if 'huggingface' in info:
                print(f"HuggingFace: {info['huggingface']}")
            
            if 'total_problems' in info:
                print(f"总问题数: {info['total_problems']}")
            
            if 'validation_samples' in info:
                print(f"验证集: {info['validation_samples']} 样本")
            
            if 'problems' in info:
                print(f"问题数: {info['problems']}")
            
            if 'context_lengths' in info:
                print(f"上下文长度: {info['context_lengths']}")
            
            if 'task_categories' in info:
                print(f"任务类别: {', '.join(info['task_categories'])}")
            
            if 'tasks' in info and isinstance(info['tasks'], list) and len(info['tasks']) > 5:
                print(f"任务数量: {len(info['tasks'])} 个")
                print(f"示例任务: {', '.join(info['tasks'][:3])}...")
            
            if 'paper' in info:
                print(f"论文: {info['paper']}")
            
            if 'github' in info:
                print(f"GitHub: {info['github']}")
    
    print("\n\n" + "=" * 80)
    print("预训练语料")
    print("=" * 80)
    for name, info in PRETRAINING_CORPORA.items():
        print(f"\n📚 {name}")
        print(f"  描述: {info['description']}")
        print(f"  规模: {info['size']}")
        print(f"  来源: {info['source']}")
        print(f"  用途: {info['purpose']}")


def generate_download_script():
    """生成数据集下载脚本"""
    script = '''#!/bin/bash
# 数据集下载脚本 - Adaptive Deep Networks

# 创建数据目录
mkdir -p data/{longbench,math,general,code,reasoning}

echo "=========================================="
echo "下载 Long-Context Retrieval 数据集"
echo "=========================================="

# LongBench-v2
echo "下载 LongBench-v2..."
python -c "
from datasets import load_dataset
ds = load_dataset('THUDM/LongBench-v2', split='train')
ds.save_to_disk('data/longbench/longbench-v2')
print(f'LongBench-v2: {len(ds)} samples')
"

echo "=========================================="
echo "下载 Mathematical Reasoning 数据集"
echo "=========================================="

# MATH
echo "下载 MATH..."
python -c "
from datasets import load_dataset
ds_train = load_dataset('hendrycks/competition_math', split='train')
ds_test = load_dataset('hendrycks/competition_math', split='test')
ds_train.save_to_disk('data/math/math_train')
ds_test.save_to_disk('data/math/math_test')
print(f'MATH Train: {len(ds_train)}, Test: {len(ds_test)}')
"

# GSM8K
echo "下载 GSM8K..."
python -c "
from datasets import load_dataset
ds_train = load_dataset('openai/gsm8k', 'main', split='train')
ds_test = load_dataset('openai/gsm8k', 'main', split='test')
ds_train.save_to_disk('data/math/gsm8k_train')
ds_test.save_to_disk('data/math/gsm8k_test')
print(f'GSM8K Train: {len(ds_train)}, Test: {len(ds_test)}')
"

echo "=========================================="
echo "下载 General Tasks 数据集"
echo "=========================================="

# HellaSwag
echo "下载 HellaSwag..."
python -c "
from datasets import load_dataset
ds = load_dataset('Rowan/hellaswag', split='validation')
ds.save_to_disk('data/general/hellaswag')
print(f'HellaSwag: {len(ds)} samples')
"

# ARC-Challenge
echo "下载 ARC-Challenge..."
python -c "
from datasets import load_dataset
ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
ds.save_to_disk('data/general/arc_challenge')
print(f'ARC-Challenge: {len(ds)} samples')
"

# HumanEval
echo "下载 HumanEval..."
python -c "
from datasets import load_dataset
ds = load_dataset('openai/openai_humaneval', split='test')
ds.save_to_disk('data/code/humaneval')
print(f'HumanEval: {len(ds)} problems')
"

# BBH
echo "下载 BBH..."
python -c "
from datasets import load_dataset, get_dataset_config_names
configs = get_dataset_config_names('lukaemon/bbh')
for config in configs:
    ds = load_dataset('lukaemon/bbh', config, split='test')
    ds.save_to_disk(f'data/reasoning/bbh_{config}')
print(f'BBH: {len(configs)} tasks')
"

echo "=========================================="
echo "数据集下载完成"
echo "=========================================="
ls -lh data/*/
'''
    
    with open('download_datasets.sh', 'w') as f:
        f.write(script)
    
    print("\n\n下载脚本已生成: download_datasets.sh")
    print("运行命令: bash download_datasets.sh")


def save_dataset_info():
    """保存数据集信息到文件"""
    output = {
        "timestamp": datetime.now().isoformat(),
        "datasets": DATASETS,
        "pretraining_corpora": PRETRAINING_CORPORA,
        "summary": {
            "total_categories": len(DATASETS),
            "total_datasets": sum(len(datasets) for datasets in DATASETS.values()),
            "huggingface_available": [
                "LongBench-v2", "MATH", "GSM8K", "HellaSwag", 
                "ARC-Challenge", "HumanEval", "BBH"
            ],
            "local_generation": ["Needle-in-Haystack"],
            "requires_implementation": ["ZeroScrolls"]
        }
    }
    
    with open('./results/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据集信息已保存: ./results/dataset_info.json")


def main():
    """主函数"""
    import os
    os.makedirs('./results', exist_ok=True)
    
    # 打印数据集信息
    print_dataset_info()
    
    # 生成下载脚本
    generate_download_script()
    
    # 保存数据集信息
    save_dataset_info()
    
    print("\n" + "=" * 80)
    print("✓ 数据集信息整理完成")
    print("=" * 80)
    print("\n下一步:")
    print("1. 安装 datasets 库: pip install datasets")
    print("2. 运行下载脚本: bash download_datasets.sh")
    print("3. 或使用 load_dataset() 直接从 HuggingFace 加载")


if __name__ == "__main__":
    main()
