#!/bin/bash
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
