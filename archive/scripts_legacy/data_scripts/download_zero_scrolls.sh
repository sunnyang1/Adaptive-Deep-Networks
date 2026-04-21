#!/bin/bash
# ZeroScrolls 数据集下载脚本

set -e

echo "=========================================="
echo "下载 ZeroScrolls 数据集"
echo "Source: https://huggingface.co/datasets/tau/zero_scrolls"
echo "=========================================="

# 创建目录
mkdir -p data/zero_scrolls
cd data/zero_scrolls

# 基础 URL
BASE_URL="https://huggingface.co/datasets/tau/zero_scrolls/resolve/main"

# 数据文件列表
FILES=(
    "book_sum_sort.zip"
    "gov_report.zip"
    "musique.zip"
    "narrative_qa.zip"
    "qasper.zip"
    "qmsum.zip"
    "quality.zip"
    "space_digest.zip"
    "squality.zip"
    "summ_screen_fd.zip"
)

# 下载每个文件
echo ""
echo "开始下载数据文件..."
echo ""

for file in "${FILES[@]}"; do
    echo "下载: $file"
    if [ -f "$file" ]; then
        echo "  ✓ 文件已存在，跳过"
    else
        curl -L -o "$file" "$BASE_URL/$file" --progress-bar
        echo "  ✓ 下载完成"
    fi
done

# 下载 README
echo ""
echo "下载 README.md..."
curl -L -o README.md "$BASE_URL/README.md" --progress-bar
echo "✓ README 下载完成"

# 解压文件
echo ""
echo "=========================================="
echo "解压数据文件"
echo "=========================================="

for file in "${FILES[@]}"; do
    task_name="${file%.zip}"
    echo "解压: $file -> $task_name/"
    
    # 创建任务目录
    mkdir -p "$task_name"
    
    # 解压到任务目录
    unzip -q -o "$file" -d "$task_name"
    
    echo "  ✓ 解压完成"
done

echo ""
echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "数据集位置: $(pwd)"
echo ""
echo "包含的任务:"
for file in "${FILES[@]}"; do
    task_name="${file%.zip}"
    echo "  - $task_name"
done
echo ""
echo "总大小:"
du -sh .
echo ""
