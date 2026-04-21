#!/bin/bash
# 使用 curl 验证 HuggingFace 数据集

echo "======================================================================"
echo "HuggingFace 数据集验证"
echo "======================================================================"

check_dataset() {
    local repo=$1
    local name=$2
    
    echo ""
    echo "🔍 检查: $name"
    echo "   Repo: $repo"
    
    # 检查数据集页面
    status=$(curl -s -o /dev/null -w "%{http_code}" "https://huggingface.co/datasets/$repo")
    
    if [ "$status" = "200" ]; then
        echo "   ✅ 可用 (HTTP 200)"
        
        # 尝试获取文件列表
        files=$(curl -s "https://huggingface.co/api/datasets/$repo/tree/main" | grep -o '"path":"[^"]*"' | head -5 | cut -d'"' -f4)
        if [ -n "$files" ]; then
            echo "   📁 文件示例:"
            echo "$files" | head -3 | sed 's/^/      - /'
        fi
    else
        echo "   ❌ 不可用 (HTTP $status)"
    fi
}

echo ""
echo "📂 Long-Context Retrieval"
echo "======================================================================"
check_dataset "THUDM/LongBench-v2" "LongBench-v2"

echo ""
echo "📂 Mathematical Reasoning"
echo "======================================================================"
check_dataset "hendrycks/competition_math" "MATH"
check_dataset "openai/gsm8k" "GSM8K"

echo ""
echo "📂 General Tasks"
echo "======================================================================"
check_dataset "Rowan/hellaswag" "HellaSwag"
check_dataset "allenai/ai2_arc" "ARC-Challenge"
check_dataset "openai/openai_humaneval" "HumanEval"
check_dataset "lukaemon/bbh" "BBH"

echo ""
echo "======================================================================"
echo "✅ 验证完成"
echo "======================================================================"
