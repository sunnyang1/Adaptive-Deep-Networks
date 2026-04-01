"""
Adaptive Deep Networks: 实验套件

本模块提供论文验证所需的所有实验实现。

## 目录结构
- core/: 核心验证实验 (exp1-exp6)
- rabitq/: RaBitQ KV Cache 压缩实验
- benchmarks/: 模型基准测试
- utils/: 实验工具函数
- docs/: 实验文档
- results/: 实验结果输出
"""

__version__ = "0.1.0"

# 主要实验入口
def run_core_experiment(exp_id: str, **kwargs):
    """运行单个核心实验。
    
    Args:
        exp_id: 实验ID ('exp1'-'exp6')
        **kwargs: 实验参数
    
    Example:
        >>> run_core_experiment('exp1', num_samples=50)
    """
    import subprocess
    import sys
    
    script_map = {
        'exp1': 'core/exp1_representation_burial/run_exp1.py',
        'exp2': 'core/exp2_margin_analysis/run_exp2.py',
        'exp3': 'core/exp3_gradient_flow/run_exp3.py',
        'exp4': 'core/exp4_flop_equivalence/run_exp4.py',
        'exp5': 'core/exp5_synergy/run_exp5.py',
        'exp6': 'core/exp6_auxiliary/run_exp6.py',
    }
    
    if exp_id not in script_map:
        raise ValueError(f"Unknown experiment: {exp_id}")
    
    script = script_map[exp_id]
    args = [sys.executable, script]
    
    for key, value in kwargs.items():
        args.extend([f"--{key}", str(value)])
    
    subprocess.run(args)


def run_all_core_experiments(quick: bool = False, device: str = 'cuda'):
    """运行所有核心实验。
    
    Args:
        quick: 是否使用快速模式
        device: 运行设备 ('cuda' 或 'cpu')
    """
    import subprocess
    import sys
    
    args = [sys.executable, 'run_all_experiments.py']
    if quick:
        args.append('--quick')
    args.extend(['--device', device])
    
    subprocess.run(args)


__all__ = [
    'run_core_experiment',
    'run_all_core_experiments',
]
