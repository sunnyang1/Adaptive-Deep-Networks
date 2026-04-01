#!/usr/bin/env python3
"""
Memory Profiler for Real Model Validation

精确测量模型推理过程中的内存使用情况。

Usage:
    python memory_profiler.py --checkpoint checkpoints/adb_medium.pt --context-length 65536
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MemoryProfiler:
    """
    内存分析器，用于测量模型推理的内存占用。
    
    功能：
    - 测量 GPU 显存使用
    - 测量 KV Cache 大小
    - 测量激活值内存
    - 生成内存使用报告
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.measurements = []
        
        if device == 'cuda' and torch.cuda.is_available():
            self.gpu_available = True
        else:
            self.gpu_available = False
            if device == 'cuda':
                print("Warning: CUDA not available, falling back to CPU")
                self.device = 'cpu'
    
    @contextmanager
    def measure(self, label: str):
        """
        上下文管理器：测量代码块的内存使用。
        
        Example:
            with profiler.measure("inference"):
                output = model(input_ids)
        """
        # 记录开始时的内存
        if self.gpu_available:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_allocated = torch.cuda.memory_allocated()
            start_reserved = torch.cuda.memory_reserved()
        else:
            start_allocated = 0
            start_reserved = 0
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # 记录结束时的内存
            if self.gpu_available:
                torch.cuda.synchronize()
                end_allocated = torch.cuda.memory_allocated()
                end_reserved = torch.cuda.memory_reserved()
                peak_allocated = torch.cuda.max_memory_allocated()
            else:
                end_allocated = 0
                end_reserved = 0
                peak_allocated = 0
            
            elapsed = time.time() - start_time
            
            measurement = {
                'label': label,
                'start_allocated_gb': start_allocated / (1024**3),
                'end_allocated_gb': end_allocated / (1024**3),
                'peak_allocated_gb': peak_allocated / (1024**3),
                'delta_allocated_gb': (end_allocated - start_allocated) / (1024**3),
                'start_reserved_gb': start_reserved / (1024**3),
                'end_reserved_gb': end_reserved / (1024**3),
                'elapsed_time': elapsed
            }
            
            self.measurements.append(measurement)
    
    def measure_inference(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        num_tokens: int = 1
    ) -> Dict:
        """
        测量单次推理的内存使用。
        
        Args:
            model: 模型
            input_ids: 输入token IDs
            num_tokens: 生成的token数
        
        Returns:
            内存测量结果
        """
        model.eval()
        
        with torch.no_grad():
            with self.measure("inference"):
                # 前向传播
                if hasattr(model, 'generate'):
                    output = model.generate(input_ids, max_length=input_ids.shape[1] + num_tokens)
                else:
                    output = model(input_ids)
        
        return self.measurements[-1]
    
    def profile_context_scaling(
        self,
        model: torch.nn.Module,
        context_lengths: List[int],
        batch_size: int = 1
    ) -> Dict:
        """
        测量不同上下文长度下的内存使用。
        
        Args:
            model: 模型
            context_lengths: 上下文长度列表
            batch_size: 批次大小
        
        Returns:
            各长度的内存测量结果
        """
        results = {
            'context_lengths': context_lengths,
            'batch_size': batch_size,
            'measurements': []
        }
        
        vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
        
        print("="*70)
        print("MEMORY PROFILING - Context Scaling")
        print("="*70)
        print(f"\n{'Context':<12} {'Peak Memory':<15} {'KV Cache Est.':<15} {'Time':<10}")
        print("-"*55)
        
        for ctx_len in context_lengths:
            # 生成随机输入
            input_ids = torch.randint(0, vocab_size, (batch_size, ctx_len), device=self.device)
            
            # 清理缓存
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 测量
            measurement = self.measure_inference(model, input_ids, num_tokens=1)
            
            # 估计KV Cache大小
            if hasattr(model, 'config'):
                num_layers = model.config.num_layers
                num_heads = model.config.num_heads
                head_dim = model.config.dim // num_heads
                
                # KV Cache = 2 * num_layers * batch * seq_len * num_heads * head_dim * 2 bytes
                kv_cache_bytes = 2 * num_layers * batch_size * ctx_len * num_heads * head_dim * 2
                kv_cache_gb = kv_cache_bytes / (1024**3)
            else:
                kv_cache_gb = 0
            
            result = {
                'context_length': ctx_len,
                'peak_memory_gb': measurement['peak_allocated_gb'],
                'kv_cache_est_gb': kv_cache_gb,
                'time': measurement['elapsed_time']
            }
            
            results['measurements'].append(result)
            
            ctx_str = f"{ctx_len//1024}K" if ctx_len >= 1024 else str(ctx_len)
            print(f"{ctx_str:<12} {measurement['peak_allocated_gb']:>10.2f}GB   "
                  f"{kv_cache_gb:>10.2f}GB   {measurement['elapsed_time']:>7.2f}s")
        
        return results
    
    def compare_compression(
        self,
        model_standard: torch.nn.Module,
        model_rabitq: torch.nn.Module,
        context_length: int = 8192
    ) -> Dict:
        """
        对比标准模型和 RaBitQ 压缩模型的内存使用。
        
        Returns:
            对比结果
        """
        vocab_size = getattr(model_standard, 'config', {}).get('vocab_size', 32000)
        input_ids = torch.randint(0, vocab_size, (1, context_length), device=self.device)
        
        print("="*70)
        print("MEMORY COMPARISON: Standard vs RaBitQ")
        print("="*70)
        print(f"\nContext length: {context_length:,} tokens")
        
        results = {}
        
        # 测试标准模型
        print("\nStandard Model:")
        if self.gpu_available:
            torch.cuda.empty_cache()
        mem_std = self.measure_inference(model_standard, input_ids)
        results['standard'] = mem_std
        print(f"  Peak memory: {mem_std['peak_allocated_gb']:.2f} GB")
        
        # 测试 RaBitQ 模型
        print("\nRaBitQ Model:")
        if self.gpu_available:
            torch.cuda.empty_cache()
        mem_rabitq = self.measure_inference(model_rabitq, input_ids)
        results['rabitq'] = mem_rabitq
        print(f"  Peak memory: {mem_rabitq['peak_allocated_gb']:.2f} GB")
        
        # 计算压缩比
        compression_ratio = mem_std['peak_allocated_gb'] / mem_rabitq['peak_allocated_gb']
        results['compression_ratio'] = compression_ratio
        
        print(f"\nCompression Ratio: {compression_ratio:.2f}x")
        print(f"Memory Saved: {mem_std['peak_allocated_gb'] - mem_rabitq['peak_allocated_gb']:.2f} GB")
        
        return results
    
    def get_summary(self) -> Dict:
        """获取内存测量的摘要统计"""
        if not self.measurements:
            return {}
        
        peak_memories = [m['peak_allocated_gb'] for m in self.measurements]
        
        return {
            'num_measurements': len(self.measurements),
            'peak_memory_max_gb': max(peak_memories),
            'peak_memory_avg_gb': np.mean(peak_memories),
            'total_elapsed_time': sum(m['elapsed_time'] for m in self.measurements)
        }
    
    def save_results(self, results: Dict, output_path: Path):
        """保存分析结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    from model_loader import load_adb_model
    
    parser = argparse.ArgumentParser(description='Memory Profiler')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--context-lengths', type=int, nargs='+',
                       default=[4096, 8192, 16384, 32768],
                       help='Context lengths to profile')
    parser.add_argument('--output', type=str, default='results/memory_profile.json')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 加载模型
    print("Loading model...")
    model, config = load_adb_model(
        checkpoint_path=args.checkpoint,
        model_size=args.size,
        device=args.device
    )
    
    # 创建分析器
    profiler = MemoryProfiler(device=args.device)
    
    # 运行分析
    results = profiler.profile_context_scaling(model, args.context_lengths)
    
    # 添加摘要
    results['summary'] = profiler.get_summary()
    
    # 保存结果
    profiler.save_results(results, Path(args.output))


if __name__ == '__main__':
    main()
