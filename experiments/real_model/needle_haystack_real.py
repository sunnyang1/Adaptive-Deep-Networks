#!/usr/bin/env python3
"""
Needle-in-Haystack Real Model Validation

真实模型的长上下文检索能力验证。

Paper Table 4:
- 4K: 98.5%
- 16K: 91.3%
- 64K: 78.2%
- 128K: 68.2%
- Average: 86.9%

Extended (ADB-TurboQuant):
- 256K: 65%
- 512K: 55%
- 1M: 45%
"""

import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .datasets.needle_dataset import NeedleDataset


class NeedleHaystackValidator:
    """
    Needle-in-Haystack 验证器（真实模型版本）。
    
    使用实际模型进行长上下文检索测试。
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        tokenizer=None
    ):
        """
        Args:
            model: 预加载的 ADB 模型
            device: 计算设备
            tokenizer: tokenizer（可选，如果不提供则使用简单分割）
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.dataset = NeedleDataset(seed=42)
    
    def _tokenize(self, text: str) -> List[int]:
        """将文本转为 token IDs"""
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        
        # 简单的基于字符的 tokenization
        # 实际使用时应替换为正确的 tokenizer
        tokens = []
        words = text.split()
        for word in words:
            # 简化的 tokenization：每个单词约 1.3 tokens
            tokens.extend([hash(word) % 32000])  # 模拟词汇表
        return tokens
    
    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> str:
        """
        生成模型输出。
        
        Args:
            input_ids: 输入 token IDs [batch, seq_len]
            max_new_tokens: 最大生成 token 数
        
        Returns:
            生成的文本
        """
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # 使用模型的 generate 方法
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # greedy
                )
                # 解码输出
                output_text = self._decode(output_ids[0, input_ids.shape[1]:])
            else:
                # 简化版本：只返回前向传播的结果
                logits = self.model(input_ids)
                # 取最后一个位置的预测
                next_token = logits[0, -1].argmax().item()
                output_text = str(next_token)
        
        return output_text
    
    def _decode(self, token_ids: torch.Tensor) -> str:
        """将 token IDs 解码为文本"""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids)
        
        # 简化解码
        return " ".join([f"{t.item()}" for t in token_ids])
    
    def evaluate_sample(self, sample, max_new_tokens: int = 50) -> Dict:
        """
        评估单个样本。
        
        Args:
            sample: NeedleSample
            max_new_tokens: 最大生成 token 数
        
        Returns:
            评估结果
        """
        # Tokenize
        prompt = sample.format_prompt()
        input_ids = self._tokenize(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        # 生成
        start_time = time.time()
        output_text = self._generate(input_tensor, max_new_tokens)
        inference_time = time.time() - start_time
        
        # 评估
        evaluation = sample.evaluate(output_text)
        evaluation['inference_time'] = inference_time
        evaluation['prompt_tokens'] = len(input_ids)
        
        return evaluation
    
    def run_test(
        self,
        context_lengths: List[int],
        num_samples: int = 10,
        max_new_tokens: int = 50
    ) -> Dict:
        """
        运行完整测试。
        
        Args:
            context_lengths: 测试的上下文长度列表
            num_samples: 每个长度的样本数
            max_new_tokens: 最大生成 token 数
        
        Returns:
            测试结果汇总
        """
        print("\n" + "="*70)
        print("NEEDLE-IN-HAYSTACK TEST")
        print("="*70)
        
        results = {
            'num_samples': num_samples,
            'results': {},
            'details': []
        }
        
        all_accuracies = []
        
        for ctx_len in context_lengths:
            print(f"\n{'='*70}")
            print(f"Context Length: {ctx_len:,} tokens ({ctx_len//1024}K)")
            print(f"{'='*70}")
            
            # 创建数据集
            samples = self.dataset.create_dataset(
                context_tokens=ctx_len,
                num_samples=num_samples,
                depth_distribution="uniform"
            )
            
            # 评估每个样本
            sample_results = []
            correct_count = 0
            
            for i, sample in enumerate(samples):
                print(f"  Sample {i+1}/{num_samples}...", end=' ', flush=True)
                
                result = self.evaluate_sample(sample, max_new_tokens)
                sample_results.append({
                    'depth_percent': sample.needle_depth_percent,
                    'correct': result['correct'],
                    'score': result['score'],
                    'inference_time': result['inference_time']
                })
                
                if result['correct']:
                    correct_count += 1
                    print("✓")
                else:
                    print("✗")
            
            # 计算该长度的统计
            accuracy = correct_count / num_samples * 100
            all_accuracies.append(accuracy)
            
            results['results'][ctx_len] = {
                'accuracy': accuracy,
                'correct': correct_count,
                'total': num_samples,
                'avg_inference_time': np.mean([r['inference_time'] for r in sample_results])
            }
            
            print(f"  Accuracy: {accuracy:.1f}% ({correct_count}/{num_samples})")
            print(f"  Avg inference time: {results['results'][ctx_len]['avg_inference_time']:.2f}s")
        
        # 汇总
        results['average_accuracy'] = np.mean(all_accuracies)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n{'Context':<12} {'Accuracy':<12} {'Target':<12} {'Status':<10}")
        print("-"*55)
        
        targets = {
            4096: 98.5,
            16384: 91.3,
            65536: 78.2,
            131072: 68.2
        }
        
        for ctx_len in context_lengths:
            acc = results['results'][ctx_len]['accuracy']
            target = targets.get(ctx_len, None)
            
            if target is not None:
                status = "✅ PASS" if abs(acc - target) < 5.0 else "⚠️ CLOSE" if abs(acc - target) < 10.0 else "❌ FAIL"
            else:
                status = "N/A"
            
            ctx_str = f"{ctx_len//1024}K" if ctx_len >= 1024 else f"{ctx_len}"
            target_str = f"{target:.1f}%" if target else "N/A"
            print(f"{ctx_str:<12} {acc:>8.1f}%    {target_str:<12} {status:<10}")
        
        print(f"\nAverage Accuracy: {results['average_accuracy']:.1f}% (Target: 86.9%)")
        
        return results


def main():
    import argparse
    from model_loader import load_adb_model
    
    parser = argparse.ArgumentParser(description='Needle-in-Haystack Real Model Test')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[4096, 16384, 65536])
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='results/needle_haystack_real.json')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, config = load_adb_model(
        checkpoint_path=args.checkpoint,
        model_size=args.size,
        device=args.device
    )
    
    print(f"Model loaded: {config}")
    
    validator = NeedleHaystackValidator(model, device=args.device)
    results = validator.run_test(
        context_lengths=args.lengths,
        num_samples=args.num_samples
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
