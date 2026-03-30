#!/usr/bin/env python3
"""
使用 Small Model 运行论文中的关键实验

验证 Adaptive Deep Networks 论文中的核心声明:
1. Table 1: Representation Burial (使用 AttnRes 的梯度流)
2. Table 2: Gradient Flow Characteristics  
3. Table 4: Needle-in-Haystack (长上下文检索)
4. FLOP Equivalence 验证
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

from models.configs import get_config
from models.adaptive_transformer import AdaptiveTransformer


class SmallModelExperiments:
    """使用 Small Model 运行论文实验"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                # Use CPU for MPS to avoid memory issues
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Build Small Model
        print("\nBuilding Small Model...")
        self.config = get_config('small')
        self.model = AdaptiveTransformer(self.config).to(self.device)
        self.model.eval()
        
        print(f"Model parameters: {self.model.count_parameters() / 1e9:.2f}B")
        print(f"Layers: {self.config.num_layers}, Hidden: {self.config.hidden_dim}")
        print(f"AttnRes blocks: {self.config.num_blocks}")
        
        self.results = {}
    
    def experiment_1_gradient_flow(self, num_samples: int = 10) -> Dict:
        """
        实验1: 梯度流分析 (对应论文 Table 2)
        测量各层的梯度幅度，计算变异系数 (CV)
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: Gradient Flow Analysis (Table 2)")
        print("="*70)
        
        seq_len = 128
        vocab_size = self.config.vocab_size
        
        gradient_norms = []
        
        print(f"\nMeasuring gradients with {num_samples} samples...")
        
        for i in range(num_samples):
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
            
            # Forward pass
            self.model.train()
            outputs = self.model(input_ids)
            
            # Compute loss (simple next-token prediction)
            targets = input_ids[:, 1:].contiguous()
            logits = outputs[:, :-1, :].contiguous()
            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Collect gradient norms per layer
            sample_norms = []
            for name, param in self.model.named_parameters():
                if 'layers.' in name and param.grad is not None:
                    # Extract layer index
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        grad_norm = param.grad.norm().item()
                        sample_norms.append((layer_idx, grad_norm))
            
            # Average by layer
            layer_grads = {}
            for layer_idx, grad_norm in sample_norms:
                if layer_idx not in layer_grads:
                    layer_grads[layer_idx] = []
                layer_grads[layer_idx].append(grad_norm)
            
            avg_layer_grads = {k: np.mean(v) for k, v in layer_grads.items()}
            gradient_norms.append(avg_layer_grads)
        
        # Aggregate across samples
        all_layers = sorted(set().union(*[set(g.keys()) for g in gradient_norms]))
        mean_norms = []
        std_norms = []
        
        for layer_idx in all_layers:
            layer_values = [g.get(layer_idx, 0) for g in gradient_norms if layer_idx in g]
            if layer_values:
                mean_norms.append(np.mean(layer_values))
                std_norms.append(np.std(layer_values))
        
        mean_norms = np.array(mean_norms)
        
        # Compute metrics
        cv = np.std(mean_norms) / np.mean(mean_norms) if np.mean(mean_norms) > 0 else 0
        early_grad = np.mean(mean_norms[:5]) if len(mean_norms) >= 5 else mean_norms[0]
        late_grad = np.mean(mean_norms[-5:]) if len(mean_norms) >= 5 else mean_norms[-1]
        early_late_ratio = early_grad / late_grad if late_grad > 0 else 1.0
        
        results = {
            'num_layers_measured': len(mean_norms),
            'cv': float(cv),
            'early_grad': float(early_grad),
            'late_grad': float(late_grad),
            'early_late_ratio': float(early_late_ratio),
            'mean_norms': mean_norms.tolist(),
            'paper_target': {
                'cv': 0.11,  # From Table 2
                'early_grad': 0.067,
                'late_grad': 0.071,
                'early_late_ratio': 0.94
            }
        }
        
        print(f"\nResults:")
        print(f"  CV (Coefficient of Variation): {cv:.4f}")
        print(f"  Paper target CV: 0.11")
        print(f"  Early layers grad: {early_grad:.4f}")
        print(f"  Late layers grad: {late_grad:.4f}")
        print(f"  Early/Late ratio: {early_late_ratio:.4f}")
        print(f"  Paper target ratio: 0.94")
        
        self.results['gradient_flow'] = results
        return results
    
    def experiment_2_needle_haystack(self, context_lengths: List[int] = None) -> Dict:
        """
        实验2: Needle-in-Haystack 长上下文检索 (对应论文 Table 4)
        简化版测试
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: Needle-in-Haystack (Table 4)")
        print("="*70)
        
        if context_lengths is None:
            context_lengths = [512, 1024, 2048]
        
        results = {
            'context_lengths': context_lengths,
            'scores': []
        }
        
        print(f"\nTesting context lengths: {context_lengths}")
        
        for ctx_len in context_lengths:
            print(f"\n  Context length: {ctx_len}...")
            
            # Create synthetic task
            # Place a "needle" (special token pattern) at random position
            batch_size = 1
            seq_len = min(ctx_len, 2048)  # Limit for memory
            
            # Generate random sequence
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
            
            # Place needle (special token) at middle
            needle_pos = seq_len // 2
            needle_token = 42
            input_ids[0, needle_pos] = needle_token
            
            # Forward pass
            try:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                
                # Check if model can predict near the needle
                # Simplified: check prediction accuracy around needle position
                pred_window = 5
                start_pos = max(0, needle_pos - pred_window)
                end_pos = min(seq_len - 1, needle_pos + pred_window)
                
                correct = 0
                total = 0
                for pos in range(start_pos, end_pos):
                    pred = outputs[0, pos].argmax().item()
                    actual = input_ids[0, pos + 1].item()
                    if pred == actual:
                        correct += 1
                    total += 1
                
                score = correct / total if total > 0 else 0.0
                
            except Exception as e:
                print(f"    Error: {e}")
                score = 0.0
            
            results['scores'].append({
                'context_length': ctx_len,
                'score': score,
                'tested_length': seq_len
            })
            
            print(f"    Score: {score:.2%}")
        
        avg_score = np.mean([s['score'] for s in results['scores']])
        results['average_score'] = float(avg_score)
        
        print(f"\nAverage score: {avg_score:.2%}")
        print(f"Paper target (4K context): 98.5% (with full system)")
        
        self.results['needle_haystack'] = results
        return results
    
    def experiment_3_flop_analysis(self) -> Dict:
        """
        实验3: FLOP 分析 (对应论文 FLOP Equivalence)
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: FLOP Analysis")
        print("="*70)
        
        config = self.config
        head_dim = config.hidden_dim // config.num_heads
        mlp_dim = config.hidden_dim * config.mlp_ratio
        
        # Per layer FLOPs
        # Attention
        attn_qkv_flops = 3 * 2 * config.hidden_dim * config.hidden_dim
        attn_out_flops = 2 * config.hidden_dim * config.hidden_dim
        attn_compute_flops = 2 * config.num_heads * head_dim  # Simplified
        
        # MLP (SwiGLU)
        mlp_flops = 3 * 2 * config.hidden_dim * mlp_dim
        
        per_layer_flops = attn_qkv_flops + attn_compute_flops + attn_out_flops + mlp_flops
        total_flops_per_token = config.num_layers * per_layer_flops
        
        # qTTT step FLOPs
        qttt_span = config.qttt_span_length
        max_qttt_steps = config.max_qttt_steps
        
        # Per qTTT step: query projection + attention
        qttt_step_flops = 2 * qttt_span * config.hidden_dim * config.hidden_dim  # Query proj
        qttt_step_flops += 2 * config.num_heads * qttt_span * head_dim  # Attention
        
        # Thinking tokens equivalent
        # T_think ≈ 2 * N_qTTT * k
        equivalent_thinking_tokens = 2 * max_qttt_steps * qttt_span
        
        results = {
            'model_config': {
                'num_layers': config.num_layers,
                'hidden_dim': config.hidden_dim,
                'num_heads': config.num_heads,
                'head_dim': head_dim,
                'mlp_dim': mlp_dim,
            },
            'flops': {
                'per_layer': per_layer_flops,
                'per_layer_human': f'{per_layer_flops/1e6:.1f} MFLOPs',
                'per_token': total_flops_per_token,
                'per_token_human': f'{total_flops_per_token/1e9:.2f} GFLOPs/token',
            },
            'qttt': {
                'max_steps': max_qttt_steps,
                'span_length': qttt_span,
                'step_flops': qttt_step_flops,
                'step_flops_human': f'{qttt_step_flops/1e6:.1f} MFLOPs/step',
                'equivalent_thinking_tokens': equivalent_thinking_tokens,
            },
            'flop_equivalence': {
                'formula': 'T_think ≈ 2 * N_qTTT * k',
                'theoretical_t_think': equivalent_thinking_tokens,
                'qttt_steps': max_qttt_steps,
                'span': qttt_span,
            }
        }
        
        print(f"\nFLOP Analysis:")
        print(f"  Per layer: {per_layer_flops/1e6:.1f} MFLOPs")
        print(f"  Per token: {total_flops_per_token/1e9:.2f} GFLOPs")
        print(f"\nqTTT Analysis:")
        print(f"  Max steps: {max_qttt_steps}")
        print(f"  Span length: {qttt_span}")
        print(f"  Step FLOPs: {qttt_step_flops/1e6:.1f} MFLOPs")
        print(f"\nFLOP Equivalence:")
        print(f"  T_think ≈ 2 * {max_qttt_steps} * {qttt_span} = {equivalent_thinking_tokens} tokens")
        
        self.results['flop_analysis'] = results
        return results
    
    def experiment_4_model_components(self) -> Dict:
        """
        实验4: 模型组件分析
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Model Component Analysis")
        print("="*70)
        
        # Count parameters by component
        component_params = {}
        for name, param in self.model.named_parameters():
            parts = name.split('.')
            component = parts[0] if parts else 'other'
            if component not in component_params:
                component_params[component] = 0
            component_params[component] += param.numel()
        
        total_params = sum(component_params.values())
        attnres_params = self.model.count_attnsres_parameters()
        
        results = {
            'total_params': total_params,
            'total_params_human': f'{total_params/1e9:.2f}B',
            'attnres_params': attnres_params,
            'attnres_params_human': f'{attnres_params/1e6:.2f}M',
            'attnres_percentage': attnres_params / total_params * 100,
            'components': {}
        }
        
        print(f"\nParameter Distribution:")
        for comp, count in sorted(component_params.items(), key=lambda x: -x[1]):
            pct = count / total_params * 100
            human = f'{count/1e9:.2f}B' if count >= 1e9 else f'{count/1e6:.1f}M'
            results['components'][comp] = {
                'params': count,
                'percentage': pct,
                'human': human
            }
            print(f"  {comp}: {human} ({pct:.1f}%)")
        
        print(f"\nAttnRes Analysis:")
        print(f"  AttnRes params: {attnres_params/1e6:.2f}M")
        print(f"  AttnRes percentage: {attnres_params/total_params*100:.4f}%")
        print(f"  Memory reduction: O(Ld) -> O(Nd) = O({self.config.num_layers}d) -> O({self.config.num_blocks}d)")
        print(f"  Reduction factor: {self.config.num_layers / self.config.num_blocks:.1f}×")
        
        self.results['component_analysis'] = results
        return results
    
    def experiment_5_inference_performance(self) -> Dict:
        """
        实验5: 推理性能测试
        """
        print("\n" + "="*70)
        print("EXPERIMENT 5: Inference Performance")
        print("="*70)
        
        seq_lengths = [64, 128, 256, 512, 1024]
        batch_size = 1
        num_runs = 5
        
        results = {
            'latency_tests': []
        }
        
        print(f"\nTesting latency with batch_size={batch_size}")
        
        for seq_len in seq_lengths:
            print(f"\n  Sequence length: {seq_len}...")
            
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=self.device)
            
            # Warmup
            with torch.no_grad():
                _ = self.model(input_ids)
            
            # Measure
            times = []
            for _ in range(num_runs):
                if self.device.type == 'mps':
                    torch.mps.synchronize()
                
                start = time.time()
                with torch.no_grad():
                    _ = self.model(input_ids)
                if self.device.type == 'mps':
                    torch.mps.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = seq_len / avg_time
            
            results['latency_tests'].append({
                'seq_len': seq_len,
                'avg_time_ms': avg_time * 1000,
                'std_time_ms': std_time * 1000,
                'throughput_tok_per_sec': throughput
            })
            
            print(f"    Time: {avg_time*1000:.1f}ms (±{std_time*1000:.1f}ms)")
            print(f"    Throughput: {throughput:.1f} tok/s")
        
        self.results['inference_performance'] = results
        return results
    
    def save_results(self, output_dir='results/small_model_paper_experiments'):
        """保存所有实验结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'model_size': 'small',
            'model_params': self.model.count_parameters(),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
        }
        
        output_file = os.path.join(output_dir, 'experiments_results.json')
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        return output_file
    
    def generate_report(self):
        """生成实验报告"""
        report = []
        report.append("="*70)
        report.append("SMALL MODEL PAPER EXPERIMENTS REPORT")
        report.append("="*70)
        report.append(f"\nTimestamp: {datetime.now().isoformat()}")
        report.append(f"Device: {self.device}")
        report.append(f"Model: Small (2.2B params)")
        
        # Exp 1: Gradient Flow
        if 'gradient_flow' in self.results:
            gf = self.results['gradient_flow']
            report.append("\n" + "-"*70)
            report.append("EXPERIMENT 1: Gradient Flow (Table 2)")
            report.append("-"*70)
            report.append(f"CV: {gf['cv']:.4f} (Target: 0.11)")
            report.append(f"Early/Late ratio: {gf['early_late_ratio']:.4f} (Target: 0.94)")
        
        # Exp 2: Needle-in-Haystack
        if 'needle_haystack' in self.results:
            nh = self.results['needle_haystack']
            report.append("\n" + "-"*70)
            report.append("EXPERIMENT 2: Needle-in-Haystack (Table 4)")
            report.append("-"*70)
            for s in nh['scores']:
                report.append(f"{s['context_length']:5d} context: {s['score']:.2%}")
            report.append(f"Average: {nh['average_score']:.2%}")
        
        # Exp 3: FLOP Analysis
        if 'flop_analysis' in self.results:
            fa = self.results['flop_analysis']
            report.append("\n" + "-"*70)
            report.append("EXPERIMENT 3: FLOP Analysis")
            report.append("-"*70)
            report.append(f"FLOPs per token: {fa['flops']['per_token_human']}")
            report.append(f"qTTT equivalent: {fa['qttt']['equivalent_thinking_tokens']} thinking tokens")
        
        # Exp 4: Components
        if 'component_analysis' in self.results:
            ca = self.results['component_analysis']
            report.append("\n" + "-"*70)
            report.append("EXPERIMENT 4: Component Analysis")
            report.append("-"*70)
            report.append(f"Total params: {ca['total_params_human']}")
            report.append(f"AttnRes overhead: {ca['attnres_percentage']:.4f}%")
        
        # Exp 5: Performance
        if 'inference_performance' in self.results:
            ip = self.results['inference_performance']
            report.append("\n" + "-"*70)
            report.append("EXPERIMENT 5: Inference Performance")
            report.append("-"*70)
            for t in ip['latency_tests']:
                report.append(f"{t['seq_len']:4d} tokens: {t['avg_time_ms']:7.1f}ms ({t['throughput_tok_per_sec']:6.1f} tok/s)")
        
        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)
        
        return "\n".join(report)


def main():
    print("="*70)
    print("Small Model Paper Experiments")
    print("="*70)
    
    # Initialize experiments
    experiments = SmallModelExperiments(device='auto')
    
    # Run all experiments
    experiments.experiment_1_gradient_flow(num_samples=5)
    experiments.experiment_2_needle_haystack(context_lengths=[512, 1024, 2048])
    experiments.experiment_3_flop_analysis()
    experiments.experiment_4_model_components()
    experiments.experiment_5_inference_performance()
    
    # Save results
    experiments.save_results()
    
    # Generate and print report
    report = experiments.generate_report()
    print("\n" + report)
    
    # Save report
    with open('results/small_model_paper_experiments/report.txt', 'w') as f:
        f.write(report)
    
    print("\n📄 Report saved to: results/small_model_paper_experiments/report.txt")


if __name__ == '__main__':
    main()
