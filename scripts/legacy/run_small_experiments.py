#!/usr/bin/env python3
"""
Small Model Experiments - Optimized for CPU/MPS
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import torch
import json
import time
import numpy as np
from datetime import datetime

from models.configs import get_config
from models.adaptive_transformer import AdaptiveTransformer


def run_experiments():
    print('='*70)
    print('Small Model Experiments')
    print('='*70)
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    
    # Build model
    config = get_config('small')
    print(f'\nBuilding Small Model...')
    print(f'  Layers: {config.num_layers}, Hidden: {config.hidden_dim}')
    
    model = AdaptiveTransformer(config).to(device)
    model.eval()
    
    total_params = model.count_parameters()
    attnres_params = model.count_attnsres_parameters()
    print(f'  Parameters: {total_params / 1e9:.2f}B')
    print(f'  AttnRes overhead: {attnres_params / 1e6:.2f}M ({attnres_params/total_params*100:.4f}%)')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_config': {
            'num_layers': config.num_layers,
            'hidden_dim': config.hidden_dim,
            'num_heads': config.num_heads,
            'num_blocks': config.num_blocks,
            'vocab_size': config.vocab_size,
            'max_seq_len': config.max_seq_len,
            'max_qttt_steps': config.max_qttt_steps,
            'qttt_span_length': config.qttt_span_length,
        },
        'model_metrics': {
            'total_params': total_params,
            'attnres_params': attnres_params,
            'attnres_percentage': attnres_params / total_params * 100,
        }
    }
    
    # Experiment 1: Latency vs Sequence Length (shorter sequences)
    print('\n' + '-'*70)
    print('Experiment 1: Latency vs Sequence Length')
    print('-'*70)
    
    seq_lengths = [64, 128, 256, 512]
    batch_size = 1
    num_runs = 3
    
    latency_results = []
    for seq_len in seq_lengths:
        print(f'  seq_len={seq_len}: ', end='', flush=True)
        try:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            
            # Warmup
            with torch.no_grad():
                _ = model(input_ids)
            if device.type == 'mps':
                torch.mps.synchronize()
            
            # Measure
            times = []
            for _ in range(num_runs):
                if device.type == 'mps':
                    torch.mps.empty_cache()
                
                start = time.time()
                with torch.no_grad():
                    _ = model(input_ids)
                if device.type == 'mps':
                    torch.mps.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = seq_len / avg_time
            
            result = {
                'seq_len': seq_len,
                'avg_time_ms': round(avg_time * 1000, 2),
                'std_time_ms': round(std_time * 1000, 2),
                'throughput_tok_per_sec': round(throughput, 1),
            }
            latency_results.append(result)
            
            print(f'{avg_time*1000:.1f}ms ({throughput:.1f} tok/s)')
        except Exception as e:
            print(f'Error: {e}')
            latency_results.append({'seq_len': seq_len, 'error': str(e)})
    
    results['latency_vs_seq_len'] = latency_results
    
    # Experiment 2: Component Analysis
    print('\n' + '-'*70)
    print('Experiment 2: Component Analysis')
    print('-'*70)
    
    component_params = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        component = parts[0] if parts else 'other'
        component_params[component] = component_params.get(component, 0) + param.numel()
    
    component_analysis = {}
    for comp, count in sorted(component_params.items(), key=lambda x: -x[1]):
        pct = count / total_params * 100
        human = f'{count/1e9:.2f}B' if count >= 1e9 else f'{count/1e6:.1f}M'
        component_analysis[comp] = {
            'params': count,
            'human_readable': human,
            'percentage': round(pct, 2)
        }
        print(f'  {comp}: {human} ({pct:.1f}%)')
    
    results['component_analysis'] = component_analysis
    
    # Experiment 3: FLOP Estimation
    print('\n' + '-'*70)
    print('Experiment 3: FLOP Estimation')
    print('-'*70)
    
    head_dim = config.hidden_dim // config.num_heads
    mlp_dim = config.hidden_dim * config.mlp_ratio
    
    # Per layer FLOPs
    attn_qkv_flops = 3 * 2 * config.hidden_dim * config.hidden_dim  # Q, K, V projections
    attn_out_flops = 2 * config.hidden_dim * config.hidden_dim  # Output projection
    attn_compute_flops = 2 * config.num_heads * head_dim  # Attention computation (simplified)
    mlp_flops = 3 * 2 * config.hidden_dim * mlp_dim  # SwiGLU: gate, up, down
    
    per_layer_flops = attn_qkv_flops + attn_compute_flops + attn_out_flops + mlp_flops
    total_flops_per_token = config.num_layers * per_layer_flops
    
    flop_analysis = {
        'per_layer_flops': per_layer_flops,
        'per_layer_flops_human': f'{per_layer_flops/1e6:.1f} MFLOPs',
        'total_layers': config.num_layers,
        'total_flops_per_token': total_flops_per_token,
        'total_flops_per_token_human': f'{total_flops_per_token/1e9:.2f} GFLOPs/token',
        'breakdown': {
            'attn_qkv': attn_qkv_flops,
            'attn_compute': attn_compute_flops,
            'attn_out': attn_out_flops,
            'mlp': mlp_flops,
        }
    }
    
    print(f'  Per layer: {per_layer_flops/1e6:.1f} MFLOPs')
    print(f'  Total: {total_flops_per_token/1e9:.2f} GFLOPs/token')
    
    results['flop_analysis'] = flop_analysis
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_file = 'results/small_model_experiments.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print('\n' + '='*70)
    print(f'Results saved to: {output_file}')
    print('='*70)
    
    # Print summary
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Model: Small (2.2B params)')
    print(f'Device: {device}')
    print(f'\nLatency Results:')
    for r in latency_results:
        if 'error' not in r:
            print(f'  {r["seq_len"]:4d} tokens: {r["avg_time_ms"]:8.1f}ms ({r["throughput_tok_per_sec"]:6.1f} tok/s)')
    print(f'\nFLOPs: {total_flops_per_token/1e9:.2f} GFLOPs/token')
    print(f'AttnRes overhead: {attnres_params/total_params*100:.4f}%')
    print('='*70)
    
    return results


if __name__ == '__main__':
    run_experiments()
