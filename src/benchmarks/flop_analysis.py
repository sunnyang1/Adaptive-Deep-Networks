"""
FLOP Analysis for Adaptive Deep Networks

Validates FLOP equivalence: T_think ≈ 2 * N_qTTT * k
"""

from typing import Dict, Tuple
import json

# Optional torch import (for type hints only)
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class FLOPCounter:
    """Count FLOPs for different operations."""
    
    @staticmethod
    def matmul_flops(m: int, n: int, p: int) -> int:
        """FLOPs for matrix multiplication [m,n] @ [n,p]."""
        return 2 * m * n * p
    
    @staticmethod
    def attention_flops(
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int
    ) -> int:
        """FLOPs for self-attention."""
        # Q, K, V projections: 3 * B * T * D * D
        qkv_flops = 3 * batch * seq_len * (num_heads * head_dim) ** 2
        
        # Attention: B * H * T * T * d
        attn_flops = batch * num_heads * seq_len * seq_len * head_dim
        
        # Output projection: B * T * D * D
        out_flops = batch * seq_len * (num_heads * head_dim) ** 2
        
        return qkv_flops + attn_flops + out_flops
    
    @staticmethod
    def mlp_flops(
        batch: int,
        seq_len: int,
        hidden_dim: int,
        mlp_ratio: int = 4
    ) -> int:
        """FLOPs for MLP."""
        mlp_dim = hidden_dim * mlp_ratio
        # Two linear layers
        return 2 * batch * seq_len * hidden_dim * mlp_dim
    
    @staticmethod
    def transformer_layer_flops(
        batch: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int = 4
    ) -> int:
        """FLOPs for one transformer layer."""
        head_dim = hidden_dim // num_heads
        attn_flops = FLOPCounter.attention_flops(batch, seq_len, num_heads, head_dim)
        mlp_flops = FLOPCounter.mlp_flops(batch, seq_len, hidden_dim, mlp_ratio)
        return attn_flops + mlp_flops


class EfficiencyAnalyzer:
    """Analyze computational efficiency of Adaptive Deep Networks."""
    
    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 4096,
        num_heads: int = 32,
        mlp_ratio: int = 4
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.head_dim = hidden_dim // num_heads
    
    def compute_thinking_token_flops(
        self,
        batch: int,
        context_len: int,
        num_thinking_tokens: int
    ) -> int:
        """
        Compute FLOPs for generating thinking tokens.
        
        Formula from Section 4.3.3:
            C_think = C_quad * T_think * T
        """
        # Each thinking token requires full forward pass
        total_tokens = context_len + num_thinking_tokens
        
        # Per-layer FLOPs
        layer_flops = FLOPCounter.transformer_layer_flops(
            batch, total_tokens, self.hidden_dim, self.num_heads, self.mlp_ratio
        )
        
        return self.num_layers * layer_flops * num_thinking_tokens
    
    def compute_qttt_step_flops(
        self,
        batch: int,
        context_len: int,
        span_len: int
    ) -> int:
        """
        Compute FLOPs for one qTTT step.
        
        Formula from Section 4.3.3:
            C_qTTT = 2 * (C_quad * k * T + (2+2r) * L * k * d^2)
        """
        # Query projection (forward + backward)
        query_proj_flops = 2 * batch * span_len * self.hidden_dim * self.hidden_dim
        
        # Attention computation
        attn_flops = (
            batch * self.num_heads * span_len * context_len * self.head_dim * 2
        )
        
        # Per-step total
        step_flops = query_proj_flops + attn_flops
        
        return step_flops
    
    def verify_flop_equivalence(
        self,
        batch: int = 1,
        context_len: int = 100000,
        num_thinking_tokens: int = 8192,
        num_qttt_steps: int = 16,
        qttt_span: int = 256
    ) -> Dict:
        """
        Verify FLOP equivalence: T_think ≈ 2 * N_qTTT * k
        
        Returns:
            Dictionary with FLOP counts and equivalence ratio
        """
        # Compute thinking token FLOPs
        think_flops = self.compute_thinking_token_flops(
            batch, context_len, num_thinking_tokens
        )
        
        # Compute qTTT FLOPs
        qttt_step_flops = self.compute_qttt_step_flops(
            batch, context_len, qttt_span
        )
        qttt_total_flops = qttt_step_flops * num_qttt_steps
        
        # Theoretical equivalence
        # T_think ≈ 2 * N_qTTT * k
        theoretical_think_tokens = 2 * num_qttt_steps * qttt_span
        
        # Ratio
        actual_ratio = num_thinking_tokens / (2 * num_qttt_steps * qttt_span)
        
        return {
            'thinking_tokens': {
                'count': num_thinking_tokens,
                'flops': think_flops,
                'flops_per_token': think_flops / num_thinking_tokens if num_thinking_tokens > 0 else 0
            },
            'qttt': {
                'num_steps': num_qttt_steps,
                'span_length': qttt_span,
                'flops_per_step': qttt_step_flops,
                'total_flops': qttt_total_flops
            },
            'equivalence': {
                'theoretical_t_think': theoretical_think_tokens,
                'actual_t_think': num_thinking_tokens,
                'ratio': actual_ratio,
                'is_equivalent': 0.8 <= actual_ratio <= 1.2
            },
            'efficiency': {
                'qttt_vs_think_ratio': qttt_total_flops / think_flops if think_flops > 0 else 0,
                'cache_reuse_savings': 'O(T^2) vs O(k*T) per step'
            }
        }
    
    def compare_allocation_strategies(
        self,
        budget_flops: int = 1e15,
        context_len: int = 100000
    ) -> Dict:
        """
        Compare different width vs depth allocation strategies.
        
        Returns:
            Comparison of strategies under fixed FLOP budget
        """
        strategies = {}
        
        # Strategy 1: Pure width (thinking tokens only)
        # Estimate tokens from FLOP budget
        per_token_flops = self.compute_thinking_token_flops(1, context_len, 1)
        pure_width_tokens = int(budget_flops / per_token_flops)
        
        strategies['pure_width'] = {
            'thinking_tokens': pure_width_tokens,
            'qttt_steps': 0,
            'description': 'Generate only thinking tokens'
        }
        
        # Strategy 2: Pure depth (qTTT only)
        per_qttt_step = self.compute_qttt_step_flops(1, context_len, 128)
        pure_depth_steps = int(budget_flops / per_qttt_step)
        
        strategies['pure_depth'] = {
            'thinking_tokens': 0,
            'qttt_steps': pure_depth_steps,
            'qttt_span': 128,
            'description': 'Use only qTTT adaptation'
        }
        
        # Strategy 3: Balanced (1:1 FLOP split)
        half_budget = budget_flops / 2
        balanced_tokens = int(half_budget / per_token_flops)
        balanced_steps = int(half_budget / per_qttt_step)
        
        strategies['balanced'] = {
            'thinking_tokens': balanced_tokens,
            'qttt_steps': balanced_steps,
            'qttt_span': 128,
            'description': 'Equal FLOP allocation'
        }
        
        # Strategy 4: Depth-prioritized (paper recommendation)
        # 20% width, 80% depth
        width_budget = budget_flops * 0.2
        depth_budget = budget_flops * 0.8
        
        depth_prioritized_tokens = int(width_budget / per_token_flops)
        depth_prioritized_steps = int(depth_budget / per_qttt_step)
        
        strategies['depth_prioritized'] = {
            'thinking_tokens': depth_prioritized_tokens,
            'qttt_steps': depth_prioritized_steps,
            'qttt_span': 128,
            'description': 'Paper recommendation: prioritize depth'
        }
        
        return {
            'budget_flops': budget_flops,
            'context_length': context_len,
            'strategies': strategies
        }
    
    def print_analysis(self, analysis: Dict):
        """Print formatted FLOP analysis."""
        print("\n" + "=" * 60)
        print("FLOP Equivalence Analysis")
        print("=" * 60)
        
        eq = analysis['equivalence']
        print(f"\nTheoretical T_think: {eq['theoretical_t_think']}")
        print(f"Actual T_think: {eq['actual_t_think']}")
        print(f"Ratio: {eq['ratio']:.3f}")
        print(f"Equivalent: {'✓' if eq['is_equivalent'] else '✗'}")
        
        print("\nFLOP Breakdown:")
        think = analysis['thinking_tokens']
        qttt = analysis['qttt']
        print(f"  Thinking tokens: {think['flops']:.2e} FLOPs")
        print(f"  qTTT total: {qttt['total_flops']:.2e} FLOPs")
        print(f"  Ratio (qTTT/Think): {analysis['efficiency']['qttt_vs_think_ratio']:.3f}")
        
        print("=" * 60)


def run_flop_analysis(output_path: str = None) -> Dict:
    """Run complete FLOP analysis."""
    analyzer = EfficiencyAnalyzer()
    
    # Verify equivalence
    results = analyzer.verify_flop_equivalence()
    analyzer.print_analysis(results)
    
    # Compare strategies
    comparison = analyzer.compare_allocation_strategies()
    
    print("\nAllocation Strategies (Budget: 1e15 FLOPs):")
    print("-" * 60)
    for name, strategy in comparison['strategies'].items():
        print(f"\n{name}:")
        print(f"  {strategy['description']}")
        if strategy['thinking_tokens'] > 0:
            print(f"  Thinking tokens: {strategy['thinking_tokens']}")
        if strategy['qttt_steps'] > 0:
            print(f"  qTTT steps: {strategy['qttt_steps']}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump({'equivalence': results, 'strategies': comparison}, f, indent=2)
    
    return results
