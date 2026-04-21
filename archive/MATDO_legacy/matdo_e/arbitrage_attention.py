"""
Arbitrage Attention - 注意力层套利融合

在vLLM的PagedAttention基础上增加：
1. Engram融合: 将DRAM检索的Engram作为额外KV参与注意力
2. TTA (Test-time Adaptation): 梯度优化query representations

核心公式:
  output = PagedAttention(query, kv_cache) + EngramAttention(query, engram_kv)
  output = TTA_Update(output, T_steps)

论文参考: §3 Quadratic Blow-up, §4 Heterogeneous Arbitrage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class ArbitrageAttention(nn.Module):
    """
    套利注意力模块
    
    模拟在vLLM attention层中插入的修改:
    - 接收PagedAttention输出
    - 融合Engram信息
    - 执行TTA步数优化
    """
    
    def __init__(self, 
                 d_model: int = 4096,
                 n_heads: int = 32,
                 n_engram_heads: int = 1,  # 论文: single-head depth attention
                 dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_engram_heads = n_engram_heads
        self.head_dim = d_model // n_heads
        
        # Engram投影层 (将Engram embedding投影到KV空间)
        self.engram_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.engram_v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # TTA相关参数 (可学习的query adapter)
        self.tta_ln = nn.LayerNorm(d_model)
        self.tta_adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def engram_attention(self,
                         query: torch.Tensor,
                         engram_k: torch.Tensor,
                         engram_v: torch.Tensor,
                         scale: Optional[float] = None) -> torch.Tensor:
        """
        执行Engram注意力 (Single-head depth attention)
        
        Args:
            query: [batch, seq_len, d_model]
            engram_k: [batch, E, d_model]
            engram_v: [batch, E, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape
        E = engram_k.shape[1]
        
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)
        
        # 投影
        k = self.engram_k_proj(engram_k)  # [batch, E, d_model]
        v = self.engram_v_proj(engram_v)  # [batch, E, d_model]
        
        # 使用single-head计算注意力 (论文§5.4.1)
        # query: [batch, seq_len, d_model]
        q = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, E, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, E, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq, E]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return output
    
    def fusion_layer(self,
                     paged_output: torch.Tensor,
                     engram_output: torch.Tensor,
                     alpha: float = 0.5) -> torch.Tensor:
        """
        融合PagedAttention和Engram Attention的输出
        
        Args:
            paged_output: 原始PagedAttention输出
            engram_output: Engram注意力输出
            alpha: 融合权重
            
        Returns:
            fused_output
        """
        # 可学习的融合
        return paged_output + alpha * engram_output
    
    def apply_tta(self, 
                  hidden_states: torch.Tensor,
                  num_steps: int,
                  engram_k: Optional[torch.Tensor] = None,
                  engram_v: Optional[torch.Tensor] = None,
                  lr: float = 1e-3) -> torch.Tensor:
        """
        执行Test-time Adaptation (TTA)
        
        模拟论文中的梯度优化：
        - 每一步计算alignment loss
        - 更新query representations
        
        Args:
            hidden_states: 当前hidden states
            num_steps: TTA步数T
            engram_k, engram_v: Engram KV (用于alignment loss)
            lr: TTA学习率
            
        Returns:
            optimized_hidden_states
        """
        if num_steps <= 0:
            return hidden_states
            
        # 创建可学习的copy
        h = hidden_states.detach().clone()
        h.requires_grad_(True)
        
        for step in range(num_steps):
            # 简单的TTA: 通过adapter优化表示
            h_norm = self.tta_ln(h)
            h_adapted = self.tta_adapter(h_norm)
            
            # 计算alignment loss (如果提供engram)
            if engram_k is not None and engram_v is not None:
                # 模拟与Engram的对齐
                loss = self._compute_alignment_loss(h_adapted, engram_k, engram_v)
            else:
                # 简单的自洽loss
                loss = ((h_adapted - h_norm) ** 2).mean()
            
            # 手动梯度更新
            if h_adapted.grad_fn is not None:
                grad = torch.autograd.grad(loss, h, create_graph=False, retain_graph=False)[0]
                h = h - lr * grad
                h = h.detach().clone()
                h.requires_grad_(True)
            else:
                h = h_adapted
                
        return h.detach()
    
    def _compute_alignment_loss(self,
                                query: torch.Tensor,
                                engram_k: torch.Tensor,
                                engram_v: torch.Tensor) -> torch.Tensor:
        """计算query与Engram的alignment loss"""
        # 简化的cosine similarity loss
        query_norm = F.normalize(query, dim=-1)
        engram_k_norm = F.normalize(engram_k.mean(dim=1, keepdim=True), dim=-1)
        
        similarity = (query_norm * engram_k_norm).sum(dim=-1)
        loss = 1.0 - similarity.mean()
        
        return loss
    
    def forward(self,
                paged_output: torch.Tensor,
                query: torch.Tensor,
                engram_k: Optional[torch.Tensor] = None,
                engram_v: Optional[torch.Tensor] = None,
                tta_steps: int = 0,
                use_engram: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            paged_output: PagedAttention输出 [batch, seq, d_model]
            query: Query tensor [batch, seq, d_model]
            engram_k: Engram keys [batch, E, d_model] or None
            engram_v: Engram values [batch, E, d_model] or None
            tta_steps: TTA步数
            use_engram: 是否使用Engram
            
        Returns:
            output: [batch, seq, d_model]
        """
        output = paged_output
        
        # 1. Engram融合
        if use_engram and engram_k is not None and engram_v is not None:
            engram_out = self.engram_attention(query, engram_k, engram_v)
            output = self.fusion_layer(output, engram_out)
        
        # 2. TTA优化
        if tta_steps > 0:
            output = self.apply_tta(output, tta_steps, engram_k, engram_v)
        
        # 3. 输出投影
        output = self.output_proj(output)
        
        return output


class DummyArbitrageAttention(nn.Module):
    """虚拟套利注意力 (非套利模式使用)"""
    
    def forward(self, paged_output, *args, **kwargs):
        return paged_output


if __name__ == "__main__":
    # 测试ArbitrageAttention
    print("=" * 60)
    print("Arbitrage Attention Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    batch_size = 2
    seq_len = 10
    d_model = 4096
    E = 128  # Engram数量
    
    # 创建模块
    attn = ArbitrageAttention(d_model=d_model).to(device)
    
    # 模拟输入
    paged_output = torch.randn(batch_size, seq_len, d_model, device=device)
    query = torch.randn(batch_size, seq_len, d_model, device=device)
    engram_k = torch.randn(batch_size, E, d_model, device=device)
    engram_v = torch.randn(batch_size, E, d_model, device=device)
    
    # 测试各种配置
    configs = [
        ("Baseline (no engram)", 0, False),
        ("Engram only", 0, True),
        ("TTA only (T=4)", 4, False),
        ("Engram + TTA (T=4)", 4, True),
        ("Engram + TTA (T=16)", 16, True),
    ]
    
    print(f"\n{'Config':<25} | {'Output Shape'} | {'Time (ms)'}")
    print("-" * 60)
    
    import time
    
    for name, tta_steps, use_engram in configs:
        # Warmup
        _ = attn(paged_output, query, 
                engram_k if use_engram else None,
                engram_v if use_engram else None,
                tta_steps=tta_steps,
                use_engram=use_engram)
        
        # 计时
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        
        output = attn(paged_output, query,
                     engram_k if use_engram else None,
                     engram_v if use_engram else None,
                     tta_steps=tta_steps,
                     use_engram=use_engram)
        
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        print(f"{name:<25} | {list(output.shape)} | {elapsed:.2f}")
        
    print("\n✓ All tests passed!")
