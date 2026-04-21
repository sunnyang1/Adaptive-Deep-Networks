"""
实验测量工具函数
提供各种定量测量功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class LayerContribution:
    """层贡献度测量结果"""
    layer_idx: int
    grad_norm: float
    relative_contribution: float
    attention_pattern: Optional[torch.Tensor] = None


@dataclass
class MarginStats:
    """Margin统计结果"""
    mean: float
    std: float
    min_val: float
    max_val: float
    success_rate: float
    threshold: float


def measure_representation_burial(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_layer_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    测量Representation Burial现象
    
    Args:
        model: 待测模型
        input_ids: 输入token IDs [batch, seq_len]
        target_layer_indices: 要测量的层索引，None表示所有层
        
    Returns:
        {
            'layer_contributions': List[LayerContribution],
            'signal_attenuation_rate': float,  # (C1 - CL) / C1
            'effective_depth': int,  # 贡献度降到50%时的层数
            'cv': float  # 变异系数
        }
    """
    model.eval()
    layer_outputs = []
    hooks = []
    
    # 注册前向钩子捕获层输出
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        layer_outputs.append(output.detach().requires_grad_(True))
    
    # 获取所有transformer层
    layers = []
    for name, module in model.named_modules():
        if 'layer' in name.lower() or 'block' in name.lower():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.Module)):
                layers.append(module)
    
    if target_layer_indices is None:
        target_layer_indices = list(range(len(layers)))
    
    # 注册钩子
    for idx in target_layer_indices:
        if idx < len(layers):
            hook = layers[idx].register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # 前向传播
    outputs = model(input_ids)
    if isinstance(outputs, dict):
        logits = outputs.get('logits', outputs.get('hidden_states'))
    else:
        logits = outputs
    
    # 计算loss（简单的语言建模loss）
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # 计算各层梯度
    contributions = []
    for i, output in enumerate(layer_outputs):
        if output is not None and output.requires_grad:
            try:
                grad = torch.autograd.grad(
                    loss, output, 
                    retain_graph=True, 
                    allow_unused=True
                )[0]
                if grad is not None:
                    contribution = torch.norm(grad, dim=-1).mean().item()
                    contributions.append((target_layer_indices[i], contribution))
            except:
                pass
    
    # 清理钩子
    for hook in hooks:
        hook.remove()
    
    if not contributions:
        return {
            'layer_contributions': [],
            'signal_attenuation_rate': 0.0,
            'effective_depth': 0,
            'cv': 0.0
        }
    
    # 计算相对贡献度
    first_contribution = contributions[0][1]
    relative_contributions = [
        LayerContribution(
            layer_idx=idx,
            grad_norm=contrib,
            relative_contribution=contrib / first_contribution if first_contribution > 0 else 0
        )
        for idx, contrib in contributions
    ]
    
    # 计算信号衰减率
    last_contribution = contributions[-1][1]
    signal_attenuation_rate = (first_contribution - last_contribution) / first_contribution \
        if first_contribution > 0 else 0
    
    # 计算有效深度（贡献度降到50%时的层数）
    effective_depth = len(contributions)
    for i, contrib in enumerate(relative_contributions):
        if contrib.relative_contribution < 0.5:
            effective_depth = i
            break
    
    # 计算变异系数
    grad_norms = [c.grad_norm for c in relative_contributions]
    cv = np.std(grad_norms) / np.mean(grad_norms) if np.mean(grad_norms) > 0 else 0
    
    return {
        'layer_contributions': relative_contributions,
        'signal_attenuation_rate': signal_attenuation_rate,
        'effective_depth': effective_depth,
        'cv': cv
    }


def measure_attention_margin(
    model: nn.Module,
    input_ids: torch.Tensor,
    query_position: int,
    target_positions: List[int],
    distractor_positions: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    测量attention logit margin
    
    Args:
        model: 待测模型
        input_ids: 输入token IDs
        query_position: query token位置
        target_positions: 目标token位置列表
        distractor_positions: 干扰token位置列表，None表示非目标位置
        
    Returns:
        {
            'margin': float,  # 目标与最大干扰的logit差
            'target_logit': float,
            'max_distractor_logit': float,
            'attention_mass_on_target': float
        }
    """
    model.eval()
    
    with torch.no_grad():
        # 获取attention logits
        outputs = model(input_ids, output_attentions=True)
        
        if hasattr(outputs, 'attentions') and outputs.attentions:
            # 使用最后一层的attention
            last_layer_attn = outputs.attentions[-1]  # [batch, heads, seq, seq]
            
            # 取第一个batch和平均所有heads
            attn_weights = last_layer_attn[0].mean(dim=0)  # [seq, seq]
            
            # 获取query位置的attention分布
            query_attn = attn_weights[query_position]  # [seq]
            
            # 转换为logits (取log)
            logits = torch.log(query_attn + 1e-10)
            
            # 计算目标位置的平均logit
            target_logits = [logits[pos].item() for pos in target_positions]
            target_logit = np.mean(target_logits)
            
            # 确定干扰位置
            if distractor_positions is None:
                seq_len = logits.size(0)
                distractor_positions = [i for i in range(seq_len) 
                                       if i not in target_positions and i != query_position]
            
            # 计算最大干扰logit
            if distractor_positions:
                distractor_logits = [logits[pos].item() for pos in distractor_positions]
                max_distractor_logit = max(distractor_logits)
            else:
                max_distractor_logit = -float('inf')
            
            # 计算margin
            margin = target_logit - max_distractor_logit
            
            # 计算attention mass在目标上的比例
            attention_mass_on_target = sum([query_attn[pos].item() for pos in target_positions])
            
            return {
                'margin': margin,
                'target_logit': target_logit,
                'max_distractor_logit': max_distractor_logit,
                'attention_mass_on_target': attention_mass_on_target
            }
        else:
            # 如果模型不输出attention，返回空值
            return {
                'margin': 0.0,
                'target_logit': 0.0,
                'max_distractor_logit': 0.0,
                'attention_mass_on_target': 0.0
            }


def analyze_margin_distribution(
    model: nn.Module,
    test_samples: List[Dict],
    success_threshold: float = 0.5
) -> Dict[str, MarginStats]:
    """
    分析成功和失败样本的margin分布
    
    Args:
        model: 待测模型
        test_samples: 测试样本列表，每个样本包含input_ids, query_pos, target_pos, label
        success_threshold: 判定成功的阈值
        
    Returns:
        {
            'success': MarginStats,
            'failure': MarginStats,
            'overall': MarginStats
        }
    """
    success_margins = []
    failure_margins = []
    
    for sample in test_samples:
        result = measure_attention_margin(
            model,
            sample['input_ids'],
            sample['query_position'],
            sample['target_positions']
        )
        margin = result['margin']
        
        # 根据样本标注或预测判断成功与否
        is_success = sample.get('retrieval_success', margin > success_threshold)
        
        if is_success:
            success_margins.append(margin)
        else:
            failure_margins.append(margin)
    
    def compute_stats(margins):
        if not margins:
            return MarginStats(0, 0, 0, 0, 0, 0)
        return MarginStats(
            mean=np.mean(margins),
            std=np.std(margins),
            min_val=np.min(margins),
            max_val=np.max(margins),
            success_rate=len([m for m in margins if m > success_threshold]) / len(margins),
            threshold=np.median(margins) if len(margins) > 0 else 0
        )
    
    return {
        'success': compute_stats(success_margins),
        'failure': compute_stats(failure_margins),
        'overall': compute_stats(success_margins + failure_margins)
    }


def measure_gradient_statistics(
    model: nn.Module,
    batch: torch.Tensor,
    loss_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    测量各层梯度统计信息
    
    Args:
        model: 待测模型
        batch: 输入batch
        loss_fn: 损失函数，None使用默认的交叉熵
        
    Returns:
        {
            'layer_stats': List[Dict],
            'cv': float,  # 变异系数
            'early_late_ratio': float,  # 早期/晚期梯度比
            'gradient_flow_score': float  # 梯度流评分
        }
    """
    model.train()
    model.zero_grad()
    
    # 前向传播
    outputs = model(batch)
    if isinstance(outputs, dict):
        logits = outputs.get('logits', outputs.get('hidden_states'))
    else:
        logits = outputs
    
    # 计算loss
    if loss_fn is None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
    else:
        loss = loss_fn(outputs, batch)
    
    # 反向传播
    loss.backward()
    
    # 收集各层梯度统计
    layer_stats = []
    layer_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            layer_stats.append({
                'param_name': name,
                'grad_norm': grad_norm,
                'grad_mean': param.grad.mean().item(),
                'grad_std': param.grad.std().item()
            })
            layer_grad_norms.append(grad_norm)
    
    # 计算变异系数
    if layer_grad_norms:
        cv = np.std(layer_grad_norms) / np.mean(layer_grad_norms)
    else:
        cv = 0.0
    
    # 计算早期/晚期梯度比（假设层按顺序命名）
    if len(layer_grad_norms) >= 2:
        early_grad = np.mean(layer_grad_norms[:len(layer_grad_norms)//4])
        late_grad = np.mean(layer_grad_norms[-len(layer_grad_norms)//4:])
        early_late_ratio = early_grad / late_grad if late_grad > 0 else 0
    else:
        early_late_ratio = 0.0
    
    # 计算梯度流评分 (1 - CV，越高表示越均匀)
    gradient_flow_score = 1.0 - min(cv, 1.0)
    
    model.zero_grad()
    
    return {
        'layer_stats': layer_stats,
        'cv': cv,
        'early_late_ratio': early_late_ratio,
        'gradient_flow_score': gradient_flow_score
    }


def measure_actual_flops(
    model: nn.Module,
    input_ids: torch.Tensor,
    config: Dict[str, Any]
) -> int:
    """
    测量实际FLOP消耗
    
    Args:
        model: 待测模型
        input_ids: 输入
        config: 配置，包含N_qttt, T_think等
        
    Returns:
        实际FLOP数
    """
    # 使用PyTorch profiler
    from torch.profiler import profile, ProfilerActivity
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True
    ) as prof:
        with torch.no_grad():
            # 标准前向
            _ = model(input_ids)
            
            # 如果配置中有qTTT步骤，模拟额外计算
            n_qttt = config.get('N_qttt', 0)
            if n_qttt > 0:
                k = config.get('k', 128)
                # 模拟qTTT的额外FLOP
                for _ in range(min(n_qttt, 3)):  # 只测量3步作为样本
                    _ = model(input_ids[:, :k])
    
    # 统计FLOP
    total_flops = sum(evt.flops for evt in prof.events() if evt.flops is not None)
    
    # 根据实际步数 extrapolate
    if n_qttt > 3:
        total_flops = total_flops * (n_qttt / 3)
    
    return int(total_flops)


def compute_flop_equivalent_config(
    total_flops: float,
    context_len: int,
    model_config: Dict[str, Any],
    strategy: str = 'depth_priority'
) -> Dict[str, int]:
    """
    根据总FLOP预算生成等效配置
    
    Args:
        total_flops: 总FLOP预算
        context_len: 上下文长度
        model_config: 模型配置，包含hidden_dim, num_layers等
        strategy: 分配策略 ('pure_width', 'pure_depth', 'balanced', 'depth_priority')
        
    Returns:
        {'N_qttt': int, 'T_think': int, 'k': int}
    """
    d = model_config.get('hidden_dim', 4096)
    L = model_config.get('num_layers', 32)
    k = model_config.get('k', 128)
    
    # 计算attention FLOP系数 (简化估算)
    C_quad = 2 * d * d  # 每次attention的FLOP per token
    
    T = context_len
    
    if strategy == 'pure_width':
        # 全部用于thinking tokens
        T_think = int(total_flops / (C_quad * T))
        return {'N_qttt': 0, 'T_think': T_think, 'k': k}
    
    elif strategy == 'pure_depth':
        # 全部用于qTTT
        N_qttt = int(total_flops / (2 * C_quad * k * T))
        return {'N_qttt': N_qttt, 'T_think': 0, 'k': k}
    
    elif strategy == 'balanced':
        # 各50%
        N_qttt = int(total_flops / (4 * C_quad * k * T))
        T_think = int(total_flops / (2 * C_quad * T))
        return {'N_qttt': N_qttt, 'T_think': T_think, 'k': k}
    
    elif strategy == 'depth_priority':
        # 80% depth, 20% width
        N_qttt = int(0.8 * total_flops / (2 * C_quad * k * T))
        T_think = int(0.2 * total_flops / (C_quad * T))
        return {'N_qttt': N_qttt, 'T_think': T_think, 'k': k}
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_synergy_score(
    full_result: float,
    component_results: Dict[str, float],
    baseline: float = 0.0
) -> Dict[str, float]:
    """
    计算组件协同效应分数
    
    Args:
        full_result: 完整系统结果
        component_results: 各组件单独效果 {'AttnRes': x, 'qTTT': y, ...}
        baseline: 基线结果
        
    Returns:
        {
            'synergy_gain': float,  # 协同增益
            'synergy_coefficient': float,  # 协同系数
            'additive_prediction': float,  # 叠加预测
            'actual': float  # 实际结果
        }
    """
    # 计算各组件相对基线的增益
    component_gains = {
        k: max(0, v - baseline) for k, v in component_results.items()
    }
    
    # 叠加预测
    additive_prediction = baseline + sum(component_gains.values())
    
    # 协同增益
    synergy_gain = full_result - additive_prediction
    
    # 协同系数
    if additive_prediction > baseline:
        synergy_coefficient = (full_result - baseline) / (additive_prediction - baseline)
    else:
        synergy_coefficient = 1.0
    
    return {
        'synergy_gain': synergy_gain,
        'synergy_coefficient': synergy_coefficient,
        'additive_prediction': additive_prediction,
        'actual': full_result,
        'component_gains': component_gains
    }
