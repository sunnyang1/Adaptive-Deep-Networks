"""
Training Utilities

Shared training functions and checkpoint management.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


class CheckpointError(Exception):
    """Error saving or loading checkpoint."""
    pass


@dataclass
class Checkpoint:
    """Checkpoint data structure."""
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    loss: float
    metrics: Dict[str, float]
    config: Dict[str, Any]


class CheckpointManager:
    """
    Manage model checkpoints with versioning and cleanup.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_checkpoints: int = 5,
        keep_best: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.best_metric = float('inf')
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            metrics: Additional metrics
            config: Model configuration
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
            'config': config or {},
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        try:
            torch.save(checkpoint, latest_path)
        except (OSError, IOError) as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
        
        # Save epoch-specific
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        try:
            torch.save(checkpoint, epoch_path)
        except (OSError, IOError) as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")
        
        # Save best if applicable
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return epoch_path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optional optimizer to load
            checkpoint_path: Specific checkpoint (None for latest)
        
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        
        if not checkpoint_path.exists():
            raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()


def compute_loss(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute loss for a batch.
    
    Args:
        model: Model
        batch: Batch dictionary with 'input_ids' and 'labels'
        criterion: Loss criterion
        device: Device
    
    Returns:
        Loss tensor
    """
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    forward_kwargs = model_forward_kwargs or {}
    outputs = model(input_ids, **forward_kwargs)
    
    # Handle different output formats
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    else:
        logits = outputs
    
    # Compute loss
    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    return loss


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    model_forward_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Single training step.
    
    Args:
        model: Model
        batch: Batch data
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device
        gradient_accumulation_steps: Number of steps to accumulate gradients
    
    Returns:
        Loss value
    """
    loss = compute_loss(
        model,
        batch,
        criterion,
        device,
        model_forward_kwargs=model_forward_kwargs,
    )
    
    # Scale loss for gradient accumulation
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Update weights only after accumulation
    # Note: optimizer.step() and zero_grad() should be called
    # outside this function after accumulation
    
    return loss.item()


def get_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95)
) -> torch.optim.Optimizer:
    """Create AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """Create linear warmup + cosine decay scheduler."""
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
