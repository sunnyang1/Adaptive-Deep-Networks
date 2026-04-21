"""
Visualization Utilities for Experiments

Standardized plotting functions and color schemes.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None


# Unified color scheme for architectures
ARCHITECTURE_COLORS = {
    'prenorm': '#e74c3c',      # Red
    'postnorm': '#3498db',     # Blue
    'deepnorm': '#2ecc71',     # Green
    'attnres': '#9b59b6',      # Purple
    'adb': '#f39c12',          # Orange
    'baseline': '#95a5a6',     # Gray
    'ttt_linear': '#1abc9c',   # Teal
}

ARCHITECTURE_LABELS = {
    'prenorm': 'PreNorm',
    'postnorm': 'PostNorm',
    'deepnorm': 'DeepNorm',
    'attnres': 'AttnRes',
    'adb': 'ADB',
    'baseline': 'Baseline',
    'ttt_linear': 'TTT-Linear',
}


class FigureManager:
    """Context manager for creating and saving figures."""
    
    def __init__(
        self,
        name: str,
        output_dir: Path,
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 300,
        style: str = 'seaborn-v0_8-darkgrid'
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for FigureManager")
        
        self.name = name
        self.output_dir = Path(output_dir)
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.fig = None
        self.ax = None
    
    def __enter__(self):
        """Setup figure."""
        plt.style.use(self.style)
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        return self.fig, self.ax
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save and cleanup."""
        if exc_type is None:  # No exception
            self.save()
        plt.close(self.fig)
    
    def save(self):
        """Save figure to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        return output_path


def plot_architecture_comparison(
    data: Dict[str, Dict[str, float]],
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    log_scale: bool = False
) -> Path:
    """Plot bar chart comparing architectures."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping visualization")
        return output_path
    
    architectures = list(data.keys())
    values = [data[arch].get(metric, 0) for arch in architectures]
    colors = [ARCHITECTURE_COLORS.get(arch, '#95a5a6') for arch in architectures]
    labels = [ARCHITECTURE_LABELS.get(arch, arch) for arch in architectures]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Architecture', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel or metric, fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric} by Architecture', fontsize=14, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(axis='y', alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_training_curves(
    data: Dict[str, List[float]],
    output_path: Path,
    xlabel: str = 'Steps',
    ylabel: str = 'Value',
    title: Optional[str] = None
) -> Path:
    """Plot training curves for multiple architectures."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping visualization")
        return output_path
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for arch, values in data.items():
        color = ARCHITECTURE_COLORS.get(arch, '#95a5a6')
        label = ARCHITECTURE_LABELS.get(arch, arch)
        ax.plot(values, label=label, color=color, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title or 'Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    output_path: Path,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    cbar_label: Optional[str] = None
) -> Path:
    """Plot heatmap."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not installed, skipping visualization")
        return output_path
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=12, fontweight='bold')
    
    # Add values in cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="w", fontsize=8)
    
    ax.set_title(title or 'Heatmap', fontsize=14, fontweight='bold')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path
