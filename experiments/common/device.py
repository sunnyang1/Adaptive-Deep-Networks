"""
Device Management for Experiments

Unified device selection and management.
"""

import torch
import os
from typing import Union, Optional
from contextlib import contextmanager


def get_device(preference: str = "auto") -> torch.device:
    """
    Get torch device based on preference.
    
    Args:
        preference: Device preference ('auto', 'cuda', 'cpu', 'mps')
    
    Returns:
        torch.device instance
    
    Examples:
        >>> device = get_device('auto')  # Best available
        >>> device = get_device('cuda')  # Force CUDA
        >>> device = get_device('cpu')   # Force CPU
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    return torch.device(preference)


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get information about the device.
    
    Returns:
        Dictionary with device information
    """
    if device is None:
        device = get_device("auto")
    
    info = {
        'device_type': str(device),
        'device_name': 'N/A',
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'memory_allocated_gb': 0.0,
        'memory_reserved_gb': 0.0,
    }
    
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info['device_name'] = props.name
        info['compute_capability'] = f"{props.major}.{props.minor}"
        info['total_memory_gb'] = props.total_memory / (1024**3)
        info['memory_allocated_gb'] = torch.cuda.memory_allocated(device) / (1024**3)
        info['memory_reserved_gb'] = torch.cuda.memory_reserved(device) / (1024**3)
    
    return info


class DeviceManager:
    """
    Context manager for device configuration.
    
    Handles device setup, deterministic mode, and memory management.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "auto",
        seed: int = 42,
        deterministic: bool = False,
        max_memory_gb: Optional[float] = None
    ):
        self.device = get_device(device) if isinstance(device, str) else device
        self.seed = seed
        self.deterministic = deterministic
        self.max_memory_gb = max_memory_gb
        self._original_state = {}
    
    def __enter__(self):
        """Setup device configuration."""
        # Set random seeds
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        # Store original state
        self._original_state['deterministic'] = torch.are_deterministic_algorithms_enabled()
        self._original_state['benchmark'] = torch.backends.cudnn.benchmark
        
        # Configure deterministic mode
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        
        # Set memory limit if specified
        if self.max_memory_gb is not None and torch.cuda.is_available():
            max_bytes = int(self.max_memory_gb * 1024**3)
            torch.cuda.set_per_process_memory_fraction(
                max_bytes / torch.cuda.get_device_properties(0).total_memory
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original state."""
        # Restore deterministic settings
        if 'deterministic' in self._original_state:
            torch.use_deterministic_algorithms(self._original_state['deterministic'])
        if 'benchmark' in self._original_state:
            torch.backends.cudnn.benchmark = self._original_state['benchmark']
        
        # Cleanup CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def empty_cache(self):
        """Empty CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_summary(self) -> dict:
        """Get memory usage summary."""
        if not torch.cuda.is_available():
            return {'cuda_available': False}
        
        return {
            'cuda_available': True,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3),
        }


@contextmanager
def cuda_memory_tracking():
    """
    Context manager to track CUDA memory usage.
    
    Example:
        >>> with cuda_memory_tracking() as tracker:
        ...     model(input)
        ... print(f"Peak memory: {tracker.peak_gb:.2f} GB")
    """
    tracker = MemoryTracker()
    try:
        yield tracker
    finally:
        tracker.finalize()


class MemoryTracker:
    """Track CUDA memory usage."""
    
    def __init__(self):
        self.peak_gb = 0.0
        self.start_gb = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_gb = torch.cuda.memory_allocated() / (1024**3)
    
    def finalize(self):
        """Finalize tracking and get peak usage."""
        if torch.cuda.is_available():
            self.peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
    
    @property
    def current_gb(self) -> float:
        """Current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    @property
    def delta_gb(self) -> float:
        """Memory change from start."""
        return self.current_gb - self.start_gb
