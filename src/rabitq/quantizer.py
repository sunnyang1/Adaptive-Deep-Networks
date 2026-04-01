"""
Lloyd-Max Optimal Scalar Quantization.

Pre-computes optimal centroids for the bell-curve distribution that results
from Hadamard transform of high-dimensional vectors.
"""

import torch
from typing import Tuple, Optional


class LloydMaxQuantizer:
    """
    Lloyd-Max optimal scalar quantizer.
    
    Iteratively optimizes centroids to minimize MSE for a given distribution.
    For rotated unit vectors, coordinates follow approximately Gaussian(0, 1/d).
    
    Usage:
        >>> quantizer = LloydMaxQuantizer(num_bits=4)
        >>> quantizer.fit(samples)  # Fit on data
        >>> indices, dequantized = quantizer.quantize(x)
    """
    
    def __init__(self, num_bits: int, max_iter: int = 100, device: str = 'cpu'):
        """
        Initialize quantizer.
        
        Args:
            num_bits: Number of bits (levels = 2^num_bits)
            max_iter: Maximum Lloyd-Max iterations
            device: 'cpu', 'cuda', or 'mps'
        """
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.max_iter = max_iter
        self.device = device
        
        self.centroids: Optional[torch.Tensor] = None
        self.boundaries: Optional[torch.Tensor] = None
        self._is_fitted = False
    
    def fit(self, data: torch.Tensor) -> 'LloydMaxQuantizer':
        """
        Fit quantizer on data using Lloyd-Max algorithm.
        
        Args:
            data: Training data tensor of any shape
            
        Returns:
            self for chaining
        """
        flat_data = data.reshape(-1).to(self.device)
        
        # Initialize centroids uniformly over data range
        min_val = flat_data.min().item()
        max_val = flat_data.max().item()
        self.centroids = torch.linspace(min_val, max_val, self.num_levels, device=self.device)
        
        # Lloyd-Max iterations
        for iteration in range(self.max_iter):
            # Assign to nearest centroid
            distances = torch.abs(flat_data.unsqueeze(1) - self.centroids)
            assignments = distances.argmin(dim=1)
            
            # Update centroids to mean of assigned points
            new_centroids = torch.zeros_like(self.centroids)
            for i in range(self.num_levels):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = flat_data[mask].mean()
                else:
                    # Empty bin: interpolate from neighbors
                    if i > 0 and i < self.num_levels - 1:
                        new_centroids[i] = (self.centroids[i-1] + self.centroids[i+1]) / 2
                    else:
                        new_centroids[i] = self.centroids[i]
            
            # Check convergence
            change = (new_centroids - self.centroids).abs().max().item()
            self.centroids = new_centroids
            
            if change < 1e-6:
                break
        
        # Compute decision boundaries (midpoints between centroids)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2
        self._is_fitted = True
        
        return self
    
    def fit_gaussian(self, mean: float = 0.0, std: float = 1.0, 
                     num_samples: int = 100000) -> 'LloydMaxQuantizer':
        """
        Fit on Gaussian distribution.
        
        For rotated unit vectors in d dimensions, coordinates follow
        approximately Gaussian(0, 1/sqrt(d)).
        
        Args:
            mean: Mean of Gaussian
            std: Standard deviation
            num_samples: Number of samples for fitting
            
        Returns:
            self for chaining
        """
        samples = torch.randn(num_samples, device=self.device) * std + mean
        return self.fit(samples)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode tensor to quantization indices.
        
        Args:
            x: Input tensor of any shape
            
        Returns:
            Indices as uint8 tensor with same shape
        """
        if not self._is_fitted:
            raise RuntimeError("Quantizer must be fitted before encoding. Call fit() first.")
        
        original_shape = x.shape
        x_flat = x.reshape(-1, 1)
        
        # Find nearest centroid
        distances = torch.abs(x_flat - self.centroids)
        indices = distances.argmin(dim=1)
        
        return indices.reshape(original_shape).to(torch.uint8)
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode indices back to values.
        
        Args:
            indices: Quantization indices (uint8)
            
        Returns:
            Dequantized values with same shape as indices
        """
        if not self._is_fitted:
            raise RuntimeError("Quantizer must be fitted before decoding. Call fit() first.")
        
        return self.centroids[indices.long()]
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor and return both indices and dequantized values.
        
        Args:
            x: Input tensor
            
        Returns:
            (indices, dequantized) tuple
        """
        indices = self.encode(x)
        dequantized = self.decode(indices)
        return indices, dequantized
    
    @property
    def is_fitted(self) -> bool:
        """Return True if quantizer has been fitted."""
        return self._is_fitted
    
    def to(self, device: str) -> 'LloydMaxQuantizer':
        """Move quantizer to device."""
        self.device = device
        if self.centroids is not None:
            self.centroids = self.centroids.to(device)
        if self.boundaries is not None:
            self.boundaries = self.boundaries.to(device)
        return self
