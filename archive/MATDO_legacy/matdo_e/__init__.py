"""
MATDO-E: Memory Arbitrage via Test-time Dynamic Optimization with Engrams

vLLM集成模块，实现四维优化 (R, M, T, E) 和异构内存套利。
"""

from .solver import MATDOESolver, OptimalConfig
from .engram_manager import EngramManager
from .arbitrage_attention import ArbitrageAttention
from .scheduler import MATDOEScheduler

__all__ = [
    'MATDOESolver',
    'OptimalConfig', 
    'EngramManager',
    'ArbitrageAttention',
    'MATDOEScheduler',
]
