"""
Block Attention Residuals (AttnRes) Module

Provides:
- BlockAttnRes: Two-phase attention over block-level representations
- PseudoQueryManager: Learned depth-wise retrieval vectors
- PolarPseudoQueryManager: Polar-coordinate pseudo-queries for qTTT
"""

from .block_attnres import BlockAttnRes
from .pseudo_query import PseudoQueryManager, PseudoQueryInitializer, AttentionWeightMonitor
from .polar_pseudo_query import (
    PolarPseudoQuery,
    PolarPseudoQueryManager,
    PseudoQueryPolarAdapter,
    create_pseudo_query_manager,
)

__all__ = [
    "BlockAttnRes",
    "PseudoQueryManager",
    "PseudoQueryInitializer",
    "AttentionWeightMonitor",
    "PolarPseudoQuery",
    "PolarPseudoQueryManager",
    "PseudoQueryPolarAdapter",
    "create_pseudo_query_manager",
]

# Backward compatibility aliases
PolarQueryAdapter = PseudoQueryPolarAdapter
