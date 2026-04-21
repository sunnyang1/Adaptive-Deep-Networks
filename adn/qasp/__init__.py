"""ADN QASP - 质量感知Stiefel投影

QASP是ADN的扩展模块，包含：
- 矩阵级查询自适应（Stiefel流形优化）
- 信息质量评分
- 质量加权的AttnRes和Engram
"""

from adn.qasp.stiefel import (
    matrix_sign_function,
    newton_schulz,
    project_to_stiefel,
    stiefel_projection,
)
from adn.qasp.matrix_qasp import MatrixQASP, QASPConfig, PonderGate, matrix_qasp_update
from adn.qasp.quality_score import QualityScore, compute_quality_score
from adn.qasp.value_weighted_attnres import ValueWeightedAttnRes
from adn.qasp.value_weighted_engram import ValueWeightedEngram
from adn.qasp.models import (
    QASPLayer,
    QASPTransformer,
    QASPTransformerConfig,
    create_qasp_transformer,
)

__all__ = [
    # Stiefel
    "stiefel_projection",
    "matrix_sign_function",
    "newton_schulz",
    "project_to_stiefel",
    # Matrix QASP
    "MatrixQASP",
    "QASPConfig",
    "PonderGate",
    "matrix_qasp_update",
    # Quality score
    "QualityScore",
    "compute_quality_score",
    # Value-weighted components
    "ValueWeightedAttnRes",
    "ValueWeightedEngram",
    # Models
    "QASPLayer",
    "QASPTransformer",
    "QASPTransformerConfig",
    "create_qasp_transformer",
]
