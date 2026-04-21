# PRD: Align RaBitQ with VectorDB-NTU/RaBitQ-Library

## Objective
Refactor the PyTorch-based RaBitQ implementation to closely match the API, data formats, and estimator formulas of the official C++ reference implementation (VectorDB-NTU/RaBitQ-Library). Also evaluate how to call the C++ library externally.

## Key Gaps Identified

### 1. Estimator API mismatch
- **Current**: `estimate_inner_product` quantizes the query into multi-bit planes and uses popcount. This is NOT how RaBitQ-Library estimates distances.
- **Reference**: Query stays as float (only random-rotated). Distance = `F_add + G_add + F_rescale * (ip + G_kBxSumq)`. Error bound = `F_error * G_error`.

### 2. Missing data formats
- **Reference**: 5 formats (scalar replacement, full single, batch FastScan, split single, split batch).
- **Current**: Only an ad-hoc full-single equivalent with no split/incremental support.

### 3. Missing `faster_config` factory
- **Reference**: `faster_config(dim, total_bits)` pre-computes `t_const`.
- **Current**: Has `RabitqConfig` but no `faster_config` wrapper.

### 4. External call evaluation
- The C++ library is header-only (Eigen + custom).
- Possible integration paths: pybind11 wrapper, torch.utils.cpp_extension, ctypes shared lib, or keep pure Python with aligned API.

## User Stories

### US-201: Rewrite estimator.py to match reference formulas
- Add `BatchQuery`, `SplitSingleQuery`, `SplitBatchQuery` query preprocessing classes.
- Add `full_est_dist`, `split_single_estdist`, `split_batch_estdist`, `split_distance_boosting`.
- Remove query-side multi-bit quantization; use pure float inner-product against codes.

### US-202: Add Format-1 scalar quantization API
- Add `quantize_scalar` and `dequantize_scalar` to `src/rabitq/quantizer.py`.

### US-203: Add `faster_config` factory and improve `RabitqConfig`
- Add `faster_config(dim, total_bits)` in `src/rabitq/api.py`.
- Ensure `RabitqConfig` stores `t_const` and `metric_type` (L2 / IP).

### US-204: Add split formats and incremental estimation
- Add `QuantizedVectorSplit` dataclass (separate bin_data and ex_data).
- Add `quantize_split_single` and `split_single_estdist` Python equivalents.

### US-205: Evaluate external C++ integration
- Document trade-offs of pybind11 vs torch.cpp_extension vs ctypes.
- Create a minimal proof-of-concept pybind11 wrapper for `quantize_full_single` and `full_est_dist`.
- Verify compilation and correctness against Python implementation.

### US-206: Green tests
- All existing RaBitQ tests pass.
- New estimator tests pass against reference formulas.
