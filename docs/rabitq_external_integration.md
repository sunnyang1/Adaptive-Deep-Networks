# RaBitQ External Integration Evaluation

## Objective
Evaluate how to call the reference C++ implementation (VectorDB-NTU/RaBitQ-Library) from Python.

## Reference Implementation Analysis

### Repository Structure
- **URL**: https://github.com/VectorDB-NTU/RaBitQ-Library
- **Language**: C++17 (header-only with bundled Eigen)
- **Key Features**:
  - 5 data formats (scalar, full single, batch FastScan, split single, split batch)
  - Multi-bit quantization (1-bit binary + extended bits)
  - FastScan kernels (AVX2/AVX-512) for batch processing
  - Distance estimators with theoretical error bounds
  - Support for L2, Inner Product, Cosine similarity

### Dependencies
- **Eigen3**: Bundled in `third_party/`
- **OpenMP**: Used for parallelization (requires `-fopenmp`)
- **Platform**: Linux (GCC), Windows (MSVC), macOS (with libomp or GCC)

## Integration Options

### Option 1: torch.utils.cpp_extension (Recommended for PyTorch Projects)

**Approach**: Use `torch.utils.cpp_extension.load_inline()` or `load()` to compile C++ sources on-the-fly.

**Pros**:
- Automatic compilation caching
- Built-in pybind11 support
- Integrates with PyTorch's build system
- Easy distribution (single Python file)

**Cons**:
- Requires Ninja
- Requires C++ compiler at runtime
- First compilation is slow (~10-30 seconds)
- OpenMP issues on macOS (requires libomp)

**Feasibility**: ✅ High (for Linux), ⚠️ Medium (for macOS with libomp)

**Example**:
```python
from torch.utils.cpp_extension import load_inline

cpp_source = r'''
#include <torch/extension.h>
#include "rabitqlib/quantization/rabitq_impl.hpp"

py::dict quantize_one_bit(...) { ... }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &quantize_one_bit);
}
'''

rabitq_cpp = load_inline(
    name="rabitq_binding",
    cpp_sources=cpp_source,
    extra_include_paths=["third_party/rabitq-lib/include"],
    extra_cflags=["-O3", "-fopenmp"],
)
```

### Option 2: pybind11 + setuptools (Recommended for Production)

**Approach**: Create a proper Python package with compiled extension module.

**Pros**:
- Pre-compiled wheels (no runtime compilation)
- Better error messages
- Can be published to PyPI
- Supports all platforms with proper CI

**Cons**:
- More setup code (CMake or setuptools)
- Requires build environment for distribution

**Feasibility**: ✅ High

**Setup**:
```bash
# setup.py or pyproject.toml with scikit-build-core
[build-system]
requires = ["scikit-build-core", "pybind11"]
```

### Option 3: ctypes / cffi

**Approach**: Compile a small C wrapper as shared library, load with ctypes.

**Pros**:
- No pybind11 dependency
- Smaller binary size
- Works with any C compiler

**Cons**:
- Manual memory management
- More boilerplate code
- Type safety issues

**Feasibility**: ✅ Medium (more work, less safe)

### Option 4: Subprocess + Shared Memory

**Approach**: Run C++ executable, communicate via files or shared memory.

**Pros**:
- No Python binding complexity
- Can use pre-built binary

**Cons**:
- High latency (process spawn per call)
- Complex serialization
- Not suitable for fine-grained calls

**Feasibility**: ❌ Low (not suitable for KV cache compression)

### Option 5: Keep Pure Python (Current)

**Pros**:
- No compilation issues
- Works everywhere
- Easy debugging

**Cons**:
- ~10-100x slower than C++ for batch operations
- No FastScan optimization
- Higher memory usage

**Feasibility**: ✅ High (correctness verified, speed sacrificed)

## Recommended Approach

### Short Term (Development/Research)
**Option 1** (`torch.utils.cpp_extension`) for quick experimentation.

```python
# Add to requirements: ninja, torch
# On macOS: brew install libomp
```

### Long Term (Production)
**Option 2** (pybind11 + setuptools) with pre-built wheels for Linux/macOS/Windows.

**Repository Layout**:
```
external/rabitq_cpp/
├── CMakeLists.txt
├── rabitq_binding.cpp      # pybind11 wrapper
├── include/                # RaBitQ-Library headers (submodule)
└── setup.py                # Build script
```

### Build Instructions (for Option 2)

**Linux**:
```bash
git submodule add https://github.com/VectorDB-NTU/RaBitQ-Library.git external/rabitq_cpp/include
pip install pybind11
python setup.py build_ext --inplace
```

**macOS** (with libomp):
```bash
brew install libomp pybind11
export CXXFLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib -lomp"
python setup.py build_ext --inplace
```

## Performance Expectations

Based on reference implementation benchmarks:

| Operation | Python (current) | C++ (reference) | Speed-up |
|-----------|------------------|-----------------|----------|
| 1-bit quantize (single) | ~0.1 ms | ~0.01 ms | ~10x |
| 1-bit quantize (batch 32) | ~3 ms | ~0.05 ms | ~60x (FastScan) |
| Distance estimate (single) | ~0.05 ms | ~0.005 ms | ~10x |
| Distance estimate (batch) | ~1 ms | ~0.01 ms | ~100x (FastScan) |

## API Alignment Status

Our Python implementation now closely matches the reference C++ API:

| C++ API (rabitqlib) | Python API (src.rabitq) | Status |
|---------------------|-------------------------|--------|
| `faster_config()` | `faster_config()` | ✅ Added |
| `quantize_scalar()` | `quantize_scalar()` | ✅ Added |
| `quantize_full_single()` | `quantize_vector()` | ✅ Compatible |
| `full_est_dist()` | `full_est_dist()` | ✅ Added |
| `split_single_estdist()` | `split_single_estdist()` | ✅ Added |
| `split_batch_estdist()` | `split_batch_estdist()` | ✅ Added |
| `SplitSingleQuery` | `SplitSingleQuery` | ✅ Added |
| `SplitBatchQuery` | `SplitBatchQuery` | ✅ Added |
| FastScan kernels | Not implemented | ⚠️ Future work |

## Conclusion

1. **Pure Python implementation is sufficient** for correctness validation and small-scale experiments.

2. **C++ integration is recommended** for production deployment with large batch sizes (e.g., IVF index building, HNSW graph construction).

3. **Integration complexity is low** on Linux, moderate on macOS (requires OpenMP setup), well-documented on Windows (MSVC).

4. **Minimal PoC provided** in `scripts/eval_rabitq_cpp_binding.py` (requires Linux or macOS with libomp).

## Next Steps

1. Create `external/rabitq_cpp/` with pybind11 wrapper for critical paths (FastScan batch quantization)
2. Add CI workflow to build wheels for Linux/macOS/Windows
3. Add fallback to pure Python if C++ extension not available
4. Benchmark end-to-end KV cache compression with C++ backend
