"""
Evaluate calling RaBitQ-Library (C++) from Python via torch.utils.cpp_extension.

This script compiles a minimal pybind11 extension that wraps the reference
C++ implementation's one-bit quantization kernel and compares it against our
Python implementation for correctness and speed.
"""

import torch
import platform
import sys

# ---------------------------------------------------------------------------
# Platform check
# ---------------------------------------------------------------------------
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

if IS_MACOS:
    print("=" * 60)
    print("NOTE: macOS detected.")
    print("This script requires OpenMP which is not available in Apple Clang.")
    print("Please either:")
    print("  1. Install libomp: brew install libomp")
    print("  2. Use GCC: export CC=gcc-13 CXX=g++-13")
    print("  3. Run on Linux for full evaluation")
    print("=" * 60)
    sys.exit(0)

from torch.utils.cpp_extension import load_inline
import numpy as np
import time

# ---------------------------------------------------------------------------
# Minimal C++ source wrapping rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor
# ---------------------------------------------------------------------------
CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>
#include "rabitqlib/quantization/rabitq_impl.hpp"

// Quantize a batch of float vectors using the reference C++ one-bit kernel.
// Args:
//   data: [num, dim] float32
//   centroid: [dim] float32
//   metric_type: 0 = L2, 1 = IP
// Returns:
//   dict with { "binary_code": [num, dim] uint8, "f_add": [num] float,
//               "f_rescale": [num] float, "f_error": [num] float }
py::dict quantize_one_bit_batch_cpp(torch::Tensor data, torch::Tensor centroid, int metric_type) {
    TORCH_CHECK(data.dim() == 2, "data must be [num, dim]");
    TORCH_CHECK(centroid.dim() == 1, "centroid must be [dim]");
    int64_t num = data.size(0);
    int64_t dim = data.size(1);
    TORCH_CHECK(centroid.size(0) == dim, "dimension mismatch");

    data = data.contiguous().to(torch::kCPU).to(torch::kFloat32);
    centroid = centroid.contiguous().to(torch::kCPU).to(torch::kFloat32);

    auto binary_code = torch::zeros({num, dim}, torch::dtype(torch::kUInt8));
    auto f_add = torch::zeros({num}, torch::dtype(torch::kFloat32));
    auto f_rescale = torch::zeros({num}, torch::dtype(torch::kFloat32));
    auto f_error = torch::zeros({num}, torch::dtype(torch::kFloat32));

    const float* data_ptr = data.data_ptr<float>();
    const float* cent_ptr = centroid.data_ptr<float>();
    uint8_t* code_ptr = binary_code.data_ptr<uint8_t>();
    float* fadd_ptr = f_add.data_ptr<float>();
    float* fres_ptr = f_rescale.data_ptr<float>();
    float* ferr_ptr = f_error.data_ptr<float>();

    rabitqlib::MetricType mt = (metric_type == 0) ? rabitqlib::METRIC_L2 : rabitqlib::METRIC_IP;

    for (int64_t i = 0; i < num; ++i) {
        std::vector<int> bin_code(dim);
        rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor(
            data_ptr + i * dim,
            cent_ptr,
            static_cast<size_t>(dim),
            bin_code.data(),
            fadd_ptr[i],
            fres_ptr[i],
            ferr_ptr[i],
            mt
        );
        for (int64_t j = 0; j < dim; ++j) {
            code_ptr[i * dim + j] = static_cast<uint8_t>(bin_code[j]);
        }
    }

    py::dict out;
    out["binary_code"] = binary_code;
    out["f_add"] = f_add;
    out["f_rescale"] = f_rescale;
    out["f_error"] = f_error;
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_one_bit_batch_cpp", &quantize_one_bit_batch_cpp, "Reference C++ one-bit quantizer");
}
"""

# ---------------------------------------------------------------------------
# Compile the extension on-the-fly
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/Users/michelleye/Documents/Adaptive-Deep-Networks"
INCLUDE_DIRS = [
    f"{PROJECT_ROOT}/third_party/rabitq-lib/include",
]

print("Compiling minimal C++ binding via torch.utils.cpp_extension ...")
rabitq_cpp = load_inline(
    name="rabitq_cpp_binding",
    cpp_sources=CPP_SOURCE,
    extra_include_paths=INCLUDE_DIRS,
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)
print("Compilation done.")

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
from src.rabitq.quantizer import _compute_one_bit_factors
from src.rabitq.packing import pack_binary_code


def py_quantize_one_bit_batch(data: torch.Tensor, centroid: torch.Tensor, metric_type: str):
    num, dim = data.shape
    binary_codes = []
    f_adds, f_rescales, f_errors = [], [], []
    for i in range(num):
        residual = data[i] - centroid
        bin_code = (residual >= 0).to(torch.uint8)
        f_add, f_rescale, f_error, _ = _compute_one_bit_factors(
            residual, centroid, bin_code, metric_type
        )
        binary_codes.append(bin_code)
        f_adds.append(f_add)
        f_rescales.append(f_rescale)
        f_errors.append(f_error)
    return {
        "binary_code": torch.stack(binary_codes),
        "f_add": torch.tensor(f_adds, dtype=torch.float32),
        "f_rescale": torch.tensor(f_rescales, dtype=torch.float32),
        "f_error": torch.tensor(f_errors, dtype=torch.float32),
    }


def main():
    dim = 128
    num = 100
    data = torch.randn(num, dim)
    centroid = torch.zeros(dim)
    mt_str = "ip"
    mt_int = 1  # IP

    # Warm-up C++ compilation cache
    _ = rabitq_cpp.quantize_one_bit_batch_cpp(data[:1], centroid, mt_int)

    # Correctness
    cpp_out = rabitq_cpp.quantize_one_bit_batch_cpp(data, centroid, mt_int)
    py_out = py_quantize_one_bit_batch(data, centroid, mt_str)

    assert torch.equal(cpp_out["binary_code"], py_out["binary_code"])
    assert torch.allclose(cpp_out["f_add"], py_out["f_add"], atol=1e-4)
    assert torch.allclose(cpp_out["f_rescale"], py_out["f_rescale"], atol=1e-4)
    assert torch.allclose(cpp_out["f_error"], py_out["f_error"], atol=1e-4)
    print("Correctness check PASSED.")

    # Benchmark
    trials = 50
    t0 = time.time()
    for _ in range(trials):
        _ = rabitq_cpp.quantize_one_bit_batch_cpp(data, centroid, mt_int)
    t_cpp = (time.time() - t0) / trials

    t0 = time.time()
    for _ in range(trials):
        _ = py_quantize_one_bit_batch(data, centroid, mt_str)
    t_py = (time.time() - t0) / trials

    print(f"C++ one-bit quantize  {num}x{dim}: {t_cpp*1000:.3f} ms")
    print(f"Python one-bit quantize {num}x{dim}: {t_py*1000:.3f} ms")
    print(f"Speed-up: {t_py/t_cpp:.2f}x")


if __name__ == "__main__":
    main()
