"""
Unit tests for benchmarks module.
"""

import pytest


class TestNeedleHaystackImport:
    """Import tests for needle-in-haystack benchmark."""

    def test_needle_haystack_import(self):
        """Test that needle_haystack module can be imported."""
        from src.benchmarks import needle_haystack

        assert needle_haystack is not None

    def test_benchmark_class_import(self):
        """Test that benchmark class can be imported."""
        try:
            from src.benchmarks.needle_haystack import NeedleHaystackBenchmark

            assert NeedleHaystackBenchmark is not None
        except (ImportError, AttributeError) as e:
            pytest.skip(f"NeedleHaystackBenchmark not available: {e}")


class TestMathEvalImport:
    """Import tests for math evaluation."""

    def test_math_eval_import(self):
        """Test that math_eval module can be imported."""
        try:
            from src.benchmarks import math_eval

            assert math_eval is not None
        except ImportError as e:
            pytest.skip(f"math_eval not available: {e}")


class TestFLOPAnalysisImport:
    """Import tests for FLOP analysis."""

    def test_flop_analysis_import(self):
        """Test that flop_analysis module can be imported."""
        from src.benchmarks import flop_analysis

        assert flop_analysis is not None


class TestBenchmarkModules:
    """Integration tests for benchmark modules."""

    def test_all_benchmarks_importable(self):
        """Test that all benchmark modules can be imported."""
        from src.benchmarks import needle_haystack
        from src.benchmarks import flop_analysis

        assert needle_haystack is not None
        assert flop_analysis is not None

        # math_eval may fail due to external dependencies
        try:
            from src.benchmarks import math_eval

            assert math_eval is not None
        except ImportError:
            pass  # Acceptable if optional dependency missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
