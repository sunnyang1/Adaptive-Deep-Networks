#!/usr/bin/env python3
"""ADN统一评估入口"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="ADN Evaluation")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--benchmarks", nargs="+", choices=["math", "needle", "flop", "all"], default=["all"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/eval")
    args = parser.parse_args()
    
    print(f"Evaluating ADN model: {args.model_size}")
    print(f"Benchmarks: {args.benchmarks}")


if __name__ == "__main__":
    main()
