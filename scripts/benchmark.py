#!/usr/bin/env python3
"""ADN基准测试入口"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="ADN Benchmarks")
    parser.add_argument("--suite", choices=["quick", "full", "paper"], default="quick")
    parser.add_argument("--output", type=str, default="results/benchmarks")
    args = parser.parse_args()
    
    print(f"Running benchmark suite: {args.suite}")


if __name__ == "__main__":
    main()
