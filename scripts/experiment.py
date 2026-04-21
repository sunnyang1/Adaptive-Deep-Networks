#!/usr/bin/env python3
"""ADN实验入口"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="ADN Experiments")
    parser.add_argument("--category", choices=["core", "qasp", "matdo", "paper", "all"], default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--output-dir", type=str, default="results/experiments")
    args = parser.parse_args()
    
    if args.list:
        print("Available experiments:")
        print("  - core: AttnRes, qTTT, RaBitQ, Engram validation")
        print("  - qasp: QASP ablations and benchmarks")
        print("  - matdo: MATDO-E resource model validation")
        print("  - paper: Paper reproduction experiments")
        return
    
    print(f"Running experiments: category={args.category}, quick={args.quick}")


if __name__ == "__main__":
    main()
