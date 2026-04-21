#!/usr/bin/env python3
"""ADN生成入口（支持QASP）"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="ADN Generation")
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--use-qasp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    print(f"Generating with ADN{'+QASP' if args.use_qasp else ''}")
    if args.dry_run:
        print("Dry run mode - printing resolved request only")


if __name__ == "__main__":
    main()
