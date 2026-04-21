#!/usr/bin/env python3
"""
MATDO-E环境检查脚本
运行此脚本检查A100环境是否准备就绪

使用方法:
    python scripts/setup/check_env.py
"""

import sys
import subprocess
from pathlib import Path

# 颜色代码
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_success(text):
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    print(f"{YELLOW}⚠{RESET} {text}")


def check_gpu():
    """检查GPU"""
    print_header("GPU Check")

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print_success(f"PyTorch CUDA available")
            print(f"  Device count: {device_count}")

            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")

                if "A100" in name:
                    print_success("A100 GPU detected")
                    if mem >= 79:  # 80GB A100
                        print_success("80GB A100 confirmed")
                    else:
                        print_warning(f"A100 has {mem:.1f}GB (expected 80GB)")
                else:
                    print_warning(f"Not A100: {name}")
            return True
        else:
            print_error("CUDA not available in PyTorch")
            return False
    except ImportError:
        print_error("PyTorch not installed")
        return False
    except Exception as e:
        print_error(f"GPU check failed: {e}")
        return False


def check_cuda_version():
    """检查CUDA版本"""
    print_header("CUDA Version Check")

    try:
        import torch

        cuda_version = torch.version.cuda
        print(f"  PyTorch CUDA version: {cuda_version}")

        # 检查是否>=11.8
        major, minor = map(int, cuda_version.split(".")[:2])
        if major > 11 or (major == 11 and minor >= 8):
            print_success("CUDA version >= 11.8")
            return True
        else:
            print_warning("CUDA version < 11.8, some features may not work")
            return False
    except Exception as e:
        print_error(f"CUDA check failed: {e}")
        return False


def check_python_version():
    """检查Python版本"""
    print_header("Python Version Check")

    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print_success("Python version OK (>=3.10)")
        return True
    elif version.major == 3 and version.minor >= 8:
        print_warning("Python version OK but recommend >=3.10")
        return True
    else:
        print_error("Python version too old (need >=3.8)")
        return False


def check_dependencies():
    """检查关键依赖"""
    print_header("Dependencies Check")

    required = {
        "torch": "PyTorch",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "matplotlib": "Matplotlib",
        "tqdm": "tqdm",
    }

    optional = {
        "transformers": "Hugging Face Transformers",
        "datasets": "Hugging Face Datasets",
        "accelerate": "Hugging Face Accelerate",
        "faiss": "Faiss (GPU)",
    }

    all_ok = True

    print("Required packages:")
    for pkg, name in required.items():
        try:
            __import__(pkg)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} not installed")
            all_ok = False

    print("\nOptional packages:")
    for pkg, name in optional.items():
        try:
            __import__(pkg)
            print_success(f"{name}")
        except ImportError:
            print_warning(f"{name} not installed (optional)")

    return all_ok


def check_disk_space():
    """检查磁盘空间"""
    print_header("Disk Space Check")

    try:
        import shutil

        total, used, free = shutil.disk_usage("/")

        total_gb = total / (2**30)
        used_gb = used / (2**30)
        free_gb = free / (2**30)

        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used:  {used_gb:.1f} GB ({used/total*100:.1f}%)")
        print(f"  Free:  {free_gb:.1f} GB")

        if free_gb >= 100:
            print_success("Disk space sufficient (>=100GB)")
            return True
        elif free_gb >= 50:
            print_warning("Disk space may be tight (50-100GB)")
            return True
        else:
            print_error("Disk space insufficient (<50GB)")
            return False
    except Exception as e:
        print_error(f"Disk check failed: {e}")
        return False


def check_matdo_e_modules():
    """检查MATDO-E模块"""
    print_header("MATDO-E Modules Check")

    try:
        # 添加项目路径
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        from experiments.matdo.matdo_e.solver import MATDOESolver
        from experiments.matdo.common.config import config

        print_success("MATDO-E modules can be imported")

        # 测试求解器
        solver = MATDOESolver()
        opt = solver.solve(0.95)
        print_success(f"MATDO-E solver working (rho=0.95, arbitrage={opt.is_arbitrage})")

        # 检查套利不等式
        if config.check_arbitrage_inequality():
            print_success("Arbitrage inequality satisfied")
        else:
            print_warning("Arbitrage inequality not satisfied")

        return True
    except Exception as e:
        print_error(f"MATDO-E check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print(f"{BLUE}MATDO-E Environment Checker{RESET}")
    print(f"Checking if your A100 environment is ready...")

    checks = [
        ("Python Version", check_python_version),
        ("GPU", check_gpu),
        ("CUDA Version", check_cuda_version),
        ("Disk Space", check_disk_space),
        ("Dependencies", check_dependencies),
        ("MATDO-E Modules", check_matdo_e_modules),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"{name} check crashed: {e}")
            results.append((name, False))

    # 总结
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print(f"\n{GREEN}✓ Environment is ready for MATDO-E!{RESET}")
        print("\nNext steps:")
        print("  1. See docs/guides/MATDO_E_A100_BEGINNER_GUIDE.md for training instructions")
        print("  2. Run: python scripts/train.py --config configs/train_small_example.yaml")
        return 0
    elif passed >= total * 0.7:
        print(f"\n{YELLOW}⚠ Environment mostly ready, but some issues found{RESET}")
        print("\nPlease fix the failed checks above")
        return 1
    else:
        print(f"\n{RED}✗ Environment not ready{RESET}")
        print("\nPlease run: bash scripts/setup/a100_setup.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
