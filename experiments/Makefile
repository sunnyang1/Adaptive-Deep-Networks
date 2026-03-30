# Adaptive Deep Networks - 实验运行 Makefile

.PHONY: help list quick full core turboquant clean

help:
	@echo "Adaptive Deep Networks 实验运行"
	@echo ""
	@echo "可用目标:"
	@echo "  make list       - 列出所有实验"
	@echo "  make quick      - 快速运行所有实验（减少计算量）"
	@echo "  make full       - 完整运行所有实验"
	@echo "  make core       - 仅运行核心实验"
	@echo "  make turboquant - 仅运行TurboQuant验证"
	@echo "  make clean      - 清理实验结果"
	@echo ""
	@echo "选项:"
	@echo "  DEVICE=cpu      - 使用CPU运行"
	@echo "  TIMEOUT=30      - 设置超时（分钟）"

list:
	python run_all.py --list

quick:
	python run_all.py --quick

full:
	python run_all.py

core:
	python run_all.py --category core

turboquant:
	python run_all.py --category turboquant

clean:
	rm -rf results/unified/*
	@echo "已清理 unified 实验结果"

clean-all:
	rm -rf results/*
	@echo "已清理所有实验结果"

# 带选项的运行
quick-cpu:
	python run_all.py --quick --device cpu

full-cpu:
	python run_all.py --device cpu
