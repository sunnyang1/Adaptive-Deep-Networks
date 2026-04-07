# Project Organization Guide

## 目录结构

### 仓库顶层一览（当前）

与应用代码/文档直接相关的顶层项（不含 `.git`）：


| 路径                | 说明                                              |
| ----------------- | ----------------------------------------------- |
| `archive/`        | 历史文档归档                                          |
| `configs/`        | 训练与实验 YAML/JSON                                 |
| `data/`           | 数据集与加载脚本                                        |
| `docs/`           | 项目文档、指南、`papers/` 论文稿、`project/` PRD 与进度        |
| `experiments/`    | 实验代码（core、matdo、validation、engram 等）            |
| `related_papers/` | 本地参考 PDF（多数被 `.gitignore` 忽略，不进入版本库）            |
| `results/`        | 实验与验证输出                                         |
| `scripts/`        | 工具脚本（setup、training、evaluation、benchmarks）      |
| `src/`            | 库源码（attnres、gating、engram、models、qttt、rabitq 等） |
| `tasks/`          | 任务与 PRD 说明（Markdown）                            |
| `tests/`          | 单元 / 集成 / E2E 测试                                |
| `third_party/`    | 第三方依赖（如 `rabitq-lib`）                           |
| `.github/`        | CI 工作流                                          |
| `.workbuddy/`     | 本地 WorkBuddy 配置（可选）                             |


### 核心文件 (根目录)

根目录仅保留项目核心文件（论文与新手文档见 `docs/papers/`、`docs/guides/`）：


| 文件                        | 说明                          |
| ------------------------- | --------------------------- |
| `README.md`               | 项目主文档，包含快速开始指南、安装说明、训练和评估命令 |
| `AGENTS.md`               | AI Agent 开发指南，包含架构设计决策和实现细节 |
| `PROJECT_ORGANIZATION.md` | 本文件，项目结构和文件说明               |
| `LICENSE`                 | Apache 2.0 许可证              |
| `Makefile`                | 构建脚本，包含格式化、测试、清理等命令         |
| `pyproject.toml`          | Python 项目配置，包含依赖和工具配置       |
| `requirements.txt`        | Python 依赖列表                 |
| `.gitignore`              | Git 忽略规则                    |


### 归档目录 (`archive/`)

存放历史文件和备份：

```
archive/
├── old_summaries/                # 旧的总结文档
│   ├── CHAPTER3_CODE_REVIEW.md   # 第3章代码审查报告
│   ├── E2E_TESTING_COMPLETE.md   # E2E测试完成报告
│   ├── E2E_TEST_RESULTS.md       # E2E测试结果
│   ├── FIXES_SUMMARY.md          # 修复总结
│   ├── OPTIMIZATIONS_APPLIED.md  # 已应用优化
│   ├── OPTIMIZATIONS_COMPLETE.md # 优化完成报告
│   ├── OPTIMIZATION_FIXES.md     # 优化修复
│   ├── PAPER_CODE_GAP_ANALYSIS.md # 论文代码差距分析
│   ├── PAPER_REVISIONS.md        # 论文修订
│   ├── PAPER_UPDATES_SUMMARY.md  # 论文更新总结
│   └── REVISION_SUMMARY.md       # 修订总结
├── progress_tracking/            # (预留)
└── project_docs/                 # (预留)
```

### 配置目录 (`configs/`)

实验和训练配置文件：


| 文件                                            | 说明                     |
| --------------------------------------------- | ---------------------- |
| `ds_config_h20.json`                          | DeepSpeed H20 GPU 配置文件 |
| `experiments/default.yaml`                    | 默认实验配置                 |
| `experiments/exp1_representation_burial.yaml` | Exp1: 表示埋葬实验配置         |
| `experiments/exp2_margin_analysis.yaml`       | Exp2: 边际分析实验配置         |
| `experiments/exp3_gradient_flow.yaml`         | Exp3: 梯度流实验配置          |
| `experiments/validation_targets.yaml`         | 验证目标配置                 |


### 数据目录 (`data/`)

数据集存储：

```
data/
└── zero_scrolls/                 # ZeroSCROLLS 长文本数据集
    ├── README.txt                # 数据集说明
    ├── DATASET_DOWNLOADED.md     # 下载标记文件
    ├── zero_scrolls_loader.py    # 数据集加载器
    ├── book_sum_sort/            # 书籍摘要排序数据集
    ├── gov_report/               # 政府报告数据集
    ├── musique/                  # 多跳问答数据集
    ├── narrative_qa/             # 叙事问答数据集
    ├── qasper/                   # 学术论文问答数据集
    ├── qmsum/                    # 查询聚焦摘要数据集
    ├── quality/                  # 多项选择问答数据集
    ├── space_digest/             # 太空摘要数据集
    ├── squality/                 # 科学论文质量数据集
    └── summ_screen_fd/           # 剧本摘要数据集
```

### 文档目录 (`docs/`)

项目文档和技术文档：

#### 核心文档


| 文件                                        | 说明            |
| ----------------------------------------- | ------------- |
| `ARCHITECTURE.md`                         | 系统架构文档        |
| `ATTNRES_INTEGRATION_STATUS.md`           | AttnRes 集成状态  |
| `INCREMENTAL_KV_IMPLEMENTATION.md`        | 增量 KV 缓存实现文档  |
| `INFERENCE_CODE_CONSISTENCY_CHECK.md`     | 推理代码一致性检查     |
| `INFERENCE_OPTIMIZATION_COMPLETE.md`      | 推理优化完成报告      |
| `PAPER_CODE_CONSISTENCY_CHECK.md`         | 论文代码一致性检查     |
| `PAPER_INFERENCE_CONSISTENCY_ANALYSIS.md` | 论文推理一致性分析     |
| `QTTP_INTEGRATION_STATUS.md`              | QTTP 集成状态     |
| `README.md`                               | 文档目录说明        |
| `SCRIPTS_FIX_SUMMARY.md`                  | 脚本修复总结        |
| `rabitq_external_integration.md`          | RaBitQ 外部集成文档 |
| `technical_review_indepth_analysis.md`    | 技术审查深度分析      |


#### API 文档 (`docs/api/`)


| 文件          | 说明     |
| ----------- | ------ |
| `README.md` | API 文档 |


#### 审计文档 (`docs/audits/`)


| 文件                       | 说明        |
| ------------------------ | --------- |
| `FULL_CITATION_AUDIT.md` | 完整引用审计    |
| `QTTT_CITATION_AUDIT.md` | QTTT 引用审计 |


#### 指南文档 (`docs/guides/`)


| 文件                               | 说明                                    |
| -------------------------------- | ------------------------------------- |
| `A100_80G_COMPLETE_GUIDE.md`     | **A100 80G 新手完全指南** - 从环境设置到训练推理的详细步骤 |
| `MATDO_E_A100_BEGINNER_GUIDE.md` | MATDO-E / A100 入门与排障（长教程）             |
| `README_FOR_BEGINNERS.md`        | MATDO-E 新手路径与资源索引                     |
| `LARGE_MODEL_BUILD.md`           | 大模型构建指南                               |
| `MNN_TURBOQUANT_IMPROVEMENTS.md` | MNN TurboQuant 改进指南                   |
| `TURBOQUANT_REFACTORED.md`       | TurboQuant 重构指南                       |
| `TURBOQUANT_V3.md`               | TurboQuant V3 指南                      |


#### 论文相关 (`docs/papers/`)


| 文件                                                       | 说明                          |
| -------------------------------------------------------- | --------------------------- |
| `Adaptive_Deep_Networks_Experimental_Plan.md`            | ADN 实验计划                    |
| `Adaptive_Deep_Networks_Query_Optimization.md`           | 查询优化论文（主草稿）                 |
| `Adaptive_Deep_Networks_Query_Optimization-2.md`         | 查询优化论文变体稿                   |
| `Adaptive_Deep_Networks_Query_Optimization_Corrected.md` | 查询优化订正稿                     |
| `Adaptive_Deep_Networks_Query_Optimization_REVISED.md`   | 查询优化修订稿（当前主用的长篇 Markdown）   |
| `ADN_Query_Optimization_v3.md`                           | 查询优化 v3                     |
| `Adaptive_Deep_Networks_RaBitQ.md`                       | RaBitQ 论文                   |
| `Adaptive_Deep_Networks_TurboQuant.md`                   | TurboQuant 论文               |
| `Adaptive_Deep_Networks_V1.md`                           | ADN V1 版本                   |
| `matdo_paper.tex`                                        | MATDO LaTeX（经典稿）            |
| `matdo-e_paper.tex`                                      | MATDO-E LaTeX 补充稿           |
| `matdo-e_revised_paper.md` / `matdo-e_revised_paper.tex` | MATDO-E 修订稿（Markdown + TeX） |


#### 项目文档 (`docs/project/`)

```
docs/project/
├── prd/                          # 产品需求文档
│   ├── prd.json                  # 主 PRD
│   ├── prd-engram.json           # Engram PRD（JSON）
│   ├── prd-incremental-kv.json   # 增量 KV PRD
│   └── prd-optimizations.json    # 优化 PRD
├── progress/                     # 进度跟踪
│   ├── progress.txt              # 主进度
│   ├── progress-incremental-kv.txt # 增量 KV 进度
│   └── progress-optimizations.txt  # 优化进度
├── progress.txt                  # 进度总览
└── reports/                      # 项目报告
    ├── ARCHITECTURE_OPTIMIZATION.md  # 架构优化报告
    ├── GEMMA4_ADN_COMPARISON.md      # Gemma4 对比分析
    ├── IMPLEMENTATION_AUDIT.md       # 实现审计
    ├── MATDO.md                      # MATDO 论文
    ├── SOTA_CLAIMS_SUMMARY.md        # SOTA 声明总结
    └── TURBOQUANT_V3_SUMMARY.md      # TurboQuant V3 总结
```

#### 报告 (`docs/reports/`)


| 文件                             | 说明          |
| ------------------------------ | ----------- |
| `AUTODL_VERIFICATION_GUIDE.md` | AutoDL 验证指南 |
| `DATASET_VALIDATION_REPORT.md` | 数据集验证报告     |
| `DATA_VERIFICATION_REPORT.md`  | 数据验证报告      |
| `H20_4CARD_SUMMARY.md`         | H20 4卡总结    |
| `MODEL_PARAMS_REPORT.md`       | 模型参数报告      |
| `STREAMING_TRAINING_GUIDE.md`  | 流式训练指南      |
| `STREAMING_UPDATE_SUMMARY.md`  | 流式更新总结      |


#### 实现报告 (`docs/reports/implementation/`)


| 文件                                     | 说明              |
| -------------------------------------- | --------------- |
| `REFACTOR_COMPLETE.md`                 | 重构完成报告          |
| `TURBOQUANT_IMPLEMENTATION_SUMMARY.md` | TurboQuant 实现总结 |


#### 研究报告 (`docs/reports/research/`)


| 文件                              | 说明   |
| ------------------------------- | ---- |
| `research_report_2026-03-29.md` | 研究报告 |
| `run_summary_2026-03-29.md`     | 运行总结 |


#### 验证报告 (`docs/reports/validation/`)


| 文件                         | 说明        |
| -------------------------- | --------- |
| `VALIDATION_REPORT_5_2.md` | 5.2 节验证报告 |


### 实验目录 (`experiments/`)

实验代码和配置：


| 文件                           | 说明      |
| ---------------------------- | ------- |
| `MIGRATION_GUIDE.md`         | 迁移指南    |
| `Makefile`                   | 实验构建脚本  |
| `QUICKSTART.md`              | 快速开始指南  |
| `README.md`                  | 实验框架说明  |
| `REFACTORING_SUMMARY.md`     | 重构总结    |
| `__init__.py`                | 包初始化    |
| `run_experiments_unified.py` | 统一实验运行器 |


#### 通用模块 (`experiments/common/`)


| 文件                  | 说明    |
| ------------------- | ----- |
| `__init__.py`       | 包初始化  |
| `config.py`         | 通用配置  |
| `device.py`         | 设备管理  |
| `logging_config.py` | 日志配置  |
| `paths.py`          | 路径管理  |
| `visualization.py`  | 可视化工具 |


#### 核心实验 (`experiments/core/`)


| 实验            | 文件                            | 说明                |
| ------------- | ----------------------------- | ----------------- |
| 表示埋葬          | `exp1_representation_burial/` | Exp1: 验证表示埋葬问题    |
| 边际分析          | `exp2_margin_analysis/`       | Exp2: 查询边际分布分析    |
| 梯度流           | `exp3_gradient_flow/`         | Exp3: AttnRes 梯度流 |
| FLOP 等效       | `exp4_flop_equivalence/`      | Exp4: FLOP 等效性验证  |
| 协同效应          | `exp5_synergy/`               | Exp5: 组件协同效应      |
| 辅助头           | `exp6_auxiliary/`             | Exp6: 辅助头分析       |
| Table3 推理     | `table3_inference_benchmark/` | Table3 推理基准       |
| Table5 边际     | `table5_margin_distribution/` | Table5 边际分布       |
| Table8 Pareto | `table8_pareto/`              | Table8 Pareto 前沿  |


#### Engram 实验 (`experiments/engram/`)


| 文件                    | 说明                                                |
| --------------------- | ------------------------------------------------- |
| `benchmark_engram.py` | **Engram 性能对比基准** - 对比 baseline ADN vs ADN+Engram |


每个实验目录包含：`__init__.py`, `experiment.py`, `run_exp*.py`, `config.yaml`

#### 文档 (`experiments/docs/`)


| 文件                          | 说明              |
| --------------------------- | --------------- |
| `TURBOQUANT_EXPERIMENTS.md` | TurboQuant 实验文档 |
| `core_experiments.md`       | 核心实验文档          |


#### MATDO 实验 (`experiments/matdo/`)

```
matdo/
├── __init__.py
├── run_all_experiments.py        # 运行所有 MATDO 实验（含真实模型模式）
├── MATDO_E_IMPLEMENTATION.md     # MATDO-E 实现说明
├── matdo_e/                      # MATDO-E 核心（求解器、调度、Engram、套利注意力）
│   ├── __init__.py
│   ├── arbitrage_attention.py
│   ├── engram_manager.py
│   ├── scheduler.py
│   └── solver.py
├── vllm_integration/             # vLLM 相关集成与压测脚本
│   ├── run_all_vllm_experiments.py
│   ├── latency_profiler.py
│   ├── throughput_test.py
│   ├── ablation_vllm.py
│   └── accuracy_recovery.py
├── ablation/                     # 消融实验
├── common/                       # 通用模块
│   ├── __init__.py
│   ├── config.py
│   └── real_model_bridge.py      # 真实模型桥接
├── dual_hierarchy/               # 双层次分析（含 find_rho_oom.py 等）
├── online_identification/        # 在线识别
│   └── rls_estimator.py          # RLS 估计器
├── results/                      # 实验结果（按子实验分子目录）
├── shadow_price/                 # 影子价格分析
│   └── calculate_lambda2.py
├── singularity/                  # 奇点分析
│   └── measure_t_opt.py          # 测量最优 T
└── sota_comparison/              # SOTA 对比
    └── compare_baselines.py
```

#### RaBitQ 实验 (`experiments/rabitq/`)


| 文件                                      | 说明      |
| --------------------------------------- | ------- |
| `__init__.py`                           | 包初始化    |
| `compression_verification_results.json` | 压缩验证结果  |
| `config.yaml`                           | 实验配置    |
| `microbenchmark_results.json`           | 微基准结果   |
| `run_compression_verification.py`       | 压缩验证脚本  |
| `run_microbenchmarks.py`                | 微基准测试脚本 |


#### 真实模型实验 (`experiments/real_model/`)


| 文件                           | 说明                                    |
| ---------------------------- | ------------------------------------- |
| `IMPLEMENTATION_SUMMARY.md`  | 实现总结                                  |
| `README.md`                  | 说明文档                                  |
| `__init__.py`                | 包初始化                                  |
| `datasets/__init__.py`       | 数据集子包                                 |
| `datasets/needle_dataset.py` | Needle-in-Haystack 数据集（HuggingFace 等） |
| `memory_profiler.py`         | 内存分析器                                 |
| `model_loader.py`            | 模型加载器                                 |
| `needle_haystack_real.py`    | 真实模型针在干草堆测试                           |
| `validator.py`               | 验证器                                   |


#### 结果目录 (`experiments/results/`)


| 目录            | 说明            |
| ------------- | ------------- |
| `benchmarks/` | 基准测试结果        |
| `core/`       | 核心实验结果        |
| `turboquant/` | TurboQuant 结果 |


#### 实验运行器 (`experiments/runner/`)


| 文件            | 说明    |
| ------------- | ----- |
| `__init__.py` | 包初始化  |
| `base.py`     | 基础运行器 |
| `discover.py` | 实验发现  |
| `runner.py`   | 主运行器  |


#### TurboQuant 实验 (`experiments/turboquant/`)


| 文件          | 说明   |
| ----------- | ---- |
| `README.md` | 说明文档 |


#### 工具 (`experiments/utils/`)


| 文件               | 说明   |
| ---------------- | ---- |
| `__init__.py`    | 包初始化 |
| `measurement.py` | 测量工具 |


#### 验证实验 (`experiments/validation/`)


| 文件                                | 说明             |
| --------------------------------- | -------------- |
| `README.md`                       | 说明文档           |
| `base_validator.py`               | 基础验证器          |
| `extreme_context_scaling.py`      | 极端上下文扩展        |
| `progressive_context_test.py`     | 渐进上下文测试        |
| `run_all_validations.py`          | 运行所有验证         |
| `table1_representation_burial.py` | Table1 验证      |
| `table2_gradient_flow.py`         | Table2 梯度流     |
| `table3_rabitq_space_accuracy.py` | Table3 RaBitQ  |
| `table4_needle_haystack.py`       | Table4 针在干草堆   |
| `table5_query_margin.py`          | Table5 查询边际    |
| `table6_math.py`                  | Table6 MATH 基准 |
| `table7_synergy.py`               | Table7 协同效应    |
| `table8_sram_allocation.py`       | Table8 SRAM 分配 |
| `table9_coupling_effect.py`       | Table9 耦合效应    |


### 结果目录 (`results/`)

实验结果和报告：

#### 历史结果 (`results/2026-03-xx/`)


| 目录            | 说明                              |
| ------------- | ------------------------------- |
| `2026-03-23/` | 3月23日结果（数据集验证、验证结果）             |
| `2026-03-24/` | 3月24日结果（大模型配置、5.2节报告）           |
| `2026-03-30/` | 3月30日结果（实验、论文指标、小模型、TurboQuant） |


#### 核心结果 (`results/core/`)


| 目录                            | 说明                             |
| ----------------------------- | ------------------------------ |
| `table3_inference_benchmark/` | Table3 推理基准结果                  |
| `table8_pareto/`              | Table8 Pareto 结果（含配置、图表、日志、报告） |
| `test_exp/`                   | 测试实验                           |


#### 验证结果 (`results/validation/` 和 `results/validations/`)


| 文件                                 | 说明               |
| ---------------------------------- | ---------------- |
| `table1_results.json/png`          | Table1 表示埋葬验证    |
| `table2_results.json/png`          | Table2 梯度流验证     |
| `table3_results.json/png`          | Table3 RaBitQ 验证 |
| `table4_results.json/png`          | Table4 针在干草堆验证   |
| `table5_results.json/png`          | Table5 查询边际验证    |
| `table6_results.json/png`          | Table6 MATH 验证   |
| `table7_results.json/png`          | Table7 协同效应验证    |
| `table8_results.json/png`          | Table8 SRAM 分配验证 |
| `table9_results.json/png`          | Table9 耦合效应验证    |
| `extreme_context_scaling.json/png` | 极端上下文扩展结果        |


#### 其他结果文件


| 文件                                          | 说明            |
| ------------------------------------------- | ------------- |
| `README.md`                                 | 结果目录说明        |
| `benchmark_summary.md`                      | 基准总结          |
| `qttt_paper_values_quick.json`              | QTTT 论文值（快速）  |
| `qttt_paper_vs_optimized.json`              | QTTT 论文 vs 优化 |
| `rabitq_endtoend_benchmark.json`            | RaBitQ 端到端基准  |
| `small_model_experiments_experimental.json` | 小模型实验（实验版）    |
| `table5_margin_distribution/`               | Table5 边际分布   |


### 脚本目录 (`scripts/`)

实用脚本：

#### 基准测试脚本


| 文件                                     | 说明             |
| -------------------------------------- | -------------- |
| `README.md`                            | 脚本说明           |
| `benchmark_attnres_endtoend.py`        | AttnRes 端到端基准  |
| `benchmark_qttt_endtoend.py`           | QTTT 端到端基准     |
| `benchmark_qttt_paper_values.py`       | QTTT 论文值基准     |
| `benchmark_qttt_paper_values_quick.py` | QTTT 快速基准      |
| `benchmark_rabitq_endtoend.py`         | RaBitQ 端到端基准   |
| `benchmark_rabitq_final.py`            | RaBitQ 最终基准    |
| `benchmark_rabitq_kv_compression.py`   | RaBitQ KV 压缩基准 |
| `benchmark_rabitq_minimal.py`          | RaBitQ 最小基准    |
| `benchmark_rabitq_quick.py`            | RaBitQ 快速基准    |


#### Colab 脚本 (`scripts/colab/`)


| 文件                       | 说明          |
| ------------------------ | ----------- |
| `test_colab.py`          | Colab 测试    |
| `test_colab_complete.py` | 完整 Colab 测试 |


#### 通用模块 (`scripts/common/`)


| 文件                     | 说明      |
| ---------------------- | ------- |
| `__init__.py`          | 包初始化    |
| `data.py`              | 数据工具    |
| `distributed.py`       | 分布式训练工具 |
| `experiment_runner.py` | 实验运行器   |
| `paths.py`             | 路径管理    |
| `training.py`          | 训练工具    |


#### 数据脚本 (`scripts/data/`)


| 文件                         | 说明                 |
| -------------------------- | ------------------ |
| `check_datasets.sh`        | 检查数据集脚本            |
| `dataset_info.py`          | 数据集信息              |
| `dataset_validation.py`    | 数据集验证              |
| `download_datasets.sh`     | 下载数据集              |
| `download_zero_scrolls.sh` | 下载 ZeroSCROLLS     |
| `validate_datasets.py`     | 验证数据集              |
| `validate_hf_datasets.py`  | 验证 HuggingFace 数据集 |


#### 评估脚本 (`scripts/evaluation/`)


| 文件                         | 说明      |
| -------------------------- | ------- |
| `eval_5_2.py`              | 5.2 节评估 |
| `run_benchmarks.py`        | 运行基准测试  |
| `run_medium_model_eval.sh` | 中等模型评估  |
| `run_real_validation.sh`   | 真实模型验证  |
| `validate_models.py`       | 模型验证    |


#### 实验脚本 (`scripts/experiments/`)


| 文件                                     | 说明       |
| -------------------------------------- | -------- |
| `paper_metrics_summary.py`             | 论文指标总结   |
| `run_small_model_paper_experiments.py` | 小模型论文实验  |
| `test_small_model_datasets.py`         | 小模型数据集测试 |


#### 遗留脚本 (`scripts/experiments/legacy/`)


| 文件                                  | 说明                 |
| ----------------------------------- | ------------------ |
| `test_turboquant_on_small_model.py` | TurboQuant 小模型测试   |
| `test_turboquant_v3_improved.py`    | TurboQuant V3 改进测试 |
| `validate_turboquant_setup.py`      | TurboQuant 设置验证    |
| `turboquant_v3_demo.py`             | TurboQuant V3 演示   |


#### 遗留脚本 (`scripts/legacy/`)


| 文件                                    | 说明                |
| ------------------------------------- | ----------------- |
| `__init__.py`                         | 包初始化              |
| `mnn_turboquant_demo.py`              | MNN TurboQuant 演示 |
| `run_small_experiments.py`            | 小实验               |
| `run_small_model_experiments_fast.py` | 快速小模型实验           |
| `test_turboquant_small.py`            | TurboQuant 小测试    |
| `turboquant_refactored_demo.py`       | TurboQuant 重构演示   |


#### 模型脚本 (`scripts/model/`)


| 文件                               | 说明         |
| -------------------------------- | ---------- |
| `build_and_benchmark_small.py`   | 构建并基准测试小模型 |
| `build_large_model.py`           | 构建大模型      |
| `calculate_params.py`            | 计算参数（旧版）   |
| `calculate_params_v2.py`         | 计算参数 V2    |
| `calculate_training_time.py`     | 计算训练时间     |
| `run_small_model_experiments.py` | 运行小模型实验    |


#### 设置脚本 (`scripts/setup/`)


| 文件                    | 说明                                    |
| --------------------- | ------------------------------------- |
| `a100_setup.sh`       | **A100 80G 一键环境设置脚本** - 新手推荐，自动安装所有依赖 |
| `check_env.py`        | 环境与依赖检查（MATDO-E / GPU / 磁盘等）          |
| `autodl_h20_setup.sh` | AutoDL H20 设置                         |
| `autodl_setup.sh`     | AutoDL 设置                             |
| `lambda_setup.sh`     | Lambda AI 设置                          |
| `QUICKSTART.sh`       | MATDO-E A100 一键检查环境与快速启动（在仓库根目录执行）    |
| `quick_start_h20.sh`  | H20 快速开始                              |
| `quick_train.sh`      | **一键训练启动器** - 自动创建 tmux 会话并开始训练       |


#### 训练脚本 (`scripts/training/`)


| 文件                    | 说明         |
| --------------------- | ---------- |
| `__init__.py`         | 包初始化       |
| `base_trainer.py`     | 基础训练器      |
| `train_h20.py`        | H20 GPU 训练 |
| `train_large.py`      | 大模型训练      |
| `train_medium.py`     | 中等模型训练     |
| `train_model.py`      | 通用训练       |
| `train_refactored.py` | 重构版训练      |
| `train_small.py`      | 小模型训练      |
| `train_streaming.py`  | 流式训练       |
| `train_unified.py`    | 统一训练       |


#### 其他脚本


| 文件                           | 说明              |
| ---------------------------- | --------------- |
| `eval_rabitq_cpp_binding.py` | RaBitQ C++ 绑定评估 |
| `validate_rabitq.py`         | RaBitQ 验证       |
| `verify_qttt_fix.py`         | 验证 QTTT 修复      |


### 源代码目录 (`src/`)

核心源代码：

#### AttnRes (`src/attnres/`)


| 文件                      | 说明                                        |
| ----------------------- | ----------------------------------------- |
| `__init__.py`           | 包初始化，导出 BlockAttnRes、PseudoQueryManager 等 |
| `block_attnres.py`      | 块注意力残差实现                                  |
| `polar_pseudo_query.py` | 极坐标伪查询管理                                  |
| `pseudo_query.py`       | 伪查询管理器                                    |


#### 基准测试 (`src/benchmarks/`)


| 文件                   | 说明        |
| -------------------- | --------- |
| `__init__.py`        | 包初始化      |
| `flop_analysis.py`   | FLOP 分析   |
| `math_eval.py`       | MATH 基准评估 |
| `needle_haystack.py` | 针在干草堆测试   |


#### 门控 (`src/gating/`)


| 文件                  | 说明             |
| ------------------- | -------------- |
| `__init__.py`       | 包初始化，导出门控相关类   |
| `depth_priority.py` | 深度优先门控控制器      |
| `ponder_gate.py`    | Ponder Gate 实现 |
| `reconstruction.py` | 重构损失计算         |
| `threshold.py`      | 动态阈值校准         |


#### Engram (`src/engram/`)


| 文件                        | 说明                                            |
| ------------------------- | --------------------------------------------- |
| `__init__.py`             | 包初始化                                          |
| `config.py`               | Engram 配置预设（Small/Medium/Large）               |
| `engram_module.py`        | Engram 模块核心                                   |
| `compressed_tokenizer.py` | 压缩分词 / tokenizer 对接                           |
| `embeddings.py`           | n-gram 嵌入                                     |
| `ngram_hash.py`           | n-gram 哈希                                     |
| `integration.py`          | 与 `AdaptiveTransformer` / `AdaptiveLayer` 的集成 |


#### 模型 (`src/models/`)


| 文件                         | 说明                       |
| -------------------------- | ------------------------ |
| `__init__.py`              | 包初始化，导出 ModelConfig      |
| `adaptive_transformer.py`  | 自适应 Transformer 模型       |
| `configs.py`               | 模型配置（Small/Medium/Large） |
| `incremental_generator.py` | 增量生成器                    |
| `incremental_kv_cache.py`  | 增量 KV 缓存                 |
| `incremental_state.py`     | 增量状态管理                   |
| `tokenizer.py`             | 分词器                      |


#### QTTT (`src/qttt/`)


| 文件                    | 说明               |
| --------------------- | ---------------- |
| `__init__.py`         | 包初始化，导出 QTTT 相关类 |
| `adaptation.py`       | QTTT 适配实现        |
| `adaptive_config.py`  | 自适应配置            |
| `batch_adaptation.py` | 批量适配             |
| `margin_loss.py`      | 边际最大化损失          |
| `polar_adaptation.py` | 极坐标适配            |


#### RaBitQ (`src/rabitq/`)


| 文件             | 说明          |
| -------------- | ----------- |
| `README.md`    | RaBitQ 说明文档 |
| `__init__.py`  | 包初始化        |
| `api.py`       | RaBitQ API  |
| `cache.py`     | RaBitQ 缓存   |
| `estimator.py` | RaBitQ 估计器  |
| `packing.py`   | 位打包         |
| `quantizer.py` | RaBitQ 量化器  |
| `rotation.py`  | 旋转操作        |


##### 遗留代码 (`src/rabitq/legacy/`)


| 文件                     | 说明     |
| ---------------------- | ------ |
| `api_legacy.py`        | 旧版 API |
| `cache_legacy.py`      | 旧版缓存   |
| `compressor_legacy.py` | 旧版压缩器  |
| `quantizer_legacy.py`  | 旧版量化器  |
| `rotation_legacy.py`   | 旧版旋转   |


#### TurboQuant (`src/turboquant/`)


| 文件              | 说明              |
| --------------- | --------------- |
| `README.md`     | TurboQuant 说明文档 |
| `__init__.py`   | 包初始化            |
| `api.py`        | TurboQuant API  |
| `cache.py`      | TurboQuant 缓存   |
| `compressor.py` | 压缩流程封装          |
| `quantizer.py`  | TurboQuant 量化器  |
| `rotation.py`   | 旋转操作            |


##### 遗留代码 (`src/turboquant/legacy/`)


| 文件                | 说明            |
| ----------------- | ------------- |
| `README.md`       | 遗留代码说明        |
| `__init__.py`     | 包初始化          |
| `mnn_improved.py` | MNN 改进版       |
| `polar_quant.py`  | 极坐标量化         |
| `qjl.py`          | QJL 实现        |
| `tensor_core.py`  | 张量核心优化        |
| `turbo_quant.py`  | 旧版 TurboQuant |
| `v3_improved.py`  | V3 改进版        |


#### 工具 (`src/utils/`)

(预留目录，当前为空)

### 任务目录 (`tasks/`)

产品需求和任务文档：


| 文件                                         | 说明                     |
| ------------------------------------------ | ---------------------- |
| `prd-engram-integration.md`                | Engram 集成 PRD          |
| `product-brief-engram.md`                  | Engram 产品简报            |
| `implementation-readiness-check.md`        | 实现准备检查                 |
| `prd-adaptive-deep-networks-validation.md` | ADN 验证 PRD             |
| `prd-adaptive-transformer-integration.md`  | 自适应 Transformer 集成 PRD |
| `prd-batch-processing.md`                  | 批处理 PRD                |
| `prd-experiment-refactor.md`               | 实验重构 PRD               |
| `prd-incremental-kv.md`                    | 增量 KV PRD              |
| `prd-inference-optimization.md`            | 推理优化 PRD               |
| `prd-optimizations.md`                     | 优化 PRD                 |
| `prd-rabitq-library-alignment.md`          | RaBitQ 库对齐 PRD         |
| `prd-turboquant-refactor.md`               | TurboQuant 重构 PRD      |
| `pre-mortem-analysis.md`                   | 事前分析                   |
| `product-brief-incremental-kv.md`          | 增量 KV 产品简报             |
| `product-brief-inference-optimization.md`  | 推理优化产品简报               |
| `product-brief-optimizations.md`           | 优化产品简报                 |
| `product-brief.md`                         | 产品简报                   |


### 测试目录 (`tests/`)

测试代码：


| 文件             | 说明        |
| -------------- | --------- |
| `conftest.py`  | Pytest 配置 |
| `run_tests.py` | 测试运行器     |


#### 基准测试 (`tests/benchmark/`)


| 文件                              | 说明     |
| ------------------------------- | ------ |
| `test_inference_performance.py` | 推理性能测试 |


#### 基准测试2 (`tests/benchmarks/`)

(空目录)

#### E2E 测试 (`tests/e2e/`)


| 文件                       | 说明          |
| ------------------------ | ----------- |
| `test_all_components.py` | 所有组件 E2E 测试 |


#### 集成测试 (`tests/integration/`)


| 文件                                | 说明               |
| --------------------------------- | ---------------- |
| `test_inference_optimization.py`  | 推理优化集成测试         |
| `test_ponder_gate_integration.py` | Ponder Gate 集成测试 |


#### 遗留测试 (`tests/legacy/`)


| 文件                         | 说明                |
| -------------------------- | ----------------- |
| `__init__.py`              | 包初始化              |
| `test_gating_simple.py`    | 简单门控测试            |
| `test_mnn_turboquant.py`   | MNN TurboQuant 测试 |
| `test_models_simple.py`    | 简单模型测试            |
| `test_polar_components.py` | 极坐标组件测试           |
| `test_turboquant.py`       | TurboQuant 测试     |
| `test_turboquant_core.py`  | TurboQuant 核心测试   |


#### 单元测试 (`tests/unit/`)


| 文件                             | 说明                |
| ------------------------------ | ----------------- |
| `test_adaptive_qttt_config.py` | 自适应 QTTT 配置测试     |
| `test_attnres.py`              | AttnRes 单元测试      |
| `test_attnres_integration.py`  | AttnRes 集成测试      |
| `test_benchmarks.py`           | 基准测试              |
| `test_engram.py`               | **Engram 模块完整测试** |
| `test_gating.py`               | 门控测试              |
| `test_incremental_state.py`    | 增量状态测试            |
| `test_layer_specific_qttt.py`  | 层特定 QTTT 测试       |
| `test_models.py`               | 模型测试              |
| `test_ponder_gate.py`          | Ponder Gate 测试    |
| `test_qttt.py`                 | QTTT 测试           |
| `test_qttt_forward_fix.py`     | QTTT 前向修复测试       |
| `test_qttt_loss_types.py`      | QTTT 损失类型测试       |
| `test_qttt_self_supervised.py` | QTTT 自监督测试        |
| `test_rabitq.py`               | RaBitQ 测试         |


#### 重构测试


| 文件                          | 说明          |
| --------------------------- | ----------- |
| `test_rabitq_refactored.py` | RaBitQ 重构测试 |


### 第三方库 (`third_party/`)

外部依赖：

#### RaBitQ 库 (`third_party/rabitq-lib/`)

C++ RaBitQ 实现：

- `CMakeLists.txt` - CMake 配置
- `LICENSE` - 许可证
- `README.md` - 说明文档
- `docs/` - 文档
- `include/rabitqlib/` - C++ 头文件
- `python/` - Python 绑定
- `sample/` - 示例代码
- `tests/` - 测试代码

### 相关工作论文 (`related_papers/`)

相关研究论文 PDF（默认文件名示例如下；**多数 `.pdf` 被 `.gitignore` 排除**，克隆后目录可能为空或仅本地有文件）：


| 文件                                         | 说明              |
| ------------------------------------------ | --------------- |
| `01_Alon_Klartag_Optimal_compression...`   | 最优近似内积压缩        |
| `03_Sun_Test-time_training...`             | 测试时训练           |
| `04_Bansal_Test-Time_Training...`          | 长上下文 LLM 的测试时训练 |
| `07_Xiao_SmoothQuant...`                   | SmoothQuant     |
| `08_Frantar_GPTQ...`                       | GPTQ 量化         |
| `16_Gao_RaBitQ_Precursor...`               | RaBitQ 先驱       |
| `16_Gao_RaBitQ_Theoretical_Error_Bound...` | RaBitQ 理论误差界    |
| `16_Gao_RaBitQ...`                         | RaBitQ 论文       |
| `19_Pagliardini_DenseFormer...`            | DenseFormer     |
| `2504.19874.pdf`                           | 相关论文            |
| `26_Child_Generating_long_sequences...`    | 稀疏 Transformer  |
| `27_Zaheer_BigBird...`                     | BigBird         |
| `28_Wang_Linformer...`                     | Linformer       |
| `35_Zhu_Hyper-Connections...`              | 超连接             |
| `40_Graves_Adaptive_computation_time...`   | 自适应计算时间         |
| `44_Sun_TTT-Linear...`                     | TTT-Linear      |
| `58_Kimi_Team_Attention_Residuals...`      | 注意力残差           |


### GitHub 配置 (`.github/`)

GitHub Actions 工作流：


| 文件                        | 说明       |
| ------------------------- | -------- |
| `workflows/README.md`     | 工作流说明    |
| `workflows/benchmark.yml` | 基准测试工作流  |
| `workflows/pr.yml`        | PR 检查工作流 |
| `workflows/validate.yml`  | 验证工作流    |


### WorkBuddy 配置 (`.workbuddy/`)

WorkBuddy AI 助手配置：


| 文件                      | 说明       |
| ----------------------- | -------- |
| `expert-history.json`   | 专家历史     |
| `settings.local.json`   | 本地设置     |
| `memory/`               | 内存记录     |
| `teams/experiment-dev/` | 实验开发团队配置 |


---

## 文件移动记录

### 2026-04-07 根目录与任务整理


| 原位置                                      | 新位置                                | 说明                     |
| ---------------------------------------- | ---------------------------------- | ---------------------- |
| `ADN_Query_Optimization_v3.md`           | `docs/papers/`                     | 论文稿归档                  |
| `matdo-e_revised_paper.md` / `.tex`      | `docs/papers/`                     | MATDO-E 修订稿            |
| `MATDO_E_A100_BEGINNER_GUIDE.md`         | `docs/guides/`                     | 与现有 guides 一致          |
| `README_FOR_BEGINNERS.md`                | `docs/guides/`                     | 新手索引                   |
| `QUICKSTART.sh`                          | `scripts/setup/QUICKSTART.sh`      | 与安装脚本同目录               |
| `tasks/engram/prd-engram-integration.md` | `tasks/prd-engram-integration.md`  | 扁平化任务目录                |
| `tasks/engram/product-brief.md`          | `tasks/product-brief-engram.md`    | 避免与根级 product-brief 重名 |
| `tasks/engram/prd.json`                  | `docs/project/prd/prd-engram.json` | 与其他 PRD JSON 同目录       |
| `related paper/`（多为被忽略的 PDF）             | `related_papers/`                  | 去掉路径空格，便于脚本引用          |


### 已归档文件


| 原位置                       | 新位置                      | 说明     |
| ------------------------- | ------------------------ | ------ |
| `CHAPTER3_CODE_REVIEW.md` | `archive/old_summaries/` | 代码审查报告 |
| `E2E_TESTING_COMPLETE.md` | `archive/old_summaries/` | 测试完成报告 |
| `E2E_TEST_RESULTS.md`     | `archive/old_summaries/` | 测试结果   |
| `FIXES_SUMMARY.md`        | `archive/old_summaries/` | 修复总结   |
| `OPTIMIZATION_FIXES.md`   | `archive/old_summaries/` | 优化修复   |
| `OPTIMIZATIONS_*.md` (3个) | `archive/old_summaries/` | 优化文档   |
| `PAPER_*.md` (3个)         | `archive/old_summaries/` | 论文更新   |
| `REVISION_SUMMARY.md`     | `archive/old_summaries/` | 修订总结   |


### 已整理文件


| 原位置                            | 新位置                      | 说明            |
| ------------------------------ | ------------------------ | ------------- |
| `prd*.json` (3个)               | `docs/project/prd/`      | PRD 文档        |
| `progress*.txt` (3个)           | `docs/project/progress/` | 进度跟踪          |
| `ARCHITECTURE_OPTIMIZATION.md` | `docs/project/reports/`  | 架构报告          |
| `GEMMA4_ADN_COMPARISON.md`     | `docs/project/reports/`  | 对比分析          |
| `IMPLEMENTATION_AUDIT.md`      | `docs/project/reports/`  | 实现审计          |
| `MATDO.md`                     | `docs/project/reports/`  | MATDO 论文      |
| `SOTA_CLAIMS_SUMMARY.md`       | `docs/project/reports/`  | SOTA 声明       |
| `TURBOQUANT_V3_SUMMARY.md`     | `docs/project/reports/`  | TurboQuant 总结 |


### 已删除文件

- `.DS_Store` (系统文件)
- `__pycache__/` 目录 (Python 缓存)
- `.pytest_cache/` 目录 (测试缓存)

---

## 整理前后对比


| 指标                | 整理前（约 2026-04）        | 整理后（当前约定）                                 | 说明                          |
| ----------------- | --------------------- | ----------------------------------------- | --------------------------- |
| 根目录零散 Markdown/论文 | 多份在根目录                | 迁入 `docs/papers/`、`docs/guides/`          | 根目录仅保留 README、AGENTS 等核心元文件 |
| 任务 PRD            | 部分在子目录                | Engram 等并入 `tasks/` 与 `docs/project/prd/` | 见「文件移动记录」                   |
| 参考 PDF 路径         | `related paper/`（含空格） | `related_papers/`                         | 本地文件夹；PDF 通常不提交             |


---

## 使用建议

### 添加新文件时

1. **代码文件** → 放入 `src/` 相应子目录
2. **测试文件** → 放入 `tests/` 相应子目录
3. **脚本文件** → 放入 `scripts/` 或 `experiments/`
4. **项目文档** → 放入 `docs/project/` 相应子目录
5. **临时/旧文档** → 放入 `archive/`

### 查找文件

- 论文与长篇稿: `docs/papers/`
- 新手与环境指南: `docs/guides/`（含 `README_FOR_BEGINNERS.md`、`MATDO_E_A100_BEGINNER_GUIDE.md`）
- 活跃项目文档: `docs/project/`
- 历史文档: `archive/`
- 代码: `src/`（AttnRes / Gating / Engram / Models / qTTT / RaBitQ / TurboQuant）
- 测试: `tests/`
- 脚本: `scripts/`（`scripts/setup/QUICKSTART.sh`）、`experiments/`

---

*整理日期: 2026-04-05*  
*最后更新: 2026-04-07 — 同步仓库结构：`docs/papers` 全量条目、`experiments/matdo`（MATDO-E / vLLM）、`src/engram`、`scripts/setup/check_env.py`、`related_papers` 说明、顶层目录一览*