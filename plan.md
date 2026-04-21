# Adaptive-Deep-Networks 重构计划

## 项目现状分析

项目包含366个Python文件，分布在多个并行开发的目录中：
- `src/` - 旧的核心实现 (attnres, qttt, rabitq, engram, gating, models, turboquant)
- `QASP/` - QASP论文实现（矩阵级查询自适应、信息质量加权）
- `MATDO-new/` - MATDO-E论文新实现（统一资源模型、策略决策）
- `experiments/` - 实验代码（core, matdo, rabitq, validation, engram, autoresearch）
- `scripts/` - 脚本（training, evaluation, setup, legacy）
- `tests/` - 测试

## 核心问题

1. **代码分散重复**：同一功能在 src/, QASP/, MATDO-new/ 中有多份实现
2. **架构不清晰**：模块边界模糊，依赖关系混乱
3. **过时代码未清理**：大量 legacy/ 代码仍在活跃目录中
4. **入口点过多**：训练和实验有多个入口脚本
5. **QASP和MATDO-E作为独立包**：与主代码库分离，通过 integration.py 桥接

## 重构目标

1. 统一代码到清晰的模块化结构中
2. 将过时代码移入 archive/ 并不再引用
3. 按论文功能划分清晰的模块边界
4. 统一入口点和配置管理
5. 保持向后兼容（README中的命令仍然可用）

## 新架构设计

```
src/
  adn/                    # 统一包根
    __init__.py
    
    # === 核心模型 ===
    core/                 # 基础组件
      __init__.py
      config.py           # 统一配置系统
      model_config.py     # 模型配置类
      base.py             # 基础模块（RMSNorm等）
      
    # === ADN主框架（第一篇论文）===
    models/               # 自适应Transformer模型
      __init__.py
      adaptive_transformer.py
      configs.py
      tokenizer.py
      incremental_generator.py
      
    attention/            # 注意力机制
      __init__.py
      block_attnres.py    # AttnRes实现
      pseudo_query.py
      polar_pseudo_query.py
      
    qttt/                 # 查询时自适应训练
      __init__.py
      adaptation.py
      polar_adaptation.py
      batch_adaptation.py
      margin_loss.py
      adaptive_config.py
      
    quantization/         # KV缓存量化（RaBitQ）
      __init__.py
      rabitq/             # RaBitQ实现
        __init__.py
        api.py
        rotation.py
        quantizer.py
        packing.py
      
    memory/               # 外部记忆（Engram）
      __init__.py
      engram_module.py
      ngram_hash.py
      embeddings.py
      compressed_tokenizer.py
      
    gating/               # 门控机制
      __init__.py
      ponder_gate.py
      depth_priority.py
      threshold.py
      reconstruction.py
      
    # === QASP扩展（第二篇论文）===
    qasp/                 # QASP模块
      __init__.py
      stiefel.py          # Stiefel流形优化
      matrix_qasp.py      # 矩阵级QASP
      quality_score.py    # 信息质量评分
      value_weighted_attnres.py
      value_weighted_engram.py
      models/
        __init__.py
        qasp_layer.py
        qasp_transformer.py
        components.py
        
    # === MATDO-E扩展（第三篇论文）===
    matdo_e/              # MATDO-E模块
      __init__.py
      core/
        __init__.py
        config.py
        policy.py
        error_model.py
        online_estimation.py
        resource_theory.py
      modeling/
        __init__.py
        matdo_model.py
        attention.py
        blocks.py
        query_adaptation.py
        scope_memory.py
        external_memory.py
        kv_quantization.py
      runtime/
        __init__.py
        backend.py
        decode.py
        generation.py
        materialize.py
        metrics.py
        prefill.py
        state.py
      experiments/
        __init__.py
        benchmarks/
        evaluators/
        tasks/
        studies/
        
    # === 实验框架 ===
    experiments/          # 统一实验框架
      __init__.py
      runner.py
      core/
      benchmarks/
        math_eval.py
        needle_haystack.py
        flop_analysis.py
      
    # === 工具 ===
    utils/                # 通用工具
      __init__.py
      paths.py
      device.py
      logging_config.py
      visualization.py
      
scripts/                # 精简后的脚本
  training/
    train.py              # 统一训练入口
  evaluation/
    evaluate.py           # 统一评估入口
  benchmarks/
    run_benchmarks.py

tests/                  # 测试（保持结构）
  unit/
  integration/
  e2e/

archive/                # 归档代码
  legacy_src/           # 旧src实现备份
  legacy_scripts/       # 旧脚本备份
  legacy_experiments/   # 旧实验代码备份
  
configs/                # 配置文件（保持）
docs/                   # 文档（保持）
results/                # 结果（保持）
data/                   # 数据（保持）
```

## 执行阶段

### Stage 1: 准备 - 备份当前代码结构
- 将过时代码识别并移入 archive/
- 创建新的目录结构

### Stage 2: 核心迁移 - src/ 重构
- 将 src/ 下的模块按新架构重组到 src/adn/ 下
- 清理 legacy/ 目录中的代码
- 统一配置系统

### Stage 3: QASP整合
- 将 QASP/ 整合进 src/adn/qasp/
- 清理重复代码
- 移除 integration.py 桥接

### Stage 4: MATDO-E整合
- 将 MATDO-new/matdo_new/ 整合进 src/adn/matdo_e/
- 统一包引用

### Stage 5: 实验框架统一
- 统一 experiments/ 入口
- 清理过时实验

### Stage 6: 脚本清理
- 统一 scripts/ 入口
- 将过时脚本移入 archive/

### Stage 7: 配置与入口更新
- 更新 pyproject.toml
- 更新 README.md
- 确保向后兼容
