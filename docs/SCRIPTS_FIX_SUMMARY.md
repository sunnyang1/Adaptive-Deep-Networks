# 模型搭建脚本修复总结

**日期:** 2026-04-02  
**修复范围:** `scripts/training/` 和 `experiments/`

---

## 修复内容

### 1. 导入路径错误 (Critical)

**问题:** 多个脚本尝试从 `scripts.training.common` 导入模块，但正确的路径是 `scripts.common`。

**影响的文件:**
- `scripts/training/train_unified.py`
- `scripts/training/train_refactored.py`
- `experiments/core/exp3_gradient_flow/experiment.py`

**修复方式:**
```python
# 错误
from scripts.training.common import (
    get_default_paths,
    ensure_directories,
    setup_distributed,
    ...
)

# 正确
from scripts.common.paths import get_default_paths, ensure_directories
from scripts.common.distributed import setup_distributed, cleanup_distributed, is_main_process
from scripts.common.training import CheckpointManager, compute_loss, train_step
from scripts.common.data import DummyDataset, get_dataloader
```

### 2. 模型配置不一致 (High)

**问题:** `train_unified.py` 使用硬编码的配置字典，而不是 `src/models/configs.py` 中的配置类。

**修复:**
```python
# 错误
configs = {
    'small': {
        'vocab_size': 32000,
        'hidden_dim': 2048,
        'num_layers': 32,
        ...
    },
    ...
}

# 正确
from src.models.configs import get_config
config = get_config('small')  # 返回 AttnResSmallConfig
```

### 3. 占位符模型 (High)

**问题:** `train_unified.py` 使用了 `nn.TransformerEncoder` 作为占位符，而不是实际的 `AdaptiveTransformer`。

**修复:**
```python
# 错误
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(...),
    num_layers=model_config['num_layers']
)

# 正确
from src.models.adaptive_transformer import AdaptiveTransformer
model = AdaptiveTransformer(config)
```

---

## 修复验证

### 语法检查
```bash
python -c "
import py_compile
scripts = [
    'scripts/training/train_unified.py',
    'scripts/training/train_refactored.py',
    'experiments/core/exp3_gradient_flow/experiment.py',
]
for script in scripts:
    py_compile.compile(script, doraise=True)
    print(f'✓ {script}')
"
```

**结果:** ✅ 所有修复的脚本语法正确

### 导入检查
```bash
python -c "
from scripts.common.paths import get_default_paths, ensure_directories
from scripts.common.distributed import setup_distributed, is_main_process
from scripts.common.training import CheckpointManager, compute_loss, train_step
from scripts.common.data import DummyDataset, get_dataloader
from src.models.configs import get_config
from src.models.adaptive_transformer import AdaptiveTransformer
print('✓ All imports work')
"
```

**结果:** ✅ 所有导入正常

---

## 未修复的依赖问题

### 1. transformers 库 (Optional)

**问题:** `src/models/tokenizer.py` 依赖 `transformers` 库，用于加载 GPT-2 和 Llama 分词器。

**影响:** 
- 使用 HuggingFace 数据集的脚本需要安装 `transformers`
- 使用自定义分词器的脚本不受影响

**解决方案:**
```bash
pip install transformers
```

**替代方案:** 使用 `SimpleTokenizer` 或 `DummyDataset` 进行测试

---

## 推荐的模型搭建流程

### 1. 快速测试 (CPU)
```bash
python scripts/training/train_unified.py \
    --model-size small \
    --epochs 1 \
    --seq-len 128 \
    --batch-size 2
```

### 2. 完整训练 (GPU)
```bash
python scripts/training/train_small.py \
    --output-dir results/small_model \
    --epochs 5 \
    --batch-size 8 \
    --seq-len 512
```

### 3. 使用基础训练器
```bash
python scripts/training/train_small.py --help
python scripts/training/train_medium.py --help
python scripts/training/train_large.py --help
```

---

## 文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `train_unified.py` | ✅ 已修复 | 统一训练脚本 |
| `train_refactored.py` | ✅ 已修复 | 重构训练脚本 |
| `train_small.py` | ✅ 正常 | 小模型训练 |
| `train_medium.py` | ✅ 正常 | 中模型训练 |
| `train_large.py` | ✅ 正常 | 大模型训练 |
| `base_trainer.py` | ✅ 正常 | 基础训练器 |
| `exp3_gradient_flow/experiment.py` | ✅ 已修复 | 梯度流实验 |

---

## 注意事项

1. **路径问题:** 所有脚本使用 `sys.path.insert(0, ...)` 添加项目根目录到路径
2. **设备选择:** 脚本自动检测 CUDA/MPS/CPU
3. **分布式训练:** 使用 `--distributed` 标志启用
4. **数据集:** 默认使用 `DummyDataset`，可替换为 `HuggingFaceDataset`

---

## 后续建议

1. [ ] 添加 `transformers` 到 `requirements.txt`
2. [ ] 为所有训练脚本添加单元测试
3. [ ] 统一训练脚本的参数命名（如 `--epochs` vs `--num-epochs`）
4. [ ] 添加训练脚本的 README 文档
