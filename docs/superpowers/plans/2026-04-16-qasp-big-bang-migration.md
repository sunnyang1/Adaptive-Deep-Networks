# QASP Big-Bang Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `QASP/` primary pipeline for model, generation, inference, and experiments aligned to `QASP_paper.tex`, then switch documentation and primary run paths to QASP.

**Architecture:** Add a standalone `QASP/` package with focused modules for value-weighted AttnRes/Engram and matrix Stiefel adaptation, then wire scripts and experiment runner around it. Keep legacy code present for compatibility but switch default user-facing workflows to QASP entrypoints in one cutover.

**Tech Stack:** Python 3.12, PyTorch, existing project utilities/config patterns, pytest.

---

## File Structure

- Create: `QASP/__init__.py`
- Create: `QASP/configs/model.py`
- Create: `QASP/configs/qasp.py`
- Create: `QASP/configs/experiment.py`
- Create: `QASP/models/components.py`
- Create: `QASP/models/value_weighted_attnres.py`
- Create: `QASP/models/value_weighted_engram.py`
- Create: `QASP/models/qasp_layer.py`
- Create: `QASP/models/qasp_transformer.py`
- Create: `QASP/adaptation/quality_score.py`
- Create: `QASP/adaptation/stiefel.py`
- Create: `QASP/adaptation/ponder_gate.py`
- Create: `QASP/adaptation/matrix_qasp.py`
- Create: `QASP/inference/kv_cache.py`
- Create: `QASP/inference/incremental.py`
- Create: `QASP/inference/generator.py`
- Create: `QASP/experiments/runner.py`
- Create: `QASP/experiments/benchmarks/needle.py`
- Create: `QASP/experiments/benchmarks/math_eval.py`
- Create: `QASP/experiments/ablations/qasp_ablation.py`
- Create: `QASP/experiments/efficiency/profile.py`
- Create: `QASP/scripts/run_generation.py`
- Create: `QASP/scripts/run_inference.py`
- Create: `QASP/scripts/run_experiments.py`
- Create: `tests/unit/test_qasp_stiefel.py`
- Create: `tests/unit/test_qasp_quality_score.py`
- Create: `tests/unit/test_qasp_ponder_gate.py`
- Modify: `README.md`

### Task 1: Build QASP Config + Adaptation Core

**Files:**
- Create: `QASP/configs/model.py`
- Create: `QASP/configs/qasp.py`
- Create: `QASP/configs/experiment.py`
- Create: `QASP/adaptation/quality_score.py`
- Create: `QASP/adaptation/stiefel.py`
- Create: `QASP/adaptation/ponder_gate.py`
- Create: `QASP/adaptation/matrix_qasp.py`
- Test: `tests/unit/test_qasp_stiefel.py`
- Test: `tests/unit/test_qasp_quality_score.py`
- Test: `tests/unit/test_qasp_ponder_gate.py`

- [ ] **Step 1: Write failing unit tests**

```python
def test_stiefel_projection_outputs_near_orthonormal_columns():
    projected = stiefel_project(torch.randn(64, 8), num_iters=5)
    gram = projected.T @ projected
    assert torch.allclose(gram, torch.eye(8), atol=1e-2)
```

```python
def test_quality_score_is_bounded_between_zero_and_one():
    rho = compute_quality_scores(torch.randn(2, 16, 32))
    assert torch.all(rho >= 0.0)
    assert torch.all(rho <= 1.0)
```

```python
def test_ponder_gate_triggers_on_high_entropy():
    gate = PonderGate(entropy_threshold=0.8, confidence_threshold=0.6)
    logits = torch.zeros(1, 10)
    assert gate.should_adapt(logits) is True
```

- [ ] **Step 2: Run tests to verify failure**

Run:  
`pytest tests/unit/test_qasp_stiefel.py tests/unit/test_qasp_quality_score.py tests/unit/test_qasp_ponder_gate.py -v`

Expected: FAIL due to missing modules/functions.

- [ ] **Step 3: Implement minimal adaptation core**

```python
def stiefel_project(matrix: torch.Tensor, num_iters: int = 5, eps: float = 1e-6) -> torch.Tensor:
    y = matrix / (torch.linalg.norm(matrix, ord=2) + eps)
    eye = torch.eye(y.shape[1], device=y.device, dtype=y.dtype)
    for _ in range(num_iters):
        y = 0.5 * y @ (3.0 * eye - y.T @ y)
    return y
```

```python
def compute_quality_scores(hidden_states: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    freq = torch.fft.rfft(hidden_states, dim=-1)
    cutoff = max(freq.shape[-1] // 4, 1)
    low_pass = torch.zeros_like(freq)
    low_pass[..., :cutoff] = freq[..., :cutoff]
    low_component = torch.fft.irfft(low_pass, n=hidden_states.shape[-1], dim=-1)
    ratio = torch.linalg.norm(low_component, dim=-1) / (torch.linalg.norm(hidden_states, dim=-1) + eps)
    return (1.0 - ratio).clamp(min=0.0, max=1.0)
```

```python
class PonderGate:
    def should_adapt(self, logits: torch.Tensor) -> bool:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        return bool(((entropy > self.entropy_threshold) | (max_prob < self.confidence_threshold)).any().item())
```

- [ ] **Step 4: Re-run tests to verify pass**

Run:  
`pytest tests/unit/test_qasp_stiefel.py tests/unit/test_qasp_quality_score.py tests/unit/test_qasp_ponder_gate.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add QASP/configs QASP/adaptation tests/unit/test_qasp_stiefel.py tests/unit/test_qasp_quality_score.py tests/unit/test_qasp_ponder_gate.py
git commit -m "feat: add QASP adaptation core with tests"
```

### Task 2: Implement QASP Model Stack (AttnRes + Engram + Transformer)

**Files:**
- Create: `QASP/models/components.py`
- Create: `QASP/models/value_weighted_attnres.py`
- Create: `QASP/models/value_weighted_engram.py`
- Create: `QASP/models/qasp_layer.py`
- Create: `QASP/models/qasp_transformer.py`
- Create: `QASP/__init__.py`
- Test: `tests/unit/test_qasp_model_shapes.py`

- [ ] **Step 1: Write failing shape/integration test**

```python
def test_qasp_transformer_forward_shape():
    model = create_qasp_transformer()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, model.config.vocab_size)
```

- [ ] **Step 2: Run test to verify failure**

Run:  
`pytest tests/unit/test_qasp_model_shapes.py -v`

Expected: FAIL due to missing QASP model modules.

- [ ] **Step 3: Implement model modules**

```python
class ValueWeightedAttnRes(nn.Module):
    def forward(self, block_representations: torch.Tensor, query: torch.Tensor, block_quality: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btd,nbtd->nbt", query, block_representations)
        logits = logits * block_quality.unsqueeze(-1)
        alpha = torch.softmax(logits, dim=0)
        return torch.einsum("nbt,nbtd->btd", alpha, block_representations)
```

```python
class ValueWeightedEngram(nn.Module):
    def fuse(self, hidden: torch.Tensor, memory_vec: torch.Tensor, memory_quality: torch.Tensor, gate_alpha: torch.Tensor) -> torch.Tensor:
        return hidden + gate_alpha * torch.sigmoid(memory_quality).unsqueeze(-1) * memory_vec
```

```python
class QASPTransformer(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(self.norm(hidden))
```

- [ ] **Step 4: Re-run test to verify pass**

Run:  
`pytest tests/unit/test_qasp_model_shapes.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add QASP/models QASP/__init__.py tests/unit/test_qasp_model_shapes.py
git commit -m "feat: add QASP model stack with value-weighted modules"
```

### Task 3: Implement Generation + Incremental Inference Entry Paths

**Files:**
- Create: `QASP/inference/kv_cache.py`
- Create: `QASP/inference/incremental.py`
- Create: `QASP/inference/generator.py`
- Create: `QASP/scripts/run_generation.py`
- Create: `QASP/scripts/run_inference.py`
- Test: `tests/integration/test_qasp_generation_smoke.py`

- [ ] **Step 1: Write failing integration smoke test**

```python
def test_generate_returns_longer_sequence():
    model = create_qasp_transformer()
    generator = QASPGenerator(model)
    inp = torch.randint(0, model.config.vocab_size, (1, 8))
    out = generator.generate(inp, max_new_tokens=4)
    assert out.shape[1] == 12
```

- [ ] **Step 2: Run test to verify failure**

Run:  
`pytest tests/integration/test_qasp_generation_smoke.py -v`

Expected: FAIL due to missing generator/incremental modules.

- [ ] **Step 3: Implement inference modules and scripts**

```python
class QASPGenerator:
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 16) -> torch.Tensor:
        output = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.model(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            output = torch.cat([output, next_token], dim=1)
        return output
```

```python
def main() -> int:
    model = create_qasp_transformer()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 16))
    output = QASPGenerator(model).generate(input_ids, max_new_tokens=8)
    print(output.shape)
    return 0
```

- [ ] **Step 4: Re-run test to verify pass**

Run:  
`pytest tests/integration/test_qasp_generation_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add QASP/inference QASP/scripts tests/integration/test_qasp_generation_smoke.py
git commit -m "feat: add QASP generation and incremental inference entrypoints"
```

### Task 4: Implement QASP Experiment Runner + Benchmarks/Ablations/Efficiency

**Files:**
- Create: `QASP/experiments/runner.py`
- Create: `QASP/experiments/benchmarks/needle.py`
- Create: `QASP/experiments/benchmarks/math_eval.py`
- Create: `QASP/experiments/ablations/qasp_ablation.py`
- Create: `QASP/experiments/efficiency/profile.py`
- Create: `QASP/scripts/run_experiments.py`
- Test: `tests/integration/test_qasp_experiments_quick.py`

- [ ] **Step 1: Write failing runner smoke test**

```python
def test_qasp_experiment_runner_quick_writes_outputs(tmp_path):
    code = run_quick_experiments(output_dir=tmp_path)
    assert code == 0
    assert (tmp_path / "metrics.json").exists()
```

- [ ] **Step 2: Run test to verify failure**

Run:  
`pytest tests/integration/test_qasp_experiments_quick.py -v`

Expected: FAIL due to missing runner and quick output implementation.

- [ ] **Step 3: Implement experiment stack**

```python
def run_quick_experiments(output_dir: Path) -> int:
    metrics = {
        "needle_accuracy": run_needle_benchmark(quick=True),
        "math_score": run_math_eval(quick=True),
        "ablation": run_qasp_ablation(quick=True),
        "efficiency": profile_qasp(quick=True),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "config_snapshot.json").write_text(json.dumps({"mode": "quick"}, indent=2))
    (output_dir / "logs.txt").write_text("quick run complete\n")
    return 0
```

- [ ] **Step 4: Re-run test to verify pass**

Run:  
`pytest tests/integration/test_qasp_experiments_quick.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add QASP/experiments QASP/scripts/run_experiments.py tests/integration/test_qasp_experiments_quick.py
git commit -m "feat: add QASP experiment runner with quick artifacts"
```

### Task 5: Cutover README and Verify End-to-End Commands

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write failing command-level smoke checks (manual checklist)**

```python
# Manual verification checklist represented as assertions for script return codes.
assert run_command("python3 QASP/scripts/run_generation.py") == 0
assert run_command("python3 QASP/scripts/run_inference.py") == 0
assert run_command("python3 QASP/scripts/run_experiments.py --quick") == 0
```

- [ ] **Step 2: Run commands before README update to capture current behavior**

Run:
- `python3 QASP/scripts/run_generation.py`
- `python3 QASP/scripts/run_inference.py`
- `python3 QASP/scripts/run_experiments.py --quick`

Expected: all return code `0`.

- [ ] **Step 3: Update README defaults to QASP**

```markdown
## Quick Start (QASP)

python3 QASP/scripts/run_generation.py
python3 QASP/scripts/run_inference.py
python3 QASP/scripts/run_experiments.py --quick

## Legacy ADN (Deprecated)
Legacy ADN commands remain for historical reference and compatibility only.
```

- [ ] **Step 4: Re-run command-level verification**

Run:
- `python3 QASP/scripts/run_generation.py`
- `python3 QASP/scripts/run_inference.py`
- `python3 QASP/scripts/run_experiments.py --quick`

Expected: all return code `0` and quick artifacts written.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: switch primary usage docs to QASP workflows"
```
