# GitHub Actions Workflows

## Workflows

### 1. `validate.yml` - Main Validation
**Triggers:** Push to main/develop, PR to main, weekly schedule

**Jobs:**
- **Code Quality**: Black, flake8, mypy
- **Unit Tests**: Python 3.9, 3.10, 3.11 with coverage
- **FLOP Analysis**: Model-free verification of equivalence
- **Integration Tests**: Import tests, model creation
- **Documentation**: README, PRD, AGENTS.md checks
- **Summary**: Combined results

### 2. `benchmark.yml` - Performance Benchmarks
**Triggers:** Manual, monthly schedule

**Jobs:**
- Needle-in-Haystack (small scale)
- Detailed FLOP analysis

### 3. `pr.yml` - Pull Request Quick Check
**Triggers:** Pull requests

**Jobs:**
- Fast validation
- Core functionality smoke tests

## Status Badges

Add to README.md:

```markdown
![Validation](https://github.com/sunnyang1/Adaptive-Deep-Networks/workflows/Validation/badge.svg)
![Benchmark](https://github.com/sunnyang1/Adaptive-Deep-Networks/workflows/Benchmark/badge.svg)
```
