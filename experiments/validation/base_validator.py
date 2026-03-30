"""
Base Class for Paper Validation Scripts

Modernized validation scripts using YAML config for targets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yaml
from typing import Dict, Any, Optional
from abc import abstractmethod

from experiments.runner import BaseExperiment, ExperimentResult
from experiments.common import ExperimentConfig, OutputPaths, MODEL_SIZES
from experiments.core.base_core_experiment import ValidationMixin


class PaperValidator(BaseExperiment, ValidationMixin):
    """
    Base class for paper table validation.
    
    Loads validation targets from YAML config and validates against them.
    """
    
    def __init__(
        self,
        name: str,
        table_name: str,
        targets_config_path: Optional[Path] = None
    ):
        super().__init__(name, category="validation")
        self.table_name = table_name
        self.targets = {}
        self.validations = {}
        
        # Load targets from config
        if targets_config_path is None:
            targets_config_path = Path(__file__).parent.parent.parent / \
                                  "configs" / "experiments" / "validation_targets.yaml"
        
        if targets_config_path.exists():
            with open(targets_config_path, 'r') as f:
                all_targets = yaml.safe_load(f)
                self.targets = all_targets.get(table_name, {})
    
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run validation (must be implemented by subclasses)."""
        pass
    
    def validate_all(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Validate all results against targets.
        
        Args:
            results: Measured results
        
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        for key, target_value in self.targets.items():
            if key == 'tolerance':
                continue
            
            if isinstance(target_value, dict):
                # Nested targets (e.g., per-architecture)
                for sub_key, sub_target in target_value.items():
                    actual = results.get(key, {}).get(sub_key)
                    if actual is not None:
                        validation = self.validate_target(
                            actual=actual,
                            target=sub_target,
                            tolerance=self.targets.get('tolerance', 0.15),
                            name=f"{key}.{sub_key}"
                        )
                        validations[f"{key}.{sub_key}"] = validation
            else:
                # Simple target
                actual = results.get(key)
                if actual is not None:
                    validation = self.validate_target(
                        actual=actual,
                        target=target_value,
                        tolerance=self.targets.get('tolerance', 0.15),
                        name=key
                    )
                    validations[key] = validation
        
        self.validations = validations
        return validations
    
    def all_passed(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            return False
        return all(v['passed'] for v in self.validations.values())
    
    def generate_report(self, result: ExperimentResult) -> str:
        """Generate validation report."""
        lines = [
            f"# {self.table_name} Validation Report",
            "",
            f"**Status**: {'✅ PASS' if self.all_passed() else '❌ FAIL'}",
            "",
        ]
        
        # Add validation table
        if self.validations:
            lines.append(self.generate_validation_report(self.validations))
        
        # Add raw results
        lines.extend([
            "",
            "## Raw Results",
            "",
            "```json",
            str(result.metrics),
            "```",
        ])
        
        return "\n".join(lines)


class Table1Validator(PaperValidator):
    """Validator for Table 1: Representation Burial."""
    
    def __init__(self):
        super().__init__(
            name="table1_representation_burial",
            table_name="table1_representation_burial"
        )
    
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run Table 1 validation."""
        from experiments.core.exp1_representation_burial.experiment import \
            RepresentationBurialExperiment
        
        # Run the core experiment
        exp = RepresentationBurialExperiment()
        exp_result = exp.execute(config)
        
        if not exp_result.success:
            return ExperimentResult(
                name=self.name,
                success=False,
                error="Core experiment failed"
            )
        
        # Validate against targets
        architectures = exp_result.metrics.get('architectures', {})
        results = {
            arch: {
                'attenuation': data['attenuation_rate'],
                'effective_depth': data['effective_depth'],
            }
            for arch, data in architectures.items()
        }
        
        validations = self.validate_all(results)
        
        return ExperimentResult(
            name=self.name,
            success=self.all_passed(),
            metrics={
                'validations': validations,
                'all_passed': self.all_passed(),
                'results': results,
            }
        )


# CLI for running validators
def main():
    """CLI entry point for validators."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Table Validation')
    parser.add_argument('table', choices=['table1', 'table2', 'table4'])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default=None)
    
    args = parser.parse_args()
    
    # Select validator
    validators = {
        'table1': Table1Validator,
        # Add more as they are implemented
    }
    
    ValidatorClass = validators.get(args.table)
    if ValidatorClass is None:
        print(f"Validator for {args.table} not yet implemented")
        return 1
    
    # Create config
    config = ExperimentConfig(
        name=f'{args.table}_validation',
        category='validation',
        device=args.device,
        output_dir=args.output_dir or Path(f'results/validation/{args.table}')
    )
    
    # Run validation
    validator = ValidatorClass()
    result = validator.execute(config)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation {'PASSED' if result.success else 'FAILED'}")
    if 'validations' in result.metrics:
        passed = sum(1 for v in result.metrics['validations'].values() if v['passed'])
        total = len(result.metrics['validations'])
        print(f"Passed: {passed}/{total}")
    print('='*60)
    
    return 0 if result.success else 1


if __name__ == '__main__':
    exit(main())
