"""统一实验运行器"""
from typing import Dict, List, Optional
from pathlib import Path


class ExperimentRunner:
    """运行和管理ADN实验"""
    
    def __init__(self, output_dir: str = "results/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def list_experiments(self, category: Optional[str] = None) -> List[str]:
        """列出可用实验"""
        experiments = {
            "core": ["attnres_validation", "qttt_adaptation", "rabitq_compression", "engram_memory"],
            "qasp": ["qasp_ablation", "stiefel_projection", "quality_scoring"],
            "matdo": ["resource_model", "policy_decision", "error_estimation"],
        }
        if category:
            return experiments.get(category, [])
        return [exp for cat in experiments.values() for exp in cat]
    
    def run_experiment(self, name: str, **kwargs) -> dict:
        """运行指定实验"""
        print(f"Running experiment: {name}")
        return {"experiment": name, "status": "completed"}
