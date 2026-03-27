"""
MATH Dataset Evaluation

Evaluates mathematical reasoning capability on competition problems.

Based on: Section 5.2.2 of Adaptive Deep Networks paper
"""

import torch
import json
import re
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset


class MATHEvaluator:
    """
    MATH dataset evaluator.
    
    Paper results (7B model):
    - Level 1: 76.2%
    - Level 2: 66.8%
    - Level 3: 56.4%
    - Level 4: 46.2%
    - Level 5: 34.5%
    - Overall: 52.3%
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_samples: int = None,
        use_cot: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.use_cot = use_cot
        
        # Load dataset
        try:
            self.dataset = load_dataset("hendrycks/competition_math", split="test")
            if max_samples:
                self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        except (ConnectionError, ValueError, RuntimeError):
            print("Warning: Could not load MATH dataset. Using dummy data.")
            self.dataset = []
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove latex formatting
        answer = re.sub(r'\\[a-zA-Z]+', '', answer)
        answer = re.sub(r'[\$\{\}\[\]]', '', answer)
        
        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        # Extract final answer (after = if present)
        if '=' in answer:
            answer = answer.split('=')[-1]
        
        return answer.strip().lower()
    
    def extract_numerical_answer(self, text: str) -> str:
        """Extract numerical answer from generated text."""
        # Look for common patterns
        patterns = [
            r'answer[\s:]+([^\n]+)',
            r'final answer[\s:]+([^\n]+)',
            r'answer is[\s:]+([^\n]+)',
            r'\\boxed\{([^}]+)\}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Return last line as fallback
        lines = text.strip().split('\n')
        return lines[-1] if lines else ""
    
    def create_prompt(self, problem: str, level: int) -> str:
        """Create evaluation prompt."""
        if self.use_cot:
            prompt = (
                f"Solve the following math problem step by step.\n\n"
                f"Problem (Level {level}): {problem}\n\n"
                f"Solution: Let's think through this step by step.\n"
            )
        else:
            prompt = f"Problem: {problem}\nAnswer:"
        
        return prompt
    
    def evaluate_single(self, example: Dict) -> Tuple[bool, str, str]:
        """
        Evaluate single problem.
        
        Returns:
            (correct, generated_answer, expected_answer)
        """
        problem = example['problem']
        level = example.get('level', 0)
        expected = example['solution']
        
        # Create prompt
        prompt = self.create_prompt(problem, level)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512 if self.use_cot else 128,
                temperature=0.3,
                do_sample=True
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0][inputs.shape[1]:])
        
        # Extract answer
        generated_answer = self.extract_numerical_answer(generated)
        expected_answer = self.extract_numerical_answer(expected)
        
        # Normalize and compare
        gen_norm = self.normalize_answer(generated_answer)
        exp_norm = self.normalize_answer(expected_answer)
        
        # Check correctness (exact match or numerical equivalence)
        correct = gen_norm == exp_norm
        
        # Try numerical comparison
        if not correct:
            try:
                gen_num = float(gen_norm.replace(',', ''))
                exp_num = float(exp_norm.replace(',', ''))
                correct = abs(gen_num - exp_num) < 1e-3
            except (ValueError, TypeError):
                pass
        
        return correct, generated_answer, expected_answer
    
    def run(self, verbose: bool = True) -> Dict:
        """Run complete evaluation."""
        if not self.dataset:
            return {"error": "No dataset loaded"}
        
        results_by_level = {i: [] for i in range(1, 6)}
        all_results = []
        
        for example in tqdm(self.dataset, desc="Evaluating MATH"):
            level = example.get('level', 1)
            if isinstance(level, str):
                level = int(level.split()[-1]) if 'Level' in level else 1
            
            correct, gen, exp = self.evaluate_single(example)
            
            result = {
                'correct': correct,
                'problem': example['problem'][:100],
                'generated': gen,
                'expected': exp,
                'level': level
            }
            
            all_results.append(result)
            
            if level in results_by_level:
                results_by_level[level].append(result)
        
        # Calculate accuracies
        summary = {}
        for level in range(1, 6):
            level_results = results_by_level[level]
            if level_results:
                acc = sum(r['correct'] for r in level_results) / len(level_results)
                summary[f'level_{level}'] = {
                    'accuracy': acc,
                    'count': len(level_results)
                }
        
        # Overall
        overall_acc = sum(r['correct'] for r in all_results) / len(all_results)
        summary['overall'] = {
            'accuracy': overall_acc,
            'count': len(all_results)
        }
        
        summary['details'] = all_results
        
        return summary
    
    def save_results(self, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("MATH Dataset Results")
        print("=" * 60)
        print(f"{'Level':<15} {'Accuracy':<15} {'Count':<10}")
        print("-" * 60)
        
        for level in range(1, 6):
            key = f'level_{level}'
            if key in self.results:
                acc = self.results[key]['accuracy']
                count = self.results[key]['count']
                print(f"Level {level:<9} {acc * 100:>6.1f}%      {count:<10}")
        
        if 'overall' in self.results:
            acc = self.results['overall']['accuracy']
            count = self.results['overall']['count']
            print("-" * 60)
            print(f"{'Overall':<15} {acc * 100:>6.1f}%      {count:<10}")
        
        print("=" * 60)


def run_math_evaluation(
    model,
    tokenizer,
    output_dir: str = "./results"
) -> Dict:
    """Convenience function to run MATH evaluation."""
    evaluator = MATHEvaluator(model, tokenizer)
    results = evaluator.run()
    evaluator.results = results
    
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(os.path.join(output_dir, "math_results.json"))
    evaluator.print_summary()
    
    return results
