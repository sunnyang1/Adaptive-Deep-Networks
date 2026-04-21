"""
Needle-in-Haystack Benchmark

Tests long-context retrieval capability by hiding a specific fact
(needle) in a large context (haystack) and testing retrieval.

Based on: Section 5.2.1 of Adaptive Deep Networks paper
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import json
import os


class NeedleHaystackBenchmark:
    """
    Needle-in-Haystack evaluation.

    Paper results (7B model):
    - 1K: 99.5%
    - 4K: 98.2%
    - 16K: 94.1%
    - 32K: 89.3%
    - 64K: 82.5%
    - 128K: 75.8%
    - 256K: 68.2%
    - Average: 86.9%
    """

    NEEDLE_TEMPLATE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    QUESTION = "What is the best thing to do in San Francisco?"
    ANSWER = "eat a sandwich and sit in Dolores Park on a sunny day"

    def __init__(
        self,
        model,
        tokenizer,
        context_lengths: List[int] = None,
        depths_per_length: int = 10,
        num_trials: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.context_lengths = context_lengths or [1024, 4096, 16384, 32768, 65536, 131072, 262144]
        self.depths_per_length = depths_per_length
        self.num_trials = num_trials

        self.results = {}

    def generate_haystack(self, num_tokens: int) -> str:
        """Generate filler text (haystack)."""
        # Use Paul Graham essays or similar repetitive text
        filler_text = (
            "The history of technology is filled with unexpected twists and turns. "
            "What seems obvious in retrospect was often controversial at the time. "
            "The best ideas often start as toys, dismissed by serious people. "
        ) * 1000

        # Tokenize and truncate to desired length
        tokens = self.tokenizer.encode(filler_text)
        if len(tokens) < num_tokens:
            # Repeat if necessary
            multiplier = (num_tokens // len(tokens)) + 1
            tokens = (tokens * multiplier)[:num_tokens]
        else:
            tokens = tokens[:num_tokens]

        return self.tokenizer.decode(tokens)

    def insert_needle(self, haystack: str, needle: str, depth_percent: float) -> str:
        """
        Insert needle at specified depth percentage.

        Args:
            haystack: Filler text
            needle: Fact to hide
            depth_percent: 0-100, position in context

        Returns:
            Context with needle inserted
        """
        tokens = self.tokenizer.encode(haystack)
        needle_tokens = self.tokenizer.encode(needle)

        # Calculate insertion position
        insert_pos = int(len(tokens) * depth_percent / 100)
        insert_pos = max(0, min(insert_pos, len(tokens) - len(needle_tokens)))

        # Insert needle
        new_tokens = tokens[:insert_pos] + needle_tokens + tokens[insert_pos:]

        return self.tokenizer.decode(new_tokens)

    def evaluate_single(
        self, context: str, question: str, expected_answer: str
    ) -> Tuple[bool, str]:
        """
        Evaluate single needle retrieval.

        Returns:
            (correct: bool, generated_answer: str)
        """
        # Prepare prompt
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=50, temperature=0.0, do_sample=False  # Greedy
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0][inputs.shape[1] :])
        generated = generated.strip().lower()

        # Check if answer is present
        expected_lower = expected_answer.lower()
        correct = expected_lower in generated

        return correct, generated

    def run(self, verbose: bool = True) -> Dict:
        """
        Run complete benchmark.

        Returns:
            Dictionary with results per context length
        """
        results = {}

        for ctx_len in tqdm(self.context_lengths, desc="Context lengths"):
            if verbose:
                print(f"\nEvaluating context length: {ctx_len}")

            depths = np.linspace(0, 100, self.depths_per_length + 2)[1:-1]
            depth_results = []

            for depth in tqdm(depths, desc="Depths", leave=False):
                trial_results = []

                for trial in range(self.num_trials):
                    # Generate haystack
                    haystack = self.generate_haystack(ctx_len)

                    # Insert needle
                    context = self.insert_needle(haystack, self.NEEDLE_TEMPLATE, depth)

                    # Evaluate
                    correct, answer = self.evaluate_single(context, self.QUESTION, self.ANSWER)

                    trial_results.append(
                        {"correct": correct, "answer": answer, "depth": depth, "trial": trial}
                    )

                depth_results.extend(trial_results)

            # Calculate accuracy
            accuracy = sum(r["correct"] for r in depth_results) / len(depth_results)

            results[ctx_len] = {
                "accuracy": accuracy,
                "num_evaluations": len(depth_results),
                "details": depth_results,
            }

            if verbose:
                print(f"  Accuracy: {accuracy * 100:.1f}%")

        # Calculate average
        avg_accuracy = np.mean([r["accuracy"] for r in results.values()])
        results["average"] = avg_accuracy

        self.results = results
        return results

    def save_results(self, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("Needle-in-Haystack Results")
        print("=" * 60)
        print(f"{'Context Length':<20} {'Accuracy':<15}")
        print("-" * 60)

        for ctx_len in sorted(k for k in self.results.keys() if isinstance(k, int)):
            acc = self.results[ctx_len]["accuracy"]
            print(f"{ctx_len:<20} {acc * 100:>6.1f}%")

        print("-" * 60)
        print(f"{'Average':<20} {self.results.get('average', 0) * 100:>6.1f}%")
        print("=" * 60)


def run_needle_haystack_validation(model, tokenizer, output_dir: str = "./results") -> Dict:
    """Convenience function to run full validation."""
    benchmark = NeedleHaystackBenchmark(model, tokenizer)
    results = benchmark.run()

    os.makedirs(output_dir, exist_ok=True)
    benchmark.save_results(os.path.join(output_dir, "needle_haystack.json"))
    benchmark.print_summary()

    return results
