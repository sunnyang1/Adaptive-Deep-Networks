from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from matdo_new.experiments.benchmarks.needle import NeedleBenchmark
from matdo_new.experiments.evaluators.needle import NeedleEvaluator
from matdo_new.experiments.tasks.needle import NeedleDatasetAdapter, NeedleExample
from matdo_new.runtime.backend import AdaptiveTransformerRuntimeBackend
from matdo_new.runtime.state import BackendResult, MATDOState
from src.models.configs import ModelConfig


@dataclass(frozen=True)
class FakeLegacyNeedleSample:
    needle_depth_percent: float
    secret: str
    prompt: str

    def format_prompt(self) -> str:
        return self.prompt


class ScriptedNeedleDataset:
    def __init__(self, samples_by_context: dict[int, list[FakeLegacyNeedleSample]]) -> None:
        self.samples_by_context = samples_by_context
        self.calls: list[tuple[int, int, str]] = []

    def create_dataset(
        self,
        context_tokens: int,
        num_samples: int,
        depth_distribution: str = "uniform",
    ) -> list[FakeLegacyNeedleSample]:
        self.calls.append((context_tokens, num_samples, depth_distribution))
        return list(self.samples_by_context[context_tokens][:num_samples])


class SeededScriptedNeedleDataset(ScriptedNeedleDataset):
    def __init__(self, seed: int, samples_by_context: dict[int, list[FakeLegacyNeedleSample]]) -> None:
        super().__init__(samples_by_context=samples_by_context)
        self.seed = seed


class FakeTokenizer:
    def __init__(self, encodings: dict[str, tuple[int, ...]], decodings: dict[tuple[int, ...], str]) -> None:
        self.encodings = dict(encodings)
        self.decodings = dict(decodings)
        self.encode_calls: list[str] = []
        self.decode_calls: list[tuple[int, ...]] = []

    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> list[int] | torch.Tensor:
        del max_length, truncation, add_special_tokens
        self.encode_calls.append(text)
        tokens = list(self.encodings[text])
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, tokens: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(tokens, torch.Tensor):
            flattened = tuple(int(token) for token in tokens.reshape(-1).tolist())
        else:
            flattened = tuple(int(token) for token in tokens)
        self.decode_calls.append(flattened)
        return self.decodings.get(flattened, " ".join(str(token) for token in flattened))

    def __len__(self) -> int:
        token_ids = {token_id for tokens in self.encodings.values() for token_id in tokens}
        token_ids.update(token_id for tokens in self.decodings for token_id in tokens)
        return max(token_ids, default=-1) + 1


class ScriptedRuntimeBackend:
    def __init__(self, prompt_to_answer: dict[tuple[int, ...], tuple[int, ...]]) -> None:
        self.prompt_to_answer = dict(prompt_to_answer)
        self.forward_inputs: list[tuple[int, ...]] = []
        self.forward_step_inputs: list[tuple[int, ...]] = []
        self.forward_step_state_lengths: list[int] = []
        self._active_answer_by_prompt: dict[tuple[int, ...], tuple[int, ...]] = {}

    def forward(
        self,
        token_ids: tuple[int, ...],
        *,
        policy: object | None = None,
    ) -> BackendResult:
        del policy
        prompt = tuple(int(token_id) for token_id in token_ids)
        self.forward_inputs.append(prompt)
        answer = self.prompt_to_answer[prompt]
        self._active_answer_by_prompt[prompt] = answer
        initial_logits = answer[0] if answer else None
        return BackendResult(logits=initial_logits, cache={"prompt": prompt, "index": 0})

    def forward_step(
        self,
        token_ids: tuple[int, ...],
        *,
        state: MATDOState,
        policy: object | None = None,
    ) -> BackendResult:
        del policy
        emitted = tuple(int(token_id) for token_id in token_ids)
        self.forward_step_inputs.append(emitted)
        self.forward_step_state_lengths.append(state.sequence_length)
        prompt = tuple(int(token_id) for token_id in state.prompt_token_ids)
        active_answer = self._active_answer_by_prompt[prompt]
        decode_index = state.decoded_length + len(emitted)
        next_logits = (
            active_answer[decode_index]
            if decode_index < len(active_answer)
            else None
        )
        return BackendResult(
            logits=next_logits,
            cache={"prompt": prompt, "index": decode_index},
            submitted_token_count=len(emitted),
            used_incremental_cache=True,
        )


class SequentialTokenSampler:
    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self.token_ids = token_ids
        self.index = 0

    def __call__(self, logits: object | None) -> int:
        del logits
        token_id = self.token_ids[self.index]
        self.index += 1
        return int(token_id)


def _tiny_backend_config() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        hidden_dim=16,
        num_heads=4,
        num_blocks=1,
        mlp_ratio=2,
        vocab_size=64,
        max_seq_len=64,
    )


def test_dataset_adapter_normalizes_real_dataset_samples() -> None:
    adapter = NeedleDatasetAdapter()

    examples = adapter.build_examples(
        NeedleBenchmark(
            context_lengths=[256],
            num_samples=2,
            max_new_tokens=4,
            tokenizer_name="fake-tokenizer",
            model_size="t4",
            seed=7,
        )
    )

    assert len(examples) == 2
    assert all(example.target_answer for example in examples)
    assert all("Question: What is the secret passcode?" in example.prompt for example in examples)
    assert {example.metadata["context_length"] for example in examples} == {256}
    assert all(isinstance(example.metadata["needle_depth_percent"], float) for example in examples)
    assert all(example.evaluation.allow_exact_match for example in examples)
    assert all(example.evaluation.allow_subsequence_match for example in examples)


def test_dataset_adapter_reports_requested_budget_and_actual_prompt_length() -> None:
    dataset = SeededScriptedNeedleDataset(
        17,
        {
            256: [
                FakeLegacyNeedleSample(
                    needle_depth_percent=25.0,
                    secret="SECRET-X",
                    prompt="Prompt with much longer content than the requested budget label implies.",
                )
            ]
        },
    )
    tokenizer = FakeTokenizer(
        encodings={
            "Prompt with much longer content than the requested budget label implies.": (1, 2, 3, 4, 5, 6),
            "SECRET-X": (99,),
        },
        decodings={(99,): "SECRET-X"},
    )
    adapter = NeedleDatasetAdapter(dataset_factory=lambda seed: dataset)

    examples = adapter.build_examples(
        NeedleBenchmark(
            context_lengths=[256],
            num_samples=1,
            max_new_tokens=2,
            tokenizer_name="fake-tokenizer",
            model_size="t4",
            seed=17,
        ),
        tokenizer=tokenizer,
    )

    assert dataset.seed == 17
    assert dataset.calls == [(256, 1, "uniform")]
    assert len(examples) == 1
    assert examples[0].metadata["requested_context_length"] == 256
    assert examples[0].metadata["legacy_context_budget"] == 256
    assert examples[0].metadata["context_length"] == 6
    assert examples[0].metadata["prompt_token_count"] == 6


def test_needle_evaluator_uses_explicit_tokenizer_boundary_and_reports_metadata() -> None:
    benchmark = NeedleBenchmark(
        context_lengths=[256],
        num_samples=2,
        max_new_tokens=1,
        tokenizer_name="fake-tokenizer",
        model_size="t4",
    )
    examples = (
        NeedleExample(
            example_id="needle:256:0",
            prompt="Prompt A",
            target_answer="SECRET-A",
            metadata={
                "context_length": 2,
                "requested_context_length": 256,
                "prompt_token_count": 2,
                "needle_depth_percent": 10.0,
            },
            max_new_tokens=1,
        ),
        NeedleExample(
            example_id="needle:256:1",
            prompt="Prompt B",
            target_answer="SECRET-B",
            metadata={
                "context_length": 2,
                "requested_context_length": 256,
                "prompt_token_count": 2,
                "needle_depth_percent": 80.0,
            },
            max_new_tokens=4,
        ),
    )
    tokenizer = FakeTokenizer(
        encodings={
            "Prompt A": (101, 11),
            "Prompt B": (202, 22),
            "SECRET-A": (901,),
            "SECRET-B": (902, 903),
        },
        decodings={
            (901,): "SECRET-A",
            (902,): "SECRET-B",
        },
    )
    backend = ScriptedRuntimeBackend(
        {
            (101, 11): (901,),
            (202, 22): (444, 902, 903, 555),
        }
    )
    evaluator = NeedleEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda current_benchmark, current_tokenizer: backend,
        dataset_adapter=NeedleDatasetAdapter(dataset_factory=lambda: ScriptedNeedleDataset({256: []})),
        sampler_factory=lambda example, answer_token_ids: SequentialTokenSampler(
            (444, 902, 903, 555) if example.example_id.endswith(":1") else answer_token_ids
        ),
    )

    result = evaluator.evaluate(benchmark, examples=examples)

    assert tokenizer.encode_calls == ["Prompt A", "SECRET-A", "Prompt B", "SECRET-B"]
    assert tokenizer.decode_calls == [(901,), (444, 902, 903, 555)]
    assert backend.forward_inputs == [(101, 11), (202, 22)]
    assert backend.forward_step_inputs == [(901,), (444,), (902,), (903,), (555,)]
    assert result.aggregate_metrics["num_samples"] == 2
    assert result.aggregate_metrics["exact_match_rate"] == pytest.approx(0.5)
    assert result.aggregate_metrics["retrieval_success_rate"] == pytest.approx(1.0)
    assert result.aggregate_metrics["mean_prompt_tokens"] == pytest.approx(2.0)
    assert {summary.summary_id for summary in result.slice_summaries} == {
        "context-length:2",
        "needle-depth:early",
        "needle-depth:late",
    }
    assert result.example_summaries[0].metadata["context_length"] == 2
    assert result.example_summaries[0].metadata["requested_context_length"] == 256
    assert result.example_summaries[0].metadata["needle_depth_percent"] == 10.0
    assert result.example_summaries[1].metrics["exact_match"] is False
    assert result.example_summaries[1].metrics["retrieval_success"] is True
    assert result.runtime is not None
    assert result.runtime.metrics["prefill_calls"] == 2
    assert result.runtime.metrics["decode_calls"] == 5
    assert result.runtime.metrics["incremental_decode_calls"] == 5


def test_needle_evaluator_can_run_tiny_real_backend_end_to_end() -> None:
    torch.manual_seed(0)

    benchmark = NeedleBenchmark(
        context_lengths=[128],
        num_samples=1,
        max_new_tokens=2,
        tokenizer_name="fake-tokenizer",
        model_size="t4",
        use_attnres=False,
    )
    examples = (
        NeedleExample(
            example_id="needle:128:0",
            prompt="Prompt C",
            target_answer="SECRET-C",
            metadata={
                "context_length": 3,
                "requested_context_length": 128,
                "prompt_token_count": 3,
                "needle_depth_percent": 55.0,
            },
            max_new_tokens=2,
        ),
    )
    tokenizer = FakeTokenizer(
        encodings={
            "Prompt C": (7, 8, 9),
            "SECRET-C": (31, 32),
        },
        decodings={
            (31, 32): "SECRET-C",
        },
    )
    evaluator = NeedleEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda current_benchmark, current_tokenizer: (
            AdaptiveTransformerRuntimeBackend.from_backend_config(
                _tiny_backend_config(),
                use_attnres=current_benchmark.use_attnres,
            )
        ),
        dataset_adapter=NeedleDatasetAdapter(dataset_factory=lambda: ScriptedNeedleDataset({128: []})),
        sampler_factory=lambda example, answer_token_ids: SequentialTokenSampler(answer_token_ids),
    )

    result = evaluator.evaluate(benchmark, examples=examples)

    assert result.aggregate_metrics["num_samples"] == 1
    assert result.aggregate_metrics["exact_match_rate"] == pytest.approx(1.0)
    assert result.aggregate_metrics["retrieval_success_rate"] == pytest.approx(1.0)
    assert result.runtime is not None
    assert result.runtime.metrics["prefill_calls"] == 1
    assert result.runtime.metrics["decode_calls"] == 2
    assert result.runtime.metrics["submitted_tokens"] == 5


def test_needle_evaluator_does_not_count_text_substring_without_token_match() -> None:
    benchmark = NeedleBenchmark(
        context_lengths=[256],
        num_samples=1,
        max_new_tokens=1,
        tokenizer_name="fake-tokenizer",
        model_size="t4",
    )
    example = NeedleExample(
        example_id="needle:256:0",
        prompt="Prompt D",
        target_answer="SECRET-D",
        metadata={
            "context_length": 2,
            "requested_context_length": 256,
            "prompt_token_count": 2,
            "needle_depth_percent": 40.0,
        },
        max_new_tokens=1,
    )
    tokenizer = FakeTokenizer(
        encodings={
            "Prompt D": (303, 33),
            "SECRET-D": (911, 912),
        },
        decodings={
            (777,): "prefix SECRET-D suffix",
        },
    )
    backend = ScriptedRuntimeBackend({(303, 33): (777,)})
    evaluator = NeedleEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda current_benchmark, current_tokenizer: backend,
        dataset_adapter=NeedleDatasetAdapter(dataset_factory=lambda: ScriptedNeedleDataset({256: []})),
        sampler_factory=lambda example, answer_token_ids: SequentialTokenSampler((777,)),
    )

    result = evaluator.evaluate(benchmark, examples=(example,))

    assert result.aggregate_metrics["exact_match_rate"] == pytest.approx(0.0)
    assert result.aggregate_metrics["retrieval_success_rate"] == pytest.approx(0.0)
    assert result.example_summaries[0].metadata["generated_text"] == "prefix SECRET-D suffix"
    assert result.example_summaries[0].metadata["generated_token_ids"] == (777,)
    assert result.example_summaries[0].metrics["retrieval_success"] is False


def test_needle_evaluator_aligns_backend_vocab_with_tokenizer() -> None:
    benchmark = NeedleBenchmark(
        context_lengths=[128],
        num_samples=1,
        max_new_tokens=1,
        tokenizer_name="fake-tokenizer",
        model_size="t4",
        use_attnres=False,
    )
    tokenizer = FakeTokenizer(
        encodings={
            "Prompt E": (600, 601),
            "SECRET-E": (602,),
        },
        decodings={
            (602,): "SECRET-E",
        },
    )
    evaluator = NeedleEvaluator(tokenizer=tokenizer)

    backend = evaluator._build_backend(benchmark, tokenizer)

    assert isinstance(backend, AdaptiveTransformerRuntimeBackend)
    assert backend.model.config.vocab_size == len(tokenizer)
