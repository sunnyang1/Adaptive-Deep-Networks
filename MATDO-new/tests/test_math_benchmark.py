from __future__ import annotations

import pytest

from matdo_new.experiments.benchmarks.math import MathBenchmark
from matdo_new.experiments.evaluators import math as math_evaluator_module
from matdo_new.experiments.tasks.math import MathDatasetAdapter, MathExample, build_prompt
from matdo_new.runtime.state import BackendResult, MATDOState

from matdo_new.experiments.evaluators.math import MathEvaluator


class FakeTokenizer:
    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._next_id = 1

    def __len__(self) -> int:
        return 512

    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del return_tensors, max_length, truncation, add_special_tokens
        token_ids: list[int] = []
        for piece in text.split():
            token_ids.append(self._token_id(piece))
        return token_ids

    def decode(self, tokens: tuple[int, ...] | list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[int(token_id)] for token_id in tokens)

    def token_ids(self, text: str) -> tuple[int, ...]:
        return tuple(self.encode(text, add_special_tokens=False))

    def _token_id(self, piece: str) -> int:
        token_id = self._token_to_id.get(piece)
        if token_id is None:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[piece] = token_id
            self._id_to_token[token_id] = piece
        return token_id


class ScriptedRuntimeBackend:
    def __init__(
        self,
        scripts_by_prompt: dict[tuple[int, ...], tuple[int, ...]],
        *,
        use_incremental_cache: bool = True,
    ) -> None:
        self._scripts_by_prompt = dict(scripts_by_prompt)
        self._use_incremental_cache = use_incremental_cache

    def forward(
        self,
        token_ids: tuple[int, ...],
        *,
        policy: object | None = None,
    ) -> BackendResult:
        del policy
        script = self._scripts_by_prompt[tuple(token_ids)]
        first_logit = script[0] if script else None
        return BackendResult(
            logits=first_logit,
            cache={"script": script, "index": 1},
            submitted_token_count=len(token_ids),
            used_incremental_cache=False,
        )

    def forward_step(
        self,
        token_ids: tuple[int, ...],
        *,
        state: MATDOState,
        policy: object | None = None,
    ) -> BackendResult:
        del token_ids, policy
        cache = dict(state.cache)
        script = tuple(cache["script"])
        index = int(cache["index"])
        next_logit = script[index] if index < len(script) else None
        cache["index"] = index + 1
        return BackendResult(
            logits=next_logit,
            cache=cache,
            submitted_token_count=1,
            used_incremental_cache=self._use_incremental_cache,
        )


class NonDeterministicRuntimeModel:
    def sample_next_token(self, logits: object, temperature: float = 1.0) -> int:
        del logits, temperature
        return -1


def _build_math_benchmark(*, max_new_tokens: int = 2) -> MathBenchmark:
    return MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=[],
        levels=[],
        prompt_style="cot_boxed",
        max_new_tokens=max_new_tokens,
        tokenizer_name="fake-tokenizer",
        model_size="small",
    )


def _build_math_example(
    *,
    example_id: str,
    prompt: str,
    gold_answer: str,
    subject: str,
    level: int,
    max_new_tokens: int = 2,
) -> MathExample:
    return MathExample(
        example_id=example_id,
        problem=f"Problem for {example_id}",
        prompt=prompt,
        gold_solution=f"Solution for {example_id}",
        gold_answer=gold_answer,
        subject=subject,
        level=level,
        metadata={"subject": subject, "level": level},
        max_new_tokens=max_new_tokens,
    )


def test_math_benchmark_build_exposes_expected_identity_and_config() -> None:
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=["algebra", "geometry"],
        levels=[3, 5],
        prompt_style="cot_boxed",
        max_new_tokens=256,
        tokenizer_name="fake-tokenizer",
        model_size="small",
        use_attnres=False,
        seed=7,
    )

    assert benchmark.name == "math"
    assert benchmark.kind == "task"
    assert benchmark.config == {
        "split": "test",
        "max_samples": 8,
        "subjects": ("algebra", "geometry"),
        "levels": (3, 5),
        "prompt_style": "cot_boxed",
        "max_new_tokens": 256,
        "tokenizer_name": "fake-tokenizer",
        "model_size": "small",
        "use_attnres": False,
        "seed": 7,
    }


def test_math_benchmark_build_does_not_allow_name_override() -> None:
    with pytest.raises(TypeError):
        MathBenchmark.build(
            split="test",
            max_samples=8,
            subjects=["algebra"],
            levels=[3],
            prompt_style="cot_boxed",
            max_new_tokens=256,
            tokenizer_name="fake-tokenizer",
            model_size="small",
            name="custom-math",
        )


def test_math_benchmark_rejects_non_math_name_on_direct_construction() -> None:
    with pytest.raises(ValueError, match="MathBenchmark.name"):
        MathBenchmark(
            split="test",
            max_samples=8,
            subjects=("algebra",),
            levels=(3,),
            prompt_style="cot_boxed",
            max_new_tokens=256,
            tokenizer_name="fake-tokenizer",
            model_size="small",
            name="custom-math",
        )


def test_math_benchmark_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="MathBenchmark.split"):
        MathBenchmark.build(
            split="train",
            max_samples=8,
            subjects=["algebra"],
            levels=[3],
            prompt_style="cot_boxed",
            max_new_tokens=256,
            tokenizer_name="fake-tokenizer",
            model_size="small",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_samples": 0}, "MathBenchmark.max_samples"),
        ({"max_new_tokens": 0}, "MathBenchmark.max_new_tokens"),
        ({"tokenizer_name": ""}, "MathBenchmark.tokenizer_name"),
        ({"model_size": ""}, "MathBenchmark.model_size"),
    ],
)
def test_math_benchmark_rejects_invalid_runtime_and_sample_guards(
    kwargs: dict[str, object], message: str
) -> None:
    base_kwargs = {
        "split": "test",
        "max_samples": 8,
        "subjects": ["algebra"],
        "levels": [3],
        "prompt_style": "cot_boxed",
        "max_new_tokens": 256,
        "tokenizer_name": "fake-tokenizer",
        "model_size": "small",
    }

    with pytest.raises(ValueError, match=message):
        MathBenchmark.build(**(base_kwargs | kwargs))


def test_math_benchmark_rejects_invalid_prompt_style() -> None:
    with pytest.raises(ValueError, match="MathBenchmark.prompt_style"):
        MathBenchmark.build(
            split="test",
            max_samples=8,
            subjects=["algebra"],
            levels=[3],
            prompt_style="direct",
            max_new_tokens=256,
            tokenizer_name="fake-tokenizer",
            model_size="small",
        )


def test_math_dataset_adapter_filters_and_normalizes_subject_and_level() -> None:
    rows = [
        {
            "problem": "Compute 1 + 1.",
            "solution": "We add the integers and get \\boxed{2}.",
            "subject": "  Algebra ",
            "level": "Level 3",
        },
        {
            "problem": "Name a triangle.",
            "solution": "This is not the target row.",
            "subject": "Geometry",
            "level": "Level 2",
        },
    ]
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=["algebra"],
        levels=[3],
        prompt_style="cot_boxed",
        max_new_tokens=64,
        tokenizer_name="fake-tokenizer",
        model_size="small",
    )
    adapter = MathDatasetAdapter(dataset_loader=lambda split: rows)

    examples = adapter.build_examples(benchmark)

    assert len(examples) == 1
    assert examples[0].example_id == "math:algebra:3:0"
    assert examples[0].problem == "Compute 1 + 1."
    assert examples[0].subject == "algebra"
    assert examples[0].level == 3
    assert examples[0].metadata == {"subject": "algebra", "level": 3}
    assert examples[0].max_new_tokens == 64


def test_math_dataset_adapter_seed_changes_selected_subset_when_truncating() -> None:
    rows = [
        {
            "problem": f"Problem {index}",
            "solution": f"Work {index} \\boxed{{{index}}}.",
            "subject": "Algebra",
            "level": "Level 1",
        }
        for index in range(8)
    ]
    adapter = MathDatasetAdapter(dataset_loader=lambda split: rows)
    seed_one_benchmark = MathBenchmark.build(
        split="test",
        max_samples=3,
        subjects=["algebra"],
        levels=[1],
        prompt_style="cot_boxed",
        max_new_tokens=64,
        tokenizer_name="fake-tokenizer",
        model_size="small",
        seed=1,
    )
    seed_two_benchmark = MathBenchmark.build(
        split="test",
        max_samples=3,
        subjects=["algebra"],
        levels=[1],
        prompt_style="cot_boxed",
        max_new_tokens=64,
        tokenizer_name="fake-tokenizer",
        model_size="small",
        seed=2,
    )

    seed_one_ids = tuple(example.example_id for example in adapter.build_examples(seed_one_benchmark))
    seed_two_ids = tuple(example.example_id for example in adapter.build_examples(seed_two_benchmark))

    assert len(seed_one_ids) == 3
    assert len(seed_two_ids) == 3
    assert seed_one_ids != seed_two_ids


def test_math_dataset_adapter_reuses_same_seed_for_repeatable_subset_selection() -> None:
    rows = [
        {
            "problem": f"Problem {index}",
            "solution": f"Work {index} \\boxed{{{index}}}.",
            "subject": "Algebra",
            "level": "Level 1",
        }
        for index in range(8)
    ]
    adapter = MathDatasetAdapter(dataset_loader=lambda split: rows)
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=3,
        subjects=["algebra"],
        levels=[1],
        prompt_style="cot_boxed",
        max_new_tokens=64,
        tokenizer_name="fake-tokenizer",
        model_size="small",
        seed=17,
    )

    first_ids = tuple(example.example_id for example in adapter.build_examples(benchmark))
    second_ids = tuple(example.example_id for example in adapter.build_examples(benchmark))

    assert first_ids == second_ids


def test_build_prompt_cot_boxed_includes_instructions_and_problem() -> None:
    prompt = build_prompt("What is 1 + 1?", prompt_style="cot_boxed")

    assert "Solve the following math problem carefully." in prompt
    assert "What is 1 + 1?" in prompt
    assert "\\boxed{...}" in prompt


def test_math_dataset_adapter_extracts_gold_answer_from_boxed_solution() -> None:
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=[],
        levels=[],
        prompt_style="cot_boxed",
        max_new_tokens=32,
        tokenizer_name="fake-tokenizer",
        model_size="small",
    )
    adapter = MathDatasetAdapter(
        dataset_loader=lambda split: [
            {
                "problem": "Compute 1 + 1.",
                "solution": "We add the integers and get \\boxed{2}.",
                "subject": "Algebra",
                "level": "Level 1",
            }
        ]
    )

    examples = adapter.build_examples(benchmark)

    assert examples[0].gold_solution == "We add the integers and get \\boxed{2}."
    assert examples[0].gold_answer == "2"


def test_math_dataset_adapter_extracts_gold_answer_from_nested_boxed_solution() -> None:
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=[],
        levels=[],
        prompt_style="cot_boxed",
        max_new_tokens=32,
        tokenizer_name="fake-tokenizer",
        model_size="small",
    )
    adapter = MathDatasetAdapter(
        dataset_loader=lambda split: [
            {
                "problem": "Compute one half.",
                "solution": "The reduced value is \\boxed{\\frac{1}{2}}.",
                "subject": "Algebra",
                "level": "Level 1",
            }
        ]
    )

    examples = adapter.build_examples(benchmark)

    assert examples[0].gold_answer == "\\frac{1}{2}"


def test_math_dataset_adapter_extracts_gold_answer_from_final_answer_phrase() -> None:
    benchmark = MathBenchmark.build(
        split="test",
        max_samples=8,
        subjects=[],
        levels=[],
        prompt_style="cot_boxed",
        max_new_tokens=32,
        tokenizer_name="fake-tokenizer",
        model_size="small",
    )
    adapter = MathDatasetAdapter(
        dataset_loader=lambda split: [
            {
                "problem": "Compute 3 * 4.",
                "solution": "We multiply to get 12.\nFinal answer: 12",
                "subject": "Algebra",
                "level": "Level 1",
            }
        ]
    )

    examples = adapter.build_examples(benchmark)

    assert examples[0].gold_answer == "12"


def test_extract_candidate_answer_prefers_last_boxed_value() -> None:
    answer, mode = MathDatasetAdapter.extract_candidate_answer(
        "Reasoning \\boxed{11}\nMore work\nFinal form is \\boxed{\\frac{14}{2}}"
    )

    assert answer == "\\frac{14}{2}"
    assert mode == "boxed"


def test_extract_candidate_answer_falls_back_to_final_answer_phrase() -> None:
    answer, mode = MathDatasetAdapter.extract_candidate_answer(
        "We simplify the expression.\nFinal answer: 12"
    )

    assert answer == "12"
    assert mode == "final_phrase"


def test_normalize_answer_unwraps_boxed_math_formatting() -> None:
    assert MathDatasetAdapter.normalize_answer(r"$\boxed{14}$") == "14"


def test_answers_match_uses_exact_or_numeric_equality_without_substrings() -> None:
    assert MathDatasetAdapter.answers_match("14", "14") is True
    assert MathDatasetAdapter.answers_match("14", "114") is False


def test_answers_match_treats_fraction_and_decimal_as_equal() -> None:
    assert MathDatasetAdapter.answers_match("1/2", "0.5") is True


def test_answers_match_treats_latex_fraction_and_integer_as_equal() -> None:
    assert MathDatasetAdapter.answers_match("\\frac{14}{2}", "7") is True


def test_math_evaluator_aggregates_accuracy_slices_and_example_metadata() -> None:
    tokenizer = FakeTokenizer()
    examples = (
        _build_math_example(
            example_id="math:algebra:2:0",
            prompt="algebra prompt",
            gold_answer="4",
            subject="algebra",
            level=2,
        ),
        _build_math_example(
            example_id="math:geometry:3:1",
            prompt="geometry prompt",
            gold_answer="7",
            subject="geometry",
            level=3,
        ),
    )
    scripts_by_prompt = {
        tokenizer.token_ids(examples[0].prompt): tokenizer.token_ids("work \\boxed{4}"),
        tokenizer.token_ids(examples[1].prompt): tokenizer.token_ids("work \\boxed{8}"),
    }
    evaluator = MathEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda benchmark, resolved_tokenizer: ScriptedRuntimeBackend(scripts_by_prompt),
        sampler_factory=lambda example, prompt_token_ids: (lambda logits: int(logits)),
    )

    result = evaluator.evaluate(_build_math_benchmark(), examples=examples)

    assert result.aggregate_metrics["num_samples"] == 2
    assert result.aggregate_metrics["accuracy"] == 0.5
    assert result.aggregate_metrics["mean_prompt_tokens"] == 2
    assert result.aggregate_metrics["mean_generated_tokens"] == 2

    slice_ids = {summary.summary_id for summary in result.slice_summaries}
    assert "math-level:2" in slice_ids
    assert "math-level:3" in slice_ids
    assert "math-subject:algebra" in slice_ids
    assert "math-subject:geometry" in slice_ids

    example_by_id = {summary.summary_id: summary for summary in result.example_summaries}
    assert example_by_id["math:algebra:2:0"].metrics["correct"] is True
    assert example_by_id["math:geometry:3:1"].metrics["correct"] is False
    assert example_by_id["math:algebra:2:0"].metadata["candidate_answer"] == "4"
    assert example_by_id["math:algebra:2:0"].metadata["extraction_mode"] == "boxed"
    assert example_by_id["math:algebra:2:0"].metadata["generated_text"] == "work \\boxed{4}"
    assert example_by_id["math:algebra:2:0"].metadata["generated_token_ids"] == tokenizer.token_ids(
        "work \\boxed{4}"
    )


def test_math_evaluator_runtime_backed_generation_reports_runtime_metrics() -> None:
    tokenizer = FakeTokenizer()
    example = _build_math_example(
        example_id="math:number_theory:1:0",
        prompt="number theory prompt",
        gold_answer="9",
        subject="number_theory",
        level=1,
    )
    scripts_by_prompt = {
        tokenizer.token_ids(example.prompt): tokenizer.token_ids("reasoning \\boxed{9}"),
    }
    evaluator = MathEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda benchmark, resolved_tokenizer: ScriptedRuntimeBackend(scripts_by_prompt),
        sampler_factory=lambda example, prompt_token_ids: (lambda logits: int(logits)),
    )

    result = evaluator.evaluate(_build_math_benchmark(), examples=(example,))

    assert result.aggregate_metrics["accuracy"] == 1.0
    assert result.runtime is not None
    assert result.runtime.metrics["prefill_calls"] == 1
    assert result.runtime.metrics["decode_calls"] == 2


def test_math_evaluator_defaults_to_greedy_sampling_without_custom_sampler() -> None:
    tokenizer = FakeTokenizer()
    example = _build_math_example(
        example_id="math:algebra:1:0",
        prompt="default sampler prompt",
        gold_answer="6",
        subject="algebra",
        level=1,
    )
    scripts_by_prompt = {
        tokenizer.token_ids(example.prompt): tokenizer.token_ids("reasoning \\boxed{6}"),
    }
    backend = ScriptedRuntimeBackend(scripts_by_prompt)
    backend.runtime_model = NonDeterministicRuntimeModel()
    evaluator = MathEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda benchmark, resolved_tokenizer: backend,
    )

    result = evaluator.evaluate(_build_math_benchmark(), examples=(example,))

    assert result.aggregate_metrics["accuracy"] == 1.0
    example_summary = result.example_summaries[0]
    assert example_summary.metrics["correct"] is True
    assert example_summary.metadata["candidate_answer"] == "6"
    assert example_summary.metadata["generated_text"] == "reasoning \\boxed{6}"


def test_math_evaluator_uses_injected_tokenizer_and_backend_without_repo_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    example = _build_math_example(
        example_id="math:algebra:1:0",
        prompt="injected collaborators prompt",
        gold_answer="5",
        subject="algebra",
        level=1,
    )
    scripts_by_prompt = {
        tokenizer.token_ids(example.prompt): tokenizer.token_ids("reasoning \\boxed{5}"),
    }
    evaluator = MathEvaluator(
        tokenizer=tokenizer,
        backend_factory=lambda benchmark, resolved_tokenizer: ScriptedRuntimeBackend(scripts_by_prompt),
    )

    def fail_loader() -> object:
        raise AssertionError("repo bridge loader should not be touched")

    monkeypatch.setattr(math_evaluator_module, "_load_repo_bridge_classes", fail_loader)

    result = evaluator.evaluate(_build_math_benchmark(), examples=(example,))

    assert result.aggregate_metrics["accuracy"] == 1.0


def test_math_evaluator_raises_clear_error_when_default_backend_bridge_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    evaluator = MathEvaluator(tokenizer=tokenizer)

    def fail_loader() -> object:
        raise ImportError("src.models bridge unavailable")

    monkeypatch.setattr(math_evaluator_module, "_load_repo_bridge_classes", fail_loader)

    with pytest.raises(RuntimeError, match="real-model MATH evaluator requires"):
        evaluator.evaluate(_build_math_benchmark(), examples=())
