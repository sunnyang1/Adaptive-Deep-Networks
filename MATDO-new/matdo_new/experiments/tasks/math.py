from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol


@dataclass(frozen=True)
class MathExample:
    example_id: str
    problem: str
    prompt: str
    gold_solution: str
    gold_answer: str
    subject: str
    level: int
    metadata: dict[str, object] = field(default_factory=dict)
    max_new_tokens: int = 256


class DatasetLoader(Protocol):
    def __call__(self, split: str) -> Iterable[dict[str, object]]: ...


def build_prompt(problem: str, *, prompt_style: str) -> str:
    if prompt_style != "cot_boxed":
        raise ValueError(f"unsupported math prompt style: {prompt_style}")

    return (
        "Solve the following math problem carefully.\n\n"
        f"{problem}\n\n"
        "Show your reasoning, and end with your final answer in \\boxed{...}."
    )


class MathDatasetAdapter:
    _FINAL_ANSWER_PATTERNS = (
        re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE),
        re.compile(r"answer is\s+(.+)", re.IGNORECASE),
    )

    def __init__(self, dataset_loader: DatasetLoader | None = None) -> None:
        self._dataset_loader = dataset_loader or self._load_dataset_split

    def build_examples(self, benchmark: object) -> tuple[MathExample, ...]:
        rows = self._dataset_loader(str(getattr(benchmark, "split")))
        allowed_subjects = {
            self._normalize_subject(subject) for subject in getattr(benchmark, "subjects", ())
        }
        allowed_levels = {int(level) for level in getattr(benchmark, "levels", ())}
        max_samples = getattr(benchmark, "max_samples", None)
        eligible_rows: list[tuple[int, dict[str, object]]] = []
        for row_index, row in enumerate(rows):
            subject = self._normalize_subject(row["subject"])
            level = self._parse_level(row["level"])

            if allowed_subjects and subject not in allowed_subjects:
                continue
            if allowed_levels and level not in allowed_levels:
                continue

            eligible_rows.append((row_index, row))

        if max_samples is not None and len(eligible_rows) > int(max_samples):
            rng = random.Random(int(getattr(benchmark, "seed")))
            eligible_rows = rng.sample(eligible_rows, k=int(max_samples))

        examples: list[MathExample] = []
        for row_index, row in eligible_rows:
            subject = self._normalize_subject(row["subject"])
            level = self._parse_level(row["level"])
            problem = str(row["problem"]).strip()
            gold_solution = str(row["solution"]).strip()
            examples.append(
                MathExample(
                    example_id=f"math:{subject}:{level}:{row_index}",
                    problem=problem,
                    prompt=build_prompt(problem, prompt_style=str(getattr(benchmark, "prompt_style"))),
                    gold_solution=gold_solution,
                    gold_answer=self._extract_gold_answer(gold_solution),
                    subject=subject,
                    level=level,
                    metadata={"subject": subject, "level": level},
                    max_new_tokens=int(getattr(benchmark, "max_new_tokens")),
                )
            )

        return tuple(examples)

    @staticmethod
    def _load_dataset_split(split: str) -> Iterable[dict[str, object]]:
        from datasets import load_dataset

        return load_dataset("hendrycks/competition_math", split=split)

    @staticmethod
    def _normalize_subject(subject: object) -> str:
        return str(subject).strip().lower()

    @staticmethod
    def _parse_level(level: object) -> int:
        match = re.search(r"(\d+)", str(level))
        if match is None:
            raise ValueError(f"could not parse math level from {level!r}")
        return int(match.group(1))

    @staticmethod
    def _extract_gold_answer(gold_solution: str) -> str:
        answer, _mode = MathDatasetAdapter.extract_candidate_answer(gold_solution)
        return answer

    @staticmethod
    def _extract_last_boxed_payload(text: str) -> str | None:
        marker = r"\boxed{"
        search_start = 0
        last_payload: str | None = None

        while True:
            marker_index = text.find(marker, search_start)
            if marker_index == -1:
                return last_payload

            payload_start = marker_index + len(marker)
            depth = 1
            cursor = payload_start
            while cursor < len(text) and depth > 0:
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                cursor += 1

            if depth == 0:
                last_payload = text[payload_start : cursor - 1]

            search_start = payload_start

    @staticmethod
    def extract_candidate_answer(text: str) -> tuple[str, str]:
        boxed_answer = MathDatasetAdapter._extract_last_boxed_payload(text)
        if boxed_answer is not None:
            return boxed_answer.strip(), "boxed"

        phrase_answer = MathDatasetAdapter._extract_last_final_phrase_answer(text)
        if phrase_answer is not None:
            return phrase_answer, "final_phrase"

        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not non_empty_lines:
            return "", "last_line"
        return non_empty_lines[-1], "last_line"

    @staticmethod
    def normalize_answer(answer: str) -> str:
        normalized = answer.strip()

        while len(normalized) >= 2 and normalized.startswith("$") and normalized.endswith("$"):
            normalized = normalized[1:-1].strip()

        boxed_payload = MathDatasetAdapter._extract_simple_boxed_wrapper(normalized)
        if boxed_payload is not None:
            normalized = boxed_payload.strip()

        while len(normalized) >= 2 and normalized.startswith("$") and normalized.endswith("$"):
            normalized = normalized[1:-1].strip()

        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.lower()

    @staticmethod
    def answers_match(candidate: str, gold: str, *, tolerance: float = 1e-9) -> bool:
        normalized_candidate = MathDatasetAdapter.normalize_answer(candidate)
        normalized_gold = MathDatasetAdapter.normalize_answer(gold)

        if normalized_candidate == normalized_gold:
            return True

        candidate_value = MathDatasetAdapter._parse_numeric_value(normalized_candidate)
        gold_value = MathDatasetAdapter._parse_numeric_value(normalized_gold)
        if candidate_value is None or gold_value is None:
            return False

        return abs(candidate_value - gold_value) <= tolerance

    @staticmethod
    def _extract_last_final_phrase_answer(text: str) -> str | None:
        last_match: re.Match[str] | None = None
        for pattern in MathDatasetAdapter._FINAL_ANSWER_PATTERNS:
            for match in pattern.finditer(text):
                if last_match is None or match.start() >= last_match.start():
                    last_match = match

        if last_match is None:
            return None
        return last_match.group(1).strip()

    @staticmethod
    def _extract_simple_boxed_wrapper(text: str) -> str | None:
        stripped = text.strip()
        if not stripped.startswith(r"\boxed{"):
            return None

        payload = MathDatasetAdapter._extract_last_boxed_payload(stripped)
        if payload is None:
            return None

        expected = rf"\boxed{{{payload}}}"
        if stripped != expected:
            return None
        return payload

    @staticmethod
    def _parse_float(text: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_numeric_value(text: str) -> float | None:
        latex_fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", text)
        if latex_fraction is not None:
            numerator = MathDatasetAdapter._parse_numeric_value(latex_fraction.group(1).strip())
            denominator = MathDatasetAdapter._parse_numeric_value(latex_fraction.group(2).strip())
            if numerator is None or denominator is None or denominator == 0:
                return None
            return numerator / denominator

        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return MathDatasetAdapter._parse_float(text)
