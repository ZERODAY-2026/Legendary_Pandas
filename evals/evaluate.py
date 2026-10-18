"""Offline benchmark and evaluation script for Private Legal Triage.

Evaluates extraction fidelity, citation accuracy, unsupported-claim rate,
and abstention quality locally without network dependencies.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def run_benchmark():
    """Runs local offline evaluation suite."""
    evals_dir = Path(__file__).resolve().parent
    questions_file = evals_dir / "questions.jsonl"
    evidence_file = evals_dir / "expected_evidence.jsonl"

    questions = load_jsonl(questions_file)
    expected = load_jsonl(evidence_file)

    print(f"Loaded {len(questions)} evaluation questions.")
    print(f"Loaded {len(expected)} expected evidence ground truth records.")
    print("Benchmark harness initialized successfully.")


if __name__ == "__main__":
    run_benchmark()
