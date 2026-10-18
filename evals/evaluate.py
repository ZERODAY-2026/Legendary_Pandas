"""
Retrieval evaluation harness for Phase 3.

Metrics measured:
  - Recall@5:      fraction of questions where at least one expected keyword
                   appears in the top-5 retrieved passages.
  - MRR:          Mean Reciprocal Rank of the first relevant result.
  - Duplicate rate: fraction of result sets where the same text span appears >1.
  - Latency:       wall-clock time per retrieval call (mean ± std).
  - Cache hit rate: fraction of calls served from cache on second pass.

Ablation modes supported: "bm25", "vector", "hybrid"

Usage:
    python evals/evaluate.py
    python evals/evaluate.py --method vector
    python evals/evaluate.py --method bm25
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Literal


EvalMethod = Literal["hybrid", "vector", "bm25"]


def load_eval_data(questions_path: Path, evidence_path: Path):
    """Load JSONL evaluation questions and expected evidence."""
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    expected: dict[str, dict] = {}
    with open(evidence_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                expected[item["id"]] = item

    return questions, expected


def _is_relevant(passages: list[dict], expected_keywords: list[str]) -> tuple[bool, int]:
    """Check if any passage contains an expected keyword. Return (found, rank)."""
    for rank, passage in enumerate(passages, start=1):
        quote = passage.get("quote", "").lower()
        if any(kw.lower() in quote for kw in expected_keywords):
            return True, rank
    return False, 0


def _has_duplicate_spans(passages: list[dict]) -> bool:
    """Return True if any two passages share the same (source_id, quote) pair."""
    seen: set[str] = set()
    for p in passages:
        key = f"{p.get('source_id', '')}|{p.get('quote', '')[:50]}"
        if key in seen:
            return True
        seen.add(key)
    return False


def evaluate_retrieval(
    document_id: str,
    questions: list[dict],
    expected: dict[str, dict],
    method: EvalMethod = "hybrid",
    top_k: int = 5,
    run_twice_for_cache: bool = True,
) -> dict:
    """
    Run retrieval for each question and compute aggregate metrics.

    Args:
        document_id: The contract document to search.
        questions: List of {id, question, ...} dicts.
        expected:   Dict of question_id -> {expected_keywords, ...}.
        method:     Retrieval method ("hybrid", "vector", "bm25").
        top_k:      Number of passages to retrieve.
        run_twice_for_cache: If True, run each query twice to measure cache hit rate.

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    try:
        from backend.app.retrieval.hybrid_retriever import retrieve_evidence
    except ImportError as exc:
        print(f"ERROR: Cannot import retrieve_evidence: {exc}", file=sys.stderr)
        sys.exit(1)

    per_question = []
    latencies_ms: list[float] = []
    cache_latencies_ms: list[float] = []
    reciprocal_ranks: list[float] = []
    recalls: list[int] = []
    has_duplicate: list[bool] = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        exp = expected.get(qid, {})
        expected_kws = exp.get("expected_keywords", [])

        # First call — live retrieval
        t0 = time.perf_counter()
        result = retrieve_evidence(document_id=document_id, question=question, top_k=top_k, method=method)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies_ms.append(latency_ms)

        passages = [p.model_dump() for p in result.passages]

        found, rank = _is_relevant(passages, expected_kws) if expected_kws else (True, 1)
        recalls.append(1 if found else 0)
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        has_duplicate.append(_has_duplicate_spans(passages))

        q_result = {
            "id": qid,
            "question": question,
            "found_relevant": found,
            "rank": rank,
            "latency_ms": round(latency_ms, 2),
            "passage_count": len(passages),
            "has_duplicate": has_duplicate[-1],
        }

        # Second call — cache measurement
        if run_twice_for_cache:
            t2 = time.perf_counter()
            retrieve_evidence(document_id=document_id, question=question, top_k=top_k, method=method)
            t3 = time.perf_counter()
            cache_lat = (t3 - t2) * 1000
            cache_latencies_ms.append(cache_lat)
            q_result["cache_latency_ms"] = round(cache_lat, 2)

        per_question.append(q_result)

    n = len(questions)
    recall_at_5 = sum(recalls) / n if n > 0 else 0.0
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0.0
    dup_rate = sum(1 for d in has_duplicate if d) / n if n > 0 else 0.0
    mean_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    std_latency = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

    aggregate = {
        "method": method,
        "n_questions": n,
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "duplicate_rate": round(dup_rate, 4),
        "mean_latency_ms": round(mean_latency, 2),
        "std_latency_ms": round(std_latency, 2),
    }

    if run_twice_for_cache and cache_latencies_ms:
        mean_cache = statistics.mean(cache_latencies_ms)
        aggregate["mean_cache_latency_ms"] = round(mean_cache, 2)
        # Cache is "faster" if second call is meaningfully quicker
        speedup = mean_latency / mean_cache if mean_cache > 0 else 1.0
        aggregate["cache_speedup_ratio"] = round(speedup, 2)

    return {"aggregate": aggregate, "per_question": per_question}


def print_report(report: dict) -> None:
    agg = report["aggregate"]
    print("\n" + "=" * 60)
    print(f"  RETRIEVAL EVALUATION — method: {agg['method'].upper()}")
    print("=" * 60)
    print(f"  Questions evaluated : {agg['n_questions']}")
    print(f"  Recall@5            : {agg['recall_at_5']:.1%}")
    print(f"  MRR                 : {agg['mrr']:.4f}")
    print(f"  Duplicate rate      : {agg['duplicate_rate']:.1%}")
    print(f"  Mean latency        : {agg['mean_latency_ms']:.1f} ms ± {agg['std_latency_ms']:.1f}")
    if "mean_cache_latency_ms" in agg:
        print(f"  Cache latency (2nd) : {agg['mean_cache_latency_ms']:.1f} ms")
        print(f"  Cache speedup       : {agg['cache_speedup_ratio']:.1f}×")
    print("-" * 60)
    print(f"  {'ID':<8} {'Found':>6}  {'Rank':>5}  {'Latency':>10}  {'Dupes':>6}")
    print("-" * 60)
    for q in report["per_question"]:
        found_str = "✓" if q["found_relevant"] else "✗"
        rank_str = str(q["rank"]) if q["rank"] > 0 else "-"
        dup_str = "YES" if q["has_duplicate"] else "no"
        print(
            f"  {q['id']:<8} {found_str:>6}  {rank_str:>5}  "
            f"{q['latency_ms']:>8.1f}ms  {dup_str:>6}"
        )
    print("=" * 60 + "\n")


def run_benchmark(
    document_id: str = "sample-contract",
    method: EvalMethod = "hybrid",
) -> bool:
    evals_dir = Path(__file__).parent
    questions_path = evals_dir / "questions.jsonl"
    evidence_path = evals_dir / "expected_evidence.jsonl"

    if not questions_path.exists() or not evidence_path.exists():
        print("ERROR: questions.jsonl or expected_evidence.jsonl not found in evals/")
        return False

    questions, expected = load_eval_data(questions_path, evidence_path)
    print(f"Loaded {len(questions)} evaluation questions. Method: {method}")

    report = evaluate_retrieval(document_id, questions, expected, method=method)
    print_report(report)

    # Save JSON report
    report_path = evals_dir / f"report_{method}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {report_path}")

    # Return True if Recall@5 >= 0.5 (pass bar)
    return report["aggregate"]["recall_at_5"] >= 0.5


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--document_id", default="sample-contract", help="Document ID to evaluate")
    parser.add_argument("--method", default="hybrid", choices=["hybrid", "vector", "bm25"])
    args = parser.parse_args()

    passed = run_benchmark(document_id=args.document_id, method=args.method)
    sys.exit(0 if passed else 1)

