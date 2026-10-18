"""
Unit tests for JSON dataset ingestion and vector embedding pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from backend.app.document.schemas import SourceType
from scripts.ingest_json_dataset import load_json_data, records_to_chunks


def test_load_json_array(tmp_path: Path):
    data = [
        {"clause_id": "1. TERM", "text": "Lease duration is 12 months."},
        {"clause_id": "2. RENT", "text": "Monthly rent is $2,000."},
    ]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    records = load_json_data(p)
    assert len(records) == 2
    assert records[0]["clause_id"] == "1. TERM"


def test_load_json_wrapped_dict(tmp_path: Path):
    data = {
        "documents": [
            {"id": "doc_a", "content": "Sample agreement paragraph."},
        ]
    }
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    records = load_json_data(p)
    assert len(records) == 1
    assert records[0]["id"] == "doc_a"


def test_load_jsonl(tmp_path: Path):
    lines = [
        json.dumps({"id": "q1", "text": "First passage"}),
        json.dumps({"id": "q2", "text": "Second passage"}),
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")

    records = load_json_data(p)
    assert len(records) == 2


def test_records_to_chunks_conversion():
    records = [
        {
            "source_id": "s_1",
            "document_id": "doc_100",
            "clause_id": "TERM",
            "text": "The term of this lease shall be one year starting Jan 1.",
            "page_start": 1,
            "page_end": 1,
        }
    ]
    chunks = records_to_chunks(
        records=records,
        document_id="doc_default",
        source_type=SourceType.CONTRACT,
        chunk_raw_text=False,
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.source_id == "s_1"
    assert c.document_id == "doc_100"
    assert c.clause_id == "TERM"
    assert c.text == records[0]["text"]
