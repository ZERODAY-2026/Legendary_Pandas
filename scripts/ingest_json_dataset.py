"""
JSON Dataset Ingestion and Vector Embedding Script.

Embeds documents or pre-chunked records stored in JSON/JSONL format
using the local BGE-small-en-v1.5 model and indexes them into Qdrant.

Usage:
    python scripts/ingest_json_dataset.py path/to/dataset.json --document-id my_doc_01
    python scripts/ingest_json_dataset.py path/to/dataset.jsonl --document-id legal_corpus --source-type legislation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.document.chunker import chunk_document
from backend.app.document.schemas import Chunk, DocumentMetadata, ExtractedDocument, SourceType
from backend.app.retrieval.embeddings import embed_passages
from backend.app.retrieval.vector_store import index_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("json_ingest")


def load_json_data(file_path: str | Path) -> list[dict[str, Any]]:
    """Load records from a standard JSON file (list or dict) or JSONL file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    records: list[dict[str, Any]] = []

    if file_path.suffix.lower() == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning("Skipping invalid JSON line %d: %s", idx, e)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # If wrapped under a key like 'items', 'documents', 'data'
                for key in ("items", "documents", "data", "records", "chunks"):
                    if key in data and isinstance(data[key], list):
                        records = data[key]
                        break
                if not records:
                    # Single document dictionary
                    records = [data]
            else:
                raise ValueError("JSON content must be a list of records or a dictionary.")

    return records


def records_to_chunks(
    records: list[dict[str, Any]],
    document_id: str,
    source_type: SourceType,
    chunk_raw_text: bool = False,
) -> list[Chunk]:
    """
    Convert JSON records into Chunk objects.

    Supports two modes:
      1. Pre-chunked items: each record represents a chunk/clause (has 'text' or 'content').
      2. Full documents: if chunk_raw_text is True or records have long text, run the clause chunker.
    """
    chunks: list[Chunk] = []

    for idx, rec in enumerate(records):
        doc_id = str(rec.get("document_id") or rec.get("doc_id") or document_id)

        # Support CUAD / SQuAD schema: {"title": "...", "paragraphs": [{"context": "...", "qas": [...]}]}
        if "paragraphs" in rec and isinstance(rec["paragraphs"], list) and len(rec["paragraphs"]) > 0:
            para = rec["paragraphs"][0]
            text = para.get("context", "")
            title = rec.get("title")
            if title:
                doc_id = title
        else:
            text = rec.get("text") or rec.get("content") or rec.get("body") or rec.get("clause_text") or ""

        text = str(text).strip()
        if not text:
            continue

        source_id = str(rec.get("source_id") or f"{doc_id}_chunk_{idx}")
        clause_id = rec.get("clause_id") or rec.get("title") or rec.get("section")
        parent_clause_id = rec.get("parent_clause_id")
        page_start = int(rec.get("page_start") or rec.get("page") or rec.get("page_num") or 1)
        page_end = int(rec.get("page_end") or page_start)
        span_start = int(rec.get("span_start") or 0)
        span_end = int(rec.get("span_end") or len(text))

        # Check if record is a full unchunked document that should be split
        if chunk_raw_text or len(text) > settings.CHUNK_SIZE_CHARS * 2:
            extracted_doc = ExtractedDocument(
                metadata=DocumentMetadata(
                    document_id=doc_id,
                    filename=f"{doc_id[:60]}.json",
                    file_type="json",
                    file_size_bytes=len(text.encode("utf-8")),
                    page_count=page_end,
                    char_count=len(text),
                    source_type=source_type,
                ),
                raw_text=text,
                page_texts={1: text},
                page_spans={1: (0, len(text))},
            )
            doc_chunks = chunk_document(extracted_doc)
            chunks.extend(doc_chunks)
        else:
            chunk = Chunk(
                source_id=source_id,
                document_id=doc_id,
                source_type=source_type,
                page_start=page_start,
                page_end=page_end,
                clause_id=clause_id,
                parent_clause_id=parent_clause_id,
                text=text,
                span_start=span_start,
                span_end=span_end,
                chunk_index=len(chunks),
                index_version="v1",
            )
            chunks.append(chunk)

    return chunks


def ingest_json_dataset(
    file_path: str | Path,
    document_id: str = "dataset_default",
    source_type_str: str = "contract",
    chunk_raw_text: bool = False,
    batch_size: int = 32,
    limit: int | None = None,
) -> int:
    """Read a JSON dataset, generate vector embeddings, and index into Qdrant."""
    path = Path(file_path)
    logger.info("Reading JSON dataset from: %s", path.resolve())
    records = load_json_data(path)
    logger.info("Loaded %d raw records from JSON", len(records))

    if limit is not None and limit > 0:
        records = records[:limit]
        logger.info("Limiting to first %d records", len(records))

    source_type = SourceType(source_type_str.lower())
    chunks = records_to_chunks(
        records=records,
        document_id=document_id,
        source_type=source_type,
        chunk_raw_text=chunk_raw_text,
    )
    logger.info("Prepared %d chunks for embedding", len(chunks))

    if not chunks:
        logger.warning("No valid text chunks found to embed.")
        return 0

    texts = [c.text for c in chunks]
    logger.info(
        "Generating embeddings for %d passages using model '%s' (dimension=%d)...",
        len(texts),
        settings.EMBEDDING_MODEL_NAME,
        settings.EMBEDDING_DIMENSION,
    )
    vectors = embed_passages(texts, batch_size=batch_size)
    logger.info("Generated %d vector embeddings successfully", len(vectors))

    logger.info("Indexing chunks and vectors into Qdrant collection '%s'...", settings.QDRANT_COLLECTION_NAME)
    num_indexed = index_document(chunks=chunks, vectors=vectors)
    logger.info("Successfully indexed %d chunks in Qdrant vector store", num_indexed)

    return num_indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create vector embeddings for a JSON/JSONL dataset and index into Qdrant."
    )
    parser.add_argument("file", type=str, help="Path to the JSON or JSONL dataset file.")
    parser.add_argument(
        "--document-id",
        type=str,
        default="json_dataset_001",
        help="Logical document ID to group these records under in the vector store.",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="contract",
        choices=["contract", "legislation", "court_judgment"],
        help="Legal source classification (default: contract).",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="If set, breaks long text fields into overlapping clause chunks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of contracts/records to process (useful for testing).",
    )

    args = parser.parse_args()
    ingest_json_dataset(
        file_path=args.file,
        document_id=args.document_id,
        source_type_str=args.source_type,
        chunk_raw_text=args.chunk,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
