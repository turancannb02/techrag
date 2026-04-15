from __future__ import annotations

import argparse
import json
from pathlib import Path

from chain import TechRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal evaluation runner.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset with {'question','answer'} rows.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rag = TechRAG()

    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError("Dataset is empty")

    total = len(rows)
    with_sources = 0

    for row in rows:
        result = rag.ask(row["question"], top_k=args.top_k)
        if result.sources:
            with_sources += 1

    print(f"Samples: {total}")
    print(f"Answers with citations: {with_sources}/{total}")
    print("Note: this is a bootstrap evaluator. Add RAGAS or task-specific metrics next.")


if __name__ == "__main__":
    main()
