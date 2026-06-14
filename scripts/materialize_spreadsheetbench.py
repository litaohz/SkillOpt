#!/usr/bin/env python3
"""Materialize the SpreadsheetBench split_dir from the released ID manifest.

Reads the ID-only manifests under ``data/spreadsheetbench_id_split/{train,val,test}/items.json``,
joins each ``id`` against ``data/spreadsheetbench_verified_400/dataset.json`` (the
verified-400 release shipped with full task metadata: instruction,
spreadsheet_path, instruction_type, answer_position, answer_sheet,
data_position), and writes runnable items into
``data/spreadsheetbench_split/{train,val,test}/<split>.json``.

The spreadsheet files themselves stay under
``data/spreadsheetbench_verified_400/spreadsheet/<id>/`` (the configured
``data_root``); ``spreadsheet_path`` in each item is the relative path the
rollout resolves against that root.

Usage:
    python scripts/materialize_spreadsheetbench.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ID_SPLIT = PROJECT_ROOT / "data" / "spreadsheetbench_id_split"
DATA_ROOT = PROJECT_ROOT / "data" / "spreadsheetbench_verified_400"
OUT_SPLIT = PROJECT_ROOT / "data" / "spreadsheetbench_split"


def _load_ids(split: str) -> list[str]:
    with open(ID_SPLIT / split / "items.json", encoding="utf-8") as f:
        return [str(it["id"]) for it in json.load(f)]


def main() -> None:
    with open(DATA_ROOT / "dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)
    by_id: dict[str, dict] = {str(item["id"]): item for item in dataset}
    print(f"Loaded {len(by_id)} task records from {DATA_ROOT / 'dataset.json'}")

    for split in ["train", "val", "test"]:
        ids = _load_ids(split)
        items = []
        missing: list[str] = []
        for tid in ids:
            rec = by_id.get(tid)
            if rec is None:
                missing.append(tid)
                continue
            items.append(rec)
        out_dir = OUT_SPLIT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(
            f"  {split}: wrote {len(items):>4} / {len(ids):>4} items -> "
            f"{out_path.relative_to(PROJECT_ROOT)}"
            + (f"  missing={missing[:5]}..." if missing else "")
        )


if __name__ == "__main__":
    main()
