#!/usr/bin/env python3
"""Materialize the SearchQA split_dir from the released ID manifest.

Reads the ID-only manifests under ``data/searchqa_id_split/{train,val,test}/items.json``,
matches each ``id`` against the ``key`` field of the HuggingFace dataset
``lucadiliello/searchqa`` (streaming over all HF splits), and writes runnable
items into ``data/searchqa_split/{train,val,test}/<split>.json`` with the fields
consumed by the SearchQA environment: ``id``, ``question``, ``context``,
``answers``.

Usage:
    python scripts/materialize_searchqa.py
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ID_SPLIT = PROJECT_ROOT / "data" / "searchqa_id_split"
OUT_SPLIT = PROJECT_ROOT / "data" / "searchqa_split"
HF_DATASET = "lucadiliello/searchqa"
HF_SPLITS = ["train", "validation", "test"]


def _load_ids(split: str) -> list[str]:
    with open(ID_SPLIT / split / "items.json") as f:
        return [str(it["id"]) for it in json.load(f)]


def _coerce_answers(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except (ValueError, SyntaxError):
            pass
        return [raw]
    return []


def main() -> None:
    wanted: dict[str, str] = {}  # key -> split name
    ids_by_split: dict[str, list[str]] = {}
    for split in ["train", "val", "test"]:
        ids = _load_ids(split)
        ids_by_split[split] = ids
        for k in ids:
            wanted[k] = split
    print(f"Need {len(wanted)} unique ids across train/val/test")

    found: dict[str, dict] = {}
    remaining = set(wanted)
    for hf_split in HF_SPLITS:
        if not remaining:
            break
        print(f"Streaming HF split '{hf_split}' (remaining={len(remaining)}) ...")
        ds = load_dataset(HF_DATASET, split=hf_split, streaming=True)
        for row in ds:
            key = str(row["key"])
            if key in remaining:
                found[key] = {
                    "id": key,
                    "question": row.get("question", ""),
                    "context": row.get("context", ""),
                    "answers": _coerce_answers(row.get("answers", [])),
                }
                remaining.discard(key)
                if not remaining:
                    break
    print(f"Matched {len(found)}/{len(wanted)} ids (missing={len(remaining)})")

    for split in ["train", "val", "test"]:
        out_dir = OUT_SPLIT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        items = [found[k] for k in ids_by_split[split] if k in found]
        out_path = out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        print(f"  wrote {len(items):>5} items -> {out_path.relative_to(PROJECT_ROOT)}")

    if remaining:
        print(f"WARNING: {len(remaining)} ids unmatched: {list(remaining)[:5]} ...")


if __name__ == "__main__":
    main()
