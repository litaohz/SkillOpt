#!/usr/bin/env python3
"""Materialize the DocVQA split CSVs + page images from the official HF release.

DocVQA's released on-disk manifests under
``data/docvqa_id_split/{train,val,test}/items.json`` carry only IDs
(``questionId``, ``docId``, ``image_path``, ``topic`` …). The question text,
gold answers, and the page images live in the source dataset
``lmms-lab/DocVQA`` (config ``DocVQA``, split ``validation``).

This script:

1. Reads the three id-split manifests and collects the wanted ``questionId`` set
   (``docvqa_validation_10pct``: 107 / 53 / 374).
2. Streams the official ``validation`` split, and for each wanted row:
   - saves the page image to ``data/docvqa_images/q<qid>_d<docId>.png`` (the path
     the id-split records), and
   - keeps its question text + answers.
3. Writes one CSV per split into ``data/docvqa/splits/{train,val,test}/<split>.csv``
   with the columns the DocVQA dataloader consumes
   (``questionId, question, answer, image_path, topic, docId,
   ucsf_document_id, ucsf_document_page_no, source_split``).

Auth (token resolution order): ``--token`` → env
``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` → ``~/.env``
line ``hf_token=...``. ``lmms-lab/DocVQA`` is public, but a token avoids rate
limits.

Usage:
    python scripts/materialize_docvqa.py
    python scripts/materialize_docvqa.py --skip-images   # only rebuild CSVs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ID_SPLIT = PROJECT_ROOT / "data" / "docvqa_id_split"
OUT_SPLIT = PROJECT_ROOT / "data" / "docvqa" / "splits"
IMAGES_ROOT = PROJECT_ROOT / "data" / "docvqa_images"

HF_REPO = "lmms-lab/DocVQA"
HF_CONFIG = "DocVQA"
HF_SPLIT = "validation"

SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "questionId",
    "question",
    "answer",
    "image_path",
    "topic",
    "docId",
    "ucsf_document_id",
    "ucsf_document_page_no",
    "source_split",
)


# ── token resolution ───────────────────────────────────────────────────────
def _read_env_file_token(env_path: Path) -> str:
    if not env_path.is_file():
        return ""
    keys = ("hf_token", "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    pat = re.compile(r"^\s*(?:export\s+)?(" + "|".join(keys) + r")\s*=\s*(.+?)\s*$")
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(line)
        if m:
            return m.group(2).strip().strip('"').strip("'")
    return ""


def resolve_token(cli_token: str | None) -> str:
    if cli_token:
        return cli_token.strip()
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return _read_env_file_token(Path.home() / ".env")


# ── id-split manifests ─────────────────────────────────────────────────────
def _load_id_items(split: str) -> list[dict]:
    with open(ID_SPLIT / split / "items.json", encoding="utf-8") as f:
        return json.load(f)


def _image_dest(qid: str, doc_id: str, recorded: str) -> Path:
    """Prefer the path the id-split recorded; fall back to the canonical name."""
    rel = str(recorded or "").strip()
    if rel:
        path = Path(rel)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path
    return IMAGES_ROOT / f"q{qid}_d{doc_id}.png"


def build_index() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Return (qid -> id_split item) and (split -> ordered list of qids)."""
    by_qid: dict[str, dict] = {}
    split_qids: dict[str, list[str]] = {}
    for split in SPLITS:
        qids: list[str] = []
        for it in _load_id_items(split):
            qid = str(it.get("questionId") or it.get("id") or "").strip()
            if not qid:
                continue
            by_qid[qid] = it
            qids.append(qid)
        split_qids[split] = qids
        print(f"  [id-split] {split}: {len(qids)} questionIds")
    return by_qid, split_qids


# ── source streaming ───────────────────────────────────────────────────────
def stream_rows(token: str, wanted: set[str], *, save_images: bool) -> dict[str, dict]:
    """Stream the validation split; collect rows whose questionId is wanted."""
    from datasets import load_dataset

    ds = load_dataset(
        HF_REPO, HF_CONFIG, split=HF_SPLIT, streaming=True, token=token or None
    )
    if save_images:
        IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    found: dict[str, dict] = {}
    seen = 0
    for row in ds:
        seen += 1
        qid = str(row.get("questionId") or "").strip()
        if qid not in wanted or qid in found:
            if seen % 1000 == 0:
                print(f"    [stream] scanned {seen}, matched {len(found)}/{len(wanted)}")
            continue
        found[qid] = row
        if seen % 1000 == 0 or len(found) == len(wanted):
            print(f"    [stream] scanned {seen}, matched {len(found)}/{len(wanted)}")
        if len(found) == len(wanted):
            break
    print(f"  [stream] done: scanned {seen}, matched {len(found)}/{len(wanted)}")
    return found


def _answers_repr(value: object) -> str:
    """Normalize the answers field to a Python-list repr string the loader parses."""
    if isinstance(value, list):
        return repr([str(x) for x in value])
    text = str(value or "").strip()
    return text or "[]"


# ── CSV writing ────────────────────────────────────────────────────────────
def write_splits(
    by_qid: dict[str, dict],
    split_qids: dict[str, list[str]],
    rows: dict[str, dict],
    *,
    save_images: bool,
) -> None:
    total_written = 0
    total_missing = 0
    total_images = 0
    for split in SPLITS:
        out_dir = OUT_SPLIT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.csv"
        written = 0
        missing: list[str] = []
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for qid in split_qids[split]:
                src = rows.get(qid)
                meta = by_qid.get(qid, {})
                if src is None:
                    missing.append(qid)
                    continue
                doc_id = str(src.get("docId") or meta.get("docId") or "").strip()
                dest = _image_dest(qid, doc_id, meta.get("image_path", ""))
                if save_images:
                    image = src.get("image")
                    if image is not None:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not (dest.is_file() and dest.stat().st_size > 0):
                            image.convert("RGB").save(dest, format="PNG")
                        total_images += 1
                rel_image = os.path.relpath(dest, PROJECT_ROOT).replace(os.sep, "/")
                writer.writerow({
                    "questionId": qid,
                    "question": str(src.get("question") or "").strip(),
                    "answer": _answers_repr(src.get("answers")),
                    "image_path": rel_image,
                    "topic": str(meta.get("topic") or "").strip()
                    or _answers_repr(src.get("question_types")),
                    "docId": doc_id,
                    "ucsf_document_id": str(src.get("ucsf_document_id") or "").strip(),
                    "ucsf_document_page_no": str(src.get("ucsf_document_page_no") or "").strip(),
                    "source_split": str(src.get("data_split") or meta.get("source_split") or "").strip(),
                })
                written += 1
        total_written += written
        total_missing += len(missing)
        msg = f"  {split}: wrote {written}/{len(split_qids[split])} rows -> {out_path.relative_to(PROJECT_ROOT)}"
        if missing:
            msg += f"  (unmatched qids: {missing[:5]}{'...' if len(missing) > 5 else ''})"
        print(msg)
    print(f"  [csv] total rows={total_written}, images={total_images}, unmatched={total_missing}")
    if total_missing:
        print(f"  [csv] WARNING: {total_missing} questionId(s) not found in the source split.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize DocVQA split CSVs + page images.")
    ap.add_argument("--token", type=str, default=None, help="HF token (else env / ~/.env)")
    ap.add_argument("--skip-images", action="store_true", help="only rebuild CSVs; skip image save")
    args = ap.parse_args()

    token = resolve_token(args.token)
    if token:
        print(f"  [auth] using token …{token[-4:]} (len={len(token)})")
    else:
        print("  [auth] no token found; lmms-lab/DocVQA is public, continuing anonymously")

    by_qid, split_qids = build_index()
    wanted = set(by_qid.keys())
    print(f"  [id-split] {len(wanted)} unique questionIds across {sum(len(v) for v in split_qids.values())} items")

    rows = stream_rows(token, wanted, save_images=not args.skip_images)
    if not rows:
        sys.exit("ERROR: no rows matched; check token/access and source split.")

    write_splits(by_qid, split_qids, rows, save_images=not args.skip_images)

    print("\nDone. Next: run the eval, e.g.")
    print("  python scripts/eval_only.py --config configs/docvqa/default.yaml \\")
    print("    --skill outputs/empty_skill.md --split valid_unseen \\")
    print("    --split_dir data/docvqa/splits --target_model gpt-5.5 ...")


if __name__ == "__main__":
    main()
