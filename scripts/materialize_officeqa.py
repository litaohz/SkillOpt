#!/usr/bin/env python3
"""Materialize the OfficeQA split_dir + parsed-doc corpus from the gated HF release.

OfficeQA's payload (``databricks/officeqa``) is **gated** on Hugging Face: the
released on-disk manifests under ``data/officeqa_id_split/{train,val,test}/items.json``
carry only IDs (``uid``, ``category``, ``source_files``, ``source_docs``) — the
question text and ground-truth answers live in the gated ``officeqa_full.csv``,
and the parsed Treasury-Bulletin pages live in the gated
``treasury_bulletins_parsed/{transformed,jsons}/`` tree.

This script needs a Hugging Face token (see below). It then:

1. Downloads ``officeqa_full.csv`` and joins each split ID (on ``uid``) to its
   ``question`` + ``ground_truth``, writing runnable items into
   ``data/officeqa_split/{train,val,test}/<split>.json`` — the fields the
   OfficeQA dataloader consumes.
2. Sparse-downloads only the parsed docs the splits actually reference (285
   unique bulletins by default; ~570 files) into the layout the offline
   rollout expects::

       data/officeqa_docs_official/
       ├── transformed/treasury_bulletin_YYYY_MM.txt   # local glob/read/grep corpus
       └── jsons/treasury_bulletin_YYYY_MM.json        # oracle parsed pages

   Pass ``--full`` to mirror the entire parsed corpus (~700 bulletins).

Auth (token resolution order):
    1. ``--token <hf_xxx>``
    2. env ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
    3. ``~/.env`` line ``hf_token=...`` (or ``HF_TOKEN=...``)
  The token must belong to an account that has accepted the dataset terms at
  https://huggingface.co/datasets/databricks/officeqa (gated: auto — visiting
  the page and clicking "Agree and access repository" once is required; a
  token alone is not enough).

Usage:
    # full reproduction corpus (sparse docs):
    python scripts/materialize_officeqa.py

    # only rebuild the split JSON (questions+answers), skip doc download:
    python scripts/materialize_officeqa.py --skip-docs

    # mirror the entire parsed corpus, not just referenced files:
    python scripts/materialize_officeqa.py --full
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ID_SPLIT = PROJECT_ROOT / "data" / "officeqa_id_split"
OUT_SPLIT = PROJECT_ROOT / "data" / "officeqa_split"
DOCS_ROOT = PROJECT_ROOT / "data" / "officeqa_docs_official"

HF_REPO = "databricks/officeqa"
CSV_FILENAME = "officeqa_full.csv"
PARSED_PREFIX = "treasury_bulletins_parsed"
TRANSFORMED_SUBDIR = "transformed"
JSONS_SUBDIR = "jsons"

# Allow the CSV to vary its column naming; pick the first present (case-insensitive).
UID_KEYS = ("uid", "id", "question_id", "qid")
QUESTION_KEYS = ("question", "query", "prompt", "question_text")
ANSWER_KEYS = ("ground_truth", "answer", "gt", "final_answer", "label", "gold")
CATEGORY_KEYS = ("category", "difficulty", "level")
SOURCE_FILES_KEYS = ("source_files", "source_file", "files")
SOURCE_DOCS_KEYS = ("source_docs", "source_doc", "docs", "source_urls")

SPLITS = ("train", "val", "test")


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


# ── split JSON (questions + answers) ───────────────────────────────────────
def _first_key(row: dict, keys: tuple[str, ...]) -> str | None:
    lower = {str(k).lower(): k for k in row.keys()}
    for cand in keys:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _load_id_items(split: str) -> list[dict]:
    with open(ID_SPLIT / split / "items.json", encoding="utf-8") as f:
        return json.load(f)


def _parse_source_list(value) -> list[str]:
    """Mirror the dataloader's tolerant list parsing for source_files/source_docs."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            return [str(x).strip() for x in loaded if str(x).strip()]
    except json.JSONDecodeError:
        pass
    parts = re.split(r"[\r\n]+", text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]
    if "," in text and not text.lower().endswith(".txt"):
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def build_splits(csv_path: Path) -> set[str]:
    """Join split IDs to CSV question+answer; write split JSON. Return referenced .txt stems."""
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        sys.exit(f"ERROR: {csv_path} is empty.")

    uid_k = _first_key(rows[0], UID_KEYS)
    q_k = _first_key(rows[0], QUESTION_KEYS)
    a_k = _first_key(rows[0], ANSWER_KEYS)
    if not (uid_k and q_k and a_k):
        sys.exit(
            "ERROR: could not locate required columns in CSV.\n"
            f"  uid column: {uid_k!r} (tried {UID_KEYS})\n"
            f"  question column: {q_k!r} (tried {QUESTION_KEYS})\n"
            f"  answer column: {a_k!r} (tried {ANSWER_KEYS})\n"
            f"  available columns: {list(rows[0].keys())}"
        )
    sf_k = _first_key(rows[0], SOURCE_FILES_KEYS)
    sd_k = _first_key(rows[0], SOURCE_DOCS_KEYS)
    print(f"  [csv] {len(rows)} rows; join on {uid_k!r}, question={q_k!r}, answer={a_k!r}")

    by_uid: dict[str, dict] = {}
    for r in rows:
        uid = str(r.get(uid_k, "")).strip()
        if uid:
            by_uid[uid] = r

    referenced_files: set[str] = set()
    for split in SPLITS:
        id_items = _load_id_items(split)
        items: list[dict] = []
        missing: list[str] = []
        for it in id_items:
            uid = str(it.get("uid") or it.get("id") or "").strip()
            row = by_uid.get(uid)
            if row is None:
                missing.append(uid)
                continue
            # Keep id-split's curated source hints; fall back to CSV if absent.
            src_files = _parse_source_list(it.get("source_files"))
            if not src_files and sf_k:
                src_files = _parse_source_list(row.get(sf_k))
            src_docs = _parse_source_list(it.get("source_docs"))
            if not src_docs and sd_k:
                src_docs = _parse_source_list(row.get(sd_k))
            category = str(it.get("category") or "").strip()
            if not category:
                ck = _first_key(row, CATEGORY_KEYS)
                category = str(row.get(ck, "") if ck else "").strip() or "officeqa"
            items.append({
                "id": uid,
                "uid": uid,
                "question": str(row.get(q_k, "")).strip(),
                "ground_truth": str(row.get(a_k, "")).strip(),
                "category": category,
                "source_files": src_files,
                "source_docs": src_docs,
                "split": it.get("source_split") or split,
            })
            for sf in src_files:
                stem = Path(sf).stem if Path(sf).suffix else sf
                if stem:
                    referenced_files.add(stem)

        out_dir = OUT_SPLIT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        msg = f"  {split}: wrote {len(items):>4}/{len(id_items):>4} items -> {out_path.relative_to(PROJECT_ROOT)}"
        if missing:
            msg += f"  (unmatched uids: {missing[:5]}{'...' if len(missing) > 5 else ''})"
        print(msg)

    return referenced_files


# ── parsed-doc corpus (sparse or full) ─────────────────────────────────────
def _all_repo_stems(token: str) -> set[str]:
    """List every parsed bulletin stem present in the repo (anon tree API is fine)."""
    from huggingface_hub import HfApi
    api = HfApi(token=token or None)
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    stems: set[str] = set()
    for f in files:
        if f.startswith(f"{PARSED_PREFIX}/{JSONS_SUBDIR}/") and f.endswith(".json"):
            stems.add(Path(f).stem)
    return stems


def download_docs(referenced: set[str], token: str, *, full: bool) -> None:
    from huggingface_hub import hf_hub_download

    if full:
        wanted = _all_repo_stems(token)
        print(f"  [docs] --full: mirroring all {len(wanted)} parsed bulletins")
    else:
        wanted = set(referenced)
        print(f"  [docs] sparse: {len(wanted)} referenced bulletins ({len(wanted) * 2} files)")

    (DOCS_ROOT / TRANSFORMED_SUBDIR).mkdir(parents=True, exist_ok=True)
    (DOCS_ROOT / JSONS_SUBDIR).mkdir(parents=True, exist_ok=True)

    ok_txt = ok_json = miss = 0
    for i, stem in enumerate(sorted(wanted), start=1):
        for subdir, ext in ((TRANSFORMED_SUBDIR, ".txt"), (JSONS_SUBDIR, ".json")):
            repo_file = f"{PARSED_PREFIX}/{subdir}/{stem}{ext}"
            dest = DOCS_ROOT / subdir / f"{stem}{ext}"
            if dest.is_file() and dest.stat().st_size > 0:
                if ext == ".txt":
                    ok_txt += 1
                else:
                    ok_json += 1
                continue
            try:
                cached = hf_hub_download(
                    HF_REPO, repo_file, repo_type="dataset", token=token or None
                )
                shutil.copy2(cached, dest)
                if ext == ".txt":
                    ok_txt += 1
                else:
                    ok_json += 1
            except Exception as e:  # noqa: BLE001
                miss += 1
                if miss <= 10:
                    print(f"    [warn] {repo_file}: {type(e).__name__}: {str(e)[:120]}")
        if i % 25 == 0 or i == len(wanted):
            print(f"    [docs] {i}/{len(wanted)} bulletins  (txt={ok_txt} json={ok_json} miss={miss})")
    print(f"  [docs] done: {ok_txt} transformed .txt, {ok_json} parsed .json into {DOCS_ROOT.relative_to(PROJECT_ROOT)}")
    if miss:
        print(f"  [docs] WARNING: {miss} file(s) could not be downloaded (gated/auth or absent).")


# ── CSV download ───────────────────────────────────────────────────────────
def download_csv(token: str) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        cached = hf_hub_download(HF_REPO, CSV_FILENAME, repo_type="dataset", token=token or None)
    except GatedRepoError:
        sys.exit(
            f"ERROR: {HF_REPO} is gated and access was denied.\n"
            "  1) Log in at https://huggingface.co and open\n"
            "     https://huggingface.co/datasets/databricks/officeqa\n"
            "  2) Click 'Agree and access repository' to accept the terms.\n"
            "  3) Provide a read token via --token, $env:HF_TOKEN, or ~/.env (hf_token=...).\n"
            "  A token WITHOUT accepting the terms still fails."
        )
    except RepositoryNotFoundError:
        sys.exit(f"ERROR: {HF_REPO} not found or token lacks access.")
    return Path(cached)


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize OfficeQA split + parsed-doc corpus from gated HF.")
    ap.add_argument("--token", type=str, default=None, help="HF read token (else env / ~/.env)")
    ap.add_argument("--full", action="store_true", help="mirror the whole parsed corpus, not just referenced files")
    ap.add_argument("--skip-docs", action="store_true", help="only rebuild split JSON; skip parsed-doc download")
    ap.add_argument("--skip-csv", action="store_true", help="only download docs; skip CSV/split build")
    args = ap.parse_args()

    token = resolve_token(args.token)
    if not token:
        sys.exit(
            "ERROR: no Hugging Face token found.\n"
            "  Set one of: --token hf_xxx | $env:HF_TOKEN='hf_xxx' | add 'hf_token=hf_xxx' to ~/.env\n"
            "  The account must have accepted terms at\n"
            "  https://huggingface.co/datasets/databricks/officeqa (gated: auto)."
        )
    print(f"  [auth] using token …{token[-4:]} (len={len(token)})")

    referenced: set[str] = set()
    if not args.skip_csv:
        csv_path = download_csv(token)
        print(f"  [csv] downloaded -> {csv_path}")
        referenced = build_splits(csv_path)
        print(f"  [csv] splits reference {len(referenced)} unique bulletins")
    else:
        # Derive the referenced set straight from the id-split manifests.
        for split in SPLITS:
            for it in _load_id_items(split):
                for sf in _parse_source_list(it.get("source_files")):
                    stem = Path(sf).stem if Path(sf).suffix else sf
                    if stem:
                        referenced.add(stem)

    if not args.skip_docs:
        download_docs(referenced, token, full=args.full)
    else:
        print("  [docs] skipped (--skip-docs)")

    print("\nDone. Next: run the offline eval, e.g.")
    print("  python scripts/eval_only.py --config configs/officeqa/default.yaml \\")
    print("    --skill ckpt/officeqa/gpt5.5_skill.md --split valid_unseen \\")
    print("    --split_dir data/officeqa_split --data_root data/officeqa_docs_official ...")


if __name__ == "__main__":
    main()
