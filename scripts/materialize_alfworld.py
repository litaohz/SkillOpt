"""Materialize runnable ALFWorld splits from the released path manifest.

The shipped ``data/alfworld_path_split`` only stores *relative* gamefile paths
(e.g. ``json_2.1.1/valid_unseen/.../game.tw-pddl``) that are anchored at
``$ALFWORLD_DATA``. The ALFWorld TextWorld simulator needs absolute, existing
gamefile paths, so this script expands each ``gamefile`` against
``$ALFWORLD_DATA`` and writes a runnable split directory.

Prerequisites
-------------
    pip install alfworld            # pulls textworld + jericho (Linux/macOS)
    alfworld-download               # downloads $ALFWORLD_DATA/json_2.1.1 + logic
    export ALFWORLD_DATA=$HOME/.cache/alfworld

Usage
-----
    python scripts/materialize_alfworld.py
    python scripts/materialize_alfworld.py --output-dir data/alfworld_split
    python scripts/materialize_alfworld.py --alfworld-data /path/to/alfworld
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "alfworld_path_split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "alfworld_split",
    )
    parser.add_argument(
        "--alfworld-data",
        type=Path,
        default=None,
        help="Root of the ALFWorld payload (defaults to $ALFWORLD_DATA).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn instead of failing when a gamefile is not found on disk.",
    )
    return parser.parse_args()


def resolve_alfworld_data(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit
    else:
        env = os.environ.get("ALFWORLD_DATA", "").strip()
        if not env:
            raise SystemExit(
                "ALFWORLD_DATA is not set and --alfworld-data was not given.\n"
                "Run `alfworld-download` and `export ALFWORLD_DATA=$HOME/.cache/alfworld`."
            )
        root = Path(os.path.expandvars(env)).expanduser()
    root = root.expanduser()
    if not (root / "json_2.1.1").is_dir():
        raise SystemExit(
            f"ALFWorld payload not found under {root} (missing json_2.1.1/).\n"
            "Run `alfworld-download` first."
        )
    return root


def main() -> None:
    args = parse_args()
    data_root = resolve_alfworld_data(args.alfworld_data)
    print(f"ALFWORLD_DATA = {data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    missing_total = 0

    for split in SPLITS:
        src = args.manifest_dir / split / "items.json"
        with src.open(encoding="utf-8") as f:
            items = json.load(f)

        out_items: list[dict] = []
        for item in items:
            rel = str(item.get("gamefile") or "").strip()
            if not rel:
                raise SystemExit(f"{split}: item {item.get('id')!r} has empty gamefile")
            abs_path = (data_root / rel).resolve()
            if not abs_path.is_file():
                missing_total += 1
                msg = f"  [missing] {split}: {abs_path}"
                if args.allow_missing:
                    print(msg)
                else:
                    raise SystemExit(msg + "\n(use --allow-missing to skip)")
            row = dict(item)
            row["gamefile"] = str(abs_path)
            out_items.append(row)

        out_split = args.output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)
        with (out_split / "items.json").open("w", encoding="utf-8") as f:
            json.dump(out_items, f, ensure_ascii=False, indent=2)
        counts[split] = len(out_items)
        print(f"  {split}: {len(out_items)} items")

    manifest = {
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "alfworld_data": str(data_root),
        "counts": counts,
        "note": "gamefile fields are absolute paths expanded against ALFWORLD_DATA.",
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote splits to {args.output_dir.resolve()}: {counts}")
    if missing_total:
        print(f"WARNING: {missing_total} gamefiles missing on disk.")


if __name__ == "__main__":
    main()
