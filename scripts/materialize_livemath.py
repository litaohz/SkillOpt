"""Materialize runnable LiveMathematicianBench splits from the released ID manifest.

Usage
-----
    python scripts/materialize_livemath.py
    python scripts/materialize_livemath.py --output-dir data/livemathematicianbench_split
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = ("train", "val", "test")
HF_DATASET = "LiveMathematicianBench/LiveMathematicianBench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "livemathematicianbench_id_split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "livemathematicianbench_split",
    )
    parser.add_argument(
        "--dataset",
        default=HF_DATASET,
    )
    return parser.parse_args()


def load_manifest_ids(manifest_dir: Path) -> dict[str, list[str]]:
    split_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        path = manifest_dir / split / "items.json"
        with path.open(encoding="utf-8") as f:
            items = json.load(f)
        split_ids[split] = [str(item["id"]) for item in items]
    return split_ids


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'.\n"
            "Install it with:  pip install datasets"
        ) from exc

    from skillopt.envs.livemathematicianbench.dataloader import _normalize_item

    print(f"Loading {args.dataset} from Hugging Face...")
    ds = load_dataset(args.dataset)

    split_ids = load_manifest_ids(args.manifest_dir)
    wanted_ids = {item_id for ids in split_ids.values() for item_id in ids}

    # Build a lookup keyed by "{month}:{no}"
    selected: dict[str, dict] = {}
    for hf_split in ds.values():
        for row_idx, row in enumerate(hf_split):
            month = str(row.get("month", "")).strip()
            no = row.get("no", row_idx + 1)
            item_id = f"{month}:{no}" if month else str(no)
            if item_id not in wanted_ids:
                continue
            if item_id in selected:
                print(f"  [warn] duplicate id {item_id!r}, keeping first", flush=True)
                continue
            norm = _normalize_item(dict(row), row_idx=row_idx, source_path=str(row.get("source_file", "")))
            selected[item_id] = norm

    missing = sorted(wanted_ids - selected.keys())
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} IDs from the manifest. First 5: {missing[:5]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, ids in split_ids.items():
        items = [selected[item_id] for item_id in ids]
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        counts[split] = len(items)
        print(f"  {split}: {len(items)} items")

    manifest = {
        "source_manifest_dir": str(args.manifest_dir.resolve()),
        "source_dataset": args.dataset,
        "counts": counts,
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Wrote splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
