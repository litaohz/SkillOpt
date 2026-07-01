"""Causal-attribution hints for the optimizer (Direction 4, M3 building block).

Turns a measured per-unit value table (from scripts/skill_attribution.py) into a
structured guidance block that can be injected into the optimizer's analyst /
meta-skill prompts, so edits are driven by *measured* causal value instead of the
optimizer's self-report. This is the wiring for the in-loop, attribution-guided
optimizer; the full A/B training run is triggered separately.

Convention: LOO Δ > eps ⇒ keep/protect; ≈0 ⇒ redundant (safe to prune);
LOO < -eps, or add-one < -eps standalone ⇒ harmful (prune).
"""
from __future__ import annotations

import csv


def load_attribution(csv_path: str) -> list[dict]:
    """Load rows from an attribution.csv, skipping the full/empty summary rows."""
    rows: list[dict] = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("unit") in ("full", "empty"):
                continue

            def _f(key: str):
                v = r.get(key, "")
                return float(v) if v not in ("", None) else None

            rows.append({"unit": r.get("unit"), "loo": _f("loo_delta"),
                         "addone": _f("addone_delta"), "text": (r.get("text") or "")})
    return rows


def classify(rows: list[dict], eps: float = 0.01) -> dict[str, list[dict]]:
    keep, prune, harmful = [], [], []
    for r in rows:
        loo = r["loo"]
        if loo is None:
            continue
        if loo > eps:
            keep.append(r)
        elif loo < -eps or (r["addone"] is not None and r["addone"] < -eps):
            harmful.append(r)
        else:
            prune.append(r)
    return {"keep": keep, "harmful": harmful, "prune": prune}


def format_hints(rows: list[dict], eps: float = 0.01, max_chars: int = 90) -> str:
    """Render a compact, optimizer-facing guidance block. Empty string if no signal."""
    g = classify(rows, eps)
    if not any(g.values()):
        return ""
    lines = ["## Measured skill-unit value (causal, held-out — trust over intuition)"]
    if g["keep"]:
        lines.append("Protect (removing measurably hurts):")
        lines += [f"  +{r['loo']:.3f}  {r['text'][:max_chars]}" for r in g["keep"]]
    if g["harmful"]:
        lines.append("Remove (net-harmful / harmful standalone):")
        lines += [f"  {r['loo']:.3f}  {r['text'][:max_chars]}" for r in g["harmful"]]
    if g["prune"]:
        lines.append(f"Redundant (≈0 value, prefer pruning over adding more): "
                     f"{len(g['prune'])} unit(s).")
    return "\n".join(lines)


def hints_from_csv(csv_path: str, eps: float = 0.01) -> str:
    return format_hints(load_attribution(csv_path), eps)
