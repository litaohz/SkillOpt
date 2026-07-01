#!/usr/bin/env python3
"""Causal attribution over an arbitrary learned skill document (Direction 4, M1).

Unlike ``eval_skill_ablation.py`` (which ablates a fixed *template* section),
this operates on a free-form **skill .md** produced by training: it splits the
skill into atomic units (headers / list items / paragraphs) and measures each
unit's causal value by running ``eval_only.py`` with variant skill files —
no template editing, the variant is simply passed via ``--skill``.

Per unit i it reports:
  * LOO Δ    = score(full) - score(full \\ {i})   — indispensability
  * add-one Δ= score({i})  - score(empty)         — standalone value

With ``--prune`` it emits a pruned skill keeping only units whose removal hurts
(LOO Δ > 0) — a causal-attribution-guided compaction that directly targets the
"bloat" observed in long training runs (skill grows while test EM is flat).

This is the foundation for the in-loop attribution-guided optimizer: the same
per-unit value table can be injected into the optimizer's meta-skill / analyst
prompts so edits are guided by *measured* value rather than self-report.

Example (cheap validation: 8 items)::

    $env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"; $env:PYTHONIOENCODING="utf-8"
    python scripts/skill_attribution.py `
        --skill outputs/train_officeqa_gpt55_v1/best_skill.md `
        --modes loo addone --prune `
        --config configs/officeqa/default.yaml `
        --split valid_unseen --split_dir data/officeqa_split `
        --out-root outputs/attrib_officeqa_best `
        --eval-arg env.workers=8 --eval-arg env.limit=8 `
        --azure_openai_endpoint http://localhost:4141/v1 `
        --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible `
        --target_model gpt-5.5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

_UNIT_START = re.compile(r"^\s*(?:#+\s|\d+[.)]\s|[-*+]\s)")


# ── Skill → atomic units ────────────────────────────────────────────────────

def split_units(text: str, granularity: str = "unit") -> list[str]:
    """Split a skill document into atomic units.

    granularity="unit": each markdown header line, list item, or blank-line
    paragraph is its own unit. granularity="section": group each header with
    everything until the next header into one unit (coarse, cheap).
    """
    if granularity == "section":
        sections: list[str] = []
        cur: list[str] = []
        for line in text.split("\n"):
            if re.match(r"^\s*#+\s", line) and cur:
                if any(l.strip() for l in cur):
                    sections.append("\n".join(cur).rstrip())
                cur = [line]
            else:
                cur.append(line)
        if any(l.strip() for l in cur):
            sections.append("\n".join(cur).rstrip())
        return sections
    units: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        if any(line.strip() for line in buf):
            units.append("\n".join(buf).rstrip())
        buf = []

    for line in text.split("\n"):
        if _UNIT_START.match(line):
            flush()
            units.append(line.rstrip())
        elif line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return units


def render(units: list[str], keep: list[int]) -> str:
    return "\n".join(units[i] for i in keep).strip() + "\n"


# ── Eval one skill variant via eval_only.py ─────────────────────────────────

@dataclass
class RunResult:
    name: str
    hard: float | None = None
    soft: float | None = None
    n_items: int | None = None
    error: str = ""


def run_eval(base_cmd: list[str], skill_text: str, variant_out: str,
             name: str, dry_run: bool) -> RunResult:
    os.makedirs(variant_out, exist_ok=True)
    skill_path = os.path.join(variant_out, "skill.md")
    with open(skill_path, "w", encoding="utf-8") as f:
        f.write(skill_text)
    res = RunResult(name=name)
    if dry_run:
        print(f"  [dry] {name}: {len(skill_text)} chars -> {skill_path}")
        return res
    summary_path = os.path.join(variant_out, "eval_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                s = json.load(f)
            res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
            print(f"  [cache] {name}: hard={res.hard}")
            return res
        except Exception:
            pass
    cmd = list(base_cmd) + ["--skill", skill_path, "--out_root", variant_out]
    print(f"\n  [run] {name}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        res.error = f"eval_only exit {proc.returncode}"
        print(f"  [fail] {name}: {res.error}")
        return res
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
    print(f"  [done] {name}: hard={res.hard:.4f} ({time.time() - t0:.0f}s)")
    return res


def build_base_eval_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [args.python, os.path.join("scripts", "eval_only.py"),
           "--config", args.config, "--split", args.split]
    if args.split_dir:
        cmd += ["--split_dir", args.split_dir]
    if args.target_model:
        cmd += ["--target_model", args.target_model]
    if args.azure_openai_endpoint:
        cmd += ["--azure_openai_endpoint", args.azure_openai_endpoint]
    if args.azure_openai_api_key:
        cmd += ["--azure_openai_api_key", args.azure_openai_api_key]
    if args.azure_openai_auth_mode:
        cmd += ["--azure_openai_auth_mode", args.azure_openai_auth_mode]
    if args.eval_args:
        cmd += ["--cfg-options", *args.eval_args]
    return cmd


# ── Attribution driver (with content-hash dedup cache) ──────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Causal attribution over a skill document")
    ap.add_argument("--skill", required=True, help="learned skill .md to attribute")
    ap.add_argument("--modes", nargs="+", default=["loo", "addone"],
                    choices=["full", "empty", "loo", "addone"])
    ap.add_argument("--prune", action="store_true",
                    help="emit a pruned skill keeping only LOO Δ > 0 units")
    ap.add_argument("--prune-eps", type=float, default=0.0,
                    help="keep unit if LOO Δ > eps (default 0.0)")
    ap.add_argument("--granularity", choices=["unit", "section"], default="unit")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    # eval passthrough
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="valid_unseen")
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--target_model", default="")
    ap.add_argument("--azure_openai_endpoint", default="")
    ap.add_argument("--azure_openai_api_key", default="")
    ap.add_argument("--azure_openai_auth_mode", default="")
    ap.add_argument("--eval-arg", dest="eval_args", action="append", default=[])
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    with open(args.skill, encoding="utf-8") as f:
        skill_text = f.read()
    units = split_units(skill_text, args.granularity)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    print(f"  Parsed {len(units)} unit(s) from {args.skill}")

    base_cmd = build_base_eval_cmd(args)
    cache: dict[str, RunResult] = {}

    def ev(skill_variant: str, name: str) -> RunResult:
        h = hashlib.sha256(skill_variant.encode("utf-8")).hexdigest()
        if h in cache:
            prev = cache[h]
            print(f"  [dedup] {name} == {prev.name}: hard={prev.hard}")
            return RunResult(name=name, hard=prev.hard, soft=prev.soft, n_items=prev.n_items)
        r = run_eval(base_cmd, skill_variant, os.path.join(out_root, name), name, args.dry_run)
        cache[h] = r
        return r

    all_idx = list(range(len(units)))
    full = ev(render(units, all_idx), "full") if (
        "full" in args.modes or "loo" in args.modes or args.prune) else None
    empty = ev("", "empty") if ("empty" in args.modes or "addone" in args.modes) else None

    loo = {}
    if "loo" in args.modes or args.prune:
        for i in all_idx:
            loo[i] = ev(render(units, [j for j in all_idx if j != i]), f"loo_{i:03d}")
    addone = {}
    if "addone" in args.modes:
        for i in all_idx:
            addone[i] = ev(render(units, [i]), f"addone_{i:03d}")

    if args.dry_run:
        return

    full_h = full.hard if full else None
    empty_h = empty.hard if empty else None
    rows = []
    for i, unit in enumerate(units):
        loo_d = (full_h - loo[i].hard) if (full_h is not None and i in loo and loo[i].hard is not None) else None
        add_d = (addone[i].hard - empty_h) if (empty_h is not None and i in addone and addone[i].hard is not None) else None
        rows.append({"unit": i, "chars": len(unit),
                     "loo_delta": loo_d, "addone_delta": add_d,
                     "text": unit.replace("\n", " ")})

    csv_path = os.path.join(out_root, "attribution.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "chars", "loo_delta", "addone_delta", "text"])
        if full_h is not None:
            w.writerow(["full", len(skill_text), "", "", f"({len(units)} units)"])
        if empty_h is not None:
            w.writerow(["empty", 0, "", "", "(no skill)"])
        for r in rows:
            w.writerow([r["unit"], r["chars"], r["loo_delta"], r["addone_delta"], r["text"]])
    with open(os.path.join(out_root, "attribution.json"), "w", encoding="utf-8") as f:
        json.dump({"full_hard": full_h, "empty_hard": empty_h, "units": rows}, f,
                  indent=2, ensure_ascii=False)

    def fmt(x):
        return "  n/a " if x is None else f"{x:+.4f}"

    print(f"\n{'=' * 78}\n  SKILL ATTRIBUTION ({len(units)} units)\n{'=' * 78}")
    if full_h is not None:
        print(f"  full  = {full_h:.4f}  ({len(skill_text)} chars)")
    if empty_h is not None:
        print(f"  empty = {empty_h:.4f}")
    print(f"\n  {'#':>3} {'chars':>6} {'LOO Δ':>8} {'addone Δ':>9}  unit")
    for r in rows:
        t = (r["text"][:60] + "…") if len(r["text"]) > 61 else r["text"]
        print(f"  {r['unit']:>3} {r['chars']:>6} {fmt(r['loo_delta']):>8} {fmt(r['addone_delta']):>9}  {t}")

    if args.prune and full_h is not None:
        eps = args.prune_eps
        keep = [i for i in all_idx if (loo[i].hard is None) or (full_h - loo[i].hard) > eps]
        pruned = render(units, keep)
        pruned_path = os.path.join(out_root, "pruned_skill.md")
        with open(pruned_path, "w", encoding="utf-8") as f:
            f.write(pruned)
        print(f"\n  [prune] kept {len(keep)}/{len(units)} units "
              f"({len(skill_text)} -> {len(pruned)} chars). Saved: {pruned_path}")
        print("  Validate: re-run eval_only.py --skill " + pruned_path +
              " and compare hard vs full.")

    print(f"\n  CSV: {csv_path}\n{'=' * 78}")


if __name__ == "__main__":
    main()
