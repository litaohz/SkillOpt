#!/usr/bin/env python3
"""Plot a SkillOpt training run as a paper-style "epoch/step trends" figure.

Reproduces the look of the paper's `epoch-trends` figure (Train rollout /
Selection best / Unseen test vs checkpoint) for a *local* training run, and
overlays which steps the gate accepted vs rejected plus the skill-length growth.

Data sources (all from one ``out_root``):
  * ``steps/step_*/step_record.json`` — per-step ``rollout_hard`` (train),
    ``best_score`` (selection best), ``action`` (accept/reject), ``skill_len``.
  * ``summary.json`` — step-0 baselines (``baseline_selection_hard``, ``baseline_test_hard``).
  * optional ``--version-curve <csv>`` — per-version ``hard`` = Unseen test EM,
    produced by ``eval_skill_ablation.py --versions-dir`` (skill_v{k} ↔ step k).

Usage::

    python scripts/plot_training_curve.py `
        --run outputs/train_officeqa_gpt55_v1 `
        --version-curve outputs/version_curve_officeqa_gpt55/version_curve.csv `
        --title "OfficeQA (gpt-5.5)" --out outputs/train_officeqa_gpt55_v1/step_trends.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_steps(run: str) -> list[dict]:
    steps_dir = os.path.join(run, "steps")
    recs = []
    for name in sorted(os.listdir(steps_dir)):
        p = os.path.join(steps_dir, name, "step_record.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                recs.append(json.load(f))
    recs.sort(key=lambda r: r.get("step", 0))
    return recs


def load_version_test(csv_path: str) -> dict[int, float]:
    """Map step index -> unseen-test EM from a version_curve.csv (skill_v{k} ↔ step k)."""
    out: dict[int, float] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"v(\d+)", row["version"])
            if m and row.get("hard"):
                out[int(m.group(1))] = float(row["hard"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot SkillOpt step/epoch trends")
    ap.add_argument("--run", required=True, help="training out_root")
    ap.add_argument("--version-curve", help="version_curve.csv for the Unseen test line")
    ap.add_argument("--title", default="")
    ap.add_argument("--out", help="output PNG (default: <run>/step_trends.png)")
    args = ap.parse_args()

    run = os.path.abspath(args.run)
    summary = {}
    _summary_path = os.path.join(run, "summary.json")
    if os.path.exists(_summary_path):
        with open(_summary_path, encoding="utf-8") as f:
            summary = json.load(f)
    recs = load_steps(run)

    steps = [r["step"] for r in recs]
    train = [r.get("rollout_hard") for r in recs]
    sel_best = [r.get("best_score") for r in recs]
    actions = [r.get("action", "") for r in recs]
    skill_len = [r.get("skill_len") for r in recs]

    # step-0 baselines
    base_sel = summary.get("baseline_selection_hard")
    base_test = summary.get("baseline_test_hard")

    sel_x = ([0] + steps) if base_sel is not None else steps
    sel_y = ([base_sel] + sel_best) if base_sel is not None else sel_best

    test_x, test_y = [], []
    if args.version_curve:
        vmap = load_version_test(args.version_curve)
        for k in sorted(vmap):
            test_x.append(k)
            test_y.append(vmap[k])
    elif base_test is not None:
        test_x, test_y = [0], [base_test]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, train, "-o", color="#1f77b4", label="Train rollout")
    ax.plot(sel_x, sel_y, "--s", color="#d62728", label="Selection best")
    if test_x:
        ax.plot(test_x, test_y, "-.^", color="#2ca02c", label="Unseen test")

    # mark accepted vs rejected steps on the train line
    for s, y, a in zip(steps, train, actions):
        if y is None:
            continue
        if str(a).startswith("accept"):
            ax.annotate("✓", (s, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", color="green", fontsize=11, fontweight="bold")
        elif str(a).startswith("reject"):
            ax.annotate("✗", (s, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", color="#999999", fontsize=10)

    ax.set_xlabel("Step checkpoint  (✓ gate-accepted, ✗ rejected)")
    ax.set_ylabel("Hard score")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks([0] + steps)

    # skill length on a twin axis (context for "bloat")
    if any(v is not None for v in skill_len):
        ax2 = ax.twinx()
        ax2.plot(steps, skill_len, ":", color="#888888", alpha=0.7, label="Skill length (chars)")
        ax2.set_ylabel("Skill length (chars)", color="#888888")
        ax2.tick_params(axis="y", labelcolor="#888888")

    lines, labels = ax.get_legend_handles_labels()
    if any(v is not None for v in skill_len):
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lab2
    ax.legend(lines, labels, loc="lower right", fontsize=9)

    out = args.out or os.path.join(run, "step_trends.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
