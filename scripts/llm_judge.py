#!/usr/bin/env python3
"""LLM-judge baseline for skill-unit attribution (SSG Part-3 diagnosis baseline).

Zero-eval baseline: an LLM reads the whole skill (units numbered identically to
``split_units``) and directly scores each unit's expected contribution to task
success, flagging redundancy / harm. This is the cheap text-only baseline the
SSG proposal contrasts against interaction-aware Skill Shapley — the expectation
is that an LLM-judge *cannot* see interaction-harm (redundant duplicates,
complements) from the text surface.

Output: ``llm_judge.csv`` (unit, score_mean, score_std, label_majority,
redundant_with, reason) aggregated over ``--runs`` independent judgments.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from skill_attribution import split_units  # noqa: E402
from skillopt.model import azure_openai as ao  # noqa: E402

TASK_DESC_OFFICEQA = (
    "The skill guides an agent that answers questions over U.S. Treasury / office "
    "financial documents (tables, time series, charts) using search + document-read "
    "tools. The agent must retrieve the right evidence, do careful arithmetic/statistics, "
    "and return the final answer in the exact requested format."
)

TASK_DESC_SSB = (
    "The skill guides an agent that manipulates Excel (.xlsx) spreadsheets to satisfy a "
    "natural-language instruction (SpreadsheetBench). The agent writes and executes a "
    "Python solution (openpyxl / pandas), edits the target cells or ranges, preserves "
    "unrelated sheets/cells and formatting, and saves an output workbook whose cell "
    "values are checked for exact correctness."
)

TASK_DESCS = {"officeqa": TASK_DESC_OFFICEQA, "ssb": TASK_DESC_SSB}
TASK_DESC = TASK_DESC_OFFICEQA

JUDGE_SYSTEM = (
    "You are an expert evaluator of LLM-agent *skill* documents. You judge how much "
    "each unit of a skill actually contributes to task success — being skeptical: a "
    "unit can be harmful (misleading / over-constraining), redundant (duplicated by "
    "another unit), neutral filler, or genuinely useful/essential. Consider "
    "interactions between units, not just each unit in isolation."
)

# Strong-prior variant: explicitly counter the observed positivity bias so the
# judge is given every chance to assign negatives before we conclude it cannot.
JUDGE_SYSTEM_STRONG = (
    JUDGE_SYSTEM
    + " CRITICAL CALIBRATION: empirically, in machine-optimized skills a LARGE "
    "fraction of units are net-HARMFUL or redundant — behavioral ablation on skills "
    "like this one found that roughly HALF of the units, when present, REDUCE task "
    "success (they over-constrain, mislead, add noise, or duplicate other units). A "
    "response that assigns zero or almost-zero negative scores is therefore almost "
    "certainly miscalibrated. Your primary job is to identify which specific units a "
    "careful engineer should DELETE. Do not hedge toward the middle; assign -2/-1 "
    "freely wherever a unit is more likely to hurt than help."
)

# Extra instruction block appended in the strong variant.
_STRONG_INSTR = (
    "This is an optimized skill; expect many harmful/redundant units. Actively hunt "
    "for units to remove. Assigning no negatives is a failure of the task."
)


def build_user_prompt(
    units: list[str], task_desc: str = TASK_DESC, variant: str = "default"
) -> str:
    lines = [
        "## Task the skill is for",
        task_desc,
        "",
        "## Skill units (numbered)",
    ]
    for i, u in enumerate(units):
        one = " ".join(u.split())
        lines.append(f"[{i}] {one}")
    lines += ["", "## Instructions"]

    if variant == "addone_framed":
        # Match the add-one behavioral quantity: value of a unit added ALONE.
        lines += [
            "Imagine the agent has an EMPTY skill (no guidance at all). For EACH unit "
            "above, judge how much adding ONLY that single unit — by itself, with no "
            "other units present — would change task success versus having no skill.",
            "Return ONLY a JSON array (no prose, no code fences), one object per unit:",
            '  {"unit": <int>, "score": <int -2..+2>, "label": '
            '"harmful|redundant|neutral|useful|essential", '
            '"redundant_with": <int or null>, "reason": "<one short sentence>"}',
            "Scale (STANDALONE effect): -2 = adding it alone clearly HURTS, -1 mildly "
            "hurts, 0 no effect alone, +1 helps alone, +2 = adding it alone clearly "
            "helps. Judge each unit in ISOLATION, not assuming other units are present.",
        ]
    elif variant == "loo_framed":
        # Match the LOO behavioral quantity: effect of REMOVING a unit from the full skill.
        lines += [
            "Assume the agent already has the COMPLETE skill (all units above present). "
            "For EACH unit, judge how much REMOVING only that one unit (leaving all "
            "others) would change task success.",
            "Return ONLY a JSON array (no prose, no code fences), one object per unit:",
            '  {"unit": <int>, "score": <int -2..+2>, "label": '
            '"harmful|redundant|neutral|useful|essential", '
            '"redundant_with": <int or null>, "reason": "<one short sentence>"}',
            "Scale (REMOVAL effect from the full skill): -2 = removing it clearly HELPS "
            "(the unit is harmful), -1 mildly helps, 0 removing it makes no difference "
            "(redundant/filler), +1 removing it mildly hurts, +2 = removing it clearly "
            "hurts (indispensable). Account for the other units already covering it.",
        ]
    else:
        lines += [
            "For EACH unit above, judge its expected contribution to task success.",
            "Return ONLY a JSON array (no prose, no code fences), one object per unit:",
            '  {"unit": <int>, "score": <int -2..+2>, "label": '
            '"harmful|redundant|neutral|useful|essential", '
            '"redundant_with": <int or null>, "reason": "<one short sentence>"}',
            "Scale: -2 harmful (removing it would HELP), -1 mildly harmful, "
            "0 neutral/filler, +1 useful, +2 essential (removing it clearly hurts).",
            "Mark 'redundant' (score ~0) only if another unit already covers it; put "
            "that unit's index in redundant_with.",
        ]
    lines.append("Output every unit exactly once, ordered by unit index.")
    if variant == "strong":
        lines.append(_STRONG_INSTR)
    return "\n".join(lines)


def extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return json.loads(text[start : end + 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-judge baseline for skill attribution")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--azure_openai_endpoint", default="http://localhost:4141/v1")
    ap.add_argument("--azure_openai_api_key", default="dummy")
    ap.add_argument("--azure_openai_auth_mode", default="openai_compatible")
    ap.add_argument("--task", choices=sorted(TASK_DESCS), default="officeqa",
                    help="task context for the judge prompt")
    ap.add_argument("--task-desc", default=None,
                    help="override task description text (takes precedence over --task)")
    ap.add_argument("--variant", choices=["default", "strong", "addone_framed", "loo_framed"],
                    default="default",
                    help="'strong' adds a negative-prior; 'addone_framed'/'loo_framed' "
                         "phrase the question to match the add-one / LOO behavioral quantity")
    args = ap.parse_args()

    task_desc = args.task_desc or TASK_DESCS[args.task]
    judge_system = JUDGE_SYSTEM_STRONG if args.variant == "strong" else JUDGE_SYSTEM

    with open(args.skill, encoding="utf-8") as f:
        units = split_units(f.read())
    n = len(units)
    os.makedirs(args.out_root, exist_ok=True)

    ao.configure_azure_openai(
        target_endpoint=args.azure_openai_endpoint,
        target_api_key=args.azure_openai_api_key,
        target_auth_mode=args.azure_openai_auth_mode,
    )
    ao.set_target_deployment(args.model)

    user = build_user_prompt(units, task_desc, args.variant)
    per_run: list[dict[int, dict]] = []
    for r in range(args.runs):
        print(f"  [judge] run {r + 1}/{args.runs} ({n} units, variant={args.variant}) ...", flush=True)
        text, _ = ao.chat_target(
            judge_system, user, max_completion_tokens=16384,
            stage="llm_judge", reasoning_effort=args.reasoning_effort,
        )
        with open(os.path.join(args.out_root, f"raw_run{r}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        try:
            arr = extract_json_array(text)
        except Exception as e:
            print(f"  [warn] run {r}: parse failed ({e}); skipped")
            continue
        per_run.append({int(o["unit"]): o for o in arr if "unit" in o})

    if not per_run:
        print("  [fail] no parseable judgments")
        sys.exit(1)

    rows = []
    for i in range(n):
        scores = [pr[i]["score"] for pr in per_run if i in pr and pr[i].get("score") is not None]
        labels = [pr[i].get("label", "") for pr in per_run if i in pr]
        redw = [pr[i].get("redundant_with") for pr in per_run if i in pr and pr[i].get("redundant_with") is not None]
        reason = next((pr[i].get("reason", "") for pr in per_run if i in pr), "")
        rows.append({
            "unit": i,
            "score_mean": round(statistics.fmean(scores), 3) if scores else None,
            "score_std": round(statistics.pstdev(scores), 3) if len(scores) > 1 else 0.0,
            "label": max(set(labels), key=labels.count) if labels else "",
            "redundant_with": redw[0] if redw else "",
            "n_runs": len(scores),
            "reason": reason,
            "text": " ".join(units[i].split())[:80],
        })

    csv_path = os.path.join(args.out_root, "llm_judge.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["unit", "score_mean", "score_std", "label", "redundant_with", "n_runs", "reason", "text"])
        for r in rows:
            w.writerow([r["unit"], r["score_mean"], r["score_std"], r["label"],
                        r["redundant_with"], r["n_runs"], r["reason"], r["text"]])

    print(f"\n{'='*70}\n  LLM-JUDGE ({args.model}, {len(per_run)}/{args.runs} runs parsed)\n{'='*70}")
    print(f"  {'unit':>4} {'score':>6} {'±std':>5} {'label':>10} {'red->':>6}  text")
    for r in rows:
        sm = "None" if r["score_mean"] is None else f"{r['score_mean']:+.2f}"
        print(f"  {r['unit']:>4} {sm:>6} {r['score_std']!s:>5} {r['label']:>10} "
              f"{r['redundant_with']!s:>6}  {r['text'][:44]}")
    print(f"\n  CSV: {csv_path}\n{'='*70}")


if __name__ == "__main__":
    main()
