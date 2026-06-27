#!/usr/bin/env python3
"""Sentence-level skill ablation for SkillOpt.

Quantify the contribution of each individual rule / sentence in a skill (or in
the always-on prompt template's "Rules" block) by running controlled ablations
on a fixed held-out split and measuring the score delta.

Two complementary attributions are produced per rule i:

  * Leave-one-out (LOO):  score(full) - score(full \\ {i})
        -> how *indispensable* rule i is in the presence of the others.
  * Add-one-in (addone):  score(base + {i}) - score(base)
        -> the *standalone* value of rule i on top of the bare baseline.

The ablation target is a markdown section (default ``## Rules``) inside a prompt
*template* file. Each variant is produced by rewriting that template, then a
full evaluation is run via ``scripts/eval_only.py`` with an (empty) skill file.
The original template is always restored afterwards (even on error / Ctrl-C).

Process-level attribution (``--versions-dir``)
----------------------------------------------
Instead of ablating one template, evaluate every saved skill version from a
training run (``out_root/skills/skill_v*.md`` plus ``best_skill.md``) on a fixed
held-out split. The marginal delta between consecutive versions attributes score
change to each optimization step's edit. Example::

    python scripts/eval_skill_ablation.py --versions-dir outputs/train_run/skills `
        --config configs/officeqa/default.yaml --split valid_unseen `
        --split_dir data/officeqa_split --out-root outputs/version_curve `
        --eval-arg env.workers=12 --target_model gpt-5.5 `
        --azure_openai_endpoint http://localhost:4141/v1 `
        --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible

Example (OfficeQA, on the local ghc proxy with gpt-5.5)::

    $env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"
    $env:PYTHONIOENCODING="utf-8"; $env:NO_PROXY="localhost,127.0.0.1"
    python scripts/eval_skill_ablation.py `
        --template skillopt/envs/officeqa/prompts/rollout_system.md `
        --section "## Rules" `
        --modes full bare loo addone `
        --config configs/officeqa/default.yaml `
        --skill outputs/empty_skill.md `
        --split valid_unseen --split_dir data/officeqa_split `
        --out-root outputs/ablation_officeqa_gpt55 `
        --eval-arg env.workers=12 `
        --azure_openai_endpoint http://localhost:4141/v1 `
        --azure_openai_api_key dummy `
        --azure_openai_auth_mode openai_compatible `
        --target_model gpt-5.5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)


# ── Template parsing ────────────────────────────────────────────────────────

_RULE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+(.*\S)\s*$")


@dataclass
class ParsedTemplate:
    """A template split around an ablatable markdown section."""

    head: str            # text before the section heading (keeps {skill_section})
    heading: str         # e.g. "## Rules"
    rules: list[str]     # rule bodies, list markers stripped
    tail: str            # text after the rules block (e.g. "## Tool Use" ...)


def parse_template(content: str, heading: str) -> ParsedTemplate:
    head, sep, after = content.partition(heading)
    if not sep:
        raise ValueError(f"Section heading {heading!r} not found in template.")
    lines = after.split("\n")
    # lines[0] is the remainder of the heading line (usually empty)
    rules: list[str] = []
    i = 1
    while i < len(lines):
        m = _RULE_RE.match(lines[i])
        if m:
            rules.append(m.group(1).strip())
            i += 1
        elif lines[i].strip() == "" and rules:
            # blank line terminates the list
            break
        elif not rules:
            # allow a leading blank line right after the heading
            i += 1
        else:
            break
    if not rules:
        raise ValueError(f"No list items found under {heading!r}.")
    tail = "\n".join(lines[i:])
    return ParsedTemplate(head=head, heading=heading, rules=rules, tail=tail)


def render(parsed: ParsedTemplate, selected: list[int]) -> str:
    """Render the template keeping only ``selected`` rule indices (0-based)."""
    tail = parsed.tail.lstrip("\n")
    if not selected:
        # bare: drop the heading + rules entirely, keep one blank line before tail
        body = parsed.head.rstrip()
        return f"{body}\n\n{tail}" if tail else f"{body}\n"
    numbered = "\n".join(
        f"{n}. {parsed.rules[idx]}" for n, idx in enumerate(selected, start=1)
    )
    block = f"{parsed.heading}\n{numbered}\n"
    return f"{parsed.head}{block}\n{tail}" if tail else f"{parsed.head}{block}"


# ── Variant plan ────────────────────────────────────────────────────────────

@dataclass
class Variant:
    name: str
    selected: list[int]          # rule indices kept (0-based)
    rule_idx: int | None = None  # the rule this variant isolates (LOO/addone)


def build_variants(n_rules: int, modes: list[str]) -> list[Variant]:
    all_idx = list(range(n_rules))
    variants: list[Variant] = []
    seen: set[tuple[int, ...]] = set()

    def add(v: Variant) -> None:
        key = tuple(v.selected)
        # always keep distinctly-named variants; dedupe only the eval work later
        variants.append(v)
        seen.add(key)

    if "full" in modes:
        add(Variant("full", all_idx))
    if "bare" in modes:
        add(Variant("bare", []))
    if "loo" in modes:
        for i in all_idx:
            add(Variant(f"loo_r{i + 1}", [j for j in all_idx if j != i], rule_idx=i))
    if "addone" in modes:
        for i in all_idx:
            add(Variant(f"addone_r{i + 1}", [i], rule_idx=i))
    return variants


# ── Evaluation ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    selected: list[int]
    rule_idx: int | None
    hard: float | None = None
    soft: float | None = None
    n_items: int | None = None
    out_root: str = ""
    error: str = ""


def run_eval(
    variant: Variant,
    parsed: ParsedTemplate,
    template_path: str,
    base_eval_cmd: list[str],
    out_root: str,
    dry_run: bool,
) -> RunResult:
    res = RunResult(name=variant.name, selected=variant.selected, rule_idx=variant.rule_idx)
    variant_out = os.path.join(out_root, variant.name)
    res.out_root = variant_out
    rendered = render(parsed, variant.selected)

    if dry_run:
        print(f"\n{'#' * 70}\n# VARIANT {variant.name}  (rules kept: "
              f"{[i + 1 for i in variant.selected] or 'none'})\n{'#' * 70}")
        print(rendered)
        return res

    # cache: if eval_summary already exists, reuse it (resume support)
    summary_path = os.path.join(variant_out, "eval_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                s = json.load(f)
            res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
            print(f"  [cache] {variant.name}: hard={res.hard}  (reused)")
            return res
        except Exception:
            pass

    backup = template_path + ".ablation_bak"
    shutil.copy2(template_path, backup)
    try:
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        cmd = list(base_eval_cmd) + ["--out_root", variant_out]
        print(f"\n  [run] {variant.name}: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
        dt = time.time() - t0
        if proc.returncode != 0:
            res.error = f"eval_only exit {proc.returncode}"
            print(f"  [fail] {variant.name}: {res.error}")
            return res
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
        print(f"  [done] {variant.name}: hard={res.hard:.4f} soft={res.soft:.4f} "
              f"n={res.n_items} ({dt:.0f}s)")
    finally:
        shutil.move(backup, template_path)
    return res


# ── Reporting ───────────────────────────────────────────────────────────────

def write_report(parsed: ParsedTemplate, results: list[RunResult], out_root: str) -> None:
    by_name = {r.name: r for r in results}
    full = by_name.get("full")
    bare = by_name.get("bare")
    full_hard = full.hard if full else None
    bare_hard = bare.hard if bare else None

    rows = []
    for i, rule in enumerate(parsed.rules):
        loo = by_name.get(f"loo_r{i + 1}")
        add = by_name.get(f"addone_r{i + 1}")
        loo_delta = (full_hard - loo.hard) if (full_hard is not None and loo and loo.hard is not None) else None
        add_delta = (add.hard - bare_hard) if (bare_hard is not None and add and add.hard is not None) else None
        rows.append({
            "rule": i + 1,
            "text": rule,
            "loo_score": None if not loo else loo.hard,
            "loo_delta": loo_delta,
            "addone_score": None if not add else add.hard,
            "addone_delta": add_delta,
        })

    csv_path = os.path.join(out_root, "ablation_report.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "loo_score", "loo_delta", "addone_score", "addone_delta", "text"])
        if full_hard is not None:
            w.writerow(["full", full_hard, "", "", "", "(all rules)"])
        if bare_hard is not None:
            w.writerow(["bare", bare_hard, "", "", "", "(no rules)"])
        for r in rows:
            w.writerow([r["rule"], r["loo_score"], r["loo_delta"],
                        r["addone_score"], r["addone_delta"], r["text"]])

    json_path = os.path.join(out_root, "ablation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "full_hard": full_hard, "bare_hard": bare_hard,
            "rules": rows,
            "runs": [vars(r) for r in results],
        }, f, indent=2, ensure_ascii=False)

    def fmt(x):
        return "  n/a " if x is None else f"{x:+.4f}" if isinstance(x, float) else str(x)

    print(f"\n{'=' * 78}\n  SKILL ABLATION REPORT\n{'=' * 78}")
    if full_hard is not None:
        print(f"  full (all rules): {full_hard:.4f}")
    if bare_hard is not None:
        print(f"  bare (no rules) : {bare_hard:.4f}")
    print(f"\n  {'#':>2}  {'LOO Δ':>8}  {'addone Δ':>9}   rule")
    print(f"  {'-' * 2}  {'-' * 8}  {'-' * 9}   {'-' * 50}")
    for r in rows:
        txt = (r["text"][:64] + "…") if len(r["text"]) > 65 else r["text"]
        print(f"  {r['rule']:>2}  {fmt(r['loo_delta']):>8}  {fmt(r['addone_delta']):>9}   {txt}")
    print(f"\n  CSV : {csv_path}\n  JSON: {json_path}\n{'=' * 78}")


# ── Version-curve attribution ───────────────────────────────────────────────

def _run_eval_skill(base_cmd_no_skill: list[str], skill_path: str, variant_out: str) -> RunResult:
    """Run eval_only.py for one skill file (no template editing)."""
    name = os.path.splitext(os.path.basename(skill_path))[0]
    res = RunResult(name=name, selected=[], rule_idx=None, out_root=variant_out)
    summary_path = os.path.join(variant_out, "eval_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as f:
                s = json.load(f)
            res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
            print(f"  [cache] {name}: hard={res.hard}  (reused)")
            return res
        except Exception:
            pass
    cmd = list(base_cmd_no_skill) + ["--skill", skill_path, "--out_root", variant_out]
    print(f"\n  [run] {name}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        res.error = f"eval_only exit {proc.returncode}"
        print(f"  [fail] {name}: {res.error}")
        return res
    with open(summary_path, encoding="utf-8") as f:
        s = json.load(f)
    res.hard, res.soft, res.n_items = s.get("hard"), s.get("soft"), s.get("n_items")
    print(f"  [done] {name}: hard={res.hard:.4f} soft={res.soft:.4f} "
          f"n={res.n_items} ({time.time() - t0:.0f}s)")
    return res


def _discover_versions(versions_dir: str) -> list[str]:
    files = [os.path.join(versions_dir, f) for f in os.listdir(versions_dir)
             if re.match(r"skill_v\d+\.md$", f)]
    files.sort(key=lambda p: int(re.search(r"v(\d+)", os.path.basename(p)).group(1)))
    best = os.path.join(versions_dir, "best_skill.md")
    if not os.path.exists(best):
        best = os.path.join(os.path.dirname(versions_dir.rstrip("/\\")), "best_skill.md")
    if os.path.exists(best):
        files.append(os.path.abspath(best))
    return files


def run_versions(versions_dir: str, base_cmd: list[str], out_root: str, dry_run: bool) -> None:
    files = _discover_versions(versions_dir)
    if not files:
        raise ValueError(f"No skill_v*.md found under {versions_dir!r}")
    # base_cmd was built with a placeholder --skill; strip it so we can set per-file
    cmd: list[str] = []
    skip = False
    for tok in base_cmd:
        if skip:
            skip = False
            continue
        if tok == "--skill":
            skip = True
            continue
        cmd.append(tok)

    print(f"  Found {len(files)} skill version(s): {[os.path.basename(f) for f in files]}")
    if dry_run:
        return
    import hashlib
    results: list[RunResult] = []
    by_hash: dict[str, RunResult] = {}
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h in by_hash:
            # identical skill content: reuse the prior score (a gate-rejected step)
            prev = by_hash[h]
            r = RunResult(name=name, selected=[], rule_idx=None,
                          hard=prev.hard, soft=prev.soft, n_items=prev.n_items,
                          out_root=prev.out_root)
            print(f"  [dedup] {name}: identical to {prev.name} -> hard={r.hard}")
            results.append(r)
            continue
        r = _run_eval_skill(cmd, os.path.abspath(path), os.path.join(out_root, name))
        by_hash[h] = r
        results.append(r)

    # report: per-version score + marginal delta vs previous version
    csv_path = os.path.join(out_root, "version_curve.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "hard", "soft", "n_items", "marginal_delta"])
        prev = None
        for r in results:
            delta = (r.hard - prev) if (prev is not None and r.hard is not None) else None
            w.writerow([r.name, r.hard, r.soft, r.n_items, delta])
            if r.hard is not None and not r.name.startswith("best"):
                prev = r.hard
    with open(os.path.join(out_root, "version_curve.json"), "w", encoding="utf-8") as f:
        json.dump([vars(r) for r in results], f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}\n  SKILL VERSION CURVE\n{'=' * 70}")
    print(f"  {'version':<16} {'hard':>8} {'Δ vs prev':>10}")
    prev = None
    for r in results:
        delta = (r.hard - prev) if (prev is not None and r.hard is not None) else None
        ds = "    --   " if delta is None else f"{delta:+.4f}"
        hs = " n/a" if r.hard is None else f"{r.hard:.4f}"
        print(f"  {r.name:<16} {hs:>8} {ds:>10}")
        if r.hard is not None and not r.name.startswith("best"):
            prev = r.hard
    print(f"\n  CSV: {csv_path}\n{'=' * 70}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentence-level skill ablation")
    p.add_argument("--template",
                   help="Prompt template file containing the ablatable section "
                        "(required unless --versions-dir is given)")
    p.add_argument("--versions-dir",
                   help="Process-level mode: eval every skill_v*.md (+ best_skill.md) "
                        "in this dir; produces a score-vs-version curve")
    p.add_argument("--section", default="## Rules",
                   help="Markdown heading of the section to ablate (default: '## Rules')")
    p.add_argument("--modes", nargs="+", default=["full", "bare", "loo", "addone"],
                   choices=["full", "bare", "loo", "addone"])
    p.add_argument("--out-root", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print each rendered variant and exit (no evaluation)")
    # eval_only passthrough
    p.add_argument("--config", required=True)
    p.add_argument("--skill", default="outputs/empty_skill.md")
    p.add_argument("--split", default="valid_unseen")
    p.add_argument("--split_dir", default="")
    p.add_argument("--target_model", default="")
    p.add_argument("--azure_openai_endpoint", default="")
    p.add_argument("--azure_openai_api_key", default="")
    p.add_argument("--azure_openai_auth_mode", default="")
    p.add_argument("--eval-arg", dest="eval_args", action="append", default=[],
                   help="Extra --cfg-options entry, repeatable (e.g. env.workers=12)")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def build_base_eval_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [args.python, os.path.join("scripts", "eval_only.py"),
           "--config", args.config, "--skill", args.skill, "--split", args.split]
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


def main() -> None:
    args = parse_args()
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    base_cmd = build_base_eval_cmd(args)

    # ── Process-level mode: score-vs-version curve ──
    if args.versions_dir:
        run_versions(os.path.abspath(args.versions_dir), base_cmd, out_root, args.dry_run)
        return

    # ── Sentence-level mode: template-section ablation ──
    if not args.template:
        raise SystemExit("--template is required unless --versions-dir is given")
    template_path = os.path.abspath(args.template)
    with open(template_path, encoding="utf-8") as f:
        original = f.read()

    parsed = parse_template(original, args.section)
    print(f"  Parsed {len(parsed.rules)} rule(s) from {args.section!r} in {template_path}")

    # sanity: full reconstruction should match the original semantics
    if render(parsed, list(range(len(parsed.rules)))).strip() != original.strip():
        print("  [warn] full-variant reconstruction differs from original "
              "(whitespace only is OK; check --dry-run).")

    variants = build_variants(len(parsed.rules), args.modes)
    print(f"  Planned {len(variants)} variant(s): {[v.name for v in variants]}")

    results: list[RunResult] = []
    try:
        for v in variants:
            results.append(run_eval(v, parsed, template_path, base_cmd, out_root, args.dry_run))
    finally:
        # guarantee restore
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(original)

    if args.dry_run:
        return
    write_report(parsed, results, out_root)


if __name__ == "__main__":
    main()
