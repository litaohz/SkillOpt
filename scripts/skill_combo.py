#!/usr/bin/env python3
"""Interaction-aware (combinatorial) skill attribution — Direction 4 / D.

First-order attribution (LOO / add-one) measures each unit in ONE fixed context,
so it mis-judges two cases:
  * redundancy / substitutes: A,B do the same job → LOO(A)=LOO(B)=0 (each covered
    by the other) → first-order calls both useless, but removing BOTH hurts.
  * synergy / complements: A only helps when B present → add-one(A)≈0 on empty.

This tool measures *interactions* by evaluating subsets where MORE than one unit
is changed at once. Two methods:

  pairwise  — second-order interaction for a targeted set of units U:
              I(i,j) = s(full) - s(full\\{i}) - s(full\\{j}) + s(full\\{i,j})
              I < 0 → substitutes/redundant (removing both costs less than the
              sum of singles); I > 0 → complements/synergy. Reuses cached
              s(full) and s(full\\{i}) from a first-order run (--reuse-from);
              only the double-removals s(full\\{i,j}) are new.

  shapley   — Monte-Carlo Shapley over all units (budget --perms), the
              interaction-averaged value φ_i. Expensive; use a cheaper eval
              subset for the search.

All subset evaluations are cached on disk by skill-text hash (shared across
methods and runs), so nothing is ever evaluated twice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import OrderedDict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from skill_attribution import split_units, render, build_base_eval_cmd  # noqa: E402


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvalCache:
    """Persistent skill-text-hash -> score cache (shared across methods/runs)."""

    def __init__(self, out_root: str, base_cmd: list[str]):
        self.out_root = out_root
        self.base_cmd = base_cmd
        self.path = os.path.join(out_root, "eval_cache.jsonl")
        self.data: dict[str, dict] = {}
        os.makedirs(out_root, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self.data[rec["hash"]] = rec

    def backfill(self, reuse_dir: str) -> int:
        """Import (skill.md, eval_summary.json) pairs from a prior run dir."""
        n = 0
        if not reuse_dir or not os.path.isdir(reuse_dir):
            return n
        for name in os.listdir(reuse_dir):
            d = os.path.join(reuse_dir, name)
            sp = os.path.join(d, "skill.md")
            ep = os.path.join(d, "eval_summary.json")
            if os.path.exists(sp) and os.path.exists(ep):
                with open(sp, encoding="utf-8") as f:
                    h = _hash(f.read())
                if h in self.data:
                    continue
                with open(ep, encoding="utf-8") as f:
                    s = json.load(f)
                self._store(h, s.get("hard"), s.get("soft"), s.get("n_items"))
                n += 1
        return n

    def _store(self, h: str, hard, soft, n_items) -> None:
        rec = {"hash": h, "hard": hard, "soft": soft, "n_items": n_items}
        self.data[h] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def score(self, skill_text: str, name: str, dry_run: bool) -> float | None:
        h = _hash(skill_text)
        if h in self.data:
            return self.data[h].get("hard")
        if dry_run:
            print(f"  [dry] {name}: would eval {len(skill_text)} chars (hash {h[:8]})")
            return None
        variant_out = os.path.join(self.out_root, "evals", h[:12])
        os.makedirs(variant_out, exist_ok=True)
        skill_path = os.path.join(variant_out, "skill.md")
        with open(skill_path, "w", encoding="utf-8") as f:
            f.write(skill_text)
        cmd = list(self.base_cmd) + ["--skill", skill_path, "--out_root", variant_out]
        print(f"  [run] {name} (hash {h[:8]})")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_PROJECT_ROOT)
        if proc.returncode != 0:
            print(f"  [fail] {name}: exit {proc.returncode}")
            return None
        with open(os.path.join(variant_out, "eval_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        self._store(h, s.get("hard"), s.get("soft"), s.get("n_items"))
        print(f"  [done] {name}: hard={s.get('hard'):.4f} ({time.time() - t0:.0f}s)")
        return s.get("hard")


def parse_units_arg(arg: str, n: int) -> list[int]:
    if not arg or arg.lower() == "all":
        return list(range(n))
    out = []
    for tok in arg.replace(" ", "").split(","):
        if tok:
            out.append(int(tok))
    return out


def _flat_order(all_idx: list[int], rng: random.Random) -> list[int]:
    order = list(all_idx)
    rng.shuffle(order)
    return order


def _stratified_order(groups: list[list[int]], rng: random.Random) -> list[int]:
    """Owen sampling: shuffle group order, then units within each group, keeping
    each group's units contiguous in the permutation."""
    order: list[int] = []
    gs = [list(g) for g in groups]
    rng.shuffle(gs)
    for g in gs:
        rng.shuffle(g)
        order.extend(g)
    return order


def _section_groups(units: list[str]) -> tuple[list[list[int]], list[int]]:
    """Group unit indices by section — each markdown header starts a new section.
    Returns (groups, section_id_per_unit)."""
    ids: list[int] = []
    sid = 0
    seen_header = False
    for u in units:
        first = u.lstrip().split("\n", 1)[0]
        if re.match(r"^#+\s", first):
            if seen_header:
                sid += 1
            seen_header = True
        ids.append(sid)
    gm: "OrderedDict[int, list[int]]" = OrderedDict()
    for idx, s in enumerate(ids):
        gm.setdefault(s, []).append(idx)
    return list(gm.values()), ids


def _collect_marginals(order_fn, seeds, perms, empty, s_keep, n):
    """Accumulate per-unit marginal-contribution samples over permutations.
    order_fn(rng) -> a permutation of unit indices."""
    samples: list[list[float]] = [[] for _ in range(n)]
    n_perms_total = 0
    for seed in seeds:
        rng = random.Random(seed)
        for p in range(perms):
            order = order_fn(rng)
            prev = empty
            S: list[int] = []
            for idx in order:
                S.append(idx)
                cur = s_keep(S, f"seed{seed}_perm{p}_prefix{len(S)}")
                if None not in (cur, prev):
                    samples[idx].append(cur - prev)
                prev = cur
            n_perms_total += 1
    return samples, n_perms_total


def _unit_stats(samples, all_idx, units):
    """Per-unit mean +/- SE from marginal samples. Returns (rows, sum_of_phi)."""
    rows = []
    total = 0.0
    for i in all_idx:
        vals = samples[i]
        phi = statistics.fmean(vals) if vals else None
        se = (statistics.stdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else None
        if phi is not None:
            total += phi
        rows.append({"unit": i, "phi": phi, "se": se, "n": len(vals),
                     "text": units[i].replace("\n", " ")})
    return rows, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Interaction-aware skill attribution")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--method", choices=["pairwise", "shapley", "owen"], required=True)
    ap.add_argument("--units", default="all",
                    help="pairwise: comma-separated unit indices to test (e.g. 0,3,4,6)")
    ap.add_argument("--perms", type=int, default=20,
                    help="shapley: # MC permutations per seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="",
                    help="shapley: comma-separated seeds for multi-seed SE estimation "
                         "(e.g. 0,1,2); overrides --seed. Total perms = len(seeds)*--perms.")
    ap.add_argument("--reuse-from", default="",
                    help="prior attribution out_root to import cached evals from")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    # eval passthrough (same flags as skill_attribution)
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
    units = split_units(skill_text)
    n = len(units)
    out_root = os.path.abspath(args.out_root)
    base_cmd = build_base_eval_cmd(args)
    cache = EvalCache(out_root, base_cmd)
    imported = cache.backfill(os.path.abspath(args.reuse_from)) if args.reuse_from else 0
    print(f"  Parsed {n} units; cache has {len(cache.data)} entries (imported {imported}).")

    all_idx = list(range(n))

    def s_keep(keep: list[int], name: str) -> float | None:
        return cache.score(render(units, sorted(keep)), name, args.dry_run)

    if args.method == "pairwise":
        U = parse_units_arg(args.units, n)
        print(f"  Pairwise over units {U}  ({len(U)*(len(U)-1)//2} doubles)")
        full = s_keep(all_idx, "full")
        singles = {i: s_keep([j for j in all_idx if j != i], f"loo_{i}") for i in U}
        rows = []
        for i, j in itertools.combinations(U, 2):
            both = s_keep([k for k in all_idx if k not in (i, j)], f"loo_{i}_{j}")
            inter = None
            if None not in (full, singles[i], singles[j], both):
                inter = full - singles[i] - singles[j] + both
            rows.append({"i": i, "j": j,
                         "loo_i": None if singles[i] is None else round(full - singles[i], 4),
                         "loo_j": None if singles[j] is None else round(full - singles[j], 4),
                         "both_removed_score": both,
                         "interaction": None if inter is None else round(inter, 4)})
        if args.dry_run:
            return
        csv_path = os.path.join(out_root, "pairwise.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i", "j", "loo_i", "loo_j", "both_removed_score", "interaction", "verdict"])
            for r in rows:
                inter = r["interaction"]
                verdict = "?" if inter is None else (
                    "redundant/substitutes" if inter < -0.01 else
                    "synergy/complements" if inter > 0.01 else "independent")
                w.writerow([r["i"], r["j"], r["loo_i"], r["loo_j"],
                            r["both_removed_score"], inter, verdict])
        print(f"\n{'='*70}\n  PAIRWISE INTERACTIONS (full={full:.4f})\n{'='*70}")
        print(f"  {'i,j':>7} {'LOO_i':>7} {'LOO_j':>7} {'both':>7} {'I(i,j)':>8}  verdict")
        for r in rows:
            inter = r["interaction"]
            verdict = "?" if inter is None else (
                "redundant" if inter < -0.01 else "synergy" if inter > 0.01 else "independent")
            print(f"  {str(r['i'])+','+str(r['j']):>7} {r['loo_i']!s:>7} {r['loo_j']!s:>7} "
                  f"{r['both_removed_score']!s:>7} {inter!s:>8}  {verdict}")
        print(f"\n  CSV: {csv_path}\n{'='*70}")
        print("  reading: I<0 ⇒ the two are substitutes (keep one, drop the other);")
        print("           I>0 ⇒ complements (keep together); I≈0 ⇒ independent.")

    elif args.method == "shapley":
        seeds = ([int(s) for s in args.seeds.replace(" ", "").split(",") if s]
                 if args.seeds else [args.seed])
        empty = cache.score("", "empty", args.dry_run)
        full = s_keep(all_idx, "full")
        samples, n_perms_total = _collect_marginals(
            lambda rng: _flat_order(all_idx, rng), seeds, args.perms, empty, s_keep, n)
        if args.dry_run:
            return
        rows, total_phi = _unit_stats(samples, all_idx, units)
        csv_path = os.path.join(out_root, "shapley.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit", "shapley", "se", "samples", "text"])
            for r in rows:
                w.writerow([r["unit"],
                            None if r["phi"] is None else round(r["phi"], 4),
                            None if r["se"] is None else round(r["se"], 4),
                            r["n"], r["text"]])
        print(f"\n{'='*70}\n  MONTE-CARLO SHAPLEY "
              f"({len(seeds)} seed(s) x {args.perms} perms = {n_perms_total})\n{'='*70}")
        print(f"  {'unit':>4} {'phi':>8} {'±SE':>7} {'n':>4}  text")
        for r in rows:
            t = (r["text"][:50] + "…") if len(r["text"]) > 51 else r["text"]
            phi = "None" if r["phi"] is None else f"{r['phi']:+.4f}"
            se = "None" if r["se"] is None else f"{r['se']:.4f}"
            print(f"  {r['unit']:>4} {phi:>8} {se:>7} {r['n']:>4}  {t}")
        if None not in (full, empty):
            target = full - empty
            diff = total_phi - target
            verdict = "PASS" if abs(diff) < 1e-6 else f"drift {diff:+.4f} (incomplete perms?)"
            print(f"\n  Efficiency: Σφ={total_phi:+.4f}  v(full)−v(empty)={target:+.4f}  → {verdict}")
        print(f"\n  CSV: {csv_path}\n{'='*70}")

    elif args.method == "owen":
        seeds = ([int(s) for s in args.seeds.replace(" ", "").split(",") if s]
                 if args.seeds else [args.seed])
        groups, sec_ids = _section_groups(units)
        sec_label = {}
        for g in groups:
            head = units[g[0]].lstrip().split("\n", 1)[0]
            sec_label[sec_ids[g[0]]] = head[:48]
        print(f"  Owen: {len(groups)} sections (a priori unions): "
              f"{[len(g) for g in groups]} units each")
        empty = cache.score("", "empty", args.dry_run)
        full = s_keep(all_idx, "full")
        samples, n_perms_total = _collect_marginals(
            lambda rng: _stratified_order(groups, rng), seeds, args.perms, empty, s_keep, n)
        if args.dry_run:
            return
        rows, total_phi = _unit_stats(samples, all_idx, units)
        # Section-level (quotient game) value = sum of Owen values within the union.
        sec_phi: dict[int, float] = {}
        for r in rows:
            s = sec_ids[r["unit"]]
            if r["phi"] is not None:
                sec_phi[s] = sec_phi.get(s, 0.0) + r["phi"]
        csv_path = os.path.join(out_root, "owen.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit", "section", "owen", "se", "samples", "text"])
            for r in rows:
                w.writerow([r["unit"], sec_ids[r["unit"]],
                            None if r["phi"] is None else round(r["phi"], 4),
                            None if r["se"] is None else round(r["se"], 4),
                            r["n"], r["text"]])
        sec_csv = os.path.join(out_root, "owen_sections.csv")
        with open(sec_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "section_owen", "label"])
            for s in sorted(sec_phi):
                w.writerow([s, round(sec_phi[s], 4), sec_label.get(s, "")])
        print(f"\n{'='*70}\n  OWEN VALUE (two-level Shapley, sections=unions) "
              f"({len(seeds)} seed(s) x {args.perms} perms = {n_perms_total})\n{'='*70}")
        print("  -- section-level (quotient game) --")
        for s in sorted(sec_phi):
            print(f"  sec{s:>2} {sec_phi[s]:+.4f}  {sec_label.get(s, '')}")
        print("  -- unit-level (within-union Owen) --")
        print(f"  {'unit':>4} {'sec':>3} {'owen':>8} {'±SE':>7} {'n':>4}  text")
        for r in rows:
            t = (r["text"][:46] + "…") if len(r["text"]) > 47 else r["text"]
            phi = "None" if r["phi"] is None else f"{r['phi']:+.4f}"
            se = "None" if r["se"] is None else f"{r['se']:.4f}"
            print(f"  {r['unit']:>4} {sec_ids[r['unit']]:>3} {phi:>8} {se:>7} {r['n']:>4}  {t}")
        if None not in (full, empty):
            target = full - empty
            diff = total_phi - target
            verdict = "PASS" if abs(diff) < 1e-6 else f"drift {diff:+.4f} (incomplete perms?)"
            print(f"\n  Efficiency: Σφ={total_phi:+.4f}  v(full)−v(empty)={target:+.4f}  → {verdict}")
        print(f"\n  CSV: {csv_path}  |  sections: {sec_csv}\n{'='*70}")


if __name__ == "__main__":
    main()
