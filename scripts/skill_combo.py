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
import subprocess
import sys
import time

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Interaction-aware skill attribution")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--method", choices=["pairwise", "shapley"], required=True)
    ap.add_argument("--units", default="all",
                    help="pairwise: comma-separated unit indices to test (e.g. 0,3,4,6)")
    ap.add_argument("--perms", type=int, default=20, help="shapley: # MC permutations")
    ap.add_argument("--seed", type=int, default=0)
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
        rng = random.Random(args.seed)
        phi = [0.0] * n
        cnt = [0] * n
        empty = cache.score("", "empty", args.dry_run)
        for p in range(args.perms):
            order = all_idx[:]
            rng.shuffle(order)
            prev = empty
            S: list[int] = []
            for idx in order:
                S.append(idx)
                cur = s_keep(S, f"perm{p}_prefix{len(S)}")
                if None not in (cur, prev):
                    phi[idx] += cur - prev
                    cnt[idx] += 1
                prev = cur
        if args.dry_run:
            return
        rows = [{"unit": i, "shapley": round(phi[i] / cnt[i], 4) if cnt[i] else None,
                 "samples": cnt[i], "text": units[i].replace("\n", " ")} for i in all_idx]
        csv_path = os.path.join(out_root, "shapley.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit", "shapley", "samples", "text"])
            for r in rows:
                w.writerow([r["unit"], r["shapley"], r["samples"], r["text"]])
        print(f"\n{'='*70}\n  MONTE-CARLO SHAPLEY ({args.perms} perms)\n{'='*70}")
        for r in rows:
            t = (r["text"][:54] + "…") if len(r["text"]) > 55 else r["text"]
            print(f"  {r['unit']:>3} {r['shapley']!s:>8} (n={r['samples']})  {t}")
        print(f"\n  CSV: {csv_path}\n{'='*70}")


if __name__ == "__main__":
    main()
