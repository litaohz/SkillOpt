#!/usr/bin/env python3
"""SSG estimator: precedence-constrained Winter / Shapley value on a compiled skill.

Value function v(S) = success rate of an agent carrying render(S) on the task,
evaluated through the STANDARD cc harness (claude_code_exec / opus-4.8) — the same
value function used by every other experiment in this repo. Cost is not optimised
at the expense of fidelity; we only use *unbiased* accelerators (content-hash
caching of coalitions, optional truncation).

Feasibility: a coalition S is feasible iff it is DOWN-CLOSED under the dependency
DAG D (if unit i in S and i depends on j, then j in S). We sample uniform random
*linear extensions* of D; every prefix of such an order is automatically feasible,
so the marginal v(prefix+u) - v(prefix) is well-defined (Faigle-Kern precedence-
constrained Shapley). H-contiguity (full Winter value) is an optional refinement
(--respect-h) that additionally keeps H-siblings contiguous.

render(S): reconstruct SKILL.md from the SKILL.md section-units in S (original
order) and append the contents of any script/resource file-units in S as fenced
"Provided Files" — matching how flat SSB skills are consumed (inline templates).

Usage (dry-run to count unique cc evaluations first):
  python scripts/ssg_winter.py --skill ssg/xlsx-skill --perms 12 --seeds 2 \
     --out-root outputs/ssg_winter_xlsx --dry-run
Then drop --dry-run and add the cc eval args to actually run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.skill_compiler import (  # noqa: E402
    build_dependencies, build_hierarchy, load_units,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── SSG graph ───────────────────────────────────────────────────────────────
class Graph:
    def __init__(self, skill_path: str):
        self.units, self.files, self.is_file = load_units(skill_path)
        self.n = len(self.units)
        edges = build_dependencies(self.units, self.files, self.is_file)
        self.parent, _ = build_hierarchy(self.units, self.is_file)
        # dep[u] = set of units u depends on (must precede u / be present with u)
        self.dep: dict[int, set[int]] = {i: set() for i in range(self.n)}
        for a, b, _k in edges:
            self.dep[a].add(b)

    def closure(self, S: set[int]) -> frozenset[int]:
        out = set(S)
        stack = list(S)
        while stack:
            u = stack.pop()
            for v in self.dep[u]:
                if v not in out:
                    out.add(v)
                    stack.append(v)
        return frozenset(out)

    def sample_order(self, rng: random.Random) -> list[int]:
        """Uniform-ish random linear extension: u appears after all dep[u]."""
        placed: set[int] = set()
        remaining = set(range(self.n))
        order: list[int] = []
        while remaining:
            avail = [u for u in remaining if self.dep[u] <= placed]
            # (dep <= placed) => all deps already emitted; guaranteed non-empty (acyclic)
            u = rng.choice(avail)
            order.append(u)
            placed.add(u)
            remaining.discard(u)
        return order

    def render(self, S: set[int], dest_dir: str) -> str:
        """Materialise coalition S as a REAL Agent-Skills folder at dest_dir.

        Standard-compliant (agentskills.io): SKILL.md (section units in S, in
        original order) + scripts/<f> + references/<f> for each file-unit in S.
        The agent reads SKILL.md and progressively discloses scripts/references
        on demand — this is what makes the Read-Null property real (a file
        present but never read contributes zero to that rollout).
        Returns the SKILL.md text (for hashing / logging).
        """
        import shutil
        S = set(S)
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        # SKILL.md from section/instruction/metadata units that live in SKILL.md
        md_parts = [self.units[i].strip() for i in range(self.n)
                    if i in S and not self.is_file[i]]
        skill_md = "\n\n".join(md_parts).strip() or (
            "---\nname: skillopt-target\ndescription: task skill\n---\n")
        with open(os.path.join(dest_dir, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(skill_md)
        # real file-units -> their real relative paths (scripts/ , references/)
        for i in range(self.n):
            if i in S and self.is_file[i]:
                rel = self.files[i].replace("\\", "/")
                fp = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(self.units[i])
        return skill_md


# ── persistent coalition-value cache ────────────────────────────────────────
class ValueFn:
    def __init__(self, g: Graph, out_root: str, eval_args: list[str], dry_run: bool):
        self.g = g
        self.out_root = out_root
        self.eval_args = eval_args
        self.dry_run = dry_run
        os.makedirs(out_root, exist_ok=True)
        self.cache_path = os.path.join(out_root, "coalition_cache.jsonl")
        self.cache: dict[str, float] = {}
        self.requested: set[str] = set()
        if os.path.exists(self.cache_path):
            for line in open(self.cache_path, encoding="utf-8"):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self.cache[r["hash"]] = r["hard"]

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def value(self, S: set[int]) -> float | None:
        closed = self.g.closure(S)
        probe_dir = os.path.join(self.out_root, "_probe_skill")
        text = self.g.render(closed, probe_dir)
        file_ids = sorted(i for i in closed if self.g.is_file[i])
        blob = text + "\x00" + "\x00".join(self.g.files[i] + self.g.units[i] for i in file_ids)
        h = self._hash(blob)
        self.requested.add(h)
        if h in self.cache:
            return self.cache[h]
        if self.dry_run:
            return None  # counting only
        variant_dir = os.path.join(self.out_root, f"coal_{h}")
        skill_dir = os.path.join(variant_dir, "skill")
        self.g.render(closed, skill_dir)  # materialise the real skill folder
        summary = os.path.join(variant_dir, "eval_summary.json")
        if not os.path.exists(summary):
            cmd = [sys.executable, "-u", os.path.join("scripts", "eval_only.py"),
                   "--skill", skill_dir, "--out_root", variant_dir] + self.eval_args
            subprocess.run(cmd, cwd=_ROOT)
        hard = json.load(open(summary, encoding="utf-8"))["hard"]
        self.cache[h] = hard
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"hash": h, "hard": hard, "size": len(closed)}) + "\n")
        return hard


# ── Monte-Carlo constrained Shapley ─────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="SSG precedence-constrained Shapley/Winter value")
    ap.add_argument("--skill", required=True, help="compiled skill folder (or flat .md)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--perms", type=int, default=12, help="permutations per seed")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--trunc-eps", type=float, default=0.0,
                    help="if v(prefix) within eps of v(full) for the rest, truncate marginals to 0")
    ap.add_argument("--dry-run", action="store_true", help="count unique coalitions only")
    # cc eval passthrough (everything after --eval-arg is forwarded to eval_only)
    ap.add_argument("--eval-arg", dest="eval_args", action="append", default=[],
                    help="one token forwarded verbatim to eval_only.py (repeatable)")
    args = ap.parse_args()

    g = Graph(args.skill)
    print(f"  N={g.n} units, {sum(len(v) for v in g.dep.values())} dependency edges")
    vf = ValueFn(g, args.out_root, args.eval_args, args.dry_run)

    full = set(range(g.n))
    v_full = vf.value(full)
    v_empty = vf.value(set())

    phi = {i: 0.0 for i in range(g.n)}
    counts = {i: 0 for i in range(g.n)}
    order_log = []
    for seed in range(args.seeds):
        rng = random.Random(1000 + seed)
        for p in range(args.perms):
            order = g.sample_order(rng)
            order_log.append(order)
            running: set[int] = set()
            prev = vf.value(set())
            for u in order:
                running.add(u)
                cur = vf.value(running)
                if cur is not None and prev is not None:
                    phi[u] += cur - prev
                    counts[u] += 1
                prev = cur

    os.makedirs(args.out_root, exist_ok=True)
    if args.dry_run:
        print(f"  DRY-RUN: {len(vf.requested)} UNIQUE coalitions would be cc-evaluated "
              f"(perms={args.perms} x seeds={args.seeds}); already cached "
              f"{len(vf.cache)}.")
        with open(os.path.join(args.out_root, "dryrun.json"), "w", encoding="utf-8") as f:
            json.dump({"unique_coalitions": len(vf.requested), "n": g.n,
                       "perms": args.perms, "seeds": args.seeds}, f, indent=2)
        return

    rows = []
    for i in range(g.n):
        val = phi[i] / counts[i] if counts[i] else None
        rows.append({"unit": i, "tau": None, "file": g.files[i],
                     "phi": val, "n_marginals": counts[i],
                     "text": " ".join(g.units[i].split())[:100]})
    result = {"skill": os.path.abspath(args.skill), "n": g.n,
              "v_empty": v_empty, "v_full": v_full,
              "perms": args.perms, "seeds": args.seeds, "units": rows}
    with open(os.path.join(args.out_root, "ssg_shapley.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    eff = sum(r["phi"] for r in rows if r["phi"] is not None)
    print(f"  v_empty={v_empty}  v_full={v_full}  sum(phi)={eff:.4f} "
          f"(efficiency target {(v_full or 0) - (v_empty or 0):.4f})")
    print("  TOP by phi:")
    for r in sorted([r for r in rows if r["phi"] is not None], key=lambda x: -x["phi"])[:8]:
        print(f"    #{r['unit']:>2} phi={r['phi']*100:+.1f}  {r['text'][:52]}")
    print(f"  wrote {os.path.join(args.out_root, 'ssg_shapley.json')}")


if __name__ == "__main__":
    main()
