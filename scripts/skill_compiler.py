#!/usr/bin/env python3
"""Skill compiler: compile a skill document into the SSG triple G = (N, D, H).

- N = units (via ``skill_combo.split_units``) with a type label tau(i) in
      {metadata, instruction, example, script, resource}.
- H = hierarchy tree from markdown header nesting (# > ## > ### ...); every
      non-header unit attaches to its enclosing section header.
- D = dependency DAG from deterministic static-analysis rules:
        * def-use    : a CONST_TOKEN (e.g. OUTPUT_PATH) assigned in one unit
                       (``TOKEN = ...``) and mentioned in another  (use -> def).
        * reference  : a unit's text contains another header unit's exact title
                       (section cross-reference).           (use -> referenced)
        * call       : inline ``python <path>`` / ``scripts/<file>`` where a
                       script/resource unit with that path exists. (caller -> script)
        * link       : markdown link ``[..](path)`` to a resource unit. (use -> resource)

This is a *flat-skill* compiler (single .md). Folder-form skills (SKILL.md +
scripts/ + references/) are a superset handled by the same rules once each file
becomes a unit; see docs/shapley-methodology-and-plan.md (SSG / P7).

Usage:
    python scripts/skill_compiler.py --skill ckpt/spreadsheetbench/gpt5.5_skill.md \
        --out-root outputs/ssg_compile_ssb [--dot]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.skill_combo import split_units  # noqa: E402

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
CONST_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+)\b")  # OUTPUT_PATH, INPUT_PATH
ASSIGN_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+)\s*=")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PATH_RE = re.compile(r"(?:python\s+|`)?((?:scripts|references)/[\w./-]+|\b[\w-]+\.py)\b")


def header_of(unit: str):
    """Return (level, title) if the unit starts with a markdown header, else None."""
    first = unit.strip().splitlines()[0] if unit.strip() else ""
    m = HEADER_RE.match(first)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def classify(unit: str) -> str:
    """Assign a type tau(i) to a unit with deterministic heuristics."""
    s = unit.strip()
    low = s.lower()
    h = header_of(unit)
    if h is not None:
        # top title / frontmatter-ish
        if h[0] == 1 and ("skill" in low or "frontmatter" in low):
            return "metadata"
        return "section"  # structural header node in H (not a leaf instruction)
    if s.startswith("```") or "```" in s:
        # code block: script template vs worked example
        if any(t in low for t in ("output_path", "input_path", "openpyxl.load", "wb.save", "import ")):
            return "script"
        return "example"
    if low.startswith("<!--") and low.endswith("-->"):
        return "metadata"
    if any(t in low for t in ("for example", "worked example", "e.g.,")) and "```" in s:
        return "example"
    return "instruction"


def build_hierarchy(units: list[str]):
    """Return parent[i] (index of enclosing header unit or None) + roots."""
    parent = [None] * len(units)
    # stack of (level, index) for open headers
    stack: list[tuple[int, int]] = []
    for i, u in enumerate(units):
        h = header_of(u)
        if h is not None:
            level = h[0]
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent[i] = stack[-1][1] if stack else None
            stack.append((level, i))
        else:
            parent[i] = stack[-1][1] if stack else None
    roots = [i for i in range(len(units)) if parent[i] is None]
    return parent, roots


def build_dependencies(units: list[str]):
    """Return list of edges (src, dst, kind): src depends on dst."""
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(a: int, b: int, kind: str):
        if a != b and (a, b, kind) not in seen:
            seen.add((a, b, kind))
            edges.append((a, b, kind))

    # ---- def-use on CONST tokens ----
    defs: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        for m in ASSIGN_RE.finditer(u):
            defs.setdefault(m.group(1), []).append(i)
    for i, u in enumerate(units):
        used = set(CONST_RE.findall(u))
        assigned = set(ASSIGN_RE.findall(u))
        for tok in used - assigned:
            for d in defs.get(tok, []):
                add(i, d, "def-use")

    # ---- section reference: unit mentions another header's title ----
    headers = {}
    for i, u in enumerate(units):
        h = header_of(u)
        if h is not None and len(h[1]) >= 4:
            headers[i] = h[1]
    for i, u in enumerate(units):
        body = u
        for j, title in headers.items():
            if i != j and title.lower() in body.lower() and header_of(u) is None:
                add(i, j, "reference")

    # ---- call / link to script or resource units by path ----
    path_units: dict[str, int] = {}
    for i, u in enumerate(units):
        h = header_of(u)
        # a unit that *is* a file path node (folder-form) — best effort
        for m in PATH_RE.finditer(u):
            path_units.setdefault(os.path.basename(m.group(1)), i)
    for i, u in enumerate(units):
        for m in LINK_RE.finditer(u):
            tgt = os.path.basename(m.group(1))
            if tgt in path_units:
                add(i, path_units[tgt], "link")
        for m in PATH_RE.finditer(u):
            tgt = os.path.basename(m.group(1))
            if tgt in path_units and path_units[tgt] != i:
                add(i, path_units[tgt], "call")

    return edges


def has_cycle(n: int, edges: list[tuple[int, int, str]]) -> bool:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for a, b, _ in edges:
        adj[a].append(b)
    color = [0] * n

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1 or (color[v] == 0 and dfs(v)):
                return True
        color[u] = 2
        return False

    return any(color[i] == 0 and dfs(i) for i in range(n))


def to_dot(units, parent, edges, taus) -> str:
    lines = ["digraph SSG {", "  rankdir=LR;", "  node [shape=box,fontsize=9];"]
    color = {"metadata": "gray", "section": "lightblue", "instruction": "white",
             "example": "lightyellow", "script": "lightgreen", "resource": "orange"}
    for i, u in enumerate(units):
        label = " ".join(u.split())[:34].replace('"', "'")
        lines.append(f'  n{i} [label="{i}: {label}",style=filled,'
                     f'fillcolor={color.get(taus[i], "white")}];')
    for i, p in enumerate(parent):
        if p is not None:
            lines.append(f"  n{p} -> n{i} [color=gray,style=dashed,arrowhead=none];")
    ekind = {"def-use": "red", "reference": "blue", "call": "green", "link": "purple"}
    for a, b, k in edges:
        lines.append(f'  n{a} -> n{b} [color={ekind.get(k, "black")},label="{k}"];')
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile a skill into SSG G=(N,D,H)")
    ap.add_argument("--skill", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dot", action="store_true", help="also emit graphviz .dot")
    args = ap.parse_args()

    with open(args.skill, encoding="utf-8") as f:
        units = split_units(f.read())
    n = len(units)
    taus = [classify(u) for u in units]
    parent, roots = build_hierarchy(units)
    edges = build_dependencies(units)
    cyclic = has_cycle(n, edges)

    os.makedirs(args.out_root, exist_ok=True)
    graph = {
        "skill": os.path.abspath(args.skill),
        "n_units": n,
        "acyclic": not cyclic,
        "units": [
            {"id": i, "tau": taus[i], "parent": parent[i],
             "is_header": header_of(units[i]) is not None,
             "text": " ".join(units[i].split())[:120]}
            for i in range(n)
        ],
        "H_edges": [[parent[i], i] for i in range(n) if parent[i] is not None],
        "D_edges": [{"src": a, "dst": b, "kind": k} for a, b, k in edges],
    }
    with open(os.path.join(args.out_root, "ssg_graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    if args.dot:
        with open(os.path.join(args.out_root, "ssg_graph.dot"), "w", encoding="utf-8") as f:
            f.write(to_dot(units, parent, edges, taus))

    # ---- summary ----
    from collections import Counter
    tau_counts = Counter(taus)
    kind_counts = Counter(k for _, _, k in edges)
    indeg = Counter(b for _, b, _ in edges)
    outdeg = Counter(a for a, _, _ in edges)
    isolated = [i for i in range(n) if indeg[i] == 0 and outdeg[i] == 0]
    print(f"Compiled {args.skill}")
    print(f"  N = {n} units   types = {dict(tau_counts)}")
    print(f"  H = tree with {sum(1 for p in parent if p is not None)} parent-edges, "
          f"{len(roots)} root(s)")
    print(f"  D = {len(edges)} dependency edges   kinds = {dict(kind_counts)}   "
          f"acyclic = {not cyclic}")
    print(f"  D-isolated units (no dep in/out): {len(isolated)}/{n}")
    print(f"  wrote {os.path.join(args.out_root, 'ssg_graph.json')}"
          + ("  + ssg_graph.dot" if args.dot else ""))


if __name__ == "__main__":
    main()
