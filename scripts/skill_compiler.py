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


def classify(unit: str, relfile: str | None = None, is_file: bool = False) -> str:
    """Assign a type tau(i) to a unit with deterministic heuristics."""
    if is_file and relfile:
        d = relfile.replace("\\", "/").split("/")[0]
        if relfile.endswith(".py") or d == "scripts":
            return "script"
        if d in ("references", "assets") or relfile.endswith((".md", ".txt")):
            return "resource"
    s = unit.strip()
    low = s.lower()
    h = header_of(unit)
    if h is not None:
        if h[0] == 1 and ("skill" in low or "frontmatter" in low):
            return "metadata"
        return "section"
    if s.startswith("---") and "name:" in low and "description:" in low:
        return "metadata"
    if s.startswith("```") or "```" in s:
        if any(t in low for t in ("output_path", "input_path", "openpyxl.load", "wb.save", "import ")):
            return "script"
        return "example"
    if low.startswith("<!--") and low.endswith("-->"):
        return "metadata"
    return "instruction"


def load_units(path: str):
    """Return (texts, files, is_file) for a flat .md OR a folder-form skill.

    Folder form: SKILL.md is split into section units; every file under
    scripts/ , references/ , assets/ becomes one whole-file unit.
    """
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            units = split_units(f.read())
        base = os.path.basename(path)
        return units, [base] * len(units), [False] * len(units)

    texts: list[str] = []
    files: list[str] = []
    is_file: list[bool] = []
    # 1) SKILL.md (or skill.md) split into section units
    for cand in ("SKILL.md", "skill.md"):
        sp = os.path.join(path, cand)
        if os.path.isfile(sp):
            with open(sp, encoding="utf-8") as f:
                for u in split_units(f.read()):
                    texts.append(u)
                    files.append(cand)
                    is_file.append(False)
            break
    # 2) whole-file units for scripts/ references/ assets/
    for sub in ("scripts", "references", "assets"):
        d = os.path.join(path, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            fp = os.path.join(d, name)
            if os.path.isfile(fp):
                with open(fp, encoding="utf-8", errors="replace") as f:
                    texts.append(f.read())
                files.append(f"{sub}/{name}")
                is_file.append(True)
    return texts, files, is_file


def build_hierarchy(units: list[str], is_file: list[bool] | None = None):
    """Return parent[i] (index of enclosing header unit or None) + roots.

    File-units (whole scripts/resources) are treated as roots grouped by dir and
    do not participate in the SKILL.md header stack.
    """
    if is_file is None:
        is_file = [False] * len(units)
    parent = [None] * len(units)
    stack: list[tuple[int, int]] = []
    for i, u in enumerate(units):
        if is_file[i]:
            parent[i] = None
            continue
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


def build_dependencies(units: list[str], files: list[str] | None = None,
                       is_file: list[bool] | None = None):
    """Return list of edges (src, dst, kind): src depends on dst."""
    n = len(units)
    if files is None:
        files = [""] * n
    if is_file is None:
        is_file = [False] * n
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()

    def add(a: int, b: int, kind: str):
        if a != b and (a, b, kind) not in seen:
            seen.add((a, b, kind))
            edges.append((a, b, kind))

    # ---- def-use on CONST tokens (works across files: template.py defs) ----
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
        if header_of(u) is not None:
            continue
        for j, title in headers.items():
            if i != j and title.lower() in u.lower():
                add(i, j, "reference")

    # ---- call / link to script or resource file-units by path ----
    path_units: dict[str, int] = {}
    for i in range(n):
        if is_file[i] and files[i]:
            rel = files[i].replace("\\", "/")
            path_units[rel] = i
            path_units[os.path.basename(rel)] = i
    for i, u in enumerate(units):
        if is_file[i]:
            continue
        for m in LINK_RE.finditer(u):
            tgt = m.group(1).replace("\\", "/")
            j = path_units.get(tgt) or path_units.get(os.path.basename(tgt))
            if j is not None:
                add(i, j, "link")
        for m in PATH_RE.finditer(u):
            tgt = m.group(1).replace("\\", "/")
            j = path_units.get(tgt) or path_units.get(os.path.basename(tgt))
            if j is not None:
                add(i, j, "call")

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

    units, files, is_file = load_units(args.skill)
    n = len(units)
    taus = [classify(units[i], files[i], is_file[i]) for i in range(n)]
    parent, roots = build_hierarchy(units, is_file)
    edges = build_dependencies(units, files, is_file)
    cyclic = has_cycle(n, edges)

    os.makedirs(args.out_root, exist_ok=True)
    graph = {
        "skill": os.path.abspath(args.skill),
        "n_units": n,
        "acyclic": not cyclic,
        "units": [
            {"id": i, "tau": taus[i], "file": files[i], "parent": parent[i],
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
