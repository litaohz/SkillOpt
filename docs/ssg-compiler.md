# SSG Phase 1 — Skill compiler `G = (N, D, H)`

`scripts/skill_compiler.py` deterministically compiles a skill document into the
**Skill Structure Game** triple (per `~/research/skill-theory`):

- **N** — units (`skill_combo.split_units`) with a type `τ ∈ {metadata, section,
  instruction, example, script, resource}`.
- **H** — hierarchy tree from markdown header nesting (`#` > `##` > `###`); every
  non-header unit attaches to its enclosing section.
- **D** — dependency DAG from deterministic static-analysis rules:
  - `def-use`: a `CONST_TOKEN` (e.g. `OUTPUT_PATH`) assigned in one unit and
    mentioned in another → `use → def`.
  - `reference`: a unit contains another header's exact title → `use → section`.
  - `call` / `link`: inline `python <path>` / `scripts/<file>` or markdown
    `[..](path)` resolving to a script/resource unit → `caller → target`.

Output: `ssg_graph.json` (units + τ + H parent-edges + D edges + acyclicity) and
an optional Graphviz `ssg_graph.dot`.

## Result on the flat SSB ckpt skill (key finding)

`python scripts/skill_compiler.py --skill ckpt/spreadsheetbench/gpt5.5_skill.md --out-root outputs/ssg_compile_ssb --dot`

| | value |
|---|---|
| N | 67 (metadata 1, section 8, instruction 55, script 3) |
| H | 65 parent-edges, 2 roots — full tree |
| D | **8 edges** (5 def-use, 3 call), acyclic |
| **D-isolated units** | **58 / 67 (87%)** |

**The flat ckpt skill is D-degenerate.** 87% of units carry no dependency edges,
so a Skill-Shapley / precedence-constrained Winter value on it collapses toward
the flat Shapley we already computed (D≈∅ leaves only H's contiguity constraint).
This is exactly the theory's prediction: **flat skills degenerate SSG → plain
Shapley; the value of structural attribution only appears on skills with a rich
dependency DAG.**

## Implication for the SSG showcase

To demonstrate SSG's value (where D actually constrains feasible coalitions and
changes the marginals), we need **structured skills** — SKILL.md + `scripts/` +
`references/` with real call / def-use / link edges — matching theory Part 3's
**injection experiment** (controllable ground-truth defects). Two concrete routes:

1. **Expand the SSB seed** (`skillopt/envs/spreadsheetbench/skills/initial.md`)
   into folder form (theory doc-2's worked example: `scripts/inspect.py`,
   `scripts/template.py`, `references/openpyxl-pitfalls.md`, with cross-refs).
2. **Adopt SkillsBench-style structured skills** for the injection/precision-recall
   experiment (defect types: stale API, misleading rule, redundant paraphrase,
   over-constraining rule), with baselines LOO / add-one / LLM-judge / flat Shapley.
