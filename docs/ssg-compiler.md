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

## Folder-form (structured) skill — where SSG stops degenerating

`scripts/skill_compiler.py` also accepts a **folder** (`SKILL.md` + `scripts/` +
`references/` + `assets/`): `SKILL.md` is split into section units and every file
under `scripts/`/`references/`/`assets/` becomes one whole-file unit, so
cross-file `call` / `link` / `def-use` edges fire.

We faithfully expanded the SSB **seed** (`skillopt/envs/spreadsheetbench/skills/initial.md`)
into folder form at `ssg/xlsx-skill/` (content unchanged; implicit references
physicalised into real files + explicit paths/links), then compiled it:

`python scripts/skill_compiler.py --skill ssg/xlsx-skill --out-root outputs/ssg_compile_xlsx --dot`

| | flat ckpt (67u) | **structured xlsx (27u)** |
|---|--:|--:|
| τ types present | 3 | **5 (all)** |
| D edges | 8 | **13** |
| D edge kinds | 2 (def-use, call) | **4 (def-use, reference, call, link)** |
| D-isolated units | 87% | **52%** |

Extracted D edges (all deterministic, not hand-authored): Explore-step → `inspect.py`
(call), Write-step → `template.py` (call), `INPUT_PATH`/`OUTPUT_PATH` mentions →
`template.py` (def-use), Choose-library → §Library Selection (reference), Warning →
`references/openpyxl-pitfalls.md` (link), Worked-Example → §Common Workflow +
scripts (reference/call). Acyclic; topological order exists.

**On the structured skill D is non-trivial and connected**, so a precedence-
constrained Winter value will actually differ from the flat Shapley baseline. This
`ssg/xlsx-skill/` is the substrate for the SSG estimator (next step) and, later,
the optional injection experiment.

