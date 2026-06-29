# Research Narrative — "Self-Evolving" Skills Are Low-Rank

> Working research document tying the SkillOpt reproduction findings to the
> causal-attribution / global-optimization program (Direction 4).
> Status: living doc. Evidence rows marked ✅ are measured on the local ghc
> proxy with gpt-5.5 (deterministic EM); ⏳ are planned.

---

## Thesis

**Agentic "self-evolving skill" optimization has very low effective
dimensionality.** What the method actually learns concentrates in:

- **time** — only the first 1–2 gate-accepted edits (a couple of epochs), and
- **content** — only a few sentences of the skill document,

while the remaining compute (epochs, tokens) produces text with **≈0 or even
negative causal value**. A *causal-attribution* lens both **diagnoses** this
(interventional, interaction-aware measurement) and **fixes** it (causal-guided,
cost-aware, global optimization).

These two sparsities are **the same phenomenon on two axes**: the later epochs
do nothing *because* the units they add carry no causal value (redundant with
what already exists, or harmful).

---

## Part 1 — Diagnosis (what we measured)

| Axis | Finding | Evidence | Status |
|---|---|---|---|
| Baseline validity | repo's `no-skill` is inflated by an always-on template whose hardcoded Rules paraphrase the skill | bare-prompt A/B: OfficeQA EM 0.547→0.285, DocVQA ANLS 0.918→0.860 | ✅ |
| **Content sparsity** | within a skill, ~85% of the Rules' value is a single output-format line (`<answer>`); some rules are net-negative | sentence-level LOO/add-one (172 test) | ✅ |
| **Time sparsity** | 16 epochs / 32 steps: only steps 1–2 accepted; remaining 28 steps add 0 test EM while skill bloats 887→18,775 chars (21×) | version curve + step records (7.6h, 106M tokens) | ✅ |
| Accuracy-neutral bloat | final == best test EM (0.715) despite 21× size | summary.json | ✅ |
| Mechanism ① | epoch-end `slow_update` is **force-accepted, bypassing the EM gate** | `trainer.py:1894-1901` | ✅ |
| Mechanism ② | per-step gate is greedy strict-`>` on a tiny 24-item, rollout-noisy selection set → best locks in early | `gate.py:123`, step records | ✅ |

The method **degenerates to: reflect once or twice, update the skill, then GG.**

---

## Part 2 — The instrument: causal attribution (interventional, not correlational)

Replace "score went up after this edit" (correlational) with **do-operator**
measurement (ablate a unit / subset, re-evaluate).

- **First-order** (`scripts/skill_attribution.py`): LOO Δ (indispensability) and
  add-one Δ (standalone value) per skill unit. Shows the sparsity, but is fooled
  by **redundancy** (substitutes each look useless) and **synergy** (a unit that
  only helps in context looks useless).
- **Interaction-aware / combinatorial** (`scripts/skill_combo.py`): the real
  contribution. Pairwise interaction
  `I(i,j)=s(full)−s(\i)−s(\j)+s(\{i,j})` (I<0 ⇒ substitutes, I>0 ⇒ complements),
  and budgeted Monte-Carlo Shapley. Gives the **true minimal sufficient set**,
  not the first-order illusion. All subset evals are cached by skill-text hash
  and reused across methods/runs (cheap experiments only pay for new subsets).

> Cost discipline: "cheap" = fewer **subset evaluations** (targeted pairs,
> sampled Shapley, cache reuse), **not** fewer items — interactions are
> second-order differences and need item-level statistical power.

---

## Part 3 — The bridge experiment (welds time-sparsity to content-sparsity)

Take the units **added after epoch 2** (diff of `best_skill`/`skill_v2` vs the
bloated `skill_final`) and run combinatorial attribution **only on those**.

- **Prediction**: their aggregate causal value ≈ 0, and they are mutually /
  cross-redundant (I<0) or harmful.
- **If it holds** → one-line result: *"the 28 ineffective steps produced text of
  measured causal value ≈ 0."* This turns the temporal finding into a
  mechanistic, causal statement.

Status: ⏳ (after the first-order 172 attribution of `best_skill` completes).

### Measured: combinatorial pairwise confirms no hidden synergy (✅)

Targeted pairwise over {10,12,13,17} (reusing cache; only doubles new): all
I(i,j) ≤ 0 (10↔17 −0.035, 12↔17 −0.047, 10↔12 −0.012, 12↔13 −0.029; rest ≈0) —
**zero complements**. The arithmetic rules are substitutes for each other and
redundant with #17. So the near-zero/negative LOO is *not* a redundancy illusion;
pruning loses no synergy. Interaction-aware view agrees with first-order: 1 useful
unit + 18 dead/redundant.

### Measured: the gate-selected skill is 1-of-19 useful (✅)

First-order attribution of the e16 `best_skill` (19 units, 172 test, full=0.692,
empty=0.547): a single output-format unit (#17, "match the requested output form
exactly… omit units/prose/qualifiers") carries everything — LOO Δ **+0.145**
(removing it drops to 0.547 = no-skill), add-one Δ **+0.174** (alone = 0.721,
beating the full 19-unit skill). The other 18 units are ≈0 or negative; long
arithmetic rules are standalone-harmful (add-one −0.19/−0.30). **Causal-pruned to
4 lines (470 chars) = 0.715**, ties the entire 16-epoch run (best 0.733 / final
0.715) at ~5× smaller. A 1–4 sentence skill reproduces 7.6h / 106M tokens.

---

## Part 4 — The fix: causal-guided, cost-aware, global optimization

Because value is sparse and the optimizer is blind (self-reported meta-skill +
force-accept + greedy gate):

1. **Prune** by measured value → compaction; multi-objective **Pareto(EM, tokens)**
   using each unit's (value, cost). Directly cures bloat.
2. **Guide the optimizer**: inject measured per-unit causal value into the
   meta-skill / analyst prompts (`reflect.py:309,328`, `meta_skill.py`) so edits
   target real gaps and protect high-value units — local noisy SGD → global,
   interaction-aware credit assignment.
3. **Harness-level (global) view**: attribute across *all* editable surfaces
   (template Rules, skill, tool docs, optimizer prompts), deciding *where* each
   piece of guidance should live, not just what the skill says.

---

## Contributions (paper skeleton)

1. **Method** — a harness-level causal-attribution framework (first-order →
   combinatorial / interaction-aware) for agentic skill documents.
2. **Empirical** — dual sparsity (time + content) + the mechanism (force-accept
   slow-update, greedy early lock-in, redundancy) in a published self-evolving
   skill method.
3. **Constructive** — causal-guided, cost-aware, global optimizer that matches
   accuracy at a fraction of the tokens/epochs.

**One-line pitch**: *"Self-evolving" skills are low-rank — agentic skill
optimization concentrates in a few edits and a few sentences, wastes the rest,
and a causal-attribution lens both proves it and fixes it.*

---

## Tooling

- `scripts/eval_only.py` — deterministic-EM eval primitive.
- `scripts/eval_skill_ablation.py` — template-section ablation (LOO/add-one) +
  process-level version curve (`--versions-dir`).
- `scripts/skill_attribution.py` — first-order attribution over a learned skill
  document + `--prune`.
- `scripts/skill_combo.py` — interaction-aware (pairwise / Shapley) attribution
  with a shared on-disk subset-eval cache.
- `scripts/plot_training_curve.py` — paper-style step/epoch trend figure.

## Planned experiments (cheap → expensive; all reuse the eval cache)

1. ✅ First-order attribution of the gate-selected `best_skill` (172 test).
2. ⏳ Bridge experiment — combinatorial attribution of post-epoch-2 units.
3. ⏳ Cross-surface redundancy — template Rules + skill units in one pool.
4. ⏳ Causal pruning vs full (Pareto EM/tokens) on the bloated skill.
5. ⏳ In-loop attribution-guided optimizer A/B.
