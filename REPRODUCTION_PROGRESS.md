# SkillOpt Reproduction — Progress Tracker

> Reproducing SkillOpt benchmarks on the local **GitHub Copilot (ghc) proxy** with **gpt-5.5**.
> Proxy: `http://localhost:4141/v1`, Responses API (`OPENAI_RESPONSES_API_MODELS=gpt-5.5`).
> Personal PR: **#2** on `litaohz/SkillOpt` (branch `repro/officeqa-gpt55`).
> Last updated: 2026-06-19.

---

## TL;DR — the repo's "no-skill" baseline is not a true vanilla baseline

Across **two** benchmarks we find the reported *no-skill* numbers are inflated by the
**always-on system-prompt template**, which hardcodes skill-like guidance that
**duplicates the skill the method is supposed to learn**. Measured, controlled A/B:

| Benchmark | Metric | Current template | Minimal (rules stripped) | Template's hidden boost |
|---|---|--:|--:|--:|
| OfficeQA (40-item subset) | EM | 0.500 | 0.275 | **+22.5 pts** |
| DocVQA (60-item subset) | ANLS | 0.903 | 0.853 | **+5.0 pts** |

This shrinks the apparent skill effect and is the leading explanation for why our
no-skill scores land far above the paper's vanilla rows **with the same model**.
(Note: a "parametric memory" hypothesis was tested and **ruled out** — closed-book
OfficeQA scored EM 0.0; the model is not reciting answers.)

---

## Benchmark status

| Benchmark | Status | No-skill | Best-skill | Paper vanilla | Paper best | Notes |
|---|---|--:|--:|--:|--:|---|
| **OfficeQA** | done; PR #2 | EM 54.65 | EM 70.35 | ~33 | — | 3 leakage layers (below) |
| **DocVQA** | no-skill done | ANLS 0.918 | (todo) | 78.8 | 91.2 | our vanilla ≈ paper best; template + memory |
| SearchQA / SpreadsheetBench | done earlier | — | — | — | — | other branch `repro/copilot-proxy-searchqa` |

---

## Three leakage layers inflating OfficeQA no-skill (same gpt-5.5 as paper)

1. **Template contamination (SMOKING GUN).** `prompts/rollout_system.md` always loads
   6 "Rules" that paraphrase `skills/initial.md` almost verbatim
   (e.g. "narrow to the most relevant file before reading", "compute only after
   extracting the exact operands"). Controlled A/B: **+22.5 EM pts** from those rules
   alone. The "no-skill" run is therefore not vanilla.
2. **Filename date-leakage.** Bulletins are named `treasury_bulletin_YYYY_MM.txt` and
   questions carry the date, so `glob *YYYY*` lands the answer file directly.
   Measured: **42% (453/1071) of glob calls embed a 4-digit year.** Growing the corpus
   285→697 barely moved EM (−1.75), confirming retrieval is solved by the filename, not browsing.
3. ~~**Parametric memory.**~~ **RULED OUT.** Closed-book test (50 test questions, no
   document, no tools, no benchmark naming) scored **EM 0.0 / F1 0.0** — gpt-5.5 cannot
   answer these "extract exact monthly value from a specific table, then compute" items
   from memory. So benchmark naming does *not* leak answers; the model genuinely needs
   the document. (This exonerates the model on the memorization concern.)

Ablation (all 172 test, no-skill): oracle off −1.74; file/page hints off −4.07;
corpus 285→697 −1.75; still **EM 47.1 ≫ paper ~33** even fully stripped.

Split confirmed identical to upstream (`data/officeqa_id_split/`, commit `181d71b`, 50/24/172).

## DocVQA

- No-skill **ANLS 0.9178** (binary@0.5 0.8235), n=374 — verified correct against an
  independent ANLS reference (374/374 exact match), so the scorer is fine.
- Our vanilla ≈ paper's **best-skill** (91.2), while paper vanilla is 78.8 — same model.
- Template A/B above shows ~5 ANLS pts come from the hardcoded 4 Rules; remainder is
  model strength + benchmark saturation (DocVQA val is near-ceiling for frontier VLMs).

---

## Code changes delivered (PR #2)

- `skillopt/model/azure_openai.py` — gpt-5.5 Responses-API routing; tool_call pydantic
  fix; **vision fix** (list content text+image was stringified → image dropped; now
  mapped to `input_text`/`input_image`).
- `skillopt/envs/officeqa/{rollout,adapter}.py` — env switches
  `inject_oracle_pages`, `restrict_to_source_files` (default True = legacy).
- `scripts/materialize_officeqa.py`, `scripts/materialize_docvqa.py` — data materialization.
- `scripts/eval_only.py` — utf-8 skill read (Windows fix).
- Reports: `officeqa_reproduction_report.md`, `docvqa_reproduction_report.md`.

---

## Author thread

- Asked about OfficeQA baseline protocol (model/setting/scorer). Author asked whether
  we used their split → **confirmed yes** (upstream manifest, 50/24/172).
- Author confirmed the paper also uses **gpt-5.5**. → strengthens the case that the gap
  is evaluation protocol (template + leakage), not model capability.

---

## Open todos

- [x] ~~Verify parametric memory~~ — **ruled out** (closed-book OfficeQA EM 0.0).
- [ ] Clean **vanilla** baselines with a truly minimal template (both benchmarks), full test sets.
- [ ] DocVQA best-skill run for the real skill delta.
- [ ] Draft author feedback: recommend a leakage-free no-skill baseline (bare prompt,
      anonymized filenames, no benchmark naming).
