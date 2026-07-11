# cc-harness baselines (standard Claude Code harness)

All runs use the **standard Claude Code (cc) harness** (`--target_backend claude_code_exec`),
rendering the SkillOpt skill to `.agents/skills/skillopt-target/SKILL.md` and driving `claude -p`
through the local proxy (`localhost:4141`). This removes the harness confound and gives a clean
model-vs-model comparison against the paper's cc/GPT-5.5 block (Table 1).

Note: **GPT-5.5 is not usable via cc** (Responses-API-only ⇒ 400 on `/chat/completions`), so
**opus-4.8** is our reproduction stand-in; **sonnet-4.6** is included as a second data point.

## OfficeQA (EM) & SpreadsheetBench (accuracy)

| Harness / model | OfficeQA no-skill → full (Δ) | SSB no-skill → full (Δ) |
|---|--:|--:|
| **cc / opus-4.8** (ours) | 62.8 → **69.8** (+7.0) | 51.1 → **78.6** (+27.5) |
| cc / sonnet-4.6 (ours) | — → **70.9** | — → **81.1** |
| cc / GPT-5.5 (paper Table 1) | 57.6 → **71.5** (+13.9) | 22.1 → **80.4** (+58.3) |

## Reading

- **Full-skill ceiling is model-robust**: OfficeQA lands 69.8 / 70.9 / 71.5 and SSB lands
  78.6 / 81.1 / 80.4 across opus-4.8 / sonnet-4.6 / gpt-5.5 — all clustered within ~2–3 pts.
- **Skill value shrinks as the base model strengthens**: opus-4.8's no-skill is much higher
  than gpt-5.5's (OfficeQA 62.8 vs 57.6; SSB 51.1 vs 22.1), so its skill gain is smaller
  (OfficeQA +7.0 vs +13.9; SSB +27.5 vs +58.3). The skill mostly closes a gap the stronger
  base model has already partly closed on its own.
- OfficeQA is near-saturated (oracle pages injected), so the ~70 ceiling is expected.

## Provenance

- `outputs/cc_officeqa_noskill_opus48/eval_summary.json` = 0.6279
- `outputs/cc_officeqa_full_opus48/eval_summary.json`    = 0.6977
- `outputs/cc_ssb_noskill_opus48/eval_summary.json`      = 0.5107
- `outputs/cc_ssb_full_opus48/eval_summary.json`         = 0.7857
- sonnet-4.6 cc full: see `outputs/_share/cc-results.md`
- Skill: `ckpt/{officeqa,spreadsheetbench}/gpt5.5_skill.md`; no-skill: `outputs/empty_skill.md`.
