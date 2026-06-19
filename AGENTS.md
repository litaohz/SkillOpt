# Project Memory — SkillOpt Reproduction

> Authoritative project memory for AI agents (read by GitHub Copilot CLI and Claude Code).
> Keep this concise and current. Detailed logs live in `REPRODUCTION_PROGRESS.md`.

## What this work is

Reproducing the **SkillOpt** paper's benchmarks on a **local GitHub Copilot (ghc) proxy**
with **gpt-5.5**. Personal PR: **#2** on `litaohz/SkillOpt`, branch `repro/officeqa-gpt55`.

## Key finding (do not re-litigate without new evidence)

The repo's **`no-skill` baseline is not a true vanilla baseline** — it is inflated by the
always-on system-prompt template (`skillopt/envs/<env>/prompts/rollout_system.md`), which
hardcodes "Rules" that paraphrase the to-be-learned skill (`skills/initial.md`) almost
verbatim. With a **bare prompt** (Rules stripped, everything else identical):

| Benchmark | Templated no-skill | Bare-prompt no-skill | Paper vanilla |
|---|--:|--:|--:|
| OfficeQA (EM, 172 test) | 54.65 | **28.49** | ~33 |
| DocVQA (ANLS, 374 test) | 0.918 | **0.860** | 0.788 |

- Template Rules are worth **+25.9 EM** (OfficeQA) / **+5.8 ANLS** (DocVQA).
- The **method itself reproduces**: bare-prompt skill deltas match the paper —
  SpreadsheetBench +38.2 (paper +38.9), DocVQA +10.5 (+12.4), SearchQA +6.7 (+9.6).
- **OfficeQA also has filename date-leakage**: `treasury_bulletin_YYYY_MM.txt` + dated
  questions ⇒ `glob *YYYY*` lands the answer file (42% of glob calls embed a year);
  growing the corpus 285→697 barely moves EM.
- **Model memorization ruled out**: closed-book OfficeQA (no doc/tools/naming) = EM 0.0.

## Environment / how to run

- Proxy: `http://localhost:4141/v1`; gpt-5.5 is **Responses-API only**
  (`OPENAI_RESPONSES_API_MODELS=gpt-5.5`). Always set `NO_PROXY=localhost,127.0.0.1`.
- Windows: set `PYTHONIOENCODING=utf-8`.
- Eval entry point: `scripts/eval_only.py` (all YAML keys overridable via `--cfg-options`).
- Empty skill file for no-skill runs: `outputs/empty_skill.md` (0 bytes).
- Data materialization (gated/streamed from HF; needs `HF_TOKEN` in `~/.env`):
  `scripts/materialize_officeqa.py` (add `--full` for all 697 bulletins),
  `scripts/materialize_docvqa.py`.

Example (clean bare-prompt run = temporarily strip the template's Rules, then restore it):
```powershell
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"; $env:PYTHONIOENCODING="utf-8"; $env:NO_PROXY="localhost,127.0.0.1"
python scripts/eval_only.py --config configs/officeqa/default.yaml `
  --skill outputs/empty_skill.md --split valid_unseen --split_dir data/officeqa_split `
  --cfg-options env.workers=12 `
  --azure_openai_endpoint http://localhost:4141/v1 --azure_openai_api_key dummy `
  --azure_openai_auth_mode openai_compatible --target_model gpt-5.5 `
  --out_root outputs/eval_officeqa_gpt55_noskill
```

## Code changes already delivered (PR #2)

- `skillopt/model/azure_openai.py` — gpt-5.5 Responses-API routing; tool_call pydantic
  fix; **vision fix** in `_messages_to_responses_input` (list content `text`+`image_url`
  was stringified, dropping the image; now mapped to `input_text`/`input_image`).
- `skillopt/envs/officeqa/{rollout,adapter}.py` — env switches `inject_oracle_pages` and
  `restrict_to_source_files` (both default `True` = legacy behavior).
- `scripts/materialize_officeqa.py`, `scripts/materialize_docvqa.py` (new).
- `scripts/eval_only.py` — utf-8 skill read (Windows).

## Conventions

- **Commits/PRs go to the personal fork `litaohz`** via the `commit-to-personal-repo`
  workflow: load `gh_token` from `~/.env` into `$env:GH_TOKEN`, commit with repo-local
  identity `litaohz <thomaslee3250@gmail.com>`, and drive PR ops through `gh api` (REST),
  **not** `gh pr ...` (token lacks `read:org` for GraphQL). Always add the
  `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>` trailer.
- When A/B-testing prompts, **back up the template, swap, run, then restore** and verify
  `git status` is clean for that file.
- Long eval runs (`run_batch`) support resume — re-running the same `--out_root` continues.

## Open todos

- Draft/send author feedback (already drafted in `REPRODUCTION_PROGRESS.md`): recommend
  moving procedural Rules out of the always-on template into the skill doc, and
  anonymizing OfficeQA filenames.
