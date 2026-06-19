# OfficeQA Reproduction — Status & Report

> SkillOpt paper reproduction on the local **GitHub Copilot (ghc) proxy** with **gpt-5.5**.
> Benchmarks done before this: SearchQA, SpreadsheetBench (on branch `repro/copilot-proxy-searchqa`).
> This doc = OfficeQA. Last updated 2026-06-18.

---

## RESUME HERE (next session)

- **Branch:** `repro/officeqa-gpt55` (PR #2 on litaohz fork). First commit `7eade79` already pushed; the `w/o-search` ablation below adds 2 modified files (`adapter.py`, `rollout.py`) still to commit.
- **Eval runs done and intact:** `outputs/eval_officeqa_gpt55_{noskill,best}/` plus the three `w/o-search` runs `outputs/eval_officeqa_gpt55_noskill_{nooracle,retrieval,retrieval_fullcorpus}/`.
- **Proxy:** `http://localhost:4141/v1` (ghc), model `gpt-5.5` via **Responses API** (`OPENAI_RESPONSES_API_MODELS=gpt-5.5`).
- **HF token:** in `~/.env` as `HF_TOKEN=` (gated dataset `databricks/officeqa`, terms already accepted).
- **Full corpus now materialized:** `data/officeqa_docs_official/` holds all **697** parsed bulletins (was 285 sparse).

**Key finding (w/o search):** even after stripping *all* leakage — oracle pages off, file/page hints off, full 697-doc haystack — gpt-5.5 no-skill still scores **EM 47.09**, far above the paper's ~33. The oracle/hints/corpus-size levers together only move EM ~7.5 pts (54.65 → 47.09). The dominant residual shortcut is **filename date-leakage** (`treasury_bulletin_YYYY_MM.txt` + dated questions): 42% of glob calls embed a 4-digit year, so the model jumps straight to the answer bulletin regardless of haystack size. Same model as the paper ⇒ the gap is *not* model capability; it is (a) filename leakage, (b) clean pre-parsed tables, (c) likely scorer/subset differences.

**Open todos (priority order):**
1. To truly force semantic retrieval and approach the paper's ~33, **obfuscate filenames** (hash/anonymize the bulletin stems so the date is no longer in the path) or route through the semantic `custom_search` / `azure_search` paths instead of local glob/grep.
2. Optionally re-score with the **official `reward.py` numeric-tolerance** scorer (Caveat #2) instead of strict EM.
3. Optionally fetch the paper's exact OfficeQA table numbers + protocol (does it give file hints? what scorer?) for a true side-by-side.

**New config switches (this session, both default `True` = legacy behavior):**
- `env.inject_oracle_pages` — set `false` to stop injecting the oracle parsed answer page.
- `env.restrict_to_source_files` — set `false` to drop Candidate-Files/Source/page hints **and** open the glob/read/grep tools to the whole corpus (forces self-retrieval).

---

## Results (gpt-5.5, full 172 test, `offline` mode, ghc proxy)

| Config | EM (hard) | F1 (soft) | easy | hard | avg turns |
|---|---|---|---|---|---|
| **No-skill** (empty file) | 0.5465 (94/172) | 0.6163 | 0.671 | 0.441 | 12.7 |
| **Best-skill** (`ckpt/officeqa/gpt5.5_skill.md`) | **0.7035 (121/172)** | 0.7194 | 0.835 | 0.591 | 8.1 |
| **Δ (skill effect)** | **+15.70 pts** (+27 items) | +10.31 | +16.4 | +15.0 | −4.6 |

- 0 errors / 0 crashes in both runs. Empty predictions: no-skill 4, best 2.
- **Efficiency finding:** no-skill grinds ~12.7 turns (near the 24 cap) and still loses on final formatting; best-skill converges in ~8.1 turns AND scores higher — the skill makes it *do less and get more right*. (tool-using items >2 turns: no-skill 170/172, best 57/172.)
- Skill helps both difficulty bands ~equally (+16 easy, +15 hard).

---

## w/o search ablation — why our no-skill ≫ paper's ~33 (same gpt-5.5)

Progressively removing every form of answer leakage from the **no-skill** baseline. **All rows are evaluated on the same 172 test items** (upstream split `train 50 / val 24 / test 172`); the `Corpus docs` column is the size of the *document haystack* the model may retrieve over (number of Treasury Bulletin files), **not** the number of questions:

| Run | File hint | Page hint | Oracle page | Corpus docs | Test items | EM | F1 |
|---|:--:|:--:|:--:|--:|--:|--:|--:|
| `noskill_original` | ✅ | ✅ | ✅ | 285 | 172 | 0.5465 | 0.6163 |
| `noskill_nooracle` | ✅ | ✅ | ❌ | 285 | 172 | 0.5291 | 0.5933 |
| `noskill_retrieval` (sparse) | ❌ | ❌ | ❌ | 285 | 172 | 0.4884 | 0.5490 |
| `noskill_retrieval_fullcorpus` | ❌ | ❌ | ❌ | **697** | 172 | **0.4709** | 0.5235 |

> **Corpus docs vs test items:** `172` = questions answered (the upstream test split, unchanged across all rows). `285` = only the bulletins referenced by the 246 train+val+test items (sparse `materialize`). `697` = the full `databricks/officeqa` bulletin set (`materialize --full`). Growing the haystack 285→697 tests whether retrieval gets harder — it barely does (see below).

**Reading it:**
- Oracle page off: −1.74 pts. File/page hints off: −4.07 pts. Corpus 285→697: −1.75 pts. **Total leakage budget ≈ 7.5 pts.**
- Even fully stripped (full corpus, zero hints), EM = **47.09 ≫ paper ~33**. Since the paper also uses gpt-5.5, **the gap is not model capability.**
- **Corpus size barely matters** (−1.75 over 2.4× more docs) because retrieval is solved by **filename date-leakage**, not browsing: filenames are `treasury_bulletin_YYYY_MM.txt` and questions carry the date, so `glob *YYYY*` lands the bulletin directly. Measured: **42% (453/1071) of glob calls embed a 4-digit year.**
- Residual gap to the paper is therefore attributable to: (a) filename date-leakage (the real retrieval shortcut), (b) the corpus being clean **pre-parsed tables** (`transformed/*.txt`, "recommended for RAG"), and (c) probable **scorer/subset/protocol** differences (Caveat #2).

**To approach the paper's number:** anonymize bulletin filenames (remove the date from the path) or force the semantic `custom_search`/`azure_search` retrieval path — corpus size alone will not do it.

**Split provenance (confirmed with authors' question "did you use our split?"):** yes. `data/officeqa_id_split/{train,val,test}/items.json` are the upstream git-tracked manifests (commit `181d71b "Release data split manifests"`, on `origin/main`); the manifest records `source_repo: databricks/officeqa`, `source_revision: 8ecbf18…`, `counts: 50/24/172`. `materialize_officeqa.py` only joins the CSV question/answer onto these IDs by `uid` — no re-split. Generated counts match exactly (50/24/**172 test**) and the eval log confirms `test=172`. So the split is identical; the residual gap is in the eval protocol, not the split.

Reproduce (PowerShell):
```powershell
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"; $env:PYTHONIOENCODING="utf-8"
python scripts/materialize_officeqa.py --full --skip-csv      # one-time: 697 bulletins
python scripts/eval_only.py --config configs/officeqa/default.yaml `
  --skill outputs/empty_skill.md --split valid_unseen --split_dir data/officeqa_split `
  --cfg-options env.workers=12 env.inject_oracle_pages=false env.restrict_to_source_files=false `
  --azure_openai_endpoint http://localhost:4141/v1 --azure_openai_api_key dummy `
  --azure_openai_auth_mode openai_compatible --target_model gpt-5.5 `
  --out_root outputs/eval_officeqa_gpt55_noskill_retrieval_fullcorpus
```

---

## What OfficeQA is

Local-document RAG benchmark (Databricks) over parsed **U.S. Treasury Bulletin** pages (1939–2025): dense financial tables, numeric multi-step reasoning. Each item names the source bulletin file(s) + page. SkillOpt env supports 3 modes: `offline` (local glob/read/grep tools, **config default**), `custom_search`, `azure_search`. Eval = EM + token-F1 with finance normalization (strips million/dollars, keeps %).

The skill (`ckpt/officeqa/gpt5.5_skill.md`) is a procedural cheatsheet: extract-then-compute, align by exact column header, fiscal-vs-calendar year discipline, enumerate every period, strict final formatting.

---

## ⚠️ Caveat #1 — our baseline ≠ paper baseline (oracle pages)

**This is why the paper's baseline looks "so low" vs ours.** In `offline` mode the rollout injects **oracle parsed pages** — the exact answer-bearing page — for **every item (172/172, ~10K chars avg)**. So retrieval is solved for free; we measure *reasoning given the right page*, not *retrieve+reason*. The paper's baseline almost certainly forces retrieval, so it starts much lower.

Verified: EM where oracle present = our whole set; there are no no-oracle items to compare against. To get a paper-comparable baseline, disable oracle injection (no `page=` hint / a no-oracle mode) and/or run `custom_search` mode.

## ⚠️ Caveat #2 — scorer differs from official leaderboard

Official OfficeQA (`databricks/officeqa` `reward.py`) scores with a **numeric tolerance** (`score_answer(gt, pred, tolerance=...)`). SkillOpt's local evaluator uses **strict normalized EM (tolerance 0)**. Re-scoring our runs with a ~0.5% numeric tolerance (approx): no-skill ≈ 0.773 (+39), best ≈ 0.843 (+24). So strict EM **undercounts** vs the leaderboard convention, and the gap is larger for the baseline.

**Leaderboard data note:** the official leaderboard also reads the **pre-parsed** `transformed/*.txt` / `jsons/*.json` (README: "Recommended for LLM and RAG workflows"), NOT the raw PDFs. Our corpus matches it.

---

## Data provenance (no PDFs were read)

| Item | Format | Location | In repo? |
|---|---|---|---|
| `officeqa_full.csv` (246 rows; cols `uid,question,answer,source_docs,source_files,difficulty`) | CSV | `~/.cache/huggingface/.../databricks--officeqa/.../officeqa_full.csv` | ❌ cache |
| Parsed docs (285 `.txt` + 285 `.json`, sparse) | text + structured JSON | `data/officeqa_docs_official/{transformed,jsons}/` | ✅ |
| Split (Q+A joined onto id-manifest) | JSON | `data/officeqa_split/{train,val,test}/*.json` (50/24/172) | ✅ |

`materialize_officeqa.py`: (1) downloads gated CSV, joins `uid`→question+answer onto `data/officeqa_id_split/`, writes `data/officeqa_split/`; (2) sparse-downloads only referenced `transformed/*.txt` + `jsons/*.json`. Model at eval time reads only the parsed docs (CSV answer used solely by the scorer). The 4GB raw PDFs were never downloaded.

---

## Reproduce commands (PowerShell)

```powershell
$env:HF_TOKEN  # must be set in ~/.env (gated dataset, terms accepted)
# 1) materialize split + sparse docs (one-time)
python scripts/materialize_officeqa.py

# 2) eval — Responses API for gpt-5.5
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"
$env:PYTHONIOENCODING="utf-8"
# best skill:
python scripts/eval_only.py --config configs/officeqa/default.yaml `
  --skill ckpt/officeqa/gpt5.5_skill.md `
  --split valid_unseen --split_dir data/officeqa_split `
  --cfg-options env.workers=12 `
  --azure_openai_endpoint http://localhost:4141/v1 `
  --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible `
  --target_model gpt-5.5 --out_root outputs/eval_officeqa_gpt55_best
# no-skill baseline: --skill outputs/empty_skill.md  (0-byte file)
```

Materialize flags: `--skip-docs` (only rebuild split JSON), `--skip-csv` (only docs), `--full` (mirror all ~700 bulletins).

---

## Bugs fixed this session (both required, in working tree)

1. **gpt-5.5 hit `/chat/completions` → 400** (`unsupported_api_for_model`). `main` ignored `OPENAI_RESPONSES_API_MODELS`. Ported the env-gated Responses-API routing from the repro branch into `skillopt/model/azure_openai.py`.
2. **`'dict' object has no attribute 'model_dump'`** crashed 170/172 (first baseline scored a bogus 0.0116). The Responses path returned tool_calls as plain dicts but the offline-tools rollout calls `tool_call.model_dump()`. Fixed at the source: Responses path now returns the same pydantic `ChatCompletionMessageToolCall` objects as the Chat path. (officeqa-specific — searchqa/spreadsheet never used local tools, so the repro branch never hit it.)
