# DocVQA Reproduction — Status & Report

> SkillOpt paper reproduction on the local **GitHub Copilot (ghc) proxy** with **gpt-5.5** (vision).
> Companion to `officeqa_reproduction_report.md`. Last updated 2026-06-18.

---

## RESUME HERE (next session)

- **Branch:** `repro/officeqa-gpt55` (personal PR #2). DocVQA work = new `scripts/materialize_docvqa.py`, a vision fix in `skillopt/model/azure_openai.py`, and this report.
- **Data materialized:** `data/docvqa/splits/{train,val,test}/*.csv` (107/53/374) + `data/docvqa_images/*.png` (534 page images).
- **Proxy:** `http://localhost:4141/v1` (ghc), model `gpt-5.5` via **Responses API** (`OPENAI_RESPONSES_API_MODELS=gpt-5.5`).
- **HF:** `lmms-lab/DocVQA` is public; `HF_TOKEN` in `~/.env` used to avoid rate limits.

**Open todos:**
1. Run the **best-skill** eval (skill file under `skillopt/envs/docvqa/skills/` or a trained ckpt) for the no-skill → best-skill delta.
2. Optionally compare against the paper's DocVQA row once its model/scorer protocol is known.

---

## Result (gpt-5.5, full 374 test, single-turn, ghc proxy)

| Config | ANLS (soft) | binary@0.5 (hard) | n |
|---|--:|--:|--:|
| **No-skill** (empty file) | **0.9178** | 0.8235 | 374 |

- **ANLS = 0.9178** is the standard DocVQA metric (mean normalized Levenshtein similarity, threshold 0.5). This is in line with strong vision-language models on the DocVQA validation split (~0.92).
- `binary@0.5` (`hard`) counts an item correct iff its ANLS ≥ 0.5; it is a stricter view, not the headline number.
- Split = `docvqa_validation_10pct` (upstream git-tracked manifest `data/docvqa_id_split/`, 107/53/374), a 10% subset of the official DocVQA `validation` split.

---

## What DocVQA is

Single-image document VQA over scanned business documents (UCSF Industry Documents). Each item = one document **page image** + a question; the model answers from the rendered page. Scored by **ANLS** (not EM). This SkillOpt env is **single-turn** (`max_turns: 1`): the page image is sent inline as a data-URI; the model returns the answer inside `<answer>...</answer>`.

---

## Two things were required to run it

### 1. Data materialization — `scripts/materialize_docvqa.py` (NEW)
The repo ships only id manifests (`questionId`/`docId`/`image_path`); the question text, gold answers, and page images live in `lmms-lab/DocVQA`. The script:
- reads the three id-split manifests (wanted `questionId` set, 534 items),
- streams the official `validation` split, matching by `questionId`,
- saves each page image to `data/docvqa_images/q<qid>_d<docId>.png`, and
- writes per-split CSVs (`questionId, question, answer, image_path, topic, docId, …`) to `data/docvqa/splits/{train,val,test}/`.

All 534 matched (107/53/374), 0 unmatched.

### 2. Vision plumbing fix — `skillopt/model/azure_openai.py`
gpt-5.5 is **Responses-API-only** on the proxy. `_messages_to_responses_input` stringified user `content` with `str(content)`. DocVQA builds `content` as a **list** of `{"type": "text"|"image_url"}` parts, so the image was being turned into a Python repr and **silently dropped** — the model answered blind. Fixed by mapping list content to the Responses parts `input_text` / `input_image` (and passing through `detail`). Verified end-to-end: gpt-5.5 read "CAT 42" off a synthetic PNG through this path. This fix also benefits any other multimodal env routed through the Responses API.

---

## Reproduce (PowerShell)

```powershell
$env:HF_TOKEN  # optional (public dataset); avoids rate limits
python scripts/materialize_docvqa.py            # one-time: 534 images + CSV splits

$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"; $env:PYTHONIOENCODING="utf-8"; $env:NO_PROXY="localhost,127.0.0.1"
python scripts/eval_only.py --config configs/docvqa/default.yaml `
  --skill outputs/empty_skill.md --split valid_unseen --split_dir data/docvqa/splits `
  --cfg-options env.workers=12 `
  --azure_openai_endpoint http://localhost:4141/v1 --azure_openai_api_key dummy `
  --azure_openai_auth_mode openai_compatible --target_model gpt-5.5 `
  --out_root outputs/eval_docvqa_gpt55_noskill
```
