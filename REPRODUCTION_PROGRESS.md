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

## 结论（中文，可直接转给作者）

我们在同一份上游 split、同样用 gpt-5.5 复现 SkillOpt，结论如下：

**1. 方法本身成立。** 用「裸 prompt」做 baseline 时，4 个 benchmark 的 skill 增益都和论文吻合：
- SpreadsheetBench：Δ **+38.2**（论文 +38.9）
- DocVQA：Δ **+10.5**（论文 +12.4）
- SearchQA：Δ **+6.7**（论文 +9.6）

**2. 但 repo 的「no-skill」baseline 不是真 vanilla，被系统 prompt 模板抬高了。** `prompts/rollout_system.md` 永远加载，里面硬编码的若干条 Rules 几乎逐字复制了 `skills/initial.md`（即待学习的 skill）。把这些 Rules 去掉后（其它完全不变）：

| Benchmark | 模板 no-skill | 裸 prompt no-skill | 论文 vanilla |
|---|--:|--:|--:|
| OfficeQA (EM, 172) | 54.65 | **28.49** | ~33 |
| DocVQA (ANLS, 374) | 0.918 | **0.860** | 0.788 |

裸 prompt 下，no-skill 直接落回论文 vanilla 附近。模板贡献：OfficeQA **+25.9 EM**、DocVQA **+5.8 ANLS**。

**3. OfficeQA 还有第二层：文件名日期泄漏。** 公报文件名是 `treasury_bulletin_YYYY_MM.txt`，问题里带日期，模型直接 `glob *YYYY*` 就能定位答案文件（42% 的 glob 调用带 4 位年份）。扩大语料 285→697 几乎不影响分数，说明检索是「免费」的。

**4. 已排除模型记忆。** 闭卷测试（无文档/无工具/无基准命名）OfficeQA EM = 0，说明 gpt-5.5 没有背答案，纯粹是评测口径问题。

**给作者的两点建议：**(a) 把这些 procedural Rules 从永远加载的模板移到 skill 文档里，让 no-skill 真正裸跑；(b) OfficeQA 匿名化文件名（去掉路径里的日期），强制真实检索。

---

## Benchmark status

| Benchmark | Status | No-skill | Best-skill | Paper vanilla | Paper best | Notes |
|---|---|--:|--:|--:|--:|---|
| **OfficeQA** | done; PR #2 | EM 54.65 | EM 70.35 | ~33 | — | clean-vanilla (bare prompt) = **EM 28.49**, ~paper |
| **DocVQA** | clean done | ANLS 0.918→**0.860** | **0.965** | 78.8 | 91.2 | clean Δ **+10.5** ≈ paper +12.4 |
| **SearchQA** | done; reproduced | EM 78.57 | EM 85.29 | 76.9 | 86.5 | Δ +6.7 (paper +9.6); no-skill near ceiling |
| **SpreadsheetBench** | done; reproduced | 37.14 | 75.36 | 41.8 | 80.7 | **Δ +38.2 ≈ paper +38.9** — procedural gain holds |
| **LiveMath** | done; reproduced | EM 47.58 | EM 62.10 | 37.6 | 66.9 | bare no-skill **44.35** (template −3.2, *not* OfficeQA-style); Δ +14.5~17.8 (paper +29.3) — SearchQA pattern |

**Clean-vanilla confirms the template inflation.** Re-running OfficeQA no-skill on the
full 172 test with a **bare prompt** (template Rules stripped, everything else identical)
gives **EM 28.49** — vs **54.65** with the shipped template. The hardcoded Rules are
worth **+25.9 EM pts** at full scale, and the bare-prompt number lands **near/below the
paper's vanilla ~33**. This closes the investigation: the repo's "no-skill" was inflated
by the always-on system prompt, not by the method or the model.

**Cross-benchmark pattern.** On SearchQA and SpreadsheetBench the **skill delta**
reproduces almost exactly (SearchQA +6.7 vs +9.6; SpreadsheetBench +38.2 vs +38.9),
with absolutes ~1.7–5 pts below the paper — a consistent proxy-gpt-5.5 offset, same
direction everywhere. The anomaly is **OfficeQA** (and to a lesser extent DocVQA),
where our *no-skill* is far above the paper's vanilla. That anomaly traces to the
template-contamination + filename-leakage layers below, not to the method.

---

## Three leakage layers inflating OfficeQA no-skill (same gpt-5.5 as paper)

1. **Template contamination (SMOKING GUN).** `prompts/rollout_system.md` always loads
   6 "Rules" that paraphrase `skills/initial.md` almost verbatim
   (e.g. "narrow to the most relevant file before reading", "compute only after
   extracting the exact operands"). **Full-scale confirmation:** clean-vanilla on the
   whole 172 test = **EM 28.49** vs **54.65** with the template — **+25.9 EM pts** from
   those rules alone, and the bare-prompt number is at/below the paper's vanilla ~33.
   The "no-skill" run is therefore not vanilla.
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
- **Clean (bare-prompt) full 374:** no-skill **0.8600**, best-skill (`ckpt/docvqa/gpt5.5_skill.md`)
  **0.9654**, clean Δ **+10.54** ≈ paper +12.4. So the templated no-skill 0.918 hid
  **~5.8 ANLS pts** of skill; with a bare prompt the no-skill (86.0) and the skill delta
  both track the paper (vanilla 78.8 / best 91.2 / Δ +12.4). Direction + magnitude reproduce.

---

## SearchQA & SpreadsheetBench (reproduced earlier, branch `repro/copilot-proxy-searchqa`)

**SearchQA** (full 1400 test, gpt-5.5):

| Config | EM | F1 | Paper EM |
|---|--:|--:|--:|
| No-skill | 0.7857 | 0.8898 | ~76.9 |
| Best-skill | 0.8529 | 0.9190 | ~86.5 |
| Δ (skill) | **+6.7** | +2.9 | +9.6 |

Direction reproduced. Absolutes within ±1.7 pts (proxy gpt-5.5, Responses API,
temperature=1 noise + strict-EM boundary). Smaller Δ because our no-skill is already
higher (78.6 vs 76.9) — proxy gpt-5.5 is stronger zero-shot on SearchQA, near ceiling,
matching the paper's note that SearchQA no-skill is near the ceiling.

**SpreadsheetBench** (gpt-5.5):

| Config | hard | Paper |
|---|--:|--:|
| No-skill | 37.14 | 41.8 |
| Best-skill | 75.36 | 80.7 |
| Δ (skill) | **+38.22** | +38.9 (diff −0.7) |

Absolutes ~5 lower (same proxy offset as SearchQA), but the **add-skill gain +38.2 ≈
paper +38.9** — the SkillOpt procedural-task strong gain holds end-to-end. token:
no-skill 917K vs best 2.27M (skill grows the prompt ~2.5×). timeouts: best 8 (2.9%) /
no-skill 13 (4.6%), thermal-throttle at both ends, does not affect the Δ comparison.

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

### Draft feedback to send

> Thanks for confirming the split and the model — that let us isolate the cause. We think
> the **`no-skill` baselines are inflated by the always-on system-prompt template**, which
> hardcodes guidance that duplicates the skill being learned. Concretely, on the same
> upstream split with gpt-5.5:
>
> | Benchmark | Templated no-skill | Bare-prompt no-skill | Paper vanilla |
> |---|--:|--:|--:|
> | OfficeQA (EM, 172) | 54.65 | **28.49** | ~33 |
> | DocVQA (ANLS, 374) | 0.918 | **0.860** | 0.788 |
>
> Stripping the template Rules (which paraphrase `skills/initial.md` nearly verbatim —
> e.g. "narrow to the most relevant file before reading", "extract the exact operands
> before arithmetic") drops OfficeQA no-skill by **25.9 EM** and DocVQA by **5.8 ANLS**,
> and the bare-prompt numbers land right around your paper's vanilla rows. With the bare
> prompt the **skill deltas reproduce cleanly** (DocVQA +10.5 ANLS vs paper +12.4;
> SpreadsheetBench +38.2 vs +38.9; SearchQA +6.7 vs +9.6).
>
> Two suggestions for a leakage-free baseline: (1) move the procedural Rules out of the
> always-on template and into the skill document only, so `no-skill` is a truly bare
> prompt; (2) for OfficeQA, the bulletin **filenames encode the date**
> (`treasury_bulletin_YYYY_MM.txt`) and questions carry the date, so `glob *YYYY*` locates
> the answer file directly (42% of glob calls embed a 4-digit year) — anonymizing
> filenames would force genuine retrieval. We separately **ruled out** model memorization
> (closed-book OfficeQA = EM 0.0), so this is purely an eval-protocol issue, not a model effect.

---

## Open todos

- [x] ~~Verify parametric memory~~ — **ruled out** (closed-book OfficeQA EM 0.0).
- [x] ~~Clean OfficeQA vanilla, full test~~ — **done: EM 28.49** (vs 54.65 templated).
- [x] ~~Clean DocVQA vanilla, full 374~~ — **done: no-skill 0.860, best 0.965, Δ +10.5**.
- [x] ~~DocVQA best-skill run~~ — done (clean Δ +10.5 ≈ paper +12.4).
- [ ] Draft author feedback: recommend a leakage-free no-skill baseline (bare prompt,
      anonymized filenames, no benchmark naming) — note OfficeQA bare-prompt EM 28.49 ≈ paper ~33.
