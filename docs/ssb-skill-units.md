# SpreadsheetBench — ckpt skill unit attribution (cc harness)

> 源 skill：`ckpt/spreadsheetbench/gpt5.5_skill.md`（论文官方 SkillOpt best skill，**67 units**）。
> **Harness**：标准 Claude Code（cc），`target_backend=claude_code_exec`，model=**claude-opus-4.8**（cc 跑不了 gpt-5.5）。
> **Value function v(S)**：SkillOpt 自己的 **selection set（val，40 题）** hard 分 —— 即优化器用来 gate 每次 skill 编辑的目标函数。
> **Anchors (val-40)**：empty=**0.575** → full=**0.775**（gain **+0.200**）。
> **Test-280 baseline（报告口径）**：no-skill 51.1 → full 78.6（+27.5，见 `docs/cc-harness-baselines.md`）。
> add-one Δ = 空 skill 上只加这条涨多少；LOO Δ = 从满 skill 删这条掉多少；judge = opus-4.8 读文本判 −2..+2（3 runs 均值）。

## 三个杀手级发现

1. **add-one 46/67 个单元为负** —— 大多数单元**单独**加进空 skill 反而掉分（缺前置依赖 → 误导 agent）。这是结构（DAG）动机的直接证据：单元的价值高度依赖其他单元是否在场。
2. **add-one 与 LOO 几乎零相关（Spearman = +0.043）** —— 两个 flat 端点对“哪个单元重要”意见完全相左。从空集看重要的单元，从满集看往往冗余，反之亦然。**flat 归因在结构化/冗余 skill 上是病态的。**
3. **LLM-judge 从不打负分**（mean≈+1.0，见下），却有 46 个单元 add-one 为负、20/67 个单元 LOO≈0 —— 内省式评审既识别不出“加了有害”，也识别不出“删了无损（冗余）”。

## Spearman（归因方法两两相关）

| 对比 | Spearman | 读法 |
|---|--:|---|
| judge ↔ add-one | +0.246 | 弱正相关：judge 勉强跟上“裸贡献”排序 |
| judge ↔ LOO | -0.180 | **负相关**：judge 与“边际不可替代性”背道而驰 |
| add-one ↔ LOO | +0.043 | **≈0**：两个行为端点互不一致 → 需要结构化归因 |

## Judge prompt-robustness（回应"是不是 prompt 没提示好"）

为排除"judge 从不打负分只是 prompt 没邀请负分"的质疑，加了一个 **strong 变体**（显式告知：优化过的 skill 里约半数单元是净有害/冗余、零负分几乎必然 miscalibrated、请主动找该删的单元）。

| judge 变体 | 均值 | #负分 | Spearman(add-one) | Spearman(LOO) |
|---|--:|--:|--:|--:|
| default | +0.995 | 0/67 | +0.246 | −0.180 |
| **strong** | +0.453 | 7/67 | **+0.166** ↓ | −0.147 |

强 prompt 确实压低了均值、逼出了 7 个负分 —— **但与行为真值的相关性没变好、反而略降**。更关键：

- 强判官挑的 7 个"有害"单元只有 **4 个真有害（4/7=57%）**，而基线有害率是 **46/67=69%** —— 等于**比随机还差**。
- **真正最有害的单元**（#16 库表 −12.5、#31 `Save to OUTPUT_PATH` −12.5、#4/#14/#22 `---` 分隔符 −10、#15 header −10）强判官全给了 **0~+0.7**，一个都没识别出来。
- judge 的负分集中在"看着窄"的领域规则（#58/59/63/64/66），而它们行为上只是轻微有害。

**结论：内省式归因的失效对 prompt 校准鲁棒。** 逼 judge 打负分，它会打——但打在错的单元上；真正的 boilerplate/结构性有害单元系统性识别不出。这不是 prompt artifact，而是"读文本 ≠ 懂行为贡献"的本质局限。

## Test-280 剪枝验证

把 val-40 归因的处方拿到**独立的 test-280**（cc/opus-4.8）上验证。三种归因各自剪出一个 **25 单元** skill（同预算），处方互不相同：overlap LOO∩add-one=11/25、judge∩LOO=9/25、judge∩add-one=13/25。

| skill | #units | chars | test-280 hard | SE(±) |
|---|--:|--:|--:|--:|
| full（论文原 skill） | 67 | 13287 | 0.786 | 2.45% |
| **LOO-pruned**（保留 LOO Δ>0） | 25 | 4640 | **0.800** | 2.39% |
| add-one-topK（add-one 前 25） | 25 | 6644 | 0.782 | 2.47% |
| judge-pruned（judge 前 25） | 25 | 5811 | 0.800 | 2.39% |
| no-skill | 0 | 0 | 0.511 | 2.99% |

**诚实解读（含统计显著性）：**

1. **唯一显著的效应 = 1/3 体积的无损压缩。** 三个 25 单元剪枝 skill（0.782–0.800）全都 ≈ full（0.786），且都远高于 no-skill（0.511，差 ~10 SE）。即**归因指导下把 67 单元砍到 25、篇幅砍到 35–50%，test 不掉分**——论文原 skill 有大量可删死重，任一归因都能定位到。
2. **四个变体之间的差异不显著。** 它们跨度仅 **1.8pt，而 1 SE≈2.4pt（n=280）** → LOO-pruned=judge-pruned=0.800、add-one=0.782、full=0.786 的排序**在噪声内**。⚠️ 因此**不能**用 test-280 声称"LOO 剪枝优于 add-one/judge"——粗粒度 top-25 剪枝这个任务太"容易"（把 boilerplate/分隔符删掉谁都会），区分不开各方法。
3. **judge 在粗粒度триаж上够用、在细粒度归因上失效。** judge 把结构单元（`---`、标题）打低分、内容规则打高分，所以 judge-pruned 也删对了 boilerplate → 0.800。judge 的**失效是细粒度的**（见上文 Spearman −0.18/≈0、never-negative、认不出真正有害单元），而不是"选哪 25 个留下"这种粗活。

**定位**：flat 归因（LOO/add-one）与内省归因（judge）在"砍到 1/3 不掉分"上**难分伯仲**（差异在噪声内）；真正区分它们的是**细粒度保真度**（相关性、识别有害/冗余单元），而那正是结构化方法（DAG+tree / SSG）要发力、也是粗粒度剪枝任务测不出来的地方。本节确立了 value function=val、confirmation=test（避免 test 泄漏）的验证协议。*（本节不声称 SSG 结果。）*

## judge 相关性是否由 prompt 决定?（framing invariance）

质疑：judge 相关 add-one 还是 LOO，会不会只是 prompt 怎么写？为此做**语义对齐**实验——两个变体把问法**精确对齐**到两个行为量：

- `addone_framed`：“skill 为空，只加这一条，单独能改变多少？”（= add-one 语义）
- `loo_framed`：“agent 已有完整 skill，单独删这一条会掉多少？”（= LOO 语义）

| judge prompt | 均值 | #负分 | ρ(add-one) | ρ(LOO) |
|---|--:|--:|--:|--:|
| neutral（默认，删除语义措辞） | +1.00 | 0 | +0.246 | −0.180 |
| addone_framed | +0.85 | 0 | +0.302 | −0.254 |
| loo_framed | +0.77 | 0 | +0.221 | **−0.210** |

**决定性证据**：`loo_framed` 明确按 LOO 语义问（“从满 skill 删掉它会掉多少分”），结果**仍与真实 LOO 负相关（−0.210）、与 add-one 正相关（+0.221）**。若相关性由 prompt 决定，loo_framed 应对齐 LOO（转正）——但它连符号都没翻。三个 prompt 的符号模式完全一致（+add-one / −LOO），措辞只改变幅度约 ±0.05。

**结论：judge 的相关目标不由 prompt 决定。** 不管怎么问，judge 输出的都是“这段文字看起来独不独立有用”（≈ add-one 方向），而系统性判错“边际不可替代性”（LOO）。这是“读文本 ≠ 懂行为贡献”的本质局限，不是 prompt artifact。

## 全 67 单元表（按 add-one 降序）

| # | chars | add-one Δ% | LOO Δ% | judge | label | text |
|--:|--:|--:|--:|--:|:--|:--|
| 20 | 428 | +27.5 | -5.0 | +2.0 | essential | Treat wording such as “write/fix a formula,” “SUMIFS/COUNTIFS,” “VBA,” |
| 21 | 317 | +25.0 | +2.5 | +1.0 | useful | When the user provides an existing or broken formula, use it as a sema |
| 18 | 446 | +20.0 | -5.0 | +2.0 | essential | **Formula evaluation caution**: `openpyxl` can write formulas but does |
| 56 | 277 | +20.0 | +5.0 | +1.0 | useful | For blank-sensitive formula tasks, compute the branch explicitly: if t |
| 55 | 325 | +17.5 | +5.0 | +1.0 | useful | Use existing formulas in the workbook as examples/specifications, not  |
| 53 | 347 | +15.0 | +2.5 | +1.0 | useful | <!-- SLOW_UPDATE_START --> When the user asks for a formula, macro, VB |
| 54 | 243 | +15.0 | +2.5 | +2.0 | essential | After writing, reload or inspect the saved workbook and verify that ev |
| 38 | 305 | +5.0 | -2.5 | +1.0 | useful | - For monthly or period summary grids, canonicalize period labels from |
| 41 | 277 | +5.0 | -2.5 | +1.0 | useful | - For joins, deduplication, grouping, interval lookups, lookup grids,  |
| 9 | 284 | +2.5 | +0.0 | +1.0 | useful | - Scan the used range for complete header groups, not just row 1. Tabl |
| 36 | 493 | +2.5 | +2.5 | +2.0 | essential | - Create small helper functions for comparisons and numeric parsing. N |
| 37 | 333 | +2.5 | +7.5 | +1.7 | essential | - Normalize date keys deliberately: handle `datetime`/`date` objects,  |
| 42 | 356 | +2.5 | +5.0 | +1.0 | useful | - For outputs that depend on other rows or lookup grids, make a first  |
| 49 | 309 | +2.5 | +0.0 | +2.0 | essential | - For numeric aggregation, crosstab, SUMIFS-like, and INDEX/MATCH-styl |
| 64 | 253 | +2.5 | -2.5 | +0.3 | neutral | For time-threshold rows, decide per row whether it is a normal data ro |
| 3 | 143 | +0.0 | +2.5 | +1.0 | useful | **Primary libraries**: `openpyxl` (structure-preserving read/write), ` |
| 7 | 214 | +0.0 | -5.0 | +2.0 | essential | - Inspect actual workbook data beyond the preview, including nearby ro |
| 30 | 22 | +0.0 | +5.0 | +0.0 | neutral | ## Output Requirements |
| 39 | 231 | +0.0 | -2.5 | +1.3 | useful | - For date ranges and rolling windows, infer endpoint inclusivity from |
| 40 | 317 | +0.0 | +0.0 | +1.0 | useful | - For time extraction or time-threshold logic, parse `datetime`, `time |
| 60 | 321 | +0.0 | +0.0 | +1.0 | useful | For INDEX/MATCH problems where the first row works but subsequent rows |
| 1 | 11 | -2.5 | -2.5 | +0.0 | neutral | ## Overview |
| 8 | 210 | -2.5 | +0.0 | +1.3 | useful | - Treat existing filled cells in the requested output area or adjacent |
| 11 | 82 | -2.5 | -2.5 | +1.0 | useful | 2. **Write `solution.py`** with `INPUT_PATH` and `OUTPUT_PATH` defined |
| 12 | 75 | -2.5 | +5.0 | +1.0 | useful | 3. **Execute** `python solution.py` and verify the output file was cre |
| 23 | 23 | -2.5 | +0.0 | +0.0 | neutral | ## solution.py Template |
| 26 | 76 | -2.5 | +0.0 | +1.0 | useful | wb = openpyxl.load_workbook(INPUT_PATH) ws = wb.active # or wb["SheetN |
| 27 | 30 | -2.5 | +10.0 | +0.0 | neutral | # --- perform manipulation --- |
| 32 | 90 | -2.5 | +5.0 | +1.7 | essential | - Do not hardcode row counts or column letters — iterate over actual r |
| 43 | 415 | -2.5 | -2.5 | +1.7 | essential | - For lookups, filters, joins, and label/header matching, normalize co |
| 45 | 396 | -2.5 | +0.0 | +1.0 | useful | - If the instruction includes formatting changes, apply them exactly a |
| 61 | 282 | -2.5 | +0.0 | +1.0 | useful | For multi-step macro/VBA-style requests, implement every stated operat |
| 63 | 302 | -2.5 | +0.0 | +0.0 | neutral | For residual-balancing tasks, identify data rows separately from min/m |
| 65 | 270 | -2.5 | -7.5 | +1.0 | useful | Keep scripts simple enough to run cleanly. Avoid unnecessary dynamic c |
| 0 | 39 | -5.0 | +7.5 | +0.0 | neutral | # Spreadsheet Manipulation Skill (xlsx) |
| 2 | 81 | -5.0 | +2.5 | +0.0 | neutral | This skill guides agents in manipulating Excel (.xlsx) spreadsheets us |
| 5 | 18 | -5.0 | +0.0 | +0.0 | neutral | ## Common Workflow |
| 6 | 78 | -5.0 | +0.0 | +1.0 | useful | 1. **Explore** the input file: list sheets, inspect headers, check dim |
| 10 | 238 | -5.0 | -2.5 | +2.0 | essential | - Locate tables, fields, and target ranges by header text, nearby labe |
| 19 | 166 | -5.0 | +0.0 | +1.0 | useful | ```python wb = openpyxl.load_workbook(INPUT_PATH) wb_values = openpyxl |
| 24 | 45 | -5.0 | -7.5 | +1.0 | useful | ```python import openpyxl import pandas as pd |
| 25 | 106 | -5.0 | +0.0 | +1.0 | useful | INPUT_PATH = "..." # set to the actual input path OUTPUT_PATH = "..."  |
| 34 | 36 | -5.0 | +2.5 | +0.0 | neutral | ## Matching and Target Range Hygiene |
| 46 | 238 | -5.0 | -2.5 | +2.0 | essential | - When the instruction names a destination range or columns, write der |
| 47 | 414 | -5.0 | +2.5 | +1.0 | useful | - For filtered lists, summaries, and aggregations, first collect all s |
| 50 | 246 | -5.0 | -7.5 | +1.0 | useful | - For blank-sensitive logic such as “if input is blank, output blank,” |
| 52 | 279 | -5.0 | -2.5 | +1.0 | useful | - Prefer simple, auditable row/column loops over complex workbook XML  |
| 58 | 214 | -5.0 | +0.0 | +1.0 | useful | For “every nth row” or OFFSET-style tasks, infer the source column, fi |
| 62 | 309 | -5.0 | +5.0 | +2.0 | essential | When a target range includes special rows such as `Total`, `Grand Tota |
| 13 | 66 | -7.5 | +2.5 | +1.0 | useful | 4. **Confirm** the target cells/range contain the expected values. |
| 17 | 175 | -7.5 | +0.0 | +2.0 | essential | **Warning**: `pandas.to_excel()` silently destroys existing formulas a |
| 33 | 61 | -7.5 | +5.0 | +2.0 | essential | - Preserve sheets and cells not mentioned in the instruction. |
| 35 | 225 | -7.5 | +0.0 | +2.0 | essential | - Choose the comparison operator from the instruction and examples: us |
| 48 | 123 | -7.5 | +0.0 | +1.7 | essential | - Preserve intended blanks as empty cells (`None`) rather than placeho |
| 59 | 312 | -7.5 | +2.5 | +1.0 | useful | For schedule/calendar fill tasks, build a cycle-day-to-periods mapping |
| 66 | 293 | -7.5 | +0.0 | +0.0 | neutral | If workbook cells contain arbitrary sample text that could be sensitiv |
| 4 | 3 | -10.0 | +0.0 | +0.0 | neutral | --- |
| 14 | 3 | -10.0 | +5.0 | +0.0 | neutral | --- |
| 15 | 20 | -10.0 | -2.5 | +0.0 | neutral | ## Library Selection |
| 22 | 3 | -10.0 | +5.0 | +0.0 | neutral | --- |
| 28 | 24 | -10.0 | -2.5 | +1.0 | useful | wb.save(OUTPUT_PATH) ``` |
| 29 | 3 | -10.0 | +2.5 | +0.0 | neutral | --- |
| 44 | 277 | -10.0 | -2.5 | +2.0 | essential | - When replacing a generated output area, clear only the instructed ta |
| 51 | 35 | -10.0 | -2.5 | +0.0 | neutral | ## Robustness for Simple Fill Tasks |
| 57 | 267 | -10.0 | -2.5 | +1.0 | useful | For lookup/category tasks, locate both the input rows and the lookup t |
| 16 | 237 | -12.5 | +5.0 | +1.0 | useful | \| Use case \| Library \| \|----------\|---------\| \| Preserve formulas, for |
| 31 | 35 | -12.5 | -5.0 | +1.0 | useful | - Save the result to `OUTPUT_PATH`. |

## 与 OfficeQA ckpt 的对照

| | OfficeQA ckpt (32u, gpt-5.5 direct) | SSB ckpt (67u, cc/opus-4.8 val) |
|---|--:|--:|
| judge 是否打负分 | 否（从不） | 否（从不，mean≈+1.0） |
| judge ↔ add-one | +0.246-ish（弱） | +0.246（弱） |
| judge ↔ Shapley/LOO | 负 | -0.180（负） |
| LOO 是否近乎全平（冗余） | 是 | 20/67 近零 |

两个数据集、两种 harness、两个裁判模型，**结论一致**：内省式（LLM-judge）归因系统性失效，flat 行为归因（add-one/LOO）内部又互不一致。

## 定位与下一步

- add-one / LOO / Shapley 都是 **flat baseline**（假设单元可交换、无依赖）。本表的三大发现——大量负 add-one、add-one↔LOO≈0、judge 失效——正是 flat 假设失效的症状。
- **新方法**：DAG（依赖）+ tree（层级）结构归因（SSG / precedence-constrained Winter value），条件在单元的结构位置上算边际，预期能救回被 add-one 低估的“依赖型”单元。*（本文档不声称 SSG 结果，仅作定位。）*
- **下一步**：把 add-one 的 top/bottom 单元回到 **test-280** 验证（value function 用 val、confirmation 用 test，避免 test 泄漏）。
