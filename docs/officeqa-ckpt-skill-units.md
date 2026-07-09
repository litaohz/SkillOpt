# OfficeQA 官方 ckpt skill —— 单元编号 ↔ 原句 + 归因

> 源文件：`ckpt/officeqa/gpt5.5_skill.md`（论文发布的官方 best skill，**32 units**）。
> 评测：172 题 test、确定性 EM，gpt-5.5。**full=0.680，empty=0.541**（Δ +0.139）。
> LOO Δ = 删掉这条掉多少分；add-one Δ = 在 empty 上只加这条涨多少；LLM-judge = gpt-5.5 直接判 −2..+2（3 runs 均值）。
> 数据：`skillopt-assets/officeqa-ckpt-{attribution,llmjudge}.csv`；逐题原始见 Release `officeqa-attrib-firstorder-v1`。
> ⚠️ Shapley（交互感知）暂缓（缓存留存），后续补；当前以 add-one 作扰动式代表。

| # | 字符 | LOO Δ | add-one Δ | LLM-judge | 判定 | 原句（节选） |
|---|--:|--:|--:|--:|---|---|
| 0 | 16 | -0.006 | +0.012 | +0.00 | · ≈0 | # OfficeQA Skill |
| 1 | 23 | -0.041 | -0.017 | +0.00 | · ≈0 | ## Retrieval Discipline |
| 2 | 307 | -0.052 | -0.122 | +1.00 | ❌ 死重/有害 | - When an external official time-series observation is needed, prefer … |
| 3 | 261 | -0.052 | -0.064 | +2.00 | ❌ 死重/有害 | - Treat provided/oracle parsed pages as primary evidence: if they cont… |
| 4 | 84 | +0.000 | -0.017 | +1.00 | · ≈0 | - Start by narrowing to the most likely candidate file before reading … |
| 5 | 111 | -0.023 | -0.023 | +1.00 | · ≈0 | - Prefer targeted search terms that name the exact entity, period, mea… |
| 6 | 120 | -0.023 | -0.017 | +1.00 | · ≈0 | - After a promising match, read only a small surrounding span and veri… |
| 7 | 358 | +0.006 | -0.116 | +2.00 | ❌ 死重/有害 | - If the requested date range extends beyond the provided/oracle page,… |
| 8 | 22 | -0.023 | -0.070 | +0.00 | ❌ 死重/有害 | ## Evidence Discipline |
| 9 | 78 | -0.017 | +0.047 | +2.00 | · ≈0 | - Extract the exact value from the retrieved text before doing any ari… |
| 10 | 103 | -0.035 | -0.023 | +2.00 | · ≈0 | - Keep track of each operand's period, unit, and semantic role so near… |
| 11 | 298 | +0.000 | -0.052 | +1.00 | ❌ 死重/有害 | - For Treasury financing narratives, label each amount by transaction … |
| 12 | 265 | -0.035 | -0.110 | +1.00 | ❌ 死重/有害 | - When converting currencies or scales, make a direction ledger first:… |
| 13 | 274 | +0.006 | -0.017 | +2.00 | · ≈0 | - For tables, align values by row label and exact column header, not p… |
| 14 | 106 | -0.023 | -0.012 | +1.00 | · ≈0 | - If the question asks for a transformed or derived quantity, compute … |
| 15 | 316 | -0.035 | -0.023 | +2.00 | · ≈0 | - For derived comparisons, preserve the direction and sign implied by … |
| 16 | 300 | -0.023 | -0.326 | +2.00 | ❌ 死重/有害 | - For statistical, regression, correlation, and growth-rate questions,… |
| 17 | 228 | -0.058 | -0.093 | +1.00 | ❌ 死重/有害 | - For multi-stage questions where one table determines the period/enti… |
| 18 | 246 | -0.035 | -0.227 | +2.00 | ❌ 死重/有害 | - For inclusive time-series ranges, make a period-by-period ledger cov… |
| 19 | 310 | -0.017 | -0.052 | +1.00 | ❌ 死重/有害 | - For statistical transforms over time-series windows, confirm endpoin… |
| 20 | 26 | -0.012 | +0.012 | +0.00 | · ≈0 | ## Final Answer Discipline |
| 21 | 237 | -0.012 | +0.087 | +2.00 | ✅ 有用 | - Before finalizing, enforce the requested unit and format: convert th… |
| 22 | 95 | -0.029 | -0.029 | +1.00 | · ≈0 | - Return the final answer only after one last consistency check agains… |
| 23 | 88 | -0.076 | +0.017 | +0.33 | · ≈0 | - Copy the final answer from a checked value, not from an unverified i… |
| 24 | 49 | -0.041 | -0.006 | +0.00 | · ≈0 | ## Statistical and Time-Series Calculation Checks |
| 25 | 567 | -0.023 | -0.366 | +2.00 | ❌ 死重/有害 | - Before computing any statistic, write the intended formula and denom… |
| 26 | 314 | -0.041 | -0.180 | +1.67 | ❌ 死重/有害 | - For long inclusive ranges, first enumerate the expected count of obs… |
| 27 | 268 | -0.035 | -0.017 | +1.00 | · ≈0 | - When a page contains multiple nearby sections with similar labels, u… |
| 28 | 469 | -0.041 | -0.070 | +1.00 | ❌ 死重/有害 | - For Treasury security quotations, obey the table's quote basis. If t… |
| 29 | 28 | -0.035 | -0.017 | +0.00 | · ≈0 | ## Stricter Final Formatting |
| 30 | 373 | +0.047 | +0.174 | +2.00 | ✅ 有用 | - Match any requested output template exactly. Unless the prompt expli… |
| 31 | 51 | -0.047 | -0.017 | +0.00 | · ≈0 | <!-- SLOW_UPDATE_START --> <!-- SLOW_UPDATE_END --> |

## 读法

- **#30 独大**：add-one **+0.174**（输出格式对齐规则）——ckpt 唯一稳健高价值单元，与复现 e16 的 #17 同构。
- **长统计/时序规则 = 死重/有害**：#25（add-one -0.366）, #16（add-one -0.326）, #18（add-one -0.227） 单独加明显掉分。
- 其余多在噪声带内（≈0 冗余）。**价值高度集中在一条格式规则**。

## Baseline 对比：LLM-judge 测不出有害单元

- **LLM-judge vs add-one Spearman = -0.234（负相关）**；vs LOO = +0.255。
- **LLM-judge 从不打负分**（范围 +0.00..+2.00），11/32 判 essential——无法区分。
- **最毒的单元被判成最该保留的**：

| unit | add-one | LLM-judge | |
|---|--:|--:|---|
| #25 | -0.366 | +2.00 | ❌ 反了 |
| #16 | -0.326 | +2.00 | ❌ 反了 |
| #18 | -0.227 | +2.00 | ❌ 反了 |
| #30（格式） | +0.174 | +2.00 | ✅ 都对 |

> **结论**：只有**扰动式评测**（add-one/Shapley）能测出表面好、实则有害的单元；**LLM-judge 与 LOO 都漏**。
