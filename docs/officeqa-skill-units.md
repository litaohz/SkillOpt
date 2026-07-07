# OfficeQA e16 best_skill —— 单元编号 ↔ 原句对照

> 源文件:`skillopt-assets/officeqa-e16-best_skill.md`(被归因的 19 单元 skill)。
> 数值均为 **172 题 test 集、确定性 EM**。full=0.692,empty(无 skill)=0.547。
> LOO Δ = 删掉这条掉多少分(越大越不可或缺);add-one Δ = 在 empty 上只加这条涨多少分。
> ⚠️ 噪声带:172 题时单次 EM 噪声 σ≈0.035,两次之差 σ≈0.05,故 **|Δ|≲0.03 基本在噪声内、视为 ≈0**。

| # | 字符 | LOO Δ | add-one Δ | 判定 | 原句 |
|---|--:|--:|--:|---|---|
| 0 | 16 | -0.012 | -0.017 | · ≈0(冗余/噪声内) | # OfficeQA Skill |
| 1 | 23 | +0.006 | -0.023 | · ≈0(冗余/噪声内) | ## Retrieval Discipline |
| 2 | 84 | +0.000 | -0.012 | · ≈0(冗余/噪声内) | - Start by narrowing to the most likely candidate file before reading long passages. |
| 3 | 111 | -0.017 | -0.047 | · ≈0(冗余/噪声内) | - Prefer targeted search terms that name the exact entity, period, measure, or table concept from the question. |
| 4 | 120 | -0.029 | -0.017 | · ≈0(冗余/噪声内) | - After a promising match, read only a small surrounding span and verify it matches the requested year, basis, and unit. |
| 5 | 235 | -0.012 | -0.047 | · ≈0(冗余/噪声内) | - When the question names a chart or graph, treat the plotted chart data/crossover as the target evidence; do not substitute nearby narrative summaries or adjacent tables unless they contain the same measure, period, and chart concept. |
| 6 | 22 | -0.023 | -0.017 | · ≈0(冗余/噪声内) | ## Evidence Discipline |
| 7 | 78 | +0.006 | +0.035 | · ≈0(冗余/噪声内) | - Extract the exact value from the retrieved text before doing any arithmetic. |
| 8 | 103 | -0.023 | -0.012 | · ≈0(冗余/噪声内) | - Keep track of each operand's period, unit, and semantic role so nearby proxy values are not mixed in. |
| 9 | 106 | +0.006 | -0.017 | · ≈0(冗余/噪声内) | - If the question asks for a transformed or derived quantity, compute only after confirming every operand. |
| 10 | 275 | -0.006 | -0.192 | ❌ 独立有害 | - For range-based or time-series calculations, make a checked operand list with the expected count (months, fiscal years, or year-to-year intervals) before computing; use population formulas when the prompt specifies population standard deviation or coefficient of variation. |
| 11 | 213 | -0.012 | -0.041 | · ≈0(冗余/噪声内) | - When the prompt names inclusions, exclusions, retirements, revisions, or special conventions, read the nearby table notes/footnotes and choose the row or column whose definition already matches those conditions. |
| 12 | 296 | -0.023 | -0.297 | ❌ 独立有害 | - Before statistical calculations, write down the exact requested series and formula, including whether rates are percentages, percentage points, decimals, annualized rates, or period rates; convert annualized quarterly rates to quarterly multipliers before compounding or taking geometric means. |
| 13 | 276 | -0.012 | -0.070 | ❌ 独立有害 | - For tail-risk/loss questions, compute the requested return/change distribution first, apply the specified tail probability with the correct sign convention so loss is positive, and perform currency/unit conversion only once at the end using the requested date and frequency. |
| 14 | 26 | -0.012 | -0.012 | · ≈0(冗余/噪声内) | ## Final Answer Discipline |
| 15 | 95 | -0.029 | -0.052 | ❌ 独立有害 | - Return the final answer only after one last consistency check against the retrieved evidence. |
| 16 | 88 | -0.029 | +0.029 | · ≈0(冗余/噪声内) | - Copy the final answer from a checked value, not from an unverified intermediate guess. |
| 17 | 267 | +0.145 | +0.174 | ✅ 有用(保) | - Match the requested output form exactly: if the prompt asks for just a year, decimal, rounded number, or bracketed list, return only that value/list and omit units, prose, qualifiers such as "around," and labels like "percentage points" unless explicitly requested. |
| 18 | 51 | -0.029 | -0.047 | · ≈0(冗余/噪声内) | <!-- SLOW_UPDATE_START --> <!-- SLOW_UPDATE_END --> |

**判定口径**:✅ 有用 = LOO Δ > 0.03;❌ 独立有害 = add-one Δ ≤ −0.05(单独加明显掉分);其余在噪声带内,视为 ≈0(冗余)。

**一句话**:19 条里只有 **#17** 真有用(LOO +0.145 / add-one +0.174);**#10/#12/#13/#15** 单独加明显有害(长算术规则,−0.05~−0.30);其余 ≈0 冗余。
---

## 四列对比:一阶 vs 交互(含 Shapley)

> Shapley ±SE = 10 排列 Monte-Carlo(seed0×10,`--method shapley --seeds 0 --perms 10`,复用一阶缓存;SE 由多排列样本估计,增量续跑自 perms=5)。
> **Efficiency 公理自检:Σ Shapley = +0.1453 = full−empty = +0.145 → PASS**(实现正确)。
> ⚠️ 归因**稀疏**——signal 几乎全在 #17;n=10 后 SE 收窄,#7/#16 已确定为正(见读法)。缓存实测:中段 size 5–18 几乎零命中(撞车概率 1/C(19,k)≈1e-5),故 perms 越大越贵、但结论早在 perms=3 就定型。

| # | LOO Δ | add-one Δ | Shapley ±SE (n=10) | 原句(节选) |
|---|--:|--:|--:|---|
| 0 | -0.012 | -0.017 | -0.011 ±0.007 | # OfficeQA Skill |
| 1 | +0.006 | -0.023 | -0.001 ±0.006 | ## Retrieval Discipline |
| 2 | +0.000 | -0.012 | +0.011 ±0.010 | - Start by narrowing to the most likely candidate fi |
| 3 | -0.017 | -0.047 | -0.012 ±0.007 | - Prefer targeted search terms that name the exact e |
| 4 | -0.029 | -0.017 | +0.004 ±0.014 | - After a promising match, read only a small surroun |
| 5 | -0.012 | -0.047 | -0.011 ±0.012 | - When the question names a chart or graph, treat th |
| 6 | -0.023 | -0.017 | -0.008 ±0.007 | ## Evidence Discipline |
| 7 | +0.006 | +0.035 | +0.037 ±0.008 | - Extract the exact value from the retrieved text be |
| 8 | -0.023 | -0.012 | +0.019 ±0.015 | - Keep track of each operand's period, unit, and sem |
| 9 | +0.006 | -0.017 | +0.004 ±0.007 | - If the question asks for a transformed or derived  |
| 10 | -0.006 | -0.192 | -0.048 ±0.025 | - For range-based or time-series calculations, make  |
| 11 | -0.012 | -0.041 | +0.007 ±0.007 | - When the prompt names inclusions, exclusions, reti |
| 12 | -0.023 | -0.297 | -0.091 ±0.032 | - Before statistical calculations, write down the ex |
| 13 | -0.012 | -0.070 | -0.004 ±0.012 | - For tail-risk/loss questions, compute the requeste |
| 14 | -0.012 | -0.012 | +0.009 ±0.005 | ## Final Answer Discipline |
| 15 | -0.029 | -0.052 | +0.016 ±0.016 | - Return the final answer only after one last consis |
| 16 | -0.029 | +0.029 | +0.031 ±0.010 | - Copy the final answer from a checked value, not fr |
| 17 | +0.145 | +0.174 | +0.208 ±0.020 | - Match the requested output form exactly: if the pr |
| 18 | -0.029 | -0.047 | -0.013 ±0.007 | <!-- SLOW_UPDATE_START --> <!-- SLOW_UPDATE_END --> |

**读法**:#17 全方法领先且绝对主导(**+0.208 ±0.020**,n=10,信号≈10×SE);#10(**−0.048 ±0.025**)/#12(**−0.091 ±0.032**)add-one 与 Shapley 均明显负=真死重;#7(**+0.037 ±0.008**,4.4×SE)/#16(**+0.031 ±0.010**,3.2×SE)LOO≈0 但 Shapley **确定转正**=被一阶低估、被交互捞回(perms=10 后 SE 收窄已牢固站住正号;#8/#15 仍在噪声带内)。

---

## 时间 × 空间稀疏性:e16 的全部增益 = 单次编辑

> 数据源:`outputs/train_officeqa_gpt55_e16/{summary.json,history.json,skills/skill_v*.md}`。
> **全部现成,零新增评测。** 图:`docs/officeqa-e16-sparsity.png`。

我们归因的 e16 best_skill **本身就是"时间稀疏性"的产物**,而 Shapley 是"空间稀疏性"(一阶 LOO/add-one)的升级——今天把两条轴接成了**同一事件**:

**① 时间稀疏(训练轨迹)**:32 步优化**只有 step 4 一次被接受**(`total_accepts=1`,`best_step=4`)。
best 选择分 0.708→**0.792 在 step 4 到顶后锁死**,其后 28 步全 reject、best 纹丝不动。
`best_skill.md`(=`best_origin=step_0004`,2512 字)就是这一步冻结的产物;"e16"标的是**训练跑了
16 epoch**,但 best 早在 **epoch 2/step 4** 定型,后 14 epoch 一无所获。

**② 尺寸稀疏(反直觉)**:step 4 之后 optimizer 把"current"skill 探到 **18.8K 字(赢家的 7.5×)
全被拒**——**小的早期版本打败了所有更大的**。训练末尾的肥版(`slow_update_epoch_16`,~20K 字)
并未被选中。

**③ 空间稀疏(Shapley,见上表)**:那份 2512 字/19-unit 赢家里,**~1 条(#17)独扛**,其余 ≈0 或负。

**三者是同一事件**:溯源 skill 版本发现——`skill_v0000–v0003`(step 1–3)**不含**输出格式规则(best 0.708);
`skill_v0004`(step 4)**首次出现该规则**、best 跳到 0.792。即 **step-4 那次唯一 accept 的内容,
正是引入了 Shapley 里独大的 #17**。

> **一句话**:SkillOpt 本次 OfficeQA 的全部增益 = **某早期步骤发现了一条输出格式规则**;其余 ~96%
> 的优化步与 ~90% 的文本是 bloat(含 #10/#12 这类负贡献)。Shapley(空间)+ 训练 history(时间)+
> 版本溯源(时空连接)三路三角互证。
> **局限**:n=1 run、且是复现的 e16(非官方 `ckpt` 32-unit skill)。要成为关于"SkillOpt"的一般结论,
> 需在官方 skill / 其他 env / 其他方法上复验(见 methodology 文档待办)。
