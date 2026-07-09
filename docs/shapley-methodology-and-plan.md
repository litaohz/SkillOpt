# Shapley 归因：方法论讨论纪要 + 下一步计划

> 本文落盘 2026-07-03/04 关于「skill-unit Shapley 归因」的讨论，并给出后续实验计划。
> 相关代码：`scripts/skill_combo.py`（`--method shapley` / `--method pairwise`）、
> `scripts/skill_attribution.py`（`split_units` / `render` / 一阶 LOO·add-one）。
> 相关结果：`docs/officeqa-skill-units.md`、`docs/_shapley_table.md`、
> `outputs/combo_e16best_shapley/`（含 `eval_cache.jsonl` 与 48 个子集 eval）。
> 学术参照：`~/DataShapley`（Ghorbani & Zou, *Data Shapley*, ICML 2019）。

---

## 1. 我们的 Shapley 是怎么做的（现状）

把一份 skill 文档当作**合作博弈**：

- **玩家 = skill 的原子 unit。** `split_units()` 按 markdown 结构把 `skill.md` 拆成
  原子单元——每个 header 行 / 列表项 / 空行分隔段落各一个 unit（OfficeQA e16 → 19 个）。
- **价值函数 v(S) = 子集 S 渲染出的 skill 在整个 valid split 上的 benchmark `hard` 分**
  （OfficeQA=EM，DocVQA=ANLS）。`render(units, sorted(keep))` **始终按原文档顺序**拼接
  被选中的 unit（`sorted` 是关键：渲染顺序与采样顺序解耦，v 因此是良定义的**集函数**）。
  空集 = 空 skill 分。
- **φ_i = permutation-sampling Monte-Carlo 估计**：每个随机排列里逐条 append，
  `phi[idx] += v(前缀∪{idx}) − v(前缀)`，多排列取平均。
- **全程磁盘缓存**：每个子集按 skill 文本 SHA-256 哈希缓存分数（`eval_cache.jsonl`），
  跨方法 / 跨 run 共享，`--reuse-from` 可从一阶结果 backfill —— 同一子集永不重复评测。

**配套的二阶方法** `--method pairwise`：`I(i,j)=s(full)−s(\i)−s(\j)+s(\{i,j})`，
I<0=替代/冗余，I>0=协同/互补 —— 一阶 LOO/add-one 判不出的东西。

**已知实测（OfficeQA e16，172 题 test，full=0.692，empty=0.547）：**
19 条里只有 **#17「Match the requested output form exactly」稳健为正**
（LOO +0.145 / add-one +0.174 / Shapley +0.174，三口径一致）；
#10/#12/#13/#15（长算术规则）单独加明显有害；其余 ≈0 冗余。
**Efficiency 自检通过**：ΣShapley=+0.1452 ≈ full−empty=+0.145。

---

## 2. 与学术定义（`~/DataShapley`, ICML'19）的对齐

**✅ 核心算法逐行一致**（就是 Shapley 的 permutation-sampling 定义）：

| 环节 | DataShapley `one_iteration` | 我们 `skill_combo.py` |
|---|---|---|
| 随机排列 | `np.random.permutation` | `rng.shuffle(order)` |
| 空联盟基线 | `new_score = self.random_score` | `prev = empty`（空 skill 分） |
| 逐个加入 | `for idx in idxs:` 累加 refit | `for idx in order: S.append(idx)` |
| 边际贡献 | `marginal = new_score − old_score` | `phi[idx] += cur − prev` |
| 多排列平均 | `vals_tmc = np.mean(mem_tmc, 0)` | `phi[i] / cnt[i]` |

**⚠️ 三个有意的差异（都不违背定义，是工程取舍）：**

1. **截断（TMC 的「T」）——我们没做。** 参考实现当前缀分数逼近满分
   （`|new−mean| ≤ tolerance·mean` 连续 >5 次）就提前 break、剩余边际记 0，是**加速近似**。
   我们不截断 → 更精确、但更贵。方向上更严格，不是偷工。
2. **停止准则：我们写死 `--perms`，参考用 `error(mem_tmc) < err` 收敛判据。**
   所以 perms=3 远未达到它的收敛门槛。
3. **grouping/sources：我们退化为每 unit 一源（size=1，无需除）。**
   参考支持把多点归为一 source 并 `marginal /= len(source)` —— 这正对应「section 分组」。

参考实现还用 **bagging 估 `self.tol`（分数标准差）** 来判收敛/截断；**我们目前没有量化
φ 的方差 / 误差棒**（这是下一步要补的第一件事）。

---

## 3. 三个理论问题（讨论结论）

### Q1. perms 有没有理论上合理的值？

有，服从标准采样理论。

- **方差视角（实用）**：`SE = σ_i/√m`，要压到 ε 需 **m ≈ (σ_i/ε)²**。
  经验 σ≈0.1（EM 尺度），ε=0.02 → **m≈25**。故默认 20 勉强够；
  **perms=3 的 SE≈0.1/√3≈0.058**，比中间那些 |φ|<0.03 还大 → 中间名次不可信。
- **一致性上界（保守）**：Maleki et al. (2013) Hoeffding：
  让全部 n 个 unit 同时 |φ̂−φ|≤ε（概率 1−δ）需 `m ≥ (r²/2ε²)·ln(2n/δ)`；
  代入 r≈0.3、ε=0.02、δ=0.05、n=19 → m 约几千（最坏情况，实际远小）。
- **没有魔数**：理论上「合理的 perms」应**用收敛判据检测**（参考实现 `error(mem_tmc)<err`），
  不是拍一个数。**成本上界会饱和**：n=19 精确 Shapley 需 2¹⁹≈52 万子集不可行，
  但按内容哈希缓存后，加大 perms 会开始复用旧前缀，新增评测数饱和在 ~min(perms·n, 2ⁿ)。
- **漂亮的不变量**：每个排列内边际贡献望远镜求和 → `Σ_i marginal = v(full)−v(empty)`
  **每次迭代精确成立**（efficiency 公理精确满足）。噪声只在个体间重分配总额。

### Q2. unit 是一种划分（partition）；异构划分会怎样？

- Shapley **相对于固定玩家集**定义，换 partition = 换博弈，φ 不守恒地改变，**无划分无关性**。
- 我们的 units 本就异构（整行标题 / 单条 bullet / 空占位注释 #18）→ **粒度偏置**：
  内容多的 unit 天然 |φ| 更大，仅因改动文本多；φ **没按 unit 大小归一**，跨异构 unit
  直接比大小是「苹果比橘子」。想公平比较应报 **φ/token** 或做成同质粒度。
- **空占位 #18 是 null player**：空玩家公理要求 φ=0，实测 #18=−0.008≈0 → 实现正确的免费自检。
- **唯一跨 partition 守恒量**：`Σφ_i = v(full)−v(∅)`。异构只是**重分配 credit**，不改总额。

### Q3. 先分 section 再分 unit —— Owen value（两级 Shapley）

这是有名字的构造：**Owen value**（Owen 1977, *a priori unions*）。

- **算法**：(1) 商博弈——把每个 section 当玩家做一次 Shapley → section 级价值；
  (2) section 内部——把该 section 价值用内部 Shapley 分给其 units，但**只遍历尊重联盟结构的
  排列**（其他 section 整块在场 / 整块不在）。
- **正好解决「挖空文档」问题**：Owen ≡ 只采样「每个 section 的 units 在排列中保持**连续**」
  的排列 → 中间联盟更连贯，依赖上下文的 unit 不会被无谓扒掉父标题 → 偏置和方差都更小。
- **天然两级分解**：section 级（哪段最有用）+ 段内 unit 级（段内哪条最有用）。
- **退化**：单 section 或全单元素 → Owen = 普通 Shapley；efficiency 依旧 `Σφ=v(full)−v(∅)`。
- **采样版很便宜**：先随机排 sections，再在每个 section 内随机排 units（分层洗牌）。
  这是参考实现 `sources`/grouping（一级）的**嵌套两级推广**。

---

## 4. 下一步计划（按优先级）

> 目标：把「中间名次是否可信」从定性说法变成**带误差棒的硬结论**，并用结构感知的
> Owen value 降噪、给出更贴合 skill 层级的 section↔unit 两级归因。

### P1 — 误差棒 / 收敛（对应 Q1）✅ 完成
- [x] `skill_combo.py` 支持 **`--seeds` 多 seed**；对每个 φ_i 报 **均值 ± 标准误**；
      `shapley.csv` 增列 `se`、`samples`。
- [x] **efficiency 自检打印**（ΣShapley vs full−empty）作为回归守卫 → PASS。
- [~] 原计划加大到 perms=20;**实测发现中段 size 5–18 几乎零缓存命中**(撞车概率 1/C(19,k)，
      中段≈1e-5），跑满 20 排列≈2–3 天且不改结论。**增量续跑至 perms=10(seed0×10)收口**
      （perms=5→10 只新算 perm5–9，前 5 个排列全命中缓存零重跑）：归因**稀疏**，signal 几乎
      全在 #17（**+0.208 ±0.020**，n=10，≈10×SE），#10/#12 稳健负；**#7（+0.037 ±0.008，4.4×SE）
      / #16（+0.031 ±0.010，3.2×SE）确定转正**（一阶 LOO≈0 却被交互捞回，n=10 后 SE 收窄牢固站正）。
- **结论**：`docs/officeqa-skill-units.md` 四列表已刷成 Shapley ±SE(n=10)。efficiency PASS
      (Σφ=+0.1453=full−empty)。20 排列非必需——结论早在 perms=3 定型，perms=10 仅把 #7/#16
      的正号从"疑似"变"确定"。

### P2 — Owen value / 两级 Shapley（对应 Q3）— 代码✅ / 数据run 暂缓
- [x] `--method owen`：**通用分层洗牌**（先洗 section，再洗 section 内 unit，段内连续）；
      `_section_groups` 按 markdown header 切 section；输出**两级**（section 商博弈 + 段内 unit）；
      efficiency 自检共用。已单测（4 section、段内连续、dry-run 通过）。
- [ ] **数据 run 暂缓**（用户决定，后面再推进）。
- **待量化的「缓存红利」**（P1 实测引出）：flat-Shapley 中段每条排列都产出全新子集（零命中）；
      Owen 分层把可达子集限制为「若干完整 section + 某 section 的前缀」，中段 distinct 子集数
      **远小于 C(19,k)** → 命中率大增。所以 **Owen 不仅降方差，还顺带省评测**；跑 Owen 时专门
      对比「distinct 子集数 / 命中率」相对 flat 的下降。

### P3 — 异构/归一化诊断（对应 Q2）
- [ ] `shapley.csv` 增列 `chars`/`tokens` 与 `shapley_per_100char`，标注 null-player
      自检（空占位 unit 的 φ 是否≈0）。
- [ ] 文档说明：跨异构 unit 只比「占总提升份额」，不比绝对 φ。

### P4 — 收尾
- [ ] 用新结果刷新 `docs/officeqa-skill-units.md` 的四列对比表（加 SE 列 + Owen 列）。
- [ ] 双推同步：`litaohz/SkillOpt`（personal branch）+ `gim-home/skillopt`（同名 branch 且合入 main）。

### P5 — 论文用途:作为 ablation study(用户 2026-07-07 定位)
这块归因工作定位为**未来论文的 ablation section**,已沿多轴消融:
- **内容(空间)**:LOO/add-one/**Shapley** 逐 unit → skill 可压到 ~1 条(#17),#10/#12 负贡献;
- **优化步(时间)**:训练 history → 32 步仅 step4 有效,gain 全来自单次编辑;
- **尺寸/bloat**:18.8K 字全被拒,2.5K 早期版夺冠(大≠好);
- **模板 Rules**:裸 prompt vs 模板 → Rules 抬高 baseline +25.9 EM(前期复现)。
论点:**skill 优化收益稀疏且可归因——大部分学到的 skill 惰性甚至有害**。见
`docs/officeqa-skill-units.md` 的「时间×空间稀疏性」节 + 图 `docs/officeqa-e16-sparsity.png`。
- **要 paper-grade 还差泛化(当前 n=1、且是复现 e16 非官方 ckpt)**:
  - [ ] 官方 `ckpt/<env>/gpt5.5_skill.md`(6 env)各跑一张 Shapley → SkillOpt 官方口径消融;
  - [ ] 多 env 复验"1 步定胜负 / 1 unit 独大"是否普遍;
  - [ ] **跨方法**(TextGrad/AutoSkill/…)best skill 各跑一张 = comparative ablation
        (需先拿到各方法产出的 skill:问作者要 artifact,或复现开源的 AutoSkill/TextGrad)。

### P6 — OfficeQA ablation 执行计划(focus:先 1 后 3,2026-07-07)
只聚焦 OfficeQA,暂不做多 env(P5 的 multi-env 推后)。评测统一:`configs/officeqa/default.yaml`
`--split valid_unseen --split_dir data/officeqa_split`(=172 题 test),gpt-5.5 @ localhost:4141,
`env.workers=12`——与 e16 归因**同口径**,可直接对比。

**Phase 1｜官方 ckpt OfficeQA skill 归因(32 units)**
- **1a｜一阶(LOO+add-one)**:`skill_attribution.py --skill ckpt/officeqa/gpt5.5_skill.md
  --modes loo addone --out-root outputs/attrib_ckpt_officeqa`。= full+empty+32 LOO+32 add-one ≈ 66 评测
  ≈ ~20h(detached + 巡检 + 断点续)。**先出这个**——已足够回答"价值是否集中在格式规则/是否稀疏"。
- **1b｜Shapley(perms=3)**:`skill_combo.py --method shapley --seeds 0 --perms 3
  --reuse-from outputs/attrib_ckpt_officeqa`,复用 1a 缓存,只补中段子集。
- **产出**:ckpt 的四列归因表 + 与 e16 对比(是否同样"格式规则独大"、稀疏度、死重占比)。
  注:ckpt 是作者发布的 artifact,**无本地训练 history → 只有空间维度**(时间维度是 e16 独有)。

**Phase 2｜OfficeQA 跨方法比较归因(abl-crossmethod)**——阻塞于拿到各方法的 OfficeQA best skill
- 2a:获取 TextGrad / Trace2Skill / AutoSkill / GEPA 在 OfficeQA 的产出 skill(问作者要 artifact,
  或复现开源 AutoSkill/TextGrad);
- 2b:同口径跑一阶+Shapley;
- 2c:comparative 表——稀疏度 / 死重占比 / 是否都收敛到"输出格式"规则 / 每-unit 价值。

### P7 — "标准 skill" 定义 & SSG showcase 选型(2026-07-09,关键)
**发现**:我们归因的 skill **不是"标准 Agent Skill"**。权威定义(agentskills.io/specification):
> 一个 skill = **目录**,含 `SKILL.md`(YAML frontmatter:**必填 `name`(≤64,小写-连字符)+ `description`
> (≤1024,说明做什么/何时用,带关键词供 agent 识别)**;选填 license/compatibility/metadata/allowed-tools)
> + 可选 `scripts/`(可执行代码)`references/`(文档)`assets/`(模板)。**渐进披露**:agent 先扫
> name+description,再按需加载正文/脚本/引用。

对照本仓库:
- **OfficeQA**(best_skill / ckpt):**扁平 .md 指令表**,无 frontmatter、无 name/description、无 scripts/references
  → **完全非标准**,τ 全是 instruction、D≈∅、H 两层。
- **SpreadsheetBench**(ckpt):扁平 .md **但含内联 python 代码块 + `solution.py` 模板 + INPUT_PATH/OUTPUT_PATH
  的 def-use** → **半结构化**(doc-2 的 xlsx-skill 例子就是拿它扩的),但仍是单文件、无 SKILL.md 目录。
- 结论:**SkillOpt 所有 env 的 skill 都不是标准 Agent Skill**。

**含义(对 SSG)**:扁平 skill 上 `G=(N,D,H)` **退化 → SSG ≈ 普通 Shapley**(D 空、H 浅、τ 单一、
Read-Null 无复用)。所以 OfficeQA 是"无结构角落":适合**诊断/修复应用 + 作为"无结构对照"**,但**不是
SSG 独特机器(D-closure / Read-Null / blast-radius / τ 分派)的用武之地**。

**标准 skill 的 benchmark = SkillsBench**(arXiv **2602.12670**,`benchflow-ai/skillsbench`):SKILL.md 目录式
skill、87 任务/8 域、**三条件(No Skills=V(∅) / Curated / Self-generated)**、指标含 **skill-selection-with-
distractors + compositionality**。= **理想 SSG showcase**(真 D/H/τ/Read-Null + 免费 V(∅) + distractor 正好接
我们的路由/跷跷板)。

**决策**:OfficeQA(扁平)= 诊断应用 + 无结构对照(现有 baseline 已备);**SSG 主 showcase → SkillsBench**
(或把 SpreadsheetBench skill 编译成目录形式作过渡)。工具:`scripts/show_trajectory.py` 可读任意 UID/变体的
agent 工具调用轨迹(grep/read/glob),`conversation.json` 的 `{type:tool_call,cmd,obs}` 已含读取日志雏形
(Read-Null 埋点的一半)。

---

## 5. 关键文件索引

- 归因主脚本：`scripts/skill_combo.py`（shapley / pairwise）、`scripts/skill_attribution.py`
  （split_units / render / build_base_eval_cmd / 一阶 LOO·add-one）
- 评测入口：`scripts/eval_only.py`
- 现有结果：`outputs/combo_e16best_shapley/`（`eval_cache.jsonl` + `evals/<hash>/{skill.md,eval_summary.json}`）、
  `outputs/combo_e16best_pairwise/`
- 对照表：`docs/officeqa-skill-units.md`、`docs/_shapley_table.md`
- 被归因的 skill：`skillopt-assets/officeqa-e16-best_skill.md`
- 学术参照：`~/DataShapley`（`DShap.py` 的 `one_iteration` / `_tmc_shap` / `_tol_mean_score`）

---

## 6. 长远愿景（Dream）—— 比 Owen value 更抽象的框架

> 记录 2026-07-04 讨论中逐步浮现的大图景，供后续继续讨论。**尚未定稿。**
> 一句话：把问题从「给一份 skill 做归因」重定义为
> **「在一个异构、多级、渐进加载的知识产物层次里，做 credit assignment 与最优放置」**。

### 6.1 统一命题

所有分支——skill 内 section→unit、harness 层放置、跨任务 skill、skill↔memory——
是**同一个问题在不同尺度**：一个**多级嵌套划分**上的合作博弈价值分配 + 放置优化。

```
memory  vs  skill            ← 类型/作用域(事实性 vs 程序性)
  └─ task / domain           ← OfficeQA / DocVQA / coding / ...
       └─ 单个 skill
            └─ skill 内 unit
```

- **数学对象的升级链**：Shapley(无结构) → **Owen value**(一级先验联盟) →
  **Winter Level Structure Value / LS-value, 1989**(任意层嵌套划分)。
  上面这个四级层次，精确对应 Winter LS-value。P2 的通用分层采样器就是它的最小实现，
  逐层加 strata 即可上探到 harness / task / memory 级。

### 6.2 跷跷板效应（seesaw / negative transfer）

- 把多个任务的 skill 放一起联合优化/评测时，改善一个任务可能拖垮另一个 = 负迁移。
- **仪器 = 任务 × unit 的向量值边际贡献矩阵**：对角线=自任务价值，
  **非对角=跨任务迁移（正=协同，负=跷跷板）**。
- **致命约束：v 千万别标量化**（多任务分数一平均，跷跷板当场被抹平）。
- **两个跷跷板要分开**：
  - **推理时**：库固定，eval 时加 skill_A 拉低 task_B —— 现有 Shapley/Owen 工具直接测。
  - **优化时**：联合优化共享库时为 A 改写共享 unit 而 regress B —— 需联合优化 + 逐任务跟踪。
  - 打法：用推理时归因**诊断**，据此**缓解**优化时跷跷板（冻结高跨任务正价值 unit、
    隔离/路由高干扰 unit）。

### 6.3 渐进加载 → 冲突在路由层，不在内容（关键架构约束）

生产 agent（如 Hermes）的 skill **渐进式加载**：只有 `name + description` 常驻 context，
内容按需加载。因此：

- **跨 skill 冲突/跷跷板发生在 `name+description`（路由层）**，不是内容；
  内容不互相打架、也不显著占 context（推翻了「上下文稀释」「内容互相干扰」假设）。
- **价值因式分解**：
  `V(skill_k) = P(正确路由到 k | name+desc) × V(content_k | 已加载)`
  - 左因子：归因目标是常驻 manifest 的 (name, description)；ablate 某条描述看路由分布偏移。
    **库级跷跷板住在这里。**
  - 右因子：就是现在对单份 skill 做的 unit 级 Shapley/Owen —— **照搬不变**。
- 推论：现在打磨的 OfficeQA unit 级归因是「右因子」；多任务时只需外套一层
  「description 路由归因」当「左因子」，两层解耦，**现有工作零浪费**。
- **检索混淆警告**：几十个 skill 不全在 context，有 retriever 在选；不拆开路由面，
  会把「没检中」误判成「skill 没用」。

### 6.4 skill↔memory：可证伪假设

- **skill↔skill（同域）** 倾向**替代**（负交互，冗余）；
- **skill↔memory** 倾向**互补**（正交互）：memory 给事实、skill 给程序，缺一不可。
- 若矩阵符号确为「域内 skill 负、skill-memory 正」，即得一个可指导架构的干净结论
  （程序性进 skill、事实性进 memory、跨题共性进 harness 模板）。

### 6.5 放置作为优化（把归因变成决策）

有了 per-unit φ + 跨面/跨任务交互 + per-layer 成本（always-on token 成本、可迁移性、
learned vs handwritten），「该放哪层」= 一个 assignment / facility-location 优化：
`place(u) = argmax_ℓ ( φ_u^(ℓ) − cost_ℓ(u) )`。
这给「把程序性 Rules 从 always-on 模板挪进 skill」等作者反馈一个**可优化的定量依据**。

### 6.6 研究弧线（按信噪比/成本，Dream 阶段，暂缓）

- **Phase A｜证实跷跷板**：3–4 任务各自优化后 skill，建向量值 task×skill 贡献矩阵，
  证明非对角负项存在。便宜、直接延续 OfficeQA、单独可成文。
- **Phase B｜机理归因**：pairwise/Owen + 中性 filler 对照，把负项判为 替代/干扰/稀释。
- **Phase C｜Hermes 生产**：条件 Shapley/LOO 做剪枝 + 干扰路由 + 层放置；显式建模路由面。
- **Phase D｜skill-memory**：memory 作独立 union 类型，验证 6.4 符号假设。

> **当前决定（用户）**：先把 OfficeQA 的 P1（误差棒）/P2（Owen）简单版做完收口，
> 再推进本节 Dream。本节 Phase A–D 暂列为 backlog。

### 6.7 复用 Data-Shapley 的迁移实验（Phase C 的首选切入）

> 来源：Ghorbani & Zou (2019) Data Shapley 论文的域自适应表——`Original`=用**全部** source
> 数据训练在 target 上；`Adapted`=用 **Shapley 筛过的** source 子集重训。5 个 source→target
> 全部提升（最猛 Email→SMS 68.4→86.4）。核心动作：**用 target 小验证集当 v()，给每个
> source 数据点算 Shapley，剔除负值点。**

**精确映射到 agent：**

| Data Shapley 迁移 | 我们的 agent |
|---|---|
| source 训练数据点 | 库里候选 **skill / unit**（可跨任务） |
| target 小验证集 | **目标 agent 任务**的小 val 集 |
| v() = target 上 model 性能 | v() = 目标任务 eval（skill 子集渲染后 rollout） |
| 点对 target 的 Shapley | skill/unit 对 target 的 **transfer Shapley** |
| 剔负值 → **重训** | 剔负 transfer → **重装库（无需重训）** |
| Original(全 source) vs Adapted(筛后) | 「全 skill 一股脑塞」 vs 「Shapley 筛后的库」 |

**为什么对 agent 更贴（更干净）：**
1. **无权重重训**——adaptation 只是重组库 + 跑 eval，v() 纯反映知识内容，无优化器噪声。
2. **直接回答生产问题**——"给定 target agent，几十个预置 skill 装哪些？"
3. **负值 skill = 跷跷板的可操作版**——Shapley 不只诊断跷跷板，还自带「删除即修复」处方。

**算法（现有 `skill_combo.py` 机器几乎不用改）：**
```
给定:候选 skill 集 K,target 小 val D_val,target held-out test D_test
1. v(S) = 用子集 S⊆K 组装库,在 D_val 上 eval
2. 对每个 skill_k 算 TMC/Owen Shapley φ_k (w.r.t. D_val)   ← 复用 skill_combo
3. 保留 φ_k>τ 的 skill(丢负值/低值)→ adapted 库 K*
4. D_test 上对比:全量 K vs K* vs 空库                    ← 即 Original vs Adapted 表
```
把 "units（一份 skill 的 unit）" 换成 "多份 skill"，`render` 换成 "组装选中 skill 的库"，
v() 指向 target eval；缓存 / efficiency 自检 / Owen 分层全部复用。

**三个必须警惕的坑：**
1. **target-val 过拟合** → 必须留 **held-out target test**（val 估值、test 报告，同论文）。
2. **渐进加载 → 估值粒度**：库策展选择对象是 `name+description`（路由层），库级 transfer
   Shapley 应在**路由粒度**算；unit 级 transfer 仅在「已装入」条件下有意义（接 6.3 因式分解）。
3. **token 成本**：每个 v() 是一次 LLM rollout（非训 logistic 回归）→ val 要小、
   TMC 截断 / Owen 分层降 perms 关键。

**一句话故事**：*Skill Shapley for cross-task transfer* —— 用 target 效用给 skill 估值、
筛出正迁移库，在 prompt 层复现 Data-Shapley 域自适应的收益，且无需重训。
