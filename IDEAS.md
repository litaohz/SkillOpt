# Ideas / 研究思路

## 用 MCTS 分析 skill 各部分的作用（Eval / 归因框架）

**核心想法**：把一个 skill 文档拆成若干「部分」（section / bullet / rule），
用 MCTS（蒙特卡洛树搜索）在「包含/剔除哪些部分」的组合空间里搜索，
通过 rollout 的实际评测得分来判断每个部分到底起了什么作用。

**定位**：这是一个分析方法 + eval framework，而不是训练方法——
目标是给已有 skill 做**归因（attribution）/ 可解释性**，量化每个片段的边际贡献。

### 为什么用 MCTS
- skill 部分之间存在**交互/依赖**（某条规则只有配合另一条才有用），
  单独做 ablation（一次删一条）会漏掉组合效应。
- 组合空间是指数级（2^N 个子集），全搜不可行；MCTS 能在大空间里
  优先探索高价值组合，用 UCT 平衡探索/利用。

### 框架草图
- **状态**：当前选中的 skill 片段子集。
- **动作**：加入 / 移除某个片段。
- **reward**：用该子集组成的 skill 跑 benchmark rollout 的得分（可复用
  现有 envs 的评测管线，如 officeqa / docvqa / spreadsheetbench）。
- **输出**：每个片段的贡献值（如 Shapley 风格的边际贡献、出现在高分组合里的频率），
  以及最优/最小有效子集。

### 可能的产出
- 片段重要性排序 → 指导 skill 精简（删冗余）/ 强化（保留关键）。
- 发现负作用片段（拖累得分的规则）。
- 为 SkillOpt 的 rewrite / merge 流程提供归因信号。

### 待办 / 开放问题
- reward 噪声大、rollout 贵 → 需要预算控制、缓存、共享 rollout。
- 片段粒度怎么切（按 section？按 bullet？）。
- 和经典 ablation / Shapley value 估计的对比基线。
- 是否结合现有 `skillopt/prompts/ranking*.md` 的排序信号。

---

## 方向 4（最高优先级）：用 causal 归因指导 optimizer —— 从局部 SGD 到全局优化

**动机（来自 OfficeQA 复现观察）**：SkillOpt 现在的 reflect 是「带噪的一阶局部梯度」，
meta-skill 的「which edits helped / failed」是 **optimizer 自报的主观猜测**，没有客观度量；
epoch 末 slow_update 还会 force-accept 未验证的内容 → 16-epoch 里 32 步只有 1 步真正有效、
skill 却膨胀 21×（见 `OVERVIEW.md`）。

**核心想法**：把句子级 / 子集级的 **实测因果价值**（LOO / add-one / Shapley，已在
`scripts/eval_skill_ablation.py` 落地一阶基线）反哺进优化器：

- → 注入 `meta_skill_context` 与 analyst 的 user prompt：让 optimizer 知道现有 skill 里
  **哪几句真有价值（保）、哪些是死重（删）、哪些有害（必删）**，生成 edits 时有据可依。
- → 给 consolidation / compaction 一个**客观剪枝信号**，替换现在 LLM 的盲目压缩 → 治「膨胀」。
- → 框架定位：SkillOpt 的 reflect ≈ greedy-local；causal 信号 ≈ **全局 credit assignment**，
  相当于给优化器加「哪些参数真有用」的导航/正则。

**最小闭环**：`ablation/Shapley 算每句价值 → 结构化注释挂回 skill/meta → optimizer 下一步据此 prune+edit`，
形成「测量 → 归因 → 指导更新」的循环。

**与现有代码的接口**：
- 归因器：`scripts/eval_skill_ablation.py`（扩 Shapley/MCTS）。
- 注入点：`skillopt/gradient/reflect.py:309,328`（analyst user prompt）、
  `skillopt/optimizer/meta_skill.py`（meta-skill context）。
- 评测原语：`scripts/eval_only.py`（确定性 EM）。

### 暂缓的对照实验（research notes，优先级低于方向 4）
1. slow_update 加 EM gate（`optimizer.slow_update_gate_with_selection=True`）的 A/B → 坐实「强制更新致膨胀」。
2. 同一 skill 重复 rollout → 量化 gate 的 rollout 噪声 / 误拒率（EM 确定性，噪声在生成端 + 24 题小样本）。
3. `eval_skill_ablation --versions-dir` 在测试集上扫 16-epoch 各版本 → 验证膨胀对精度「无害但费 token」。

