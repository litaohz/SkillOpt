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
