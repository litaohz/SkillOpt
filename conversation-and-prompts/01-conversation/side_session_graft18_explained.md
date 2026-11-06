# Parallel Status & L0-17 SFT + L18-31 OPSD Mechanism Explanation

- **Session ID**: `20261103_161746_d350b135`
- **Model**: `gemini-4-argon` (via GitHub Copilot Enterprise) + `gpt-6.1-sol (effort=max)` (for Weight-Space Mechanistic Diagnosis)
- **Date**: 2026-11-03 (GMT+8)
- **Mode**: Clean Dialogue (User & Assistant Turns)

---

## 🗣️ Turn 1 · Tao (`2026-11-03 16:17:46 GMT+8`, msg `#2251`)

帮我看看这个session 是正常的吗？

## 🗣️ Turn 2 · Tao (`2026-11-03 16:18:01 GMT+8`, msg `#2256`)

<https://rikiyat.github.io/blog/weight-space-programming/|Sparks of Weight-Space Programming: LLMs can natively read &amp; write neural network weights> opd已经过时了？

📎 [Sparks of Weight-Space Programming: LLMs can natively read &amp; write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/)
   We find that frontier LLMs are surprisingly good at natively reading and writing neural network weights, well past what we’d expect from humans or conventional algorithms.

## 🤖 Hermes Agent (`2026-11-03 16:18:40 GMT+8`, msg `#2263`)

**结论先行：OPD（On-Policy Distillation / OPSD）完全没有过时。** 这篇 MIT CSAIL（Omar Khattab 组，昨天 11 月 2 日刚发）的 *Sparks of Weight-Space Programming* 非常惊艳，但它干的其实是**自动化机械可解释性 + 定向表示手术（Mechanistic Surgery / Steering）**，而不是能替代梯度蒸馏的通用知识迁移。

原文自己在 **Experiment 8 的附加对照实验（Additional experiment）** 里把边界交底得非常诚实：

### 1. 为什么 Qwen3.5-0.8B 能从 7.9% 暴涨到 48.1%？（去路障 vs 加能力）
* **真实病因**：Qwen3.5-0.8B 原本就会算不少数学题，但它有严重的复读死循环（Thinking loop）——在 Base 模型里，**93.0% 的回答因为死循环撞上了 32,768 token 上限被截断**，导致算不出最终答案。
* **Astra 干了什么**：GPT-6 Astra 在 Python 沙盒里对比了正常推理和死循环样本的激活，定位到第 15 层和第 19 层 MLP 里负责写“死循环方向”的 64 个神经元（共 131,072 个权重，占总参数 0.016%），然后做了一个类似 **Refusal Direction Ablation（单方向投影消除，Arditi et al., 2024）** 的手术，把这个方向减掉了。
* **真实收益构成**：原文明确给出了统计——在救回来的 579 道题（rescues）里，**576 道原本是撞上 token 上限截断的题**；而**原本能正常结束输出的题目，准确率没有任何变化（unchanged）**。因为切掉的是通用复读回路，连带 IFEval 也从 21.0% 涨到了 62.3%。

### 2. 撞上“真不会”的模型：Gemma 3 270M 完全推不动
紧接着作者做了一个极其关键的对照（*Raising math capability of a model that does not loop*）：
* 他们换了 **Gemma 3 270M**——这个模型几乎不死循环（只有 8.5% 撞到长度上限），它做错数学题就是单纯的**算术算错、代数变形不会**。
* 给 Astra 同样的 50 次 forward pass 预算去改权重，结果 **MATH-500 从 6.8% 变成 7.0%（最高 7.7%），基本原地踏步**。
* **作者的原话结论**：
  > *"Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."*
  > （编辑权重可以移除掩盖原有能力的障碍，但**无法凭空加上模型原本不具备的能力**。）

### 3. 那为什么它从零手写 Breakout / 9×9 围棋（Exp 5 & 6）能赢 PPO？
* 在小网络（几万到 150 万参数）且输入通道语义极其规整（10×10 网格、9×9 棋盘）的任务上，Astra 本质上是**把人类可读的启发式符号代码“手工编译”进了 ReLU/tanh 矩阵**（比如第一层算球落点与挡板坐标差、围棋里前几层用卷积式连接数气/提子/算真眼，最后一层加权打分）。
* 当任务变成自然语言或复杂多跳推理（分布高度弥散在几亿参数的叠加态 Superposition 里，没有单一神经元对应单一符号规则）时，LLM 没法靠“手写矩阵”把新知识塞进去。

---

### 4. 对照 OPD / OPSD 的生态位：不仅没过时，反而互补
* **当小模型是“底座脑子不够 / 策略分布没对齐 / 召回与结构化能力缺失”**（即 *lack an ability*）：Weight-Space Programming 束手无策（如 Gemma 3 270M），必须靠 **OPD / OPSD** 在学生自己的 rollout 轨迹上用 Teacher Forcing 逐 token 传授分布。
* **当小模型（或蒸馏后的模型）出现“局部病理性退化”**（比如特定句式死循环、过度拒答、某个 head 带来的讨好偏置 sycophancy）：以前如果再跑一轮 SFT/RL 去修，很容易灾难性遗忘；现在可以让 Frontier LLM 直接挂 Python 探针做 **Weight-Space Surgery**，切掉 0.01% 的病灶权重，零梯度、零遗忘修好。

一句话总结：这篇工作真正颠覆的是**“靠人手写电路分析工具的传统 Mechanistic Interpretability”**和**“单点行为热修复（Model Hotfix）”**，但要把大模型的能力真正压进小模型里，OPD 依然是不可替代的主力。

## 🗣️ Turn 3 · Tao (`2026-11-03 16:19:13 GMT+8`, msg `#2265`)

嗯 再这个profile 有个session 再帮我做相关的research 你看看正常吗

## 🤖 Hermes Agent (`2026-11-03 16:21:51 GMT+8`, msg `#2294`)

**完全正常，那个 session（`@session:research/20261103_144924_d07315d5`）没有死，正在满负荷干活。**

之所以从 15:17 你说「好 试试看吧」到现在约 1 小时没在 Slack 里回话，是因为它把整个 Weight-Space 诊断 + 嫁接实验端到端串在一个长 turn 里跑了（目前已走了 **44 次 API 调用、63 次工具调用**，最后心跳就在几秒前）。

### 1. 它在干什么 & 已经产出了什么（本地与远端已核验）
* **Step 1 · 提取权重差异档案（已完成）**：对比了 `Qwen3.5-4B` 上 `OPSD-Astra` 与 `SFT-Astra` 各层 LoRA 的有效权重方向、奇异值比和词表投影，生成了 `Q:/wiki/misc/tabgroup-opd-20261026/weight_space_dossier.json`。
* **Step 2 · 调 `gpt-6.1-sol (xhigh)` 做权重空间机制诊断（已完成）**：落盘在 `Q:/wiki/misc/tabgroup-opd-20261026/sol_weight_diagnosis.md`（1.3 万字，给出了层段解耦假说和零训练正交投影/层段嫁接公式）。
* **Step 3 · 在两个经典顽疾样本上做零训练手术探针（已完成）**：跑了 `run_surgical_probe.py` 和 `run_surgical_probe2.py` 验证可行性。
* **Step 4 · 正在 A100（`<A100_DEVBOX_IP>`）上跑全量 61 题（45 v1 + 16 v2.1）零训练嫁接评测（进行中）**：
  * 远端进程 `eval_surgical_r1.py`（PID `3651599`）正在跑，GPU 显存占用 26.5 GB。
  * 一共测 5 个零训练 Weight-Space 手术候选，每个候选跑完 61 题约需 **5 分钟**（合计约 25 分钟）。

### 2. 刚刚在 A100 上跑出来的实时成绩（前 2 个候选已出炉！）
我已经直接去 A100 的 `/home/<SSH_USER>/work/eval_surgical_r1.log` 看了实时日志，**第 2 个嫁接候选在 v2.1 上已经同时干翻了 OPSD 和 SFT**：

* **基线对照（Run-1 缓存）**：
  * `0_sft2000`：v1 `P=79.23% / R=68.26%` ｜ v2.1 `P=58.33% / R=45.71%`
  * `1_opsd_astra`：v1 `P=85.48% / R=72.52%` ｜ v2.1 `P=69.09% / R=59.36%`
  * `2_sft_astra`：v1 `P=79.55% / R=64.19%` ｜ v2.1 `P=65.08% / R=60.25%`
* **已跑完的零训练权重嫁接（Zero-Training Weight Grafting）**：
  * `graft_14 (L0-13 SFT + L14-31 OPSD)`：v1 `P=83.33% / R=68.37%` ｜ v2.1 `P=66.10% / R=60.33%`
  * **`graft_18 (L0-17 SFT + L18-31 OPSD)`**：v1 `P=82.87% / R=69.19%` ｜ **v2.1 `Init P = 69.84% / Recall = 64.52%`**（v2.1 的 P 和 R 全部超越了单独的 OPSD-Astra 和 SFT-Astra！）
* **剩余进度**：正在跑第 3 个（`graft_22`），后面还有 `rev_graft_18` 和 `svd_blend_50_50`，预计再过 **10～15 分钟** A100 全部跑完，那个 session 就会把完整分析发回它的 Slack 线程里。

不需要中断或重启它，让它自然收尾即可。

## 🗣️ Turn 4 · Tao (`2026-11-03 16:29:57 GMT+8`, msg `#2299`)

`L0-17 SFT + L18-31 OPSD 这个是什么意思`

## 🤖 Hermes Agent (`2026-11-03 16:30:14 GMT+8`, msg `#2300`)

`Qwen3.5-4B` 一共有 **32 层 Transformer 层（编号从 L0 到 L31）**。

`L0-17 SFT + L18-31 OPSD` 的意思就是**跨模型“拼积木”（按层嫁接 LoRA 权重，零训练）**：
* **前 18 层（第 0 到 17 层，早中层）**：直接拷入 **SFT-Astra** 训练出来的 LoRA 权重。
* **后 14 层（第 18 到 31 层，深层/输出端）**：直接拷入 **OPSD-Astra** 训练出来的 LoRA 权重。

---

### 为什么要这么切？（背后的机制依据）

另一个 session 在对比两个模型的权重差异档案（`weight_space_dossier.json`）并让 `gpt-6.1-sol` 诊断时，发现这两个模型在不同层段干的活截然不同，而且各有优劣：

1. **早中层（L0～L17）负责“看懂网页、把大杂烩拆细”**：
   * 在复杂多主题的 **v2.1** 数据集上，OPSD 有个毛病——喜欢把很多标签页塞进一个笼统的大组里（比如笼统起名叫 `Soccer` 或 `Grok`），导致 v2.1 召回率偏低（59.36%）。
   * 而 **SFT-Astra** 直接拟合了 Astra 给出的细粒度分组答案，它在前中层建立的特征能把大桶拆成更细的子主题（比如把 Grok 拆成多个具体的子话题）。

2. **深层（L18～L31）负责“最终命名措辞与输出校准”**：
   * SFT-Astra 在第 21～31 层学到了非常激进的造词偏好（在词表投影里疯狂往 `&`、`学习与`、`Learning` 这类复合词方向推），导致它在常规 **v1** 上过度拆分、胡乱造复合词，把 v1 的 Precision（从 85.48% 跌到 79.55%）和 Recall（从 72.52% 跌到 64.19%）都搞崩了。
   * 而 **OPSD-Astra** 因为是在学生自己的轨迹上做温和的分布对齐，它的后半段（L18～L31）命名非常干净、克制。

3. **在 L18 一刀切开拼起来的效果**：
   * 用 **L0–17 的 SFT-Astra** 在前半程把网页的细粒度主题聚类边界刻画出来；
   * 到了后半程交给 **L18–31 的 OPSD-Astra**，抑制住 SFT 乱造 `A & B` 复合词的冲动，稳健地输出组名。
   * 所以一分钱训练没花、一行梯度没算，拼出来的模型在 v2.1 上直接拿到了 **Init P = 69.84% / Recall = 64.52%**，比单独的 SFT 和单独的 OPSD 都高。
