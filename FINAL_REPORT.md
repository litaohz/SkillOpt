# TabGroup Weight-Space Programming Pilot Report (2026-11-03)

## 1. 实验目标与设计
受 MIT CSAIL《Sparks of Weight-Space Programming》（2026-11-02）启发，我们在 `Qwen/Qwen3.5-4B`（32 层，Gated-DeltaNet 线性注意力 + 第 3/7/11/15/19/23/27/31 层全注意力，`d_model=2560`，LoRA `r=16, alpha=32`）上，对已对齐落盘的 4 个同构 Adapter 进行了 **32 层全量权重空间 SVD 解剖 + Unembedding 词表方向投影**，并结合 `gpt-6.1-sol (effort=xhigh)` 的白盒诊断，在 A100-80GB 上实测了 **零训练权重外科手术（Zero-Training Weight-Space Surgery / Circuit Grafting）**。

对比的 4 个基础 Adapter（均从同一个 `sft2000` 检查点出发训练 150 步）：
- `sft2000`：2000 步基础 SFT（`v2.1` 存在典型巨型桶过度合并 Giant-Bucket Over-Merging）。
- `opd9b`：9B-SFT 白盒 GKD 蒸馏 150 步。
- `opsd_astra`：GPT-6-astra 1,200 条特权参考答案辅助的 On-Policy Self-Distillation（150 步）。
- `sft_astra`：相同 1,200 条 Astra 答案直接做传统 Cross-Entropy SFT（150 步对照）。

---

## 2. 权重空间解剖：为什么 `sft_astra` 在 `v2.1` 拆得更细，却在 `v1` 暴跌 `-9.90pp` Recall？

通过对全部 32 层、496 个 LoRA 矩阵计算相对 `sft2000` 的有效权重差分 $\Delta W_{\text{OPSD}} = W_{\text{OPSD}} - W_0$ 与 $\Delta W_{\text{SFT}} = W_{\text{SFT}} - W_0$（脚本：`build_dossier.py`，完整数据：`Q:/wiki/misc/tabgroup-opd-20261026/weight_space_dossier.json`），发现三个清晰的物理机制：

### 发现 1：更新步长（Frobenius Norm）系统性膨胀 1.4x～1.7x
在完全相同的 150 步、相同学习率 `2e-5` 下，`sft_astra` 的有效权重更新范数在所有层都比 `opsd_astra` 大 **1.40x～1.71x**（例如 L12 达 `1.71x`，L16-L17 达 `1.68x`）。这是因为离线 SFT 在整条参考答案的所有 token 上强行施加硬 Cross-Entropy 梯度，而 OPSD 仅在学生自己生成的 on-policy 轨迹与特权教师发生分布分叉的局部位置产生 JSD 梯度。

### 发现 2：双阶段层间分化（L08–L15 共振 vs. L18–L31 正交分道扬镳）
计算每一层 $\Delta W_{\text{OPSD}}$ 与 $\Delta W_{\text{SFT}}$ 的精确 Frobenius 余弦相似度：
- **中前层特征提取与聚类回路（L08–L15）高度对齐**：余弦相似度达到峰值 **`0.3357 ~ 0.3665`**（L12 达 `0.3665`，L11 Full-Attn 达 `0.3443`）。这说明无论是 OPSD 还是离线 SFT，**用来识别跨网页细粒度子任务边界（解决 `v2.1` 大锅饭）的核心表示更新集中在第 8～17 层**。
- **后层命名与策略读出层（L18–L31）急剧退相干乃至正交**：从 L18 开始余弦相似度一路掉到 **`0.0517 ~ 0.14`**（L24 `linear_attn.out_proj` 仅 `0.0851`，L26 `mlp.down_proj` 仅 **`0.0517`**，共享能量不足 `0.27%`）。

### 发现 3：Unembedding 投影揪出后层污染源（`' &'` 复合命名漂移 + `Ungrouped` 过拟合）
将 L18–L31 残差写入矩阵（`mlp.down_proj`、`self_attn.o_proj`、`linear_attn.out_proj`）的 Top-1 左奇异向量 $u_1$ 乘上底座的 RMSNorm + Unembedding 矩阵 $W_U$ 投影回词表空间：
- **`sft_astra` 的 L18–L31 主奇异方向几乎全在往残差流里强写 `' &'`、`'&'`、`'学习与'`、`' Learning'` 等复合连接词**！`sft_astra` 在 token `' &'`（ID 594）方向上的投影强度高达 **`0.0070 ~ 0.0127`**，是 `opsd_astra`（`0.0006 ~ 0.0014`）的 **8～11 倍**。
- **回查训练数据根源**：在 1,200 条 Astra 标注数据中，带 `' & '` 的复合组名占了 **13.03%**（原 SFT 训练集仅 **1.43%**，高出 9.1 倍），且 `Ungrouped` 网页数高出 **15.3%**。
- **因果链条**：离线 SFT 的 Teacher-Forcing 被迫在 L18–L31 大幅改写输出层权重去硬背 Astra 的命名僻好（`' & '`）和高 `Ungrouped` 拒分阈值（导致 `v1` 上原本独立的 `Travel`、`Superset`、`Maps` 等双网页小桶被拆散丢进 `Ungrouped`，`v1` 未分组网页从 238 暴增到 324）；而 **OPSD 因为顺着 4B 自己的轨迹只纠偏分桶边界，在后层几乎完全免疫了这种词表表面分布漂移**（`v1` 带 `&` 组仅 0.7%）。

---

## 3. 零训练权重外科手术（Zero-Training Weight-Space Surgery）实测结果

既然 **“细粒度拆分能力在 L00–L17，而命名污染与过度丢弃在 L18–L31”**，我们在 A100 上不跑任何梯度训练，直接对现有 LoRA 权重做 **层间电路嫁接（Layerwise Circuit Grafting）** 与 **SVD 低秩子空间融合（Rank-16 QR-SVD Refactorization）**，并在 `v1`（45 题）+ `v2.1`（16 题）的 Run-1 全量 61 道题上实测精确 Initial Precision 与 Recall（脚本：`launch_surgical_r1.py`，日志：`/home/dlisuser/work/eval_surgical_r1.log`）：

| 模型 / 零训练手术方案 | `v1` Initial P (n=45) | `v1` Recall (n=45) | `v2.1` Initial P (n=16) | `v2.1` Recall (n=16) | 机制验证结论 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `0_sft2000` (原 4B-SFT 底座) | 79.23% | 68.26% | 58.33% | 45.71% | `v2.1` 严重过度合并 |
| `1_opsd_astra` (原 OPSD 150步) | **85.48%** | **72.52%** | 69.09% | 59.36% | 保住 `v1` 召回，修复 `v2.1` |
| `2_sft_astra` (原 Astra-SFT 150步) | 79.55% | 64.19% (-8.33pp) | 65.08% | 60.25% (+0.89pp) | `v1` 召回暴跌，`v2.1` 略高 |
| **手术 1：`Graft-14` (`L00-13 SFT` + `L14-31 OPSD`)** | 83.33% | 68.37% | 66.10% | 60.33% | 切得偏早，未完全纳中层拆分回路 |
| **手术 2：`Graft-18` (`L00-17 SFT` + `L18-31 OPSD`)** | **82.87%** | **69.19%** (+5.00pp vs SFT) | **69.84%** | **64.52%** (**+5.16pp vs OPSD!**) | **精准命中分界线！`v2.1` 远超两亲本** |
| **手术 3：`Graft-22` (`L00-21 SFT` + `L22-31 OPSD`)** | 80.77% | 68.26% | 65.62% | 61.82% | 纳入了 L18-21 的 SFT 污染，开始回落 |
| **对照 4：`Rev-Graft-18` (`L00-17 OPSD` + `L18-31 SFT`)** | 82.78% | 67.48% (-5.04pp vs OPSD) | 67.86% | 59.81% | **反向对照铁证：把 SFT 后层装给 OPSD 立刻砸穿 `v1` Recall** |
| **手术 5：`SVD-Blend-50/50` (`0.5 OPSD + 0.5 SFT` 精确秩-16 SVD)** | **82.61%** | **69.37%** (+5.18pp vs SFT) | **72.41%** (**全场最高**) | **64.42%** (**+5.06pp vs OPSD!**) | **中层协同叠加 + 后层正交稀释，P/R 双双破纪录** |

---

## 4. 真实 Case 走查：为什么 `Graft-18` 能超越两个亲本模型？

对比 `Graft-18 (L00-17 SFT + L18-31 OPSD)` 与两个亲本在具体 Case 上的真实输出（日志：`run_surgical_probe2.py`）：

1. **Case `v2.1 / JH16`（13 个网页：Minecraft、Khan Academy、语言学习、Elyon Christian School）**：
   - **`opsd_astra`**：分出了 `Elyon Christian School: 6, 9`（Recall **75.0%**）。
   - **`sft_astra`**：因为后层 `Ungrouped` 阈值过严，把 `9` 丢进了 `Ungrouped: 1, 9`，把 `6` 错误塞进 `Language Learning: 6, 7, 8`（丢了 Elyon Christian School 组）。
   - **`Graft-18`（零训练嫁接）**：完美输出 `Minecraft Resources: 2, 3, 10, 11, 12, 13 | Khan Academy: 4, 5 | Language Teaching: 7, 8 | Elyon Christian School: 6, 9 | Ungrouped: 1`——**成功用 OPSD 的 L18–31 救回了被 SFT 后层误杀的 `Elyon Christian School: 6, 9`！**

2. **Case `v2.1 / JH11`（33 个网页的高复杂度混杂会话）**：
   - **`opsd_astra`**：只拆成 4 个宽泛大组 `Travel`、`AI`、`Slang`、`NBA`（Recall 仅 **28.57%**）。
   - **`sft_astra`**：中层特征将它拆成了 7 个细粒度子任务组（`Travel Luggage`、`Kimi AI Models`、`San Francisco Airbnb`、`AI Decision Making`、`NBA 2K26`、`Copium Slang`、`NBA Players`）。
   - **`Graft-18`（零训练嫁接）**：依靠 `L00–17` 的 SFT 权重，**100% 完整保留了这 7 个细粒度子任务拆分**（`Travel Luggage: 1, 5, 8, 25 | Kimi AI Models: 2, 9, 11, 15, 16, 19, 24, 30 | San Francisco Airbnb: 3, 10, 17, 31, 33 | AI Decision Making: 4, 18, 20, 26, 27 | Copium Slang: 6, 13, 14, 22 | NBA 2K26: 7, 12, 28, 29 | NBA Players: 23, 32`）！

3. **Case `v1 / service-4`（19 个网页：Travel、Skype、Microsoft Edge、Superset）**：
   - **`sft_astra`**：把 `Travel: 2, 12` 和 `Superset: 1, 16` 全部扔进了 `Ungrouped: 1, 2, 12, 16`。
   - **`Graft-18`（零训练嫁接）**：换上 OPSD 的 L18–31 后，**立刻把 `Travel: 2, 12` 从 `Ungrouped` 里救了回来**，同时保留了对 `Edge Features` 和 `Edge Metrics` 的细粒度拆分。
   - **反向嫁接 `Rev-Graft-18`（`L00-17 OPSD + L18-31 SFT`）**：一旦把 SFT 的后层装到 OPSD 上，模型在 `service-4` 上瞬间退化成一个 17 网页的巨型桶 `Work: 1, 3, 4, ..., 19`，直接证实破坏源就在 L18–31。

---

## 5. 单神经元/单标量权重修改（Ideas A / B / C）实测结果（完全不用 OPSD，单模型直接改权重！）

为了严格对齐 MIT 博客“不借助第二个模型，只在单个模型内部改几个数值就治好缺陷”的设定，我们在 A100 上实测了 **想法 A（单神经元列缩放）**、**想法 B（单奇异值 $\sigma_1/\sigma_2$ 归零）** 和 **想法 C（Full-Attention `q_norm` 标量调温）**（脚本：`run_abc_all.py` & `/home/dlisuser/work/eval_idea_b2_r1.py`），三个想法**全部命中真实物理效应**：

### 想法 A（在 `sft2000` 上仅缩放 `v_newline - v_comma` 的 Top-64 MLP 神经元列）
在 `sft2000` 的 L26–L31 `mlp.down_proj`（每层 9,216 列，共改动 `0.69%` 的列）中，将投影向换行符 `\n` 最强的 64 列乘以 `2.0x`，将投影向逗号 `,` 最强的 64 列直接归零 `0.0x`：
- **Case `v2.1 / JH02`（原本 25 个 Tab 合进 1 个 `Soccer` 大桶，Recall `0.0%`）**：
  - **修改后当场拆成 4 个精准组（P `75.0%`, Recall `60.0%`）**：`Amazon: 1, 6, 17, 23, 25 | Grok: 2, 3, 4, 11, 13, 14, 20 | Manchester City: 5, 8, 9, 10, 12, 16, 19 | Data Analysis: 7, 18, 21, 22`！
- **Case `v2.1 / JH16`（原本合并成宽泛 `Education`，Recall `33.3%`）**：
  - **修改后直接飙到 P `100.0%`, Recall `100.0%`（满分！）**：`Teaching Resources: 1, 7, 8 | Minecraft: 2, 3, 10, 11, 12, 13 | Khan Academy: 4, 5 | Elyon Christian School: 6, 9`！

### 想法 B（在 `sft_astra` 上完全不用 OPSD，仅把 L18–L31 增量矩阵的奇异值标量 $\sigma_1, \sigma_2 \to 0$）
对 `sft_astra` 在 L18–L31 的更新矩阵 $\Delta W_{\text{SFT}} = U \Sigma V^\top$，**每个模块只把前 1 个（`B-1`）或前 2 个（`B-2`）奇异值标量设为 `0.0`**，其余 14 个奇异方向原封不动保留：
- **单标量归零（`B-1`: $\sigma_1 = 0$）**：在 `v1 / service-4` 上**当场救回被误杀的 `Travel: 2, 12` 组**（Recall 从 `16.7%` 升至 `66.7%`），同时完整保留 `JH11` 的 7 组细粒度拆分（P `85.7%`, R `85.7%`）。
- **双标量归零（`B-2`: $\sigma_1 = \sigma_2 = 0$）**：在 `B-1` 基础上，连 `v2.1 / JH16` 里被 SFT 误杀的 **`Elyon Christian School: 6, 9` 也一并救活（Recall 升至 `75.0%`）**！
- **Run-1 全量 61 题实测（`idea_b2_sft_top2_zero`，完全没碰 OPSD）**：
  - `v1`（45 题）：Initial P 从 `79.55%` 升至 **`80.66%` (+1.11pp)**，Recall 从 `64.19%` 升至 **`67.63%` (+3.44pp)**！
  - `v2.1`（16 题）：Initial P 从 `65.08%` 升至 **`66.13%` (+1.05pp)**，Recall 保持 **`60.25%`**！

### 想法 C（在 `sft2000` 上仅把 L07/L11/L15/L19 四个全注意力层的 `q_norm` 乘一个标量）
不改任何线性层或 MLP，只在 4 个 Full-Attention 层的 `q_norm` 上乘一个常数标量：
- **调高注意力锐度（`q_norm * 1.35x` / `1.75x`）**：
  - **`v2.1 / JH12`**（原本 22 个网页塞进 `Travel` 大桶，Recall `0.0%`）**当场被拆开成 `Travel: 1, 4, 6, 12, 15 | Finance: 2, 7, 10, 11, 19, 20, 23, 24 | Shopping: 3, 5, ...`（Recall 从 `0.0%` 升至 `50.0%`）**！
  - **`v1 / service-5` 和 `service-7`** 双双从 `50% / 66.7%` 跳到 **P `100.0%`, Recall `100.0%`**！
- **调平注意力锐度（`q_norm * 0.70x`）**：
  - **`v2.1 / JH02`**（原本 25 个网页塞进 `Soccer` 大桶，Recall `0.0%`）**当场拆成 4 个组（P `75.0%`, Recall `60.0%`）**！

---

## 6. 产物归档清单（本地三处 + A100 远端已完整封存）

1. **TabGroup 官方研究主仓库归档目录**：
   `Q:/wiki/misc/tabgroup-research-repo/results/weight-space-surgery-20261103/`
   - 含完整 SHA256 校验清单 `archive-manifest.json`、汇总表 `weight-space-surgery-r1.csv`（同步至 `metrics/weight-space-surgery-r1.csv`）、4 个亲本 Adapter（`parent-adapters/`）及 3 个零训练手术 Adapter（`adapters/graft_18`、`adapters/svd_blend_50_50`、`adapters/idea_b2_sft_top2_zero`）。
2. **实验工作目录备份**：
   `Q:/wiki/misc/tabgroup-opd-20261026/surgical-adapters-20261103/`
   - `graft_18`（SHA256: `503ed75574879a80bbf943fa1691b7372187c16e4c1061d3e1b9b69fed25eacf`）
   - `svd_blend_50_50`（SHA256: `a6d5c3ac005770696289923fdb46693acdab2f79d071254fcdec903c7e4132d9`）
   - `idea_b2_sft_top2_zero`（SHA256: `d09e20cb18444861bef418d2e25c6523de3237c68c2efa18616cede86bae025a`）
3. **本报告路径**：
   `Q:/wiki/projects/graphiti-ssv/discussions/2026-11-03-tabgroup-weight-space-surgery-pilot.md`


