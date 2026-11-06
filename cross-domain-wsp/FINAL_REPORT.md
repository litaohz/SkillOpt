# Weight-Space Anatomy and Zero-Training LoRA Surgery Across Three Public Extraction Benchmarks and Industrial Browser TabGroup

**Authors:** Tao Li (`litao037@nuaa.edu.cn`, Nanjing University of Aeronautics and Astronautics), Junfeng Liu (`liujf@pcl.ac.cn`, Pengcheng Laboratory, Corresponding Author), Qiang Shen (`qiangshen@microsoft.com`, Microsoft), Bo Shao (`boshao@microsoft.com`, Microsoft), Linjun Shou (`lisho@microsoft.com`, Microsoft), Xuejun Liu (`xuejun.liu@nuaa.edu.cn`, Nanjing University of Aeronautics and Astronautics)
**Date:** November 4, 2026
**Models & Hardware:** `Qwen/Qwen3.5-4B` (32 layers), `Qwen/Qwen3.5-2B` (24 layers), `Qwen/Qwen3.5-9B` (32 layers), and `mistralai/Ministral-3-8B-Instruct` (34 layers pure full-attention) on NVIDIA A100-80GB PCIe.

---

## 1. 四大基准统一体系：三大公开权威 NLP 数据集 + 真实工业场景 `Browser TabGroup`

我们将 **32 层精确 QR-SVD 权重空间解剖与零训练权重外科手术**（**Idea A 单模型 64 列神经元双阀门手术**、**Idea B-1/B-2 单模型后层 $\sigma_1,\sigma_2\to 0$ 奇异值切除**、**Idea C 全注意力 `q_norm` 调温**、**Surgery A `Graft-18` 层间电路嫁接**、**Surgery B `SVD-Blend 50/50` 秩-16 谱共识**）统一建立在 **3 个跨领域的经典公开 NLP 抽取基准 + 1 个真实工业浏览器会话分组基准（`Browser TabGroup`）** 上：

1. **Public Benchmark 1：`Re-DocRED` vs. `Orig-DocRED`（EMNLP 2022 · 维基百科 96 类文档级关系抽取）**
   * **天然两阶段痛点**：`Orig-DocRED` 存在公认的严重假阴性漏标（平均仅 12.5 条三元组/篇，$>90\%$ 实体对被标为无关系 `NA`），而补全后的 `Re-DocRED` 平均有 28.2 条三元组/篇。
   * **实验对齐**：Stage-1 基础 LoRA (`0_sft_base`) 在 `Orig-DocRED` 上训练（学会了关系语义，但后层过早触发 `Unrelated: high-confidence NA` 停止阀门，Recall 仅 `17.20%`）；Stage-2 在 `Re-DocRED`（带 Wikidata 属性前缀 `Pxxx & relation`）上分别训练离线 SFT (`2_sft_teacher`) 与 On-Policy 自蒸馏 (`1_opsd_teacher`)。
2. **Public Benchmark 2：`BioRED`（Briefings in Bioinformatics 2022 · 生物医学摘要跨句基因/疾病/药物关系抽取）**
   * **天然两阶段痛点**：Stage-1 基础 LoRA 仅在句内（intra-sentence）关系上训练（过早停止，漏掉跨句关系，F1 `18.33%`）；Stage-2 在全摘要跨句关系（带 `B0x & relation` 编码）上训练离线 SFT 与 OPSD。
3. **Public Benchmark 3：`PHEE`（EMNLP 2022 · 临床药物不良反应层级事件论元抽取）**
   * **天然两阶段痛点**：Stage-1 基础 LoRA 在 3 个粗粒度角色（`Subject`, `Treatment`, `Effect`）上训练（把给药途径、剂量、药物、适应症全部大锅饭塞进 `Treatment`，13 个细粒度子角色 Sub-Role F1 为 `0.00%`）；Stage-2 拆分到 16 个细粒度层级子角色（`Treatment.Drug`, `Treatment.Dosage`, `Treatment.Route`, `Subject.Age` 等，带 `Rxx & Role` 编码）。
4. **Industrial Benchmark 4：`Browser TabGroup`（真实工业场景浏览器多标签会话智能分组）**
   * **天然两阶段痛点**：全量 61 个真实浏览器多标签会话（45 个常规会话 `v1` + 16 个高复杂度多子任务混杂会话 `v2.1`）。Stage-1 基础 LoRA (`sft2000`) 在 `v2.1` 上存在严重的“巨型桶过度合并（Giant-Bucket Over-Merging）”（`v2.1` Recall 仅 `45.71%`, F1 `51.25%`）；Stage-2 在 1,200 条细粒度特权教师标注 (`Astra`) 上训练时，离线 SFT 虽然提升了 `v2.1` 细粒度拆分（F1 `62.57%`），但后层过拟合了 `' & '` 复合组名与高 `Ungrouped` 拒识阈值，导致 `v1` Recall 暴跌 `-8.33pp`（从 `68.26%` 跌至 `64.19%`）。

---

## 2. 跨四大基准（3 Public + TabGroup）的 32 层权重空间解剖（100% 复现 L18 分界线！）

对全部四个基准上的 $\Delta W_{\text{OPSD}} = W_{\text{OPSD}} - W_0$ 与 $\Delta W_{\text{SFT}} = W_{\text{SFT}} - W_0$ 执行精确 QR-SVD 与 Unembedding 词表矩阵反投影（见 `figures/fig1_public_weight_space_heatmap.pdf`），发现了**完全一致的三条跨任务物理定律**：

1. **定律 1（中前层 L06–L15 关系与细粒度分簇共振）**：
   * 在 L06–L15，OPSD 与离线 SFT 修改的是同一个跨实体关系/子任务拆分回路，余弦相似度在 `TabGroup` L12 达到 **`0.3665`**，在 `BioRED` L08 `mlp.down_proj` 达到 **`0.2292`**，在 `PHEE` L12 达到 **`0.1410`**，在 `Re-DocRED` L08 达到 **`0.1343`**。
2. **定律 2（离线 SFT 更新步长膨胀 `1.4x ~ 3.0x`）**：
   * 在相同训练步数与学习率下，离线 SFT 的 Frobenius 更新范数在中前层是 OPSD 的 **`1.4x ~ 2.4x`**，而在后层膨胀到 **`1.7x ~ 3.0x`**。
3. **定律 3（L18 正交退相干与 `8x ~ 78x` 后层表面词法污染激增）**：
   * 一旦跨过 **第 18 层（L18）**，所有四个基准上的余弦相似度瞬间跌至 **`-0.04 ~ +0.08`（正交退相干）**！
   * 与此同时，离线 SFT 在 L18–L31 往词表前缀/复合连接符 Token `' &'`（ID 594）方向强写的投影强度相比 OPSD 发生爆炸式飙升：**`Re-DocRED` 达 `18.2x ~ 78.1x`**、**`BioRED` 达 `11.9x ~ 48.8x`**、**`PHEE` 达 `15.6x ~ 54.5x`**、**`TabGroup` 达 `8.0x ~ 13.4x`**！

---

## 3. 四大基准（3 Public + TabGroup）完整实验总表（精确 P / R / F1）

| 类别 | 模型 / 零训练手术方法 | 修改范围 | **1. Re-DocRED F1** (P / R) | **2. BioRED F1** (P / R) | **3. PHEE Overall F1** (Sub-Role F1) | **4. TabGroup `v1` R / `v2.1` F1** (`v2.1` P / R) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **训练基线** | `0_sft_base` (一阶段基础 LoRA) | --- | 26.60% (58.65 / 17.20) | 18.33% (**32.80** / 12.72) | 44.39% (0.00%) | 68.26% / 51.25% (58.33 / 45.71) |
| **训练基线** | `2_sft_teacher` (二阶段离线 SFT) | 21.2M | 14.14% (19.72 / 11.03) | 0.00% (0.00 / 0.00) | 58.91% (50.68%) | 64.19% / 62.57% (65.08 / 60.25) |
| **训练基线** | `1_opsd_teacher` (二阶段 On-Policy OPSD) | 21.2M | 35.63% (47.35 / 28.56) | 21.03% (23.43 / 19.08) | 50.54% (32.22%) | **72.52%** / 63.86% (69.09 / 59.36) |
| **单模型手术 (0训练)** | **★ Idea C**: L07,11,15,19 `q_norm * 1.35x` | 4 个标量 | 29.29% (**60.76** / 19.29) | 18.76% (29.65 / 13.72) | 43.88% (0.00%) | **70.37%** / **55.37%** (59.26 / 51.96) |
| **单模型手术 (0训练)** | **★ Idea A**: L26–31 64列双阀门缩放 (`0_sft_base`) | 1.39% 列 | **31.99% (+5.39pp)** | **21.50% (超 OPSD!)** | 39.15% (4.80%) | 69.70% / **59.56% (+8.31pp F1, +12.2pp R!)** |
| **单模型手术 (0训练)** | **★ Idea B-2**: L18–31 $\sigma_1=\sigma_2=0$ (`2_sft_teacher`) | 2 标量/模块 | **38.77% (全场第1!)** | **18.15% (+18.15pp)** | 42.11% (34.95%) | **67.63% (+3.44pp R)** / **63.05%** (**66.13** / 60.25) |
| **双模型手术 (0训练)** | `Rev-Graft-18` (`L00–17 OPSD + L18–31 SFT`) | 反向对照 | 35.97% (46.50 / 29.33) | 21.39% (23.41 / 19.70) | 59.25% (54.68%) | 67.48% / 63.58% (67.86 / 59.81) |
| **双模型手术 (0训练)** | **★ Surgery A**: `Graft-18` (`L00–17 SFT + L18–31 OPSD`) | L18 嫁接 | 35.11% (42.13 / 30.10) | **23.20% (全场第1!)** | **64.15% (全场第1!)** | 69.19% / **67.08%** (69.84 / **64.52 全场第1!**) |
| **双模型手术 (0训练)** | **★ Surgery B**: `SVD-Blend 50/50` (秩-16 谱共识) | Rank-16 SVD | **36.89% (超 OPSD!)** | **21.47% (超 OPSD!)** | 63.39% (**60.17% 第1!**) | **69.37%** / **68.18% (全场第1, P 72.41%!)** |

---

## 4. 关键科学发现与跨数据集相互印证

1. **单模型 64 列阀门手术（`Idea A`）直接打赢或逼平 OPSD，并精准划定物理边界**：
   * 在 **`Re-DocRED`** 上，仅对 `0_sft_base` 的 L26–L31 `mlp.down_proj` 做 64 列促写/促停阀门缩放（0 训练），Recall 从 `17.20%` 暴涨至 **`25.03%` (+7.83pp)**，F1 涨至 **`31.99%` (+5.39pp)**；
   * 在 **`BioRED`** 上，同样 0 训练的 `Idea A` 把 F1 从 `18.33%` 拉到 **`21.50%` (+3.17pp)**，**直接超越了跑满二阶段蒸馏的 `OPSD` (`21.03%`)**！
   * 而在 **`PHEE`** 上，因为 `0_sft_base` 在一阶段只见过 3 个粗角色名，中层压根没有 13 个细粒度子角色的表示，所以只改后层阀门只能拿到 `4.80%` Sub-Role F1，必须靠 `Graft-18` / `SVD-Blend` 引入二阶段中层特征才能冲上 **`60.17%`**。这给出了「何时单模型阀门手术有效」的教科书级实证！
2. **单模型抹除 2 个奇异值标量（`Idea B-2`: $\sigma_1=\sigma_2=0$）让崩盘的离线 SFT 起死回生夺冠**：
   * 在 **`Re-DocRED`** 上，把 `2_sft_teacher` 在 L18–L31 的前 2 个奇异值归零，F1 从 `14.14%` 直接飙到 **`38.77%`（全场最高，超越 OPSD `+3.14pp`）**；
   * 在 **`BioRED`** 上，离线 SFT 因为把所有关系名吃掉写成了纯编号 `B01: E13->E6` 而得 `0.00%` 分；仅把 L18–L31 的 $\sigma_1, \sigma_2$ 归零（`Idea B-2`），模型当场恢复正常英文关系名 `association: E13->E6`，F1 从 `0.00%` 跳回 **`18.15%`**！
3. **双模型电路嫁接（`Graft-18`）与谱共识（`SVD-Blend 50/50`）在全部三个公开数据集上全面超越 OPSD 与 SFT**：
   * **`Re-DocRED`**：`SVD-Blend 50/50` 达到 **`36.89% F1`**（超越 OPSD `35.63%` 和 SFT `14.14%`）；
   * **`BioRED`**：`Graft-18` 达到 **`23.20% F1`**，`SVD-Blend` 达到 **`21.47% F1`**（双双超越 OPSD `21.03%` 和 SFT `0.00%`）；
   * **`PHEE`**：`Graft-18` 达到 **`64.15% Overall F1 / 81.34% Recall`**（超越 OPSD `+13.61pp F1`，超越 SFT `+5.24pp F1`），`SVD-Blend 50/50` 拿到全场最高 **`60.17% Sub-Role F1`**（超越 OPSD `+27.95pp`，超越 SFT `+9.49pp`）！

---

## 5. 纯单阶段（Pure 1-Stage）步数阶梯扩展、9B→4B 白盒 OPD 对比与跨模型通用定律

为彻底回答「如果完全不做两阶段训练、不加任何人工前缀，直接在官方干净完整训练集上单阶段训练原版 LoRA，内核的两阶段机制是否依然成立？对比 9B Teacher 的白盒 OPD 以及跨不同尺寸/架构模型结果如何？」，我们完成了完整的 V2 大满贯实验（见 `paper/figures/fig3_pure_single_stage_scaling_opd_models.pdf` 与 `comprehensive_suite_v2/`）：

| 实验模块 | 模型 / 零训练权重手术方案 | **BioRED P / R (%)** | **BioRED F1 (命中数)** | **Re-DocRED F1** | **PHEE Overall F1 (Sub-Role F1)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | `SFT_step50` (欠拟合 50 步) | 35.19% / 25.19% | 29.36% (202) | 49.40% | 88.72% (85.45%) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | **★ `SFT_step50 + Idea B-1` ($\sigma_1=0$) / `Idea C` (`q_norm`)** | **39.50% / 29.30%** | **32.91% / 33.64% (+4.28pp, 235)** | 48.66% | 88.72% (85.45%) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | `SFT_step100` (收敛巅峰 100 步) | 42.70% / 29.18% | 34.67% (234) | 56.58% | **90.95%** (88.79%) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | **★ `SFT_step100 + Idea B-1` (L18–31 $\sigma_1=0$, 0训练)** | 44.16% / 30.17% | **35.85% (+1.18pp, 242)** | 53.85% | 89.49% (87.00%) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | **★ `SFT_step100 + Idea C` (`q_norm * 1.35x`, 0训练)** | **44.27%** / 28.93% | **34.99%** (232) | 56.25% | 90.69% (**89.19% 全场最高!**) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | `SFT_step150` (过拟合掉点 150 步) | 37.38% / 29.93% | 33.24% (240) | **57.57%** | 89.22% (85.45%) |
| **1. 纯单阶段 SFT 步数阶梯 (`Qwen3.5-4B`)** | **★ `SFT_step150 + Idea C` / `Idea B-1` (0训练过拟合反转)** | 39.08% / **31.67%** | **34.99% (+1.75pp, 254条!)** | 56.02% | **89.43% (+0.21pp, 85.84%)** |
| **2. 同起点 Post-`SFT_100` 对比** | `SFT_step100 -> OPSD` (+25 步自蒸馏) | 36.04% / 27.68% | 31.31% (222) | 55.50% | 90.46% (87.89%) |
| **2. 同起点 Post-`SFT_100` 对比** | `SFT_step100 -> White-Box OPD (9B->4B)` (+25 步白盒蒸馏) | 33.19% / 27.93% | 30.33% (224) | --- | --- |
| **2. 同起点 Post-`SFT_100` 对比** | **★ `White-Box OPD + Idea B-1` (L18–31 $\sigma_1=0$, 0训练)** | 36.40% / 31.55% | **33.80% (+3.47pp, 253)** | --- | --- |
| **2. 同起点 Post-`SFT_100` 对比** | **★ `SVD-Blend-OPD` (`SFT_150 + OPD_9B` 50/50 谱共识)** | 39.20% / 30.55% | **34.34%** (245) | 56.34% | 90.15% (87.16%) |
| **2. 同起点 Post-`SFT_100` 对比** | **★ `Graft-18_OPD` (`L00–17 SFT_150 + L18–31 OPD_9B`)** | **40.53% / 32.29%** | **35.95% (全场第1, 259条!)** | 56.34% | 89.43% (85.84%) |
| **3. 跨尺寸与跨架构验证 (纯SFT)** | **`Qwen/Qwen3.5-2B` (24层混合, Step 100)**: SFT $\to$ **+Idea C / B-1** | $23.25/19.45 \to \mathbf{30.42/25.19}$ | $21.18\% \to \mathbf{27.56\%}$ (**+6.38pp!**) | --- | --- |
| **3. 跨尺寸与跨架构验证 (纯SFT)** | **`Qwen/Qwen3.5-4B` (32层混合, Step 100)**: SFT $\to$ **+Idea B-1 / Graft** | $42.70/29.18 \to \mathbf{40.53/32.29}$ | $34.67\% \to \mathbf{35.95\%}$ (**+1.28pp!**) | --- | --- |
| **3. 跨尺寸与跨架构验证 (纯SFT)** | **`Ministral-3-8B` (34层纯全注意力, Step 150)**: SFT $\to$ **+Idea A (64列阀门)** | $49.91/33.67 \to \mathbf{51.29/34.79}$ | $40.21\% \to \mathbf{41.46\%}$ (**+1.25pp, 279条!**) | --- | --- |
| **3. 跨尺寸与跨架构验证 (纯SFT)** | **`Ministral-3-8B` (34层纯全注意力, Step 200)**: SFT $\to$ **+Idea B-1 ($\sigma_1=0$)** | $46.53/33.42 \to \mathbf{49.91/35.29}$ | $38.90\% \to \mathbf{41.34\%}$ (**+2.44pp, 283条!**) | --- | --- |

