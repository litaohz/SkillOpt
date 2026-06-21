# SkillOpt 复现进度 (progress.md)

> 用本机 GitHub Copilot proxy 复现 SkillOpt 论文 (arXiv:2605.23904) 的评测。
> 最近更新：完成 SpreadsheetBench 全量 A/B 复现（Δ=+38.2 vs 论文 +38.9，几乎完美）。

---

## 0. 环境 / 端点

- Python 3.12，`pip install -e .` + `pip install datasets` 已装。
- **模型端点 = 本机 GitHub Copilot proxy**：`http://localhost:4141/v1`（OpenAI 兼容）。
  - 根路径返回 `Copilot API v1.2.5 - running, User: taoli1ms`。
  - `GET /v1/models` 可列模型。
  - **chat/completions 可用**：`gpt-4o / gpt-4.1 / gpt-5.4 / claude-sonnet-4.6` …
  - **`gpt-5.5` 仅支持 Responses API**（`/chat/completions` → 400 `unsupported_api_for_model`）。
    必须设 `OPENAI_RESPONSES_API_MODELS=gpt-5.5` 才能用论文主力模型。
- 接入方式：`--azure_openai_auth_mode openai_compatible` + `--azure_openai_endpoint http://localhost:4141/v1` + `--azure_openai_api_key dummy`。

## 1. 已完成的代码改动（仓库内，**待 git commit**）

1. `skillopt/model/azure_openai.py` — `_needs_responses_api()` 增加两个 env 开关（非破坏，默认行为不变）：
   - `OPENAI_RESPONSES_API_MODELS`：逗号分隔模型前缀，路由到 `/responses`。
   - `OPENAI_FORCE_RESPONSES_API=1`：强制全部走 Responses API。
2. `scripts/eval_only.py`：
   - 读 skill 文件改 `encoding="utf-8"`（修 Windows cp1252 读 curly-quote 崩溃 bug）。
   - 跑完导出 token 统计进 `eval_summary.json`（`tokens` 分 stage + `tokens_per_item`）。
3. `scripts/materialize_searchqa.py`（新增）：从 HF `lucadiliello/searchqa` 还原
   `data/searchqa_split/{train,val,test}` = 400/200/1400（全命中）。
4. `scripts/materialize_spreadsheetbench.py`（新增）：join `spreadsheetbench_id_split/`
   的 ID 到 `data/spreadsheetbench_verified_400/dataset.json`，输出
   `data/spreadsheetbench_split/{train,val,test}` = 80/40/280（全命中）。
5. SpreadsheetBench utf-8 修复：`skillopt/envs/spreadsheetbench/{rollout,adapter,react_agent}.py`
   全部 `open()` 加 `encoding="utf-8"`（同款 cp1252 bug，skill 里有 `→` 字符写盘崩溃）。

## 2. SearchQA 复现结果 ✅（gpt-5.5，全量 1400 test）

| 配置 | EM | F1 | 论文 EM |
|---|---|---|---|
| No skill（空文件） | 0.7857 | 0.8898 | ~76.9 |
| Best skill（`ckpt/searchqa/gpt5.5_skill.md`） | 0.8529 | 0.9190 | ~86.5 |
| Δ | **+6.7** | +2.9 | +9.6 |

- 方向 / 量级复现成功；绝对值偏差由 proxy 的 gpt-5.5 快照 + Responses API + temperature=1 非确定性解释。
- 产物：`outputs/eval_searchqa_gpt55_best/`、`outputs/eval_searchqa_gpt55_noskill/`。

### Cost 调研结论
- token 统计**准确**：Responses API 的 `output_tokens` **已含** `reasoning_tokens`，无低估。
- SearchQA medium effort 下 reasoning 很短（~40–60/题）；high effort reasoning ~10×。
- 全量 best-skill eval ≈ 4.44M token，97% 在 prompt 侧。proxy usage 不返回美元定价。

### SearchQA 可复现命令
```powershell
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"
$env:PYTHONIOENCODING="utf-8"
python scripts/eval_only.py --config configs/searchqa/default.yaml `
  --skill ckpt/searchqa/gpt5.5_skill.md `
  --split valid_unseen --split_dir data/searchqa_split `
  --azure_openai_endpoint http://localhost:4141/v1 `
  --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible `
  --target_model gpt-5.5 --workers 24 --out_root outputs/<name>
# no-skill 基线：--skill outputs/empty_skill.md（0 字节空文件，注意 initial.md 含占位文字不算空）
```

---

## 3. SpreadsheetBench 复现结果 ✅（gpt-5.5，全量 280 test）

| 配置 | hard | 论文 hard | Δ vs paper |
|---|---|---|---|
| No skill（`outputs/empty_skill.md`） | 0.3714 | ~41.8 | -4.7 |
| Best skill（`ckpt/spreadsheetbench/gpt5.5_skill.md`） | 0.7536 | ~80.7 | -5.3 |
| **Δ (skill effect)** | **+38.22** | **+38.9** | **-0.7** |

**Δ 几乎完美复现**：绝对值两侧都低 ~5 abs（proxy gpt-5.5 ≠ 论文 gpt-5.5，和 SearchQA 同款偏差），但"加 skill 的相对增益"+38.2 vs +38.9 几乎一致。
SkillOpt 在 procedural task 上的强增益（≈+38，远大于 SearchQA 的 +9.6）端到端成立。

- 时长 / 成本：best 23.5 min · 2.27M token；no-skill 22.7 min · 0.92M token。skill 把 prompt 涨 ~2.5×。
- timeout：尾段 proxy 慢下来，best=8/280 (2.9%) · no-skill=13/280 (4.6%)，量级相似，不影响 Δ 比较。
- 产物：`outputs/eval_spreadsheet_gpt55_best/`、`outputs/eval_spreadsheet_gpt55_noskill/`。

### SpreadsheetBench 可复现命令
```powershell
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"
$env:PYTHONIOENCODING="utf-8"
python scripts/materialize_spreadsheetbench.py    # 一次性：80/40/280 → data/spreadsheetbench_split/
python scripts/eval_only.py --config configs/spreadsheetbench/default.yaml `
  --skill ckpt/spreadsheetbench/gpt5.5_skill.md `
  --split valid_unseen --split_dir data/spreadsheetbench_split `
  --data_root data/spreadsheetbench_verified_400 `
  --azure_openai_endpoint http://localhost:4141/v1 `
  --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible `
  --target_model gpt-5.5 --workers 24 --out_root outputs/<name>
# no-skill 基线：--skill outputs/empty_skill.md
```

---

## 3.5 LiveMathematicianBench 复现结果 ✅（gpt-5.5，全量 124 test）

> 数学定理 MCQ（单选），max_turns=1，data/livemathematicianbench_split = 35/18/124。

| 配置 | EM | 论文 EM |
|---|---|---|
| No skill（模板，`outputs/empty_skill.md`） | 0.4758 (59/124) | 37.6 |
| No skill（**裸 prompt**，删模板提示行） | **0.4435 (55/124)** | 37.6 |
| Best skill（`ckpt/livemath/gpt5.5_skill.md`） | 0.621 (77/124) | 66.9 |
| Δ (skill, 模板基线) | **+14.5** | +29.3 |
| Δ (skill, 裸基线) | **+17.8** | +29.3 |

**关键结论：LiveMath 是 SearchQA 模式，不是 OfficeQA 模式。**
- 用户假设「和 OfficeQA 一样，模板把 baseline 抬高了」→ **基本不成立**。LiveMath 的
  `rollout_system.md` 只有一行可疑提示（"Reason carefully about quantifiers, hypotheses,
  extremal wording, exact equality conditions"，确实是 skill 的压缩版）。把这行删掉做裸 prompt
  A/B：no-skill 只从 **0.476 → 0.444（−3.2，在噪声内）**，远没有 OfficeQA 的 +25.9 那种污染。
- 但「我们 no-skill 高于论文 vanilla」**确实成立**：裸 prompt 44.4 仍 > 论文 37.6（+6.8）。
  这是和 SearchQA / SpreadsheetBench 同款的 **proxy-gpt-5.5 零样本更强**偏移（best 62.1 vs
  论文 66.9 = −4.8，绝对值两侧都低 ~5）。
- 所以 LiveMath 复现了**方向**（skill 大幅有效，+14.5~+17.8），但 Δ 被压缩约一半，
  因为我们的 no-skill 起点本就更高 —— 不是模板污染，是模型快照差异。

### LiveMath 代码修复（本次新增，bug fix，保留）
- `skillopt/envs/livemathematicianbench/rollout.py`：
  - **Windows 文件名 bug**：item id 形如 `202511:4` 含 `:`，`makedirs` 在 Windows 全部失败 →
    之前整批 EM=0。改为 `re.sub(r'[<>:"/\\|?*]', "_", item_id)` 生成安全目录名。
  - **cp1252 崩溃**：`results.jsonl` / `conversation.json` 的 `open()` 补 `encoding="utf-8"`
    （skill/题目含 `⊂` 等 Unicode 数学符号，写盘 charmap 崩溃，同 SpreadsheetBench 款）。

### LiveMath 可复现命令
```powershell
$env:OPENAI_RESPONSES_API_MODELS="gpt-5.5"
$env:PYTHONIOENCODING="utf-8"
python scripts/eval_only.py --config configs/livemathematicianbench/default.yaml `
  --skill ckpt/livemath/gpt5.5_skill.md `
  --split valid_unseen --split_dir data/livemathematicianbench_split `
  --cfg-options env.exec_timeout=1200 `
  --azure_openai_endpoint http://localhost:4141/v1 `
  --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible `
  --target_model gpt-5.5 --workers 10 --out_root outputs/<name>
# no-skill 基线：--skill outputs/empty_skill.md
# 注意：长推理题尾段慢，workers 调小 + exec_timeout 调大（1200~1800）避免被当 timeout=错
```

---

## 3.6 ALFWorld 复现结果 ✅（gpt-5.5，全量 134 test，**WSL 路线**）

> 长程具身文本任务（ALFRED/TextWorld）。max_steps=50，eval split = valid_unseen（OOD）。
> data/alfworld_split = 39/18/134。

| 配置 | Success Rate | 论文 SR |
|---|---|---|
| No skill（`outputs/empty_skill.md`） | **0.8060** (108/134) | 83.6 |
| Best skill（`ckpt/alfworld/gpt5.5_skill.md`） | **0.9179** (123/134) | 95.5 |
| **Δ (skill effect)** | **+11.19** | +38.9→ **+11.9** |

**Δ +11.2 ≈ 论文 +11.9（差 −0.7），几乎完美复现。** 绝对值两侧都低 ~3-4（同款 proxy-gpt-5.5
偏移，和 SearchQA/SpreadsheetBench/LiveMath 一致方向）。SkillOpt 在**长程 procedural 任务**上的
强增益端到端成立——这是迄今唯一真正 long-horizon（平均多步、需规划）的 bench，skill 让 episode
更快收敛、更少撞 50 步上限。

- token/calls：best 6.05M token / 1742 calls vs no-skill 1.71M / 2315 calls。
  **skill 把 prompt 涨 ~5×（42.6K/题 vs 8.6K/题），但 calls 反而少 25%**——因为 skill 让 agent
  少走弯路、更早 done（失败 episode 才会跑满 50 步、吃满 call 数）。completion token best 反而更少。

### ⚠️ 关键：ALFWorld 在原生 Windows 跑不了，必须用 WSL
- eval 走 `AlfredTWEnv`（TextWorld），依赖链 `alfworld → textworld → jericho`。
- **jericho 在原生 Windows 编译失败**（setup 调 make/gcc，报 `WinError 2`）。TextWorld/Jericho
  官方仅支持 Linux/macOS。这正是之前「alfworld 未跑」的根因。
- **解法 = WSL Ubuntu 22.04**：jericho/textworld/alfworld 全部正常编译安装。

### WSL 环境搭建（一次性）
```bash
# 1. 工具链
sudo apt-get install -y build-essential python3-pip python3-venv python3-dev
# 2. venv + 依赖
python3 -m venv ~/skillopt-venv && source ~/skillopt-venv/bin/activate
cd /mnt/c/Users/taoli1/SkillOpt
pip install -e . datasets alfworld omegaconf gymnasium
# 3. 下载游戏数据（~320MB：json_2.1.1 + tw-pddl + logic + detector）
export ALFWORLD_DATA=$HOME/.cache/alfworld
yes | alfworld-download
# 4. 把相对 gamefile 展开成绝对路径 → data/alfworld_split/
python scripts/materialize_alfworld.py
```

### 代理跨 WSL 边界
- WSL2 内 `localhost:4141` **访问不到** Windows 主机代理；要用默认网关 IP：
  `http://$(ip route | grep default | awk '{print $3}'):4141/v1`（本次 = `172.23.144.1:4141`，
  每次 WSL 重启可能变）。

### ALFWorld 可复现命令（WSL 内）
```bash
source ~/skillopt-venv/bin/activate
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_RESPONSES_API_MODELS=gpt-5.5
export PYTHONIOENCODING=utf-8
export ALFWORLD_WORKER_START_METHOD=spawn
python scripts/eval_only.py --config configs/alfworld/default.yaml \
  --skill ckpt/alfworld/gpt5.5_skill.md --split valid_unseen --split_dir data/alfworld_split \
  --cfg-options env.workers=8 \
  --azure_openai_endpoint http://172.23.144.1:4141/v1 \
  --azure_openai_api_key dummy --azure_openai_auth_mode openai_compatible \
  --target_model gpt-5.5 --out_root outputs/eval_alfworld_gpt55_best
# no-skill 基线：--skill outputs/empty_skill.md
```
- 注意：rollout 按 env.workers 分组锁步推进，**每组全部结束才写 conversation.json**，
  results.jsonl 要等 134 个全跑完才落（无增量 resume）。全量约 1.5~2h（失败 episode 跑满 50 步主导耗时）。
- 新增 `scripts/materialize_alfworld.py`：把 path-split 的相对 gamefile 与 $ALFWORLD_DATA 拼成绝对路径。

---

## 3.7 Token / Call 成本汇总（全量 eval，gpt-5.5）

> **口径说明**：这里的 **call = LLM API 调用次数**（不是 function/tool call）。我们的 eval 大多走
> `chat_target` 直接对话（SearchQA / LiveMath / ALFWorld 无真实工具调用）；只有 OfficeQA /
> SpreadsheetBench 用 codex backend 时才有真实工具（glob/bash/python）。
> Responses API 的 `completion_tokens` **已含 reasoning_tokens**，无低估。
> ⚠️ `eval_summary.json` 的 token 是**进程级累加**，**分多次 resume 的 run 只会记录最后一次进程**，
> 会严重低估——下表的 LiveMath 数字来自**专门一次性重跑**（`*_tok` 目录，0 timeout）才准确。

| Benchmark / 配置 | n | calls | prompt tok | completion tok | total tok | total/题 |
|---|--:|--:|--:|--:|--:|--:|
| SearchQA / no-skill（全量） | 1400 | 1400 | 1,548,399 | 130,872 | 1,679,271 | 1,199 |
| SearchQA / best（全量） | 1400 | 1400 | 4,321,799 | 148,444 | 4,470,243 | 3,193 |
| SpreadsheetBench / no-skill | 280 | 281 | 220,167 | 697,396 | 917,563 | 3,277 |
| SpreadsheetBench / best | 280 | 360 | 1,404,471 | 862,055 | 2,266,526 | 8,095 |
| LiveMath / no-skill | 124 | 124 | 102,503 | 594,806 | 697,309 | 5,624 |
| LiveMath / no-skill(bare) | 124 | 124 | 100,395 | 589,821 | 690,216 | 5,566 |
| LiveMath / best | 124 | 124 | 188,352 | 666,926 | 855,278 | 6,897 |
| ALFWorld / no-skill | 134 | 2315 | 1,151,557 | 555,982 | 1,707,539 | 12,743 |
| ALFWorld / best | 134 | 1742 | 5,706,305 | 348,552 | 6,054,857 | 45,186 |
| DocVQA / no-skill | 374 | 374 | 1,707,237 | 27,318 | 1,734,555 | 4,638 |
| DocVQA / best | 374 | 374 | 1,855,341 | 34,533 | 1,889,874 | 5,053 |
| OfficeQA / no-skill | 172 | 867 | 7,604,181 | 646,570 | 8,250,751 | 47,970 |
| OfficeQA / best | 172 | 538 | 5,214,383 | 545,376 | 5,759,759 | 33,487 |

**几个观察：**
- **skill 让 prompt 涨、但行为更高效。** SpreadsheetBench best prompt 6.4×（1.40M vs 0.22M）；
  ALFWorld best prompt 5×（5.71M vs 1.15M，42.6K/题 vs 8.6K/题）；SearchQA best prompt 2.8×（4.32M vs 1.55M）。
- **⚠️ "省 token" 要分清三个量。** 加 skill 后 prompt 几乎总是涨（skill 进 prompt），但 call 和
  completion 在 multi-turn 任务上会降，于是 **total 的走向取决于谁占主导**：
  - **ALFWorld**：best call −25%（1742 vs 2315）、completion −37%（349K vs 556K），**但 skill 太长
    （45K/题），prompt 涨 5× 压倒一切 → total 反而更高（6.05M > 1.71M）。**
  - **OfficeQA**：best call −38%（538 vs 867），省下的轮次够多，**prompt 也降 → total 净省
    （5.76M < 8.25M，−30%）。** 这才是 total 真省的案例。
  - 单轮任务（DocVQA/LiveMath/SearchQA）每题恒 1 call，best 的 total 一律略涨（纯粹 skill 进 prompt）。
- **DocVQA 几乎全在 prompt 侧（图像）**：prompt 1.71M~1.86M 但 completion 仅 27K~35K（每题~73 completion
  token），因为是单轮 VQA、答案极短。token/题 ~4.6K–5K，加 skill 仅 +0.4K。
- **OfficeQA 最贵**（no-skill 48K/题、best 33K/题）：多轮读文档 + 检索，prompt 累积最高。
- **LiveMath 几乎全在 completion 侧**（85% 是推理 token，每题 1 call、~5–6K completion）。
- **SearchQA 单题最省**（no-skill 1.2K/题、best 3.2K/题，reasoning 极短），但题量大（1400）所以全量 total 仍 1.68M→4.47M。
- 代理 usage 不返回美元定价，只能给 token。

### SearchQA best 为何这么贵 —— 根因 + 优化方向

**实测分解**（best vs no-skill，全量 1400）：
- skill 文件 `ckpt/searchqa/gpt5.5_skill.md` = **9746 字符（≈2.6K token）**。
- system prompt：best **10290 字符** vs no-skill **532 字符**；user prompt 两边相同（3683，检索文档+问题）。
- **prompt 增量 2.77M，÷1400 = 每题 +1981 token ≈ skill 本身的 token 量。**

→ **根因：SearchQA 是单轮，9746 字符的静态 cheatsheet 每题完整重发一遍，与题目难度无关。**
单轮没有"工具轮次"可省（对比 OfficeQA 多轮：skill 省下的绕路 token > skill 增的 prompt → total 净降）。

**优化方向**（按 ROI 排序）：
1. **Prompt prefix caching（根本解）**：skill 1400 题一字不变，是天然的可缓存前缀。若 proxy/模型支持
   prefix cache，这 2.6K token 只算一次、后续命中近乎免费。单轮+静态 skill 场景的标准解法。
2. **Skill 压缩**：9746 字符里有大量重叠规则（反复讲"保留精确表面形式/答案类型/关系方向"）。
   见 §3.8 压缩实验：压到 3605 字符（−63%），全量 A/B 验证 token↓38% 但 EM 掉 1.8（保留 71% 增益）。

---

## 3.8 实验：SearchQA skill 压缩的 token/精度权衡

> 动机：§3.7 发现 SearchQA best 贵在 9746 字符静态 skill 每题全量重发。手工压缩到 3605 字符
> （去重叠规则，保留核心启发式），全量 1400 A/B。压缩版在 `outputs/searchqa_skill_compressed.md`，
> **未覆盖 ckpt**。

| 配置 | skill 字符 | EM | ANLS/F1(soft) | total token | token/题 | Δ EM vs no-skill |
|---|--:|--:|--:|--:|--:|--:|
| no-skill | 0 | 0.7907 | 0.8919 | 1,679,271 | 1,199 | — |
| best（原版 ckpt） | 9746 | 0.8536 | 0.9179 | 4,470,243 | 3,193 | +6.29 |
| best（压缩 −63%） | 3605 | 0.8357 | 0.9095 | 2,789,667 | 1,993 | +4.50 |

**结论：**
- 压缩 skill 63% → **token 省 37.6%（4.47M→2.79M）**，但 **EM 掉 1.79 分**（0.854→0.836），
  仍 **保留 71% 的增益**（+4.5 / +6.3）。
- 说明：(a) skill 确有冗余——省近 4 成 token 只掉不到 2 分；(b) 但**不是无损**——被砍掉的具体规则
  （表面形式保留、关系方向陷阱的细化举例）真在贡献边际分数。
- **这是一个清晰的 token/精度 Pareto 点**，不是免费午餐。最优解可能是 **prefix caching**（保留全
  skill、token 近乎免费）而非有损压缩。压缩适合"无缓存 + 成本敏感"场景。

---

## 3.9 实验：A3 query-conditioned skill retrieval（IB 框架，**负结论**）

> 动机：把"按 query 选 skill 片段"形式化为 information bottleneck，希望同 token 下 EM 更高。
> Z = f(S, X) 是检索出的 section 子集；目标 min I(Z;S|X) − β·I(Z;Y|X)。
> 注：full skill 增益 = I(S;Y|X) 是 ceiling（对 f 是常数，从 argmin 消去）。IB 曲线 ≡ 信息平面上的
> (压缩, 相关) **Pareto 前沿**，扫 β 即扫前沿；β = 前沿斜率。

**前提探针（embedding，200 题）**：69% 的题只有 1 个 section "突出"，top-2 覆盖 → skill token 省 60%。
看似稀疏、支持检索。**但全量评测推翻了这个乐观判断。**

**全量 1400 A/B（同口径 harness，gpt-5.5）—— SearchQA 信息平面：**

| 方案 (Z) | EM | total token | 评价 |
|---|--:|--:|---|
| none（Z=∅，β→0） | 0.7907 | 1.68M | 下界 |
| **随机选 2/5 section** | **0.8314** | 2.81M | **控制组** |
| embedding 选 2（q+ctx） | 0.8250 | 2.79M | 内点 |
| hybrid bm25+emb 选 2（q-only） | 0.8257 | 3.04M | 内点（更差） |
| **静态压缩 3605 字符** | **0.8357** | 2.79M | **Pareto 上的擂台** |
| full（Z=S，β→∞） | 0.8536 | 4.47M | ceiling = I(S;Y\|X) |

**核心负结论：所有 query-conditioned 检索都被静态压缩支配，且 ≈ 随机选 section。**
- **随机(0.831) ≈ embedding(0.825) ≈ hybrid(0.826)**——"选哪些 section"几乎不影响 EM；
  起作用的是"塞了约 2 个 section 的量"本身，而非选对了哪两个。
- 检索器的 `section_keep_counts` 暴露原因：`Clue Interpretation`/`Common Clue Traps` 几乎每题都被选
  （>1000/1400），所谓"稀疏"被 k=2 下限抹平——**SearchQA 的 skill 信息是弥漫式、近各向同性的**，
  不是按 section 稀疏分布。用 IB 语言：这条任务的 **IB 曲线中段几乎是平的**，压缩率决定一切、选择策略不决定。
- **推论**：(1) 在这类"弥漫式 skill"任务上 query-conditioned retrieval 理论上就赢不了静态压缩；
  (2) 想真正逼近 ceiling 需换**更细粒度**（rule-level 而非 section-level），或干脆用 prefix caching 保 full。
- 代码：`skillopt/utils/skill_retrieval.py`（SkillRetriever / HybridSkillRetriever，含自带 BM25，
  零新依赖）；实验脚本在 session files/`a3_searchqa_ab.py` + `a3_probe.py`。

**这是一个有价值的负结论**：它给出了"何时该用 SkillOpt-style retrieval"的判据——**只有当 skill 信息
按 query 稀疏可分时才值得检索**；弥漫式 skill（如 SearchQA QA cheatsheet）应直接压缩或缓存。下一个
该测的是**信息分布更稀疏的任务**（如 SpreadsheetBench：不同操作类型用不同 skill 段，疑似真稀疏）。

---

## 3.10 实验：A3 在 SpreadsheetBench 上验证（**判据收紧：per-query 稀疏 vs 全局稀疏**）

> SpreadsheetBench 是多轮 codex 工具任务，skill 在 episode 开头注入一次，query = instruction。
> 实现：rollout 加一个 **env 门控的 per-task section 选择 hook**（`SKILLOPT_SKILL_SELECT_MAP`，默认
> 关闭、零行为变更），所有 policy 离线预计算成 `{task_id: [section_idx]}` 映射，无运行时 embedding API。
> skill 7 节，其中 Matching(5013ch) + Robustness(4359ch) 占 71% 体量。

**全量 280 test A/B（k=2，gpt-5.5）：**

| policy (Z) | EM | total token | 说明 |
|---|--:|--:|---|
| full（7 节） | 0.7536 | 2,266,526 | 基线 |
| **compress_k2（固定保 2 大节，query-无关）** | **0.7571** | 1,680,508 | **最高，省 26% token** |
| embed_k2（按 instruction 相关性选 2 节） | 0.7429 | 1,724,446 | ≈ compress |
| random_k2（随机选 2 节） | **0.5500** | 1,430,443 | **崩塌** |

**核心结论（n=280，SE≈2.6pt）：**
- **random(0.55) ≪ 其它(0.74–0.76)**：选择**确实重要**——和 SearchQA 的 random≈embed 截然不同。
- **但 compress ≈ embed ≈ full（0.74–0.76，全在噪声内）**：**query-conditioned 检索并不优于固定压缩。**
- **机制：SpreadsheetBench 是"全局稀疏"而非"per-query 稀疏"。** 存在一个**固定的**重要子集
  （Matching+Robustness，几乎每题都需要）：random 失败因常漏掉它；embed 成功因常选中它；compress
  成功因**永远**选它。所以"按 query 自适应"没有增量价值——正确策略就是"永远保同样 2 节"= 压缩。
  （embed keep 分布：Robustness 204 / Matching 141 / Output 124 / Library 63… 偶尔用 Output 换掉
  Matching，正是 embed 略低于 compress 的原因。）
- **彩蛋：compress(0.757) 略高于 full(0.754)** —— 砍掉 5 个次要节**去噪提分**，还省 26% token。

## A3 总结论（跨 SearchQA + SpreadsheetBench）

**query-conditioned skill retrieval 在两个 benchmark 上都不优于静态压缩，但机制不同**——random 控制组揭示了真相：

| | SearchQA（弥漫） | SpreadsheetBench（全局稀疏） |
|---|---|---|
| random vs embed | random ≈ embed（**选择无关**） | random ≪ embed（**选择重要**） |
| embed vs compress | embed ≈ compress | embed ≈ compress |
| 解释 | 各节信息近似等价，只有"量"重要 | 有固定重要子集，但**不随 query 变** |

**收紧后的判据：检索只在"重要 skill 子集**因 query 而变**（per-query 稀疏）"时才可能赢压缩。**
- 全局等价（SearchQA）→ 压缩=随机=检索。
- 全局稀疏但固定（SpreadsheetBench）→ 固定压缩已捕获该子集，检索无增量。
- 两个 benchmark 都不是 per-query 稀疏，所以 A3 都不成立。**但两次都得到"静态压缩可省 26–38% token
  且 EM 持平甚至更高"的实用结论**（SpreadsheetBench compress 甚至 > full）。

代码：rollout hook（`skillopt/envs/spreadsheetbench/rollout.py` 的 `_maybe_select_skill`，默认关闭）；
离线脚本 session files/`a3_ss_make_maps.py`。**下一步候选**：找真正 per-query 稀疏的任务——如
多领域混合 QA（每题属不同领域、各领域规则互斥），或把 skill 切到 **rule-level** 细粒度再检索。

---

## 4. 已知 limitations / 未做事项

- **此前复现的 bench 都不是 long-horizon**：SearchQA 单轮 RC；SpreadsheetBench 名义 max_turns=30 实测平均 1–3 turn。这些 skill 在做"程序性记忆 cheatsheet"。**ALFWorld 是唯一真正 long-horizon 的 bench（多步规划、平均数十步），已复现成功（见 §3.6，Δ +11.2 ≈ 论文 +11.9）**——但只能在 WSL 跑（jericho/textworld 不支持原生 Windows）。SWEBench 仓库**没有**（`eval_only.py` 的 ENV_REGISTRY 用 try/except 静默吞 ImportError，看 import 列表会被误导）。
- 实际可跑 env：`alfworld / docvqa / livemathematicianbench / officeqa / searchqa / spreadsheetbench`。
- 训练侧（`train.py` reflect/update loop）未跑过，本次只复现 **eval-only**。

## 5. 会话工具脚本（在 session files/，非仓库）
- `mock_openai_server.py`、`smoke_searchqa.py`、`ab_searchqa.py`（空 vs best A/B + case study）。
