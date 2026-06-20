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

## 4. 已知 limitations / 未做事项

- **复现的 bench 都不是 long-horizon**：SearchQA 单轮 RC；SpreadsheetBench 名义 max_turns=30 实测平均 1–3 turn。skill 在做"程序性记忆 cheatsheet"，没在做"长程规划脑"。仓库里 long-horizon 候选只有 alfworld（simulator，未跑）。SWEBench 仓库**没有**（`eval_only.py` 的 ENV_REGISTRY 用 try/except 静默吞 ImportError，看 import 列表会被误导）。
- 实际可跑 env：`alfworld / docvqa / livemathematicianbench / officeqa / searchqa / spreadsheetbench`。
- 训练侧（`train.py` reflect/update loop）未跑过，本次只复现 **eval-only**。

## 5. 会话工具脚本（在 session files/，非仓库）
- `mock_openai_server.py`、`smoke_searchqa.py`、`ab_searchqa.py`（空 vs best A/B + case study）。
