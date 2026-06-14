# SkillOpt 复现进度 (progress.md)

> 用本机 GitHub Copilot proxy 复现 SkillOpt 论文 (arXiv:2605.23904) 的评测。
> 最近更新：完成 SearchQA 全量复现；SpreadsheetBench 数据已下载待 materialize。

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

## 3. 下一步：SpreadsheetBench（进行中）

### 论文目标分（gpt-5.5，Table 1 / 消融）
- No skill ≈ 41.8 → SkillOpt ≈ 80.7（**+38.9**，procedural 任务，增益远大于 SearchQA）。
- test split = 280 条（train 80 / val 40 / test 280）。
- 任务性质：**agentic** — multi-round codegen，max_turns=30，真实 openpyxl/pandas 运行时，exec_timeout=600。
  ⇒ cost / 时间远高于 SearchQA，全量评测较重。

### 配置 `configs/spreadsheetbench/default.yaml`
- env: spreadsheetbench, mode=multi, max_turns=30, exec_timeout=600, workers=24
- `split_dir: data/spreadsheetbench_split`（**尚未生成**）
- `data_root: data/spreadsheetbench_verified_400`（**已下载解压，待移动到位**）
- skill: `ckpt/spreadsheetbench/gpt5.5_skill.md`（已存在）

### 数据现状（已下载）
- 已下载并解压：`data/_dl/spreadsheetbench_verified_400/`
  - `dataset.json` = 400 任务，字段：`id, instruction, spreadsheet_path, instruction_type, answer_position, answer_sheet, data_position`。
  - `spreadsheet/<id>/` 含 `*_init.xlsx` + `*_golden.xlsx` + `prompt.txt`
    （rollout.py 的 fallback glob 认 `*_init.xlsx`/`*_golden.xlsx` 命名，✅ 兼容）。
- 源：HF `KAKA22/SpreadsheetBench`，`spreadsheetbench_verified_400.tar.gz`（15MB），
  revision `ab0b742b0fc95b946f212d80ac7771b5531272e4`（与 manifest 一致）。

### 待办（下一个 session 从这里继续）
- [ ] 把 `data/_dl/spreadsheetbench_verified_400` 放到 `data/spreadsheetbench_verified_400`（= data_root）。
- [ ] materialize `data/spreadsheetbench_split/{train,val,test}`：
      用 `data/spreadsheetbench_id_split/{train,val,test}/items.json` 的 id
      join `dataset.json`，输出每个 split 一个 JSON 数组（含完整字段）。
      （仿照 `scripts/materialize_searchqa.py` 写一个 `materialize_spreadsheetbench.py`。）
- [ ] 先小批冒烟：`eval_only.py --config configs/spreadsheetbench/default.yaml
      --skill ckpt/spreadsheetbench/gpt5.5_skill.md --split valid_unseen
      --split_dir data/spreadsheetbench_split --data_root data/spreadsheetbench_verified_400
      --test_env_num 5 ... --target_model gpt-5.5`（带 `OPENAI_RESPONSES_API_MODELS=gpt-5.5`）。
- [ ] 注意 codegen 走的是 `codegen_agent.py`，确认它也用 chat_target / 支持 Responses 路由
      （`_needs_responses_api` 在 codegen_agent.py:347 已引用，应已兼容）。
- [ ] 确认 Windows 上 executor 跑 openpyxl/pandas 子进程正常（exec_timeout=600）。

## 4. 会话工具脚本（在 session files/，非仓库）
- `mock_openai_server.py`、`smoke_searchqa.py`、`ab_searchqa.py`（空 vs best A/B + case study）。
