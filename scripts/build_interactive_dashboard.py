"""Generate self-contained interactive Weight-Space Programming & LoRA Surgery Dashboard (index.html)."""
import json
from pathlib import Path

ROOT = Path('Q:/wiki/misc/tabgroup-opd-20261026')
REPO_WS = Path('Q:/wiki/misc/tabgroup-research-repo/results/weight-space-surgery-20261103')

dossier = json.loads((ROOT / 'weight_space_dossier.json').read_text(encoding='utf-8'))
bundle = json.loads((ROOT / 'complete_viz_bundle.json').read_text(encoding='utf-8'))

# Extract compact payload for HTML
showcase_ids = ['JH02', 'JH11', 'JH12', 'JH16', 'service-4-f43f114595c2', 'service-5-b48adde126b3', 'service-7-1c3a2d56655d', 'service-18-53ff1cc26ac3']

models_meta = [
    {'id': 'sft2000', 'name': '0. Base 4B-SFT (sft2000)', 'cat': 'Baseline', 'params_edited': '0', 'v1_p': 79.23, 'v1_r': 68.26, 'v2_p': 58.33, 'v2_r': 45.71, 'desc': '2000-step LoRA SFT baseline; suffers from severe giant-bucket over-merging on v2.1 (e.g., JH02 25 tabs in one Soccer bucket).'},
    {'id': 'opsd_astra', 'name': '1. OPSD-Astra (150 steps)', 'cat': 'Trained Parent', 'params_edited': '21.2M (Trained)', 'v1_p': 85.48, 'v1_r': 72.52, 'v2_p': 69.09, 'v2_r': 59.36, 'desc': 'On-Policy Self-Distillation with GPT-6-astra privileged context; preserves v1 Recall and fixes v2.1 over-merging.'},
    {'id': 'sft_astra', 'name': '2. Astra-SFT (150 steps)', 'cat': 'Trained Parent', 'params_edited': '21.2M (Trained)', 'v1_p': 79.55, 'v1_r': 64.19, 'v2_p': 65.08, 'v2_r': 60.25, 'desc': 'Off-Policy Cross-Entropy SFT on same 1,200 Astra labels; stronger middle-layer splitting on v2.1, but L18–31 overfits &/Ungrouped (-8.33pp v1 Recall drop).'},
    {'id': 'idea_a_64neurons_sft2000', 'name': 'Idea A: L26–31 64-Neuron Valve Edit (on Base SFT)', 'cat': 'Single-Model Weight Edit (0 Train)', 'params_edited': '64 cols/layer (0.69%)', 'v1_p': 79.79, 'v1_r': 69.70, 'v2_p': 61.29, 'v2_r': 57.93, 'desc': 'Zero training, no OPSD/Astra-SFT used! Scales top-64 newline-promoting columns by 2.0x and zeroes top-64 comma-promoting columns in L26–31 mlp.down_proj of Base SFT. Lifts v2.1 Recall by +12.22pp!'},
    {'id': 'idea_c_qnorm135_sft2000', 'name': 'Idea C: Full-Attn q_norm * 1.35x (on Base SFT)', 'cat': 'Single-Model Weight Edit (0 Train)', 'params_edited': '4 scalars (L07,11,15,19)', 'v1_p': 79.89, 'v1_r': 70.37, 'v2_p': 59.26, 'v2_r': 51.96, 'desc': 'Zero training! Multiplies q_norm in 4 Full-Attention layers by 1.35x to sharpen cross-tab attention clustering. Lifts v1 Recall to 70.37% (+2.11pp) and v2.1 Recall by +6.25pp.'},
    {'id': 'idea_b1_sigma1_zero', 'name': 'Idea B-1: L18–31 Top-1 Singular Value σ1=0 (on Astra-SFT)', 'cat': 'Single-Model Weight Edit (0 Train)', 'params_edited': '1 scalar/module (112 nums)', 'v1_p': 80.34, 'v1_r': 65.74, 'v2_p': 65.57, 'v2_r': 58.69, 'desc': 'Zero training, no OPSD used! Zeroes only the single largest singular value σ1 of ΔW_SFT in L18–31 to excise lexical &/Ungrouped drift.'},
    {'id': 'idea_b2_sft_top2_zero', 'name': 'Idea B-2: L18–31 Top-2 Singular Values σ1=σ2=0 (on Astra-SFT)', 'cat': 'Single-Model Weight Edit (0 Train)', 'params_edited': '2 scalars/module (224 nums)', 'v1_p': 80.66, 'v1_r': 67.63, 'v2_p': 66.13, 'v2_r': 60.25, 'desc': 'Zero training, no OPSD used! Zeroes top-2 singular values (σ1, σ2) of ΔW_SFT in L18–31. Recovers +3.44pp v1 Recall and +1.05pp v2.1 Precision while keeping 60.25% v2.1 Recall.'},
    {'id': 'graft_18', 'name': 'Surgery A: Graft-18 (L00–17 Astra-SFT + L18–31 OPSD)', 'cat': 'Dual-Adapter Circuit Surgery (0 Train)', 'params_edited': 'Layerwise Cut @ L18 (r=16)', 'v1_p': 82.87, 'v1_r': 69.19, 'v2_p': 69.84, 'v2_r': 64.52, 'desc': 'Mounts Astra-SFT middle-layer sub-task splitter (L00–17) onto OPSD clean readout policy (L18–31). Achieves 64.52% v2.1 Recall (+5.16pp over OPSD, +4.27pp over Astra-SFT)!'},
    {'id': 'svd_blend_50_50', 'name': 'Surgery B: SVD-Blend 50/50 (Rank-16 Spectral Consensus)', 'cat': 'Dual-Adapter Circuit Surgery (0 Train)', 'params_edited': 'Rank-16 QR-SVD (r=16)', 'v1_p': 82.61, 'v1_r': 69.37, 'v2_p': 72.41, 'v2_r': 64.42, 'desc': 'Exact rank-16 SVD projection of 0.5 ΔW_OPSD + 0.5 ΔW_SFT across all 32 layers. Amplifies shared L08–15 clustering directions while halving & truncating L18–31 noise. Highest v2.1 Precision (72.41%)!'},
    {'id': 'rev_graft_18', 'name': 'Control: Rev-Graft-18 (L00–17 OPSD + L18–31 Astra-SFT)', 'cat': 'Causal Ablation Control', 'params_edited': 'Reverse Cut @ L18', 'v1_p': 82.78, 'v1_r': 67.48, 'v2_p': 67.86, 'v2_r': 59.81, 'desc': 'Causal proof: transplanting Astra-SFT late layers (L18–31) onto OPSD immediately drops v1 Recall by -5.04pp (from 72.52% to 67.48%).'},
]

# Gather showcase case outputs across all models
showcase_outputs = {}
for cid in showcase_ids:
    ds = 'v2.1' if cid.startswith('JH') else 'v1'
    showcase_outputs[cid] = {}
    for m_id in ['sft2000', 'opsd_astra', 'sft_astra', 'idea_a_64neurons_sft2000', 'idea_c_qnorm135_sft2000', 'idea_b1_sigma1_zero']:
        if m_id in bundle and ds in bundle[m_id] and cid in bundle[m_id][ds]['cases']:
            showcase_outputs[cid][m_id] = bundle[m_id][ds]['cases'][cid]
    for m_id in ['graft_18', 'svd_blend_50_50', 'idea_b2_sft_top2_zero']:
        if m_id in bundle['_extra']['showcase_runs'] and cid in bundle['_extra']['showcase_runs'][m_id]:
            showcase_outputs[cid][m_id] = bundle['_extra']['showcase_runs'][m_id][cid]

payload = {
    'models': models_meta,
    'dossier': dossier['layers'],
    'neuron_spectrum': bundle['_extra']['neuron_spectrum'],
    'showcase_meta': bundle['_showcase_meta'],
    'showcase_outputs': showcase_outputs,
}

html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sparks of Weight-Space Surgery in TabGroup LoRA (Qwen3.5-4B)</title>
<style>
:root {{
  --bg: #0b1120;
  --panel: #111827;
  --panel2: #1e293b;
  --border: #334155;
  --text: #f8fafc;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --green: #10b981;
  --red: #ef4444;
  --purple: #a855f7;
  --amber: #f59e0b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 28px 24px 60px;
  font: 14.5px/1.65 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--text);
  max-width: 1420px;
  margin-inline: auto;
}}
h1 {{ font-size: 28px; margin: 0 0 6px; letter-spacing: -0.3px; }}
h2 {{ font-size: 19px; margin: 0 0 12px; color: #e2e8f0; display: flex; align-items: center; gap: 10px; }}
h3 {{ font-size: 15px; margin: 14px 0 8px; color: #cbd5e1; }}
p {{ margin: 6px 0; }}
.muted {{ color: var(--muted); }}
.badge {{
  display: inline-block;
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(56, 189, 248, 0.15);
  color: var(--accent);
  border: 1px solid rgba(56, 189, 248, 0.35);
}}
.hero {{
  background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
  border: 1px solid #3b4262;
  border-radius: 14px;
  padding: 24px 28px;
  margin-bottom: 22px;
}}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 18px;
}}
.kpi {{
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}}
.kpi .val {{ font-size: 24px; font-weight: 800; margin: 4px 0; }}
.kpi .sub {{ font-size: 12.5px; color: var(--muted); }}
section {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
  margin-bottom: 22px;
}}
.heatmap-wrap {{ overflow-x: auto; padding-bottom: 6px; }}
table.hm {{ border-collapse: separate; border-spacing: 2px; width: 100%; min-width: 1120px; font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; }}
table.hm th, table.hm td {{ padding: 6px 3px; text-align: center; border-radius: 4px; }}
table.hm th.row-lbl {{ text-align: left; padding-left: 8px; width: 195px; background: var(--panel2); color: #e2e8f0; font-weight: 600; }}
table.hm th.col-lbl {{ background: var(--panel2); color: #cbd5e1; font-weight: 600; cursor: pointer; }}
table.hm td {{ cursor: pointer; transition: transform 0.1s; }}
table.hm td:hover {{ transform: scale(1.15); outline: 2px solid #fff; z-index: 5; position: relative; }}
.cut-col {{ border-left: 2.5px dashed #ef4444 !important; }}
.zone-col {{ border-left: 2.5px dashed #10b981 !important; }}

table.metrics {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 10px; }}
table.metrics th, table.metrics td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; text-align: left; }}
table.metrics th {{ background: var(--panel2); color: #cbd5e1; font-weight: 600; }}
table.metrics tr:hover {{ background: rgba(56, 189, 248, 0.06); }}
table.metrics tr.highlight-a {{ background: rgba(16, 185, 129, 0.10); }}
table.metrics tr.highlight-b {{ background: rgba(168, 85, 247, 0.12); }}

.grid-2 {{ display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 18px; }}
.card {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
pre {{
  margin: 8px 0 0;
  padding: 12px;
  background: #090d16;
  border: 1px solid #1e293b;
  border-radius: 8px;
  font: 12.5px/1.6 ui-monospace, Consolas, monospace;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: #e2e8f0;
}}
select, button {{
  background: var(--panel2);
  color: var(--text);
  border: 1px solid #475569;
  border-radius: 7px;
  padding: 7px 12px;
  font: inherit;
  cursor: pointer;
}}
button.active {{ background: #0284c7; border-color: #38bdf8; font-weight: 600; }}
.bar-row {{ display: flex; align-items: center; gap: 8px; margin: 5px 0; font-family: ui-monospace, monospace; font-size: 12px; }}
.bar-track {{ flex: 1; height: 14px; background: #0f172a; border-radius: 4px; overflow: hidden; position: relative; }}
.bar-fill {{ height: 100%; border-radius: 4px; }}
.case-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }}
@media (max-width: 1024px) {{
  .kpi-grid, .grid-2, .case-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<div class="hero">
  <span class="badge">Weight-Space Programming · Qwen3.5-4B LoRA (21.2M Params)</span>
  <h1>Sparks of Weight-Space Surgery in Browser Tab Grouping</h1>
  <p style="margin:4px 0 6px; color:#cbd5e1; font-weight:600;">Tao Li (taoli1@microsoft.com) · Qiang Shen (qiangshen@microsoft.com) · Bo Shao (boshao@microsoft.com) · Linjun Shou (lisho@microsoft.com)</p>
  <p class="muted">从 32 层 LoRA 权重 SVD 与 Unembedding 词表反投影解剖，到「零训练单神经元阀门手术（Idea A）」与「跨范式低秩谱共识（SVD-Blend / Graft-18）」</p>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="sub">1. 定位核心物理机制</div>
      <div class="val" style="color:var(--accent)">L08–15 vs L18–31</div>
      <div class="sub">中前层共享细粒度聚类回路 (cos=0.37)；后层 L18–31 被离线 SFT 写入 8–11× <code>' &'</code> 与 <code>Ungrouped</code> 漂移</div>
    </div>
    <div class="kpi">
      <div class="sub">2. 单模型改 64 列神经元 (Idea A)</div>
      <div class="val" style="color:var(--green)">+12.22pp Recall</div>
      <div class="sub">0 训练、不借用 OPSD！仅缩放 <code>sft2000</code> L26–31 的 64 列 <code>\\n</code> vs <code>,</code> 神经元，v2.1 Recall 从 45.71% → <b>57.93%</b></div>
    </div>
    <div class="kpi">
      <div class="sub">3. 单模型抹除 σ1,σ2 标量 (Idea B-2)</div>
      <div class="val" style="color:var(--amber)">+3.44pp v1 Recall</div>
      <div class="sub">0 训练、不借用 OPSD！仅将 <code>sft_astra</code> L18–31 的前 2 个奇异值归零，v1 Recall 从 64.19% → <b>67.63%</b></div>
    </div>
    <div class="kpi">
      <div class="sub">4. 双模型嫁接与谱共识 (Graft-18 / SVD)</div>
      <div class="val" style="color:var(--purple)">72.41% P / 64.52% R</div>
      <div class="sub"><code>Graft-18</code> (v2.1 R <b>64.52%</b>) 与 <code>SVD-Blend 50/50</code> (v2.1 P <b>72.41%</b>) 双双超越所有单范式训练上限</div>
    </div>
  </div>
</div>

<section>
  <h2><span>① 32 层全网 LoRA 权重空间热力图（点击任意层/模块查看 SVD 主奇异向量词表解码）</span></h2>
  <p class="muted">对比同一个 <code>sft2000</code> 起点出发的 <code>ΔW_OPSD = W_OPSD - W_0</code> 与 <code>ΔW_SFT = W_SFT - W_0</code>（共 32 层 × 6 类核心模块，带 <code>*</code> 为 Full-Attention 层，其余为 Gated-DeltaNet 线性注意力层）。绿虚线 <code>L08</code> 为共享特征聚类区起点，红虚线 <code>L18</code> 为后层读出漂移分界线。</p>
  <div style="margin: 10px 0 12px; display:flex; gap:10px; flex-wrap:wrap;">
    <button id="btn-hm-cos" class="active" onclick="setHeatmapMode('cos')">模式 1：层间方向对齐度 cos(ΔW_OPSD, ΔW_SFT)</button>
    <button id="btn-hm-ratio" onclick="setHeatmapMode('ratio')">模式 2：更新步长膨胀比 ||ΔW_SFT|| / ||ΔW_OPSD||</button>
    <button id="btn-hm-amp" onclick="setHeatmapMode('amp')">模式 3：后层污染投影比 (' &' 词表方向 SFT/OPSD 倍数)</button>
  </div>
  <div class="heatmap-wrap">
    <table class="hm" id="heatmap-table"></table>
  </div>

  <div class="grid-2" style="margin-top: 16px;">
    <div class="card" id="inspector-card">
      <h3 style="margin-top:0">🔍 权重矩阵显微镜（点击上方热力图单元格切换）</h3>
      <div id="inspector-body"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">⚡ Idea A：单神经元「换行拆组 (\\n) vs 逗号并组 (,)」极性谱</h3>
      <p class="muted" style="font-size:12.5px">将底座 <code>mlp.down_proj</code>（形状 <code>2560 × 9216</code>）的每一列投影到词表判别方向 <code>v_newline - v_comma</code>。仅把绿色的 64 列乘 <code>2.0x</code>、红色的 64 列置 <code>0</code>，即可在 0 训练下拆开 <code>JH02</code> 25-Tab 巨型桶！</p>
      <div style="margin-bottom:8px;">
        <label>选择后层：<select id="neuron-layer-select" onchange="renderNeuronBars()">
          <option value="L31" selected>Layer 31 (Full-Attn Block MLP)</option>
          <option value="L30">Layer 30 (Linear-Attn Block MLP)</option>
          <option value="L29">Layer 29 (Linear-Attn Block MLP)</option>
          <option value="L28">Layer 28 (Linear-Attn Block MLP)</option>
          <option value="L27">Layer 27 (Full-Attn Block MLP)</option>
          <option value="L26">Layer 26 (Linear-Attn Block MLP)</option>
        </select></label>
      </div>
      <div id="neuron-bars"></div>
    </div>
  </div>
</section>

<section>
  <h2><span>② 全量 61 题（v1 Golden45 + v2.1 Hard Multi-Task）精确评测总表</span></h2>
  <p class="muted">所有零训练手术方案（Idea A / B-1 / B-2 / C 及 Graft-18 / SVD-Blend）均在与历史评测完全一致的输入 Token、Chat Template、Float32 与 A100 硬件合约下实测。</p>
  <div class="heatmap-wrap">
    <table class="metrics" id="metrics-table"></table>
  </div>
</section>

<section>
  <h2><span>③ 真实 Case 手术台对比（亲自验货：看单神经元修改与层间嫁接如何改写输出）</span></h2>
  <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
    <label>选择典型诊断案例：<select id="showcase-select" onchange="renderShowcase()">
      <option value="JH02">v2.1 / JH02（25 网页大锅饭：Soccer + Grok + Man City + 数据分析）</option>
      <option value="JH16">v2.1 / JH16（13 网页：Idea A 满分 100% & Graft-18 救回 Elyon School）</option>
      <option value="JH11">v2.1 / JH11（33 网页高难混杂：7 个细粒度子任务拆分）</option>
      <option value="JH12">v2.1 / JH12（24 网页：Idea C 单标量 q_norm*1.35x 当场拆开 Travel 大桶）</option>
      <option value="service-4-f43f114595c2">v1 / service-4（19 网页：Idea B 单奇异值 σ1=0 救回 Travel: 2, 12）</option>
      <option value="service-5-b48adde126b3">v1 / service-5（14 网页：Idea C 注意力调温跳至 100% Recall）</option>
      <option value="service-7-1c3a2d56655d">v1 / service-7（11 网页：Portable Projector / Edge / Spider-Man）</option>
    </select></label>
  </div>
  <div class="case-grid" id="showcase-grid"></div>
  <details style="margin-top:14px;">
    <summary style="cursor:pointer; color:var(--accent); font-weight:600;">展开查看本题原始打开的网页列表（Title & URL）与参考答案</summary>
    <div id="showcase-ref" style="margin-top:10px;"></div>
  </details>
</section>

<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};

const MOD_ROWS = [
  {{key: 'mlp.gate_proj', label: 'MLP GateProj'}},
  {{key: 'mlp.up_proj', label: 'MLP UpProj'}},
  {{key: 'mlp.down_proj', label: 'MLP DownProj (Writer)'}},
  {{key: 'attn_in', label: 'Attn InProj (Q/QKV)'}},
  {{key: 'attn_mid', label: 'Attn Gate/K (Z/K)'}},
  {{key: 'attn_out', label: 'Attn OutProj (Writer)'}}
];

let currentHmMode = 'cos';
let selectedLayer = 26;
let selectedModKey = 'mlp.down_proj';

function resolveMod(layerObj, key) {{
  const m = layerObj.modules;
  if (key === 'attn_in') return m['self_attn.q_proj'] ? ['self_attn.q_proj', m['self_attn.q_proj']] : ['linear_attn.in_proj_qkv', m['linear_attn.in_proj_qkv']];
  if (key === 'attn_mid') return m['self_attn.k_proj'] ? ['self_attn.k_proj', m['self_attn.k_proj']] : ['linear_attn.in_proj_z', m['linear_attn.in_proj_z']];
  if (key === 'attn_out') return m['self_attn.o_proj'] ? ['self_attn.o_proj', m['self_attn.o_proj']] : ['linear_attn.out_proj', m['linear_attn.out_proj']];
  return [key, m[key]];
}}

function colorForVal(mode, v) {{
  if (mode === 'cos') {{
    const t = Math.max(0, Math.min(1, (v - 0.05) / 0.32));
    const r = Math.round(30 + t * 195);
    const g = Math.round(60 + (1 - Math.abs(t - 0.5)*1.6) * 120);
    const b = Math.round(210 - t * 170);
    return `rgb(${{r}},${{g}},${{b}})`;
  }} else if (mode === 'ratio') {{
    const t = Math.max(0, Math.min(1, (v - 1.0) / 0.85));
    const r = Math.round(40 + t * 200);
    const g = Math.round(130 - t * 90);
    const b = Math.round(90 - t * 60);
    return `rgb(${{r}},${{g}},${{b}})`;
  }} else {{
    const t = Math.max(0, Math.min(1, (v - 1.0) / 8.5));
    const r = Math.round(35 + t * 205);
    const g = Math.round(90 - t * 65);
    const b = Math.round(140 - t * 110);
    return `rgb(${{r}},${{g}},${{b}})`;
  }}
}}

function setHeatmapMode(mode) {{
  currentHmMode = mode;
  ['cos', 'ratio', 'amp'].forEach(k => {{
    document.getElementById('btn-hm-' + k).classList.toggle('active', k === mode);
  }});
  renderHeatmap();
}}

function renderHeatmap() {{
  const tbl = document.getElementById('heatmap-table');
  let html = '<thead><tr><th class="row-lbl">Module \\\\ Layer</th>';
  for (let l = 0; l < 32; l++) {{
    const isFull = [3,7,11,15,19,23,27,31].includes(l);
    const cls = l === 18 ? 'col-lbl cut-col' : (l === 8 ? 'col-lbl zone-col' : 'col-lbl');
    html += `<th class="${{cls}}" onclick="inspectCell(${{l}}, '${{selectedModKey}}')">L${{String(l).padStart(2,'0')}}${{isFull?'*':''}}</th>`;
  }}
  html += '</tr></thead><tbody>';

  MOD_ROWS.forEach(row => {{
    html += `<tr><th class="row-lbl">${{row.label}}</th>`;
    for (let l = 0; l < 32; l++) {{
      const [actualName, m] = resolveMod(DATA.dossier[l], row.key);
      let val = 0, txt = '';
      if (currentHmMode === 'cos') {{
        val = m.cos_opsd_sft;
        txt = val.toFixed(2);
      }} else if (currentHmMode === 'ratio') {{
        val = m.norm_sft / Math.max(m.norm_opsd, 1e-6);
        txt = val.toFixed(2) + 'x';
      }} else {{
        if (m.vocab_proj_sft) {{
          val = m.vocab_proj_sft.amp_space / Math.max(m.vocab_proj_opsd.amp_space, 1e-6);
          txt = val.toFixed(1) + 'x';
        }} else {{
          val = 1.0;
          txt = '-';
        }}
      }}
      const bg = txt === '-' ? '#1e293b' : colorForVal(currentHmMode, val);
      const cls = l === 18 ? 'cut-col' : (l === 8 ? 'zone-col' : '');
      const selStyle = (l === selectedLayer && row.key === selectedModKey) ? 'outline:2px solid #fff;' : '';
      html += `<td class="${{cls}}" style="background:${{bg}};${{selStyle}}" onclick="inspectCell(${{l}}, '${{row.key}}')" title="L${{l}} ${{actualName}}">${{txt}}</td>`;
    }}
    html += '</tr>';
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
  inspectCell(selectedLayer, selectedModKey, false);
}}

function inspectCell(l, modKey, rerender=true) {{
  selectedLayer = l;
  selectedModKey = modKey;
  if (rerender) renderHeatmap();
  const [actualName, m] = resolveMod(DATA.dossier[l], modKey);
  const isFull = [3,7,11,15,19,23,27,31].includes(l);
  let body = `<p><b>当前选中：Layer ${{l}} (${{isFull ? 'Full-Attention 层' : 'Gated-DeltaNet 线性注意力层'}}) · <code>${{actualName}}</code></b></p>`;
  body += `<p>Frobenius 更新范数：<code>||ΔW_OPSD|| = ${{m.norm_opsd.toFixed(4)}}</code> ｜ <code>||ΔW_SFT|| = ${{m.norm_sft.toFixed(4)}}</code> (膨胀比 <b>${{(m.norm_sft/m.norm_opsd).toFixed(2)}}x</b>)</p>`;
  body += `<p>方向余弦相似度：<code>cos(ΔW_OPSD, ΔW_SFT) = ${{m.cos_opsd_sft.toFixed(4)}}</code> ｜ 第一主奇异值能量占比：OPSD <code>${{(m.top1_ratio_opsd*100).toFixed(1)}}%</code> / SFT <code>${{(m.top1_ratio_sft*100).toFixed(1)}}%</code></p>`;
  if (m.u1_sft_top_pos) {{
    body += `<div style="margin-top:10px; padding:10px; background:#090d16; border-radius:8px; border:1px solid #1e293b;">`;
    body += `<p style="margin:0 0 6px; color:#f87171;"><b>Astra-SFT Top-1 左奇异向量 u₁ 写入残差流的词表 Top Tokens：</b><br>正向 (+u₁): <code>${{JSON.stringify(m.u1_sft_top_pos)}}</code><br>负向 (-u₁): <code>${{JSON.stringify(m.u1_sft_top_neg)}}</code></p>`;
    body += `<p style="margin:6px 0 0; color:#38bdf8;"><b>OPSD-Astra Top-1 左奇异向量 u₁ 写入残差流的词表 Top Tokens：</b><br>正向 (+u₁): <code>${{JSON.stringify(m.u1_opsd_top_pos)}}</code><br>负向 (-u₁): <code>${{JSON.stringify(m.u1_opsd_top_neg)}}</code></p>`;
    body += `<p style="margin:6px 0 0; font-size:12px; color:var(--muted);">关键方向投影强度 (OPSD vs SFT)：<code>' &'</code>: ${{m.vocab_proj_opsd.amp_space}} vs <b>${{m.vocab_proj_sft.amp_space}}</b> ｜ <code>'\\n'</code>: ${{m.vocab_proj_opsd.newline}} vs ${{m.vocab_proj_sft.newline}} ｜ <code>','</code>: ${{m.vocab_proj_opsd.comma}} vs ${{m.vocab_proj_sft.comma}}</p>`;
    body += `</div>`;
  }} else {{
    body += `<p class="muted" style="margin-top:8px;">（注：该模块为内部投影层，输出维度非残差流 d_model=2560；请点击 <code>MLP DownProj</code> 或 <code>Attn OutProj</code> 查看直接词表投影。）</p>`;
  }}
  document.getElementById('inspector-body').innerHTML = body;
}}

function renderNeuronBars() {{
  const l = document.getElementById('neuron-layer-select').value;
  const spec = DATA.neuron_spectrum[l];
  let html = '<div style="font-size:12px; color:#10b981; font-weight:600; margin-bottom:4px;">▲ 最强促拆（Newline-Promoting，激发换行另起新组）Top-6 神经元列：</div>';
  spec.top_newline_cols.slice(0, 6).forEach(x => {{
    const w = Math.min(100, (x.score / 0.27) * 100);
    html += `<div class="bar-row"><span style="width:72px">Col ${{x.col}}</span><div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:#10b981"></div></div><span style="width:55px;text-align:right;color:#10b981">+${{x.score.toFixed(4)}}</span></div>`;
  }});
  html += '<div style="font-size:12px; color:#ef4444; font-weight:600; margin:10px 0 4px;">▼ 最强促并（Comma-Promoting，惯性连写塞入大桶）Top-6 神经元列：</div>';
  spec.top_comma_cols.slice(0, 6).forEach(x => {{
    const w = Math.min(100, (Math.abs(x.score) / 0.27) * 100);
    html += `<div class="bar-row"><span style="width:72px">Col ${{x.col}}</span><div class="bar-track"><div class="bar-fill" style="width:${{w}}%;background:#ef4444"></div></div><span style="width:55px;text-align:right;color:#ef4444">${{x.score.toFixed(4)}}</span></div>`;
  }});
  document.getElementById('neuron-bars').innerHTML = html;
}}

function renderMetricsTable() {{
  const tbl = document.getElementById('metrics-table');
  let html = `<thead><tr>
    <th>模型 / 零训练权重手术方案</th>
    <th>类别</th>
    <th>改动参数量</th>
    <th>v1 Init P (n=45)</th>
    <th>v1 Recall (n=45)</th>
    <th>v2.1 Init P (n=16)</th>
    <th>v2.1 Recall (n=16)</th>
    <th>核心物理机制与收益</th>
  </tr></thead><tbody>`;
  DATA.models.forEach(m => {{
    const cls = m.id.startsWith('idea_') ? 'highlight-a' : (m.id === 'graft_18' || m.id === 'svd_blend_50_50' ? 'highlight-b' : '');
    html += `<tr class="${{cls}}">
      <td><b>${{m.name}}</b></td>
      <td><span class="badge">${{m.cat}}</span></td>
      <td><code>${{m.params_edited}}</code></td>
      <td>${{m.v1_p.toFixed(2)}}%</td>
      <td><b>${{m.v1_r.toFixed(2)}}%</b></td>
      <td><b>${{m.v2_p.toFixed(2)}}%</b></td>
      <td><b style="color:${{m.v2_r >= 64 ? '#c084fc' : (m.v2_r >= 57.9 ? '#34d399' : '#f8fafc')}}">${{m.v2_r.toFixed(2)}}%</b></td>
      <td class="muted" style="font-size:12.5px">${{m.desc}}</td>
    </tr>`;
  }});
  html += '</tbody>';
  tbl.innerHTML = html;
}}

function renderShowcase() {{
  const cid = document.getElementById('showcase-select').value;
  const outs = DATA.showcase_outputs[cid] || {{}};
  const meta = DATA.showcase_meta[cid];
  const order = [
    ['sft2000', '0. 原 4B-SFT 底座 (sft2000)'],
    ['opsd_astra', '1. 原 OPSD-Astra (150步)'],
    ['sft_astra', '2. 原 Astra-SFT (150步)'],
    ['idea_a_64neurons_sft2000', '★ Idea A：改 sft2000 64列神经元 (0训练)'],
    ['idea_b2_sft_top2_zero', '★ Idea B-2：抹除 sft_astra σ1,σ2 (0训练)'],
    ['graft_18', '★ Surgery A：Graft-18 (L0-17 SFT + L18-31 OPSD)'],
    ['svd_blend_50_50', '★ Surgery B：SVD-Blend 50/50 (秩-16 谱共识)'],
    ['idea_c_qnorm135_sft2000', '★ Idea C：Full-Attn q_norm * 1.35x (0训练)']
  ];
  let html = '';
  order.forEach(([mid, title]) => {{
    const r = outs[mid];
    if (!r) return;
    const borderCol = mid.startsWith('idea_') ? '#10b981' : (mid === 'graft_18' || mid === 'svd_blend_50_50' ? '#a855f7' : '#334155');
    html += `<div class="card" style="border-color:${{borderCol}}">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <b>${{title}}</b>
        <span class="badge">P: ${{r.precision.toFixed(1)}}% · R: ${{r.recall.toFixed(1)}}%</span>
      </div>
      <pre>${{r.prediction}}</pre>
    </div>`;
  }});
  document.getElementById('showcase-grid').innerHTML = html;

  if (meta) {{
    let refHtml = '<div class="card"><p><b>输入编号映射 (Prompt 序号 → 真实 url_id)：</b> <code>' + JSON.stringify(meta.mapping) + '</code></p><ul style="font-size:12.5px; columns:2;">';
    meta.tabs.forEach(t => {{
      refHtml += `<li><b>[url_id=${{t.url_id}}]</b> ${{t.title}} <span class="muted">(${{t.url.slice(0,60)}}...)</span></li>`;
    }});
    refHtml += '</ul></div>';
    document.getElementById('showcase-ref').innerHTML = refHtml;
  }}
}}

renderHeatmap();
renderNeuronBars();
renderMetricsTable();
renderShowcase();
</script>
</body>
</html>
"""

(REPO_WS / 'index.html').write_text(html, encoding='utf-8')
(ROOT / 'weight_space_dashboard.html').write_text(html, encoding='utf-8')
print('Generated interactive index.html! Size:', (REPO_WS / 'index.html').stat().st_size)
