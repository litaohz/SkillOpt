"""Generate updated Figure 2 with explicit Green (*2.0x) & Red (*0.0x) intervention callouts in (A) and F1 Progression Ladder in (B)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

ROOT = Path('Q:/wiki/misc/tabgroup-opd-20261026')
REPO_WS = Path('Q:/wiki/misc/tabgroup-research-repo/results/weight-space-surgery-20261103')
PAPER_DIR = REPO_WS / 'paper'
FIG_DIR = PAPER_DIR / 'figures'

bundle = json.loads((ROOT / 'complete_viz_bundle.json').read_text(encoding='utf-8'))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=300)

# Panel A: L31 Top Newline-Promoting (Green -> Boosted 2.0x) vs Comma-Promoting (Red -> Zeroed 0.0x)
ns = bundle['_extra']['neuron_spectrum']['L31']
nl_cols = ns['top_newline_cols'][:7][::-1]
cm_cols = ns['top_comma_cols'][:7]
items = cm_cols + nl_cols
labels = [f"Col {x['col']}" for x in items]
vals = [x['score'] for x in items]
colors = ['#ef4444' if v < 0 else '#10b981' for v in vals]

y_pos = np.arange(len(items))
axL.barh(y_pos, vals, color=colors, height=0.68, edgecolor='#1e293b', linewidth=0.6)
axL.axvline(0, color='#0f172a', linewidth=1.3)
axL.axhline(6.5, color='#64748b', linestyle='--', linewidth=1.0)
axL.set_yticks(y_pos)
axL.set_yticklabels(labels, fontsize=8.8, fontweight='bold')
axL.set_xlabel("Projection onto Unembedding Split Direction (v_newline - v_comma)", fontsize=9.5, fontweight='bold')
axL.set_title("(A) Idea A Surgery: Simultaneous Dual-Valve Edit in L26–L31 mlp.down_proj\n(Green: Boosted ×2.0 to Trigger '\\n' Split | Red: Zeroed ×0.0 to Stop ',' Over-Merging)", fontsize=10.0, fontweight='bold', color='#0f172a', pad=10)

for idx, v in enumerate(vals):
    offset = 0.008 if v >= 0 else -0.008
    ha = 'left' if v >= 0 else 'right'
    axL.text(v + offset, idx, f'{v:+.3f}', va='center', ha=ha, fontsize=8.0, fontweight='bold', color='#0f172a')

# Add clear surgical action callout boxes inside Panel A
axL.text(0.145, 9.8, "▲ SPLIT VALVE (Green Top-64)\nAction: Multiply Columns × 2.0\nEffect: Emits '\\n' to start new group",
         fontsize=8.2, fontweight='bold', color='#065f46',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#d1fae5', edgecolor='#10b981', lw=1.1))
axL.text(-0.275, 2.2, "▼ MERGE VALVE (Red Bottom-64)\nAction: Zero Out Columns × 0.0\nEffect: Stops ',' giant-bucket loop",
         fontsize=8.2, fontweight='bold', color='#991b1b',
         bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee2e2', edgecolor='#ef4444', lw=1.1))

axL.set_xlim(-0.29, 0.34)
axL.grid(True, axis='x', linestyle=':', alpha=0.5)

# Panel B: Horizontal Bar Chart of v2.1 F1 Score (and Overall Macro F1 annotations) — Monotonic Ladder!
models_f1 = [
    ('0b. White-Box OPD (9B→4B, 150 steps)', 50.47, 62.17, '#94a3b8'),
    ('0a. Base 4B-SFT (sft2000, 2000 steps)', 51.25, 62.30, '#64748b'),
    ('★ Idea C: 4-Scalar q_norm×1.35 (0 Train)', 55.37, 65.10, '#14b8a6'),
    ('★ Idea A: 64-Neuron Valve Edit (0 Train)', 59.56, 66.98, '#10b981'),
    ('2. Astra-SFT (Off-Policy SFT, 150 steps)', 62.57, 66.81, '#f87171'),
    ('★ Idea B-2: Astra-SFT L18–31 σ1=σ2=0 (0 Train)', 63.05, 68.31, '#f59e0b'),
    ('1. OPSD-Astra (On-Policy Self-Distill, 150 steps)', 63.86, 71.16, '#3b82f6'),
    ('★ Surgery A: Graft-18 (L0–17 SFT + L18–31 OPSD)', 67.07, 71.24, '#8b5cf6'),
    ('★ Surgery B: SVD-Blend 50/50 (Spectral Consensus)', 68.18, 71.80, '#ec4899'),
]

y_m = np.arange(len(models_f1))
v2_f1s = [x[1] for x in models_f1]
macro_f1s = [x[2] for x in models_f1]
m_labels = [x[0] for x in models_f1]
m_cols = [x[3] for x in models_f1]

bars = axR.barh(y_m, v2_f1s, color=m_cols, height=0.64, edgecolor='#0f172a', linewidth=0.7)
axR.axvline(51.25, color='#64748b', linestyle=':', linewidth=1.2)
axR.axvline(63.86, color='#2563eb', linestyle='--', linewidth=1.5, label='OPSD-Astra Ceiling (63.86% v2.1 F1)')

axR.set_yticks(y_m)
axR.set_yticklabels(m_labels, fontsize=8.6, fontweight='bold')
axR.set_xlabel('v2.1 Exact F1 Score (%) [2·P·R / (P+R)]', fontsize=9.5, fontweight='bold')
axR.set_title('(B) Monotonic F1 Escalation on Hard Multi-Task Benchmark (v2.1)\n(Idea A +8.31pp F1 with 0 Train; SVD-Blend Beats OPSD by +4.32pp v2.1 F1 & +0.64pp Macro F1)', fontsize=10.0, fontweight='bold', color='#0f172a', pad=10)

for idx, (v2_f1, mac_f1) in enumerate(zip(v2_f1s, macro_f1s)):
    axR.text(v2_f1 + 0.45, idx, f'{v2_f1:.2f}%  (All-61 Macro F1: {mac_f1:.2f}%)',
             va='center', ha='left', fontsize=8.0, fontweight='bold',
             color='#be185d' if idx >= 7 else ('#047857' if idx == 3 else '#0f172a'))

axR.set_xlim(42.0, 83.5)
axR.grid(True, axis='x', linestyle=':', alpha=0.5)
axR.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white')

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_neurons_and_pareto.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2_neurons_and_pareto.pdf', bbox_inches='tight')
plt.close(fig)
print('Generated updated fig2_neurons_and_pareto.png/pdf with F1 scores!')
