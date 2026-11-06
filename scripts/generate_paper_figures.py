"""Generate publication-quality weight-space heatmaps, neuron spectrum, and Pareto plots."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

ROOT = Path('Q:/wiki/misc/tabgroup-opd-20261026')
REPO_WS = Path('Q:/wiki/misc/tabgroup-research-repo/results/weight-space-surgery-20261103')
PAPER_DIR = REPO_WS / 'paper'
FIG_DIR = PAPER_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

dossier = json.loads((ROOT / 'weight_space_dossier.json').read_text(encoding='utf-8'))
bundle = json.loads((ROOT / 'complete_viz_bundle.json').read_text(encoding='utf-8'))

layers = list(range(32))
# Module categories across 32 layers:
# Every layer has: mlp.gate_proj, mlp.up_proj, mlp.down_proj
# Plus Attention: Q/QKV, Z/K, V, Out/O
mod_rows = [
    ('mlp.gate_proj', 'MLP GateProj'),
    ('mlp.up_proj', 'MLP UpProj'),
    ('mlp.down_proj', 'MLP DownProj (Writer)'),
    ('attn_in', 'Attn InProj (Q / QKV)'),
    ('attn_mid', 'Attn Gate/KV (Z / K)'),
    ('attn_out', 'Attn OutProj (Writer)'),
]

cos_mat = np.zeros((len(mod_rows), 32))
ratio_mat = np.zeros((len(mod_rows), 32))

for l_obj in dossier['layers']:
    l = l_obj['layer']
    mods = l_obj['modules']
    for r_idx, (mkey, _) in enumerate(mod_rows):
        if mkey == 'attn_in':
            actual = 'self_attn.q_proj' if 'self_attn.q_proj' in mods else 'linear_attn.in_proj_qkv'
        elif mkey == 'attn_mid':
            actual = 'self_attn.k_proj' if 'self_attn.k_proj' in mods else 'linear_attn.in_proj_z'
        elif mkey == 'attn_out':
            actual = 'self_attn.o_proj' if 'self_attn.o_proj' in mods else 'linear_attn.out_proj'
        else:
            actual = mkey
        m = mods[actual]
        cos_mat[r_idx, l] = m['cos_opsd_sft']
        ratio_mat[r_idx, l] = m['norm_sft'] / max(m['norm_opsd'], 1e-6)

# Vocabulary projection matrix across 32 layers for residual writers (average of mlp.down_proj & attn_out)
vocab_keys = [
    ('amp_space', "Token ' &' (Compound Name)"),
    ('comma', "Token ',' (Merge Continuation)"),
    ('newline', "Token '\\n' (Group Split)"),
    ('ng_ungrouped', "Token 'Ungrouped'"),
]
proj_sft_mat = np.zeros((len(vocab_keys), 32))
proj_opsd_mat = np.zeros((len(vocab_keys), 32))
proj_ratio_mat = np.zeros((len(vocab_keys), 32))

for l_obj in dossier['layers']:
    l = l_obj['layer']
    mods = l_obj['modules']
    w_mods = [mods['mlp.down_proj'], mods['self_attn.o_proj'] if 'self_attn.o_proj' in mods else mods['linear_attn.out_proj']]
    for v_idx, (vkey, _) in enumerate(vocab_keys):
        po = sum(x['vocab_proj_opsd'][vkey] for x in w_mods) / 2.0
        ps = sum(x['vocab_proj_sft'][vkey] for x in w_mods) / 2.0
        proj_opsd_mat[v_idx, l] = po
        proj_sft_mat[v_idx, l] = ps
        proj_ratio_mat[v_idx, l] = ps / max(po, 1e-6)

# =========================================================================
# FIGURE 1: 3-Panel Weight-Space Anatomy Heatmap (32 Layers)
# =========================================================================
fig = plt.figure(figsize=(14, 9.2), dpi=300)
gs = gridspec.GridSpec(3, 1, height_ratios=[1.25, 1.15, 1.05], hspace=0.42)

# Panel A: Cosine Similarity Heatmap
ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(cos_mat, aspect='auto', cmap='RdYlBu_r', vmin=0.04, vmax=0.38)
ax0.set_yticks(range(len(mod_rows)))
ax0.set_yticklabels([label for _, label in mod_rows], fontsize=9.5, fontweight='bold')
ax0.set_xticks(range(32))
ax0.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax0.set_title('(A) Layer-by-Layer Cosine Similarity cos(ΔW_OPSD, ΔW_SFT): Shared Clustering Circuit (L08–L15) vs. Late Orthogonal Drift (L18–L31)', fontsize=11, fontweight='bold', pad=8, loc='left', color='#0f172a')
for i in range(len(mod_rows)):
    for j in range(32):
        val = cos_mat[i, j]
        color = 'white' if val > 0.28 or val < 0.10 else '#0f172a'
        ax0.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color, fontweight='bold' if (8 <= j <= 15 or j >= 18) else 'normal')
ax0.axvline(7.5, color='#16a34a', linestyle='--', linewidth=1.8)
ax0.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.2)
ax0.text(11.5, -0.72, 'Shared Sub-Task Clustering Zone (L08–L15, cos up to 0.37)', color='#15803d', fontsize=9, fontweight='bold', ha='center')
ax0.text(24.5, -0.72, 'Surgical Cut Boundary (L18) → Readout Drift Zone (cos down to 0.05)', color='#b91c1c', fontsize=9, fontweight='bold', ha='center')
cb0 = plt.colorbar(im0, ax=ax0, pad=0.015, fraction=0.025)
cb0.set_label('Frobenius Cosine', fontsize=8.5)

# Panel B: Vocabulary Projection Ratio (||v^T ΔW_SFT|| / ||v^T ΔW_OPSD||)
ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(proj_ratio_mat, aspect='auto', cmap='YlOrRd', vmin=1.0, vmax=9.5)
ax1.set_yticks(range(len(vocab_keys)))
ax1.set_yticklabels([label for _, label in vocab_keys], fontsize=9.5, fontweight='bold')
ax1.set_xticks(range(32))
ax1.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax1.set_title("(B) Unembedding Vocabulary Projection Ratio (||v^T ΔW_SFT|| / ||v^T ΔW_OPSD||): Late-Layer ' &' & 'Ungrouped' Overfitting in SFT", fontsize=11, fontweight='bold', pad=8, loc='left', color='#0f172a')
for i in range(len(vocab_keys)):
    for j in range(32):
        val = proj_ratio_mat[i, j]
        color = 'white' if val > 5.2 else '#0f172a'
        ax1.text(j, i, f'{val:.1f}x', ha='center', va='center', fontsize=6.5, color=color, fontweight='bold' if val >= 4.0 else 'normal')
ax1.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.2)
cb1 = plt.colorbar(im1, ax=ax1, pad=0.015, fraction=0.025)
cb1.set_label('SFT / OPSD Proj Ratio', fontsize=8.5)

# Panel C: Update Norm Curve across 32 Layers + Graft Cutoff Sweep
ax2 = fig.add_subplot(gs[2])
l_opsd_norms = [np.sqrt(sum(m['norm_opsd']**2 for m in l_obj['modules'].values())) for l_obj in dossier['layers']]
l_sft_norms = [np.sqrt(sum(m['norm_sft']**2 for m in l_obj['modules'].values())) for l_obj in dossier['layers']]
x = np.arange(32)
ax2.plot(x, l_sft_norms, marker='o', markersize=4.5, linewidth=2.2, color='#dc2626', label='||ΔW_SFT||_F (Off-Policy Astra-SFT: 1.4x–1.7x overshoot)')
ax2.plot(x, l_opsd_norms, marker='s', markersize=4.5, linewidth=2.2, color='#2563eb', label='||ΔW_OPSD||_F (On-Policy Self-Distillation: compact & stable)')
ax2.fill_between(x[8:18], l_opsd_norms[8:18], l_sft_norms[8:18], color='#bbf7d0', alpha=0.45, label='L08–L17: High-gain feature splitting amplification')
ax2.fill_between(x[18:], l_opsd_norms[18:], l_sft_norms[18:], color='#fecaca', alpha=0.45, label='L18–L31: Lexical & Ungrouped threshold contamination')
ax2.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.0)
ax2.set_xticks(range(32))
ax2.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax2.set_ylabel('Layer Frobenius Norm', fontsize=9.5, fontweight='bold')
ax2.set_title('(C) Layerwise Effective LoRA Update Magnitude ||ΔW||_F Across All 32 Layers (* denotes Full-Attention layers)', fontsize=11, fontweight='bold', pad=6, loc='left', color='#0f172a')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8.5, ncol=2, frameon=True, facecolor='white')

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig1_weight_space_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig1_weight_space_heatmap.pdf', bbox_inches='tight')
plt.close(fig)

# =========================================================================
# FIGURE 2: Single-Neuron Spectrum (L26–L31) & Pareto Frontier (v1 vs v2.1)
# =========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

# Left: L31 & L30 Top Newline-Promoting vs Comma-Promoting Neurons in mlp.down_proj
ns = bundle['_extra']['neuron_spectrum']['L31']
nl_cols = ns['top_newline_cols'][:8][::-1]
cm_cols = ns['top_comma_cols'][:8]
items = cm_cols + nl_cols
labels = [f"Col {x['col']}" for x in items]
vals = [x['score'] for x in items]
colors = ['#ef4444' if v < 0 else '#10b981' for v in vals]

y_pos = np.arange(len(items))
axL.barh(y_pos, vals, color=colors, height=0.68, edgecolor='#1e293b', linewidth=0.6)
axL.axvline(0, color='#0f172a', linewidth=1.2)
axL.set_yticks(y_pos)
axL.set_yticklabels(labels, fontsize=8.5, fontweight='bold')
axL.set_xlabel("Projection onto Unembedding Direction (v_newline - v_comma)", fontsize=9.5, fontweight='bold')
axL.set_title("(A) Idea A: Single-Neuron Readout Polarity in L31 mlp.down_proj\n(Green = Newline/Split Valve; Red = Comma/Giant-Bucket Valve)", fontsize=10.5, fontweight='bold', color='#0f172a')
for idx, v in enumerate(vals):
    offset = 0.008 if v >= 0 else -0.008
    ha = 'left' if v >= 0 else 'right'
    axL.text(v + offset, idx, f'{v:+.3f}', va='center', ha=ha, fontsize=8, fontweight='bold', color='#0f172a')
axL.set_xlim(-0.28, 0.32)
axL.grid(True, axis='x', linestyle=':', alpha=0.5)

# Right: Joint v1 Recall vs v2.1 Recall Pareto Frontier (Run-1 61 Cases)
points = [
    ('Base 4B-SFT (sft2000)', 68.26, 45.71, '#64748b', 's', 85, (-10, -14)),
    ('Astra-SFT (sft_astra)', 64.19, 60.25, '#ef4444', '^', 95, (6, -10)),
    ('OPSD-Astra (opsd_astra)', 72.52, 59.36, '#2563eb', 'o', 100, (-85, -14)),
    ('Idea A: 64-Neuron Edit\n(on Base SFT, 0 train!)', 69.70, 57.93, '#10b981', 'D', 110, (-75, -22)),
    ('Idea C: Full-Attn q_norm*1.35\n(4 scalars on Base SFT)', 70.37, 51.96, '#14b8a6', 'p', 105, (-70, 8)),
    ('Idea B-1: L18-31 σ1=0\n(1 scalar/mod on Astra-SFT)', 65.74, 58.69, '#f97316', 'v', 95, (6, -6)),
    ('Idea B-2: L18-31 σ1=σ2=0\n(2 scalars/mod on Astra-SFT)', 67.63, 60.25, '#eab308', 'v', 105, (-55, 10)),
    ('Surgery 1: Graft-18\n(L0-17 SFT + L18-31 OPSD)', 69.19, 64.52, '#7c3aed', '*', 180, (-65, 8)),
    ('Surgery 2: SVD-Blend 50/50\n(Rank-16 Spectral Consensus)', 69.37, 64.42, '#db2777', '*', 180, (8, -6)),
    ('Rev-Graft-18 Control\n(L0-17 OPSD + L18-31 SFT)', 67.48, 59.81, '#94a3b8', 'X', 85, (-60, -16)),
]

for name, v1_r, v2_r, col, mk, sz, (dx, dy) in points:
    axR.scatter(v1_r, v2_r, color=col, marker=mk, s=sz, edgecolors='#0f172a', linewidths=0.9, zorder=5)
    axR.annotate(name, (v1_r, v2_r), xytext=(dx, dy), textcoords='offset points', fontsize=7.8, fontweight='bold', color=col,
                 arrowprops=dict(arrowstyle='->', color=col, lw=0.7, alpha=0.7))

# Draw arrows for zero-training jumps
axR.annotate('', xy=(69.70, 57.93), xytext=(68.26, 45.71), arrowprops=dict(arrowstyle='->', color='#10b981', lw=1.8, ls='--'))
axR.annotate('', xy=(67.63, 60.25), xytext=(64.19, 60.25), arrowprops=dict(arrowstyle='->', color='#eab308', lw=1.8, ls='--'))
axR.annotate('', xy=(69.19, 64.52), xytext=(64.19, 60.25), arrowprops=dict(arrowstyle='->', color='#7c3aed', lw=1.8, ls='--'))

axR.set_xlabel('v1 (Golden45) Exact Recall (%)', fontsize=9.5, fontweight='bold')
axR.set_ylabel('v2.1 (Hard Multi-Task) Exact Recall (%)', fontsize=9.5, fontweight='bold')
axR.set_title('(B) Zero-Training Weight Surgery Moves the Pareto Frontier\n(Run-1 Full 61-Case Benchmark: 45 v1 + 16 v2.1)', fontsize=10.5, fontweight='bold', color='#0f172a')
axR.set_xlim(63.2, 73.6)
axR.set_ylim(44.0, 66.8)
axR.grid(True, linestyle=':', alpha=0.5)

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_neurons_and_pareto.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2_neurons_and_pareto.pdf', bbox_inches='tight')
plt.close(fig)

print('Generated fig1_weight_space_heatmap.png/pdf and fig2_neurons_and_pareto.png/pdf successfully!')
