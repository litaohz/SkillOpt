"""Fix overlapping annotations in Fig 1 and generate clean v2.1-focused Fig 1 & Fig 2."""
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

mod_rows = [
    ('mlp.gate_proj', 'MLP GateProj (FFN Switch)'),
    ('mlp.up_proj', 'MLP UpProj (FFN Feature)'),
    ('mlp.down_proj', 'MLP DownProj (Residual Writer)'),
    ('attn_in', 'Attn InProj (Query / QKV)'),
    ('attn_mid', 'Attn Gate/KV (Key / Gate Z)'),
    ('attn_out', 'Attn OutProj (Residual Writer)'),
]

cos_mat = np.zeros((len(mod_rows), 32))
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
        cos_mat[r_idx, l] = mods[actual]['cos_opsd_sft']

vocab_keys = [
    ('amp_space', "Token ' &' (Compound Name)"),
    ('comma', "Token ',' (Merge Continuation)"),
    ('newline', "Token '\\n' (Group Split)"),
    ('ng_ungrouped', "Token 'Ungrouped'"),
]
proj_ratio_mat = np.zeros((len(vocab_keys), 32))
for l_obj in dossier['layers']:
    l = l_obj['layer']
    mods = l_obj['modules']
    w_mods = [mods['mlp.down_proj'], mods['self_attn.o_proj'] if 'self_attn.o_proj' in mods else mods['linear_attn.out_proj']]
    for v_idx, (vkey, _) in enumerate(vocab_keys):
        po = sum(x['vocab_proj_opsd'][vkey] for x in w_mods) / 2.0
        ps = sum(x['vocab_proj_sft'][vkey] for x in w_mods) / 2.0
        proj_ratio_mat[v_idx, l] = ps / max(po, 1e-6)

# =========================================================================
# FIGURE 1: Clean 3-Panel Weight-Space Anatomy Heatmap (No text overlap!)
# =========================================================================
fig = plt.figure(figsize=(14.5, 10.2), dpi=300)
gs = gridspec.GridSpec(3, 1, height_ratios=[1.35, 1.10, 1.05], hspace=0.52)

# Panel A: Cosine Similarity Heatmap
ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(cos_mat, aspect='auto', cmap='RdYlBu_r', vmin=0.04, vmax=0.40)
ax0.set_yticks(range(len(mod_rows)))
ax0.set_yticklabels([label for _, label in mod_rows], fontsize=9.2, fontweight='bold')
ax0.set_xticks(range(32))
ax0.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax0.set_title('(A) Layer-by-Layer Directional Alignment cos(ΔW_OPSD, ΔW_SFT) Across All 6 Module Types', fontsize=11.5, fontweight='bold', pad=24, loc='left', color='#0f172a')

for i in range(len(mod_rows)):
    for j in range(32):
        val = cos_mat[i, j]
        color = 'white' if val > 0.29 or val < 0.10 else '#0f172a'
        ax0.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.6, color=color, fontweight='bold' if (8 <= j <= 15 or j >= 18) else 'normal')

ax0.axvline(7.5, color='#16a34a', linestyle='--', linewidth=2.0)
ax0.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.2)

# Put zone badges cleanly in the 24pt pad gap above the heatmap box using axes fraction coordinates
ax0.text(0.39, 1.04, '◄── Shared Sub-Task Clustering Zone (L08–L15, cos up to 0.41) ──►',
         transform=ax0.transAxes, color='#15803d', fontsize=9.0, fontweight='bold', ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#dcfce7', edgecolor='#86efac', lw=0.8))
ax0.text(0.78, 1.04, '◄── L18 Cut: Late Readout Drift Zone (L18–L31, cos down to 0.01) ──►',
         transform=ax0.transAxes, color='#b91c1c', fontsize=9.0, fontweight='bold', ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='#fee2e2', edgecolor='#fca5a5', lw=0.8))

cb0 = plt.colorbar(im0, ax=ax0, pad=0.012, fraction=0.022)
cb0.set_label('Frobenius Cosine', fontsize=8.5)

# Panel B: Vocabulary Projection Ratio
ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(proj_ratio_mat, aspect='auto', cmap='YlOrRd', vmin=1.0, vmax=9.5)
ax1.set_yticks(range(len(vocab_keys)))
ax1.set_yticklabels([label for _, label in vocab_keys], fontsize=9.2, fontweight='bold')
ax1.set_xticks(range(32))
ax1.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax1.set_title("(B) Unembedding Readout Projection Ratio (||v^T ΔW_SFT|| / ||v^T ΔW_OPSD||): Late-Layer ' &' & 'Ungrouped' Surge in Off-Policy SFT", fontsize=11.5, fontweight='bold', pad=10, loc='left', color='#0f172a')
for i in range(len(vocab_keys)):
    for j in range(32):
        val = proj_ratio_mat[i, j]
        color = 'white' if val > 5.2 else '#0f172a'
        ax1.text(j, i, f'{val:.1f}x', ha='center', va='center', fontsize=6.6, color=color, fontweight='bold' if val >= 4.0 else 'normal')
ax1.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.2)
cb1 = plt.colorbar(im1, ax=ax1, pad=0.012, fraction=0.022)
cb1.set_label('SFT / OPSD Proj Ratio', fontsize=8.5)

# Panel C: Update Norm Curve across 32 Layers
ax2 = fig.add_subplot(gs[2])
l_opsd_norms = [np.sqrt(sum(m['norm_opsd']**2 for m in l_obj['modules'].values())) for l_obj in dossier['layers']]
l_sft_norms = [np.sqrt(sum(m['norm_sft']**2 for m in l_obj['modules'].values())) for l_obj in dossier['layers']]
x = np.arange(32)
ax2.plot(x, l_sft_norms, marker='o', markersize=4.5, linewidth=2.2, color='#dc2626', label='||ΔW_SFT||_F (Off-Policy Astra-SFT: 1.5x–1.7x stronger middle-layer pull)')
ax2.plot(x, l_opsd_norms, marker='s', markersize=4.5, linewidth=2.2, color='#2563eb', label='||ΔW_OPSD||_F (On-Policy Self-Distillation: clean late-layer readout)')
ax2.fill_between(x[8:18], l_opsd_norms[8:18], l_sft_norms[8:18], color='#bbf7d0', alpha=0.45, label='L08–L17: High-gain sub-task feature splitting')
ax2.fill_between(x[18:], l_opsd_norms[18:], l_sft_norms[18:], color='#fecaca', alpha=0.45, label='L18–L31: Lexical & threshold contamination')
ax2.axvline(7.5, color='#16a34a', linestyle='--', linewidth=1.8)
ax2.axvline(17.5, color='#dc2626', linestyle='--', linewidth=2.2)
ax2.set_xticks(range(32))
ax2.set_xticklabels([f'L{i:02d}*' if i in (3,7,11,15,19,23,27,31) else f'L{i:02d}' for i in range(32)], fontsize=8)
ax2.set_ylabel('Layer Frobenius Norm', fontsize=9.5, fontweight='bold')
ax2.set_title('(C) Layerwise Effective LoRA Update Norm ||ΔW||_F (* denotes Full-Attention layers, others Gated-DeltaNet Linear-Attention)', fontsize=11.5, fontweight='bold', pad=8, loc='left', color='#0f172a')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8.5, ncol=2, frameon=True, facecolor='white')

fig.savefig(FIG_DIR / 'fig1_weight_space_heatmap.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig1_weight_space_heatmap.pdf', bbox_inches='tight')
plt.close(fig)
print('Saved clean fig1_weight_space_heatmap.png/pdf!')
