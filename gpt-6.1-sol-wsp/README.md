# GPT-6.1-sol Weight-Space Programming (WSP) — Prompts, Scripts & Mechanistic Diagnosis

This directory contains the exact code, rendered prompts, 32-layer weight-space SVD & vocabulary-projection tables, and full mechanistic diagnosis produced by **`gpt-6.1-sol`** (`reasoning.effort = "xhigh"` / `"max"`) during our Weight-Space Programming investigation.

## Files in This Directory

1. **`build_dossier.py`** — Exact 32-layer $\times$ 248-module Gauge-Invariant Reduced QR-SVD and RMSNorm-Unembedding vocabulary projection script executed on A100-80GB (`Qwen/Qwen3.5-4B`, $d_{\text{model}}=2560, r=16, \alpha=32$). Generates `../weight_space_dossier.json`.
2. **`find_single_neurons.py`** — Projects the 9,216 columns of late-layer `mlp.down_proj` ($[2560, 9216]$) onto the unembedding split-vs-merge direction $v_{\text{newline}} - v_{\text{comma}}$ to locate the exact 64 newline-promoting (green) and 64 comma-promoting (red) valve neurons (**Idea A**).
3. **`query_sol_xhigh.py`** — The exact Python SSE streaming script that queried `gpt-6.1-sol` (`reasoning.effort="xhigh"`) and generated the 13.7 KB mechanistic diagnosis in 260.1s.
4. **`query_sol_max_stream.py` & `query_sol_max.py`** — Full 248-module table and raw JSON dossier query scripts (`reasoning.effort="max"`).
5. **`rendered_prompt_to_gpt_6_1_sol_xhigh.md`** — The exact rendered prompt (including the 128-row residual-writer & gate SVD table) sent to `gpt-6.1-sol (effort=xhigh)`.
6. **`rendered_prompt_to_gpt_6_1_sol_248modules.md`** — The full 248-module rendered prompt table across all 32 layers.
7. **`sol_weight_diagnosis_response.md`** — Complete verbatim output from `gpt-6.1-sol` diagnosing:
   - Middle-layer shared clustering component (`L08–L15`, $\cos \in [0.34, 0.37]$) vs. late-layer orthogonal readout divergence (`L18–L31`, $\cos \in [0.01, 0.14]$);
   - Why off-policy teacher-forced SFT writes `' &'`, `'学习与'`, `'Learning'`, and elevated `Ungrouped` thresholds into the residual stream;
   - Three exact zero-training surgical recipes (**Edit 1**: Late full-attention module graft; **Edit 2**: Unembedding-null, norm-capped late graft $(I - P_T)(W_{\text{SFT}} - W_{\text{OPSD}})$; **Edit 3**: Input-activation-protected SFT innovation graft).
