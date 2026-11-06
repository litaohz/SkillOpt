```text
══════════════════════════════════════════════
MEMORY & USER PROFILE
══════════════════════════════════════════════
[Active Research Context]
- TabGroup On-Policy Distillation (OPD / OPSD):
  - Repository: Q:/wiki/misc/tabgroup-research-repo (GitHub: gim-home/tab-group-eval)
  - 8-model SFT baseline completed (tabgroup-eight-sft-complete-20261026).
  - Qwen3.5-4B trained via OPSD (Privileged-Context Self-Distillation on 1,200 GPT-6-astra reference solutions) significantly outperforms standard white-box 9B -> 4B GKD.
- TabGroup Weight-Space Anatomy & Zero-Training Surgery (2026-11-03):
  - 4B LoRA middle layers (L08-L15) exhibit high clustering directional alignment (cos = 0.41), whereas late readout layers (L18-L31) diverge (cos = 0.05-0.14) as off-policy SFT writes surface compound tokens (' &') and Ungrouped.
  - Single-model zero-training valve neuron scaling (Idea A, 64 columns in L26-L31) yields +12.22pp Recall on v2.1.
  - Dual-adapter surgery (Graft-18 and SVD-Blend) achieves 64.52% Recall and 72.41% Precision (66.85% / 68.24% F1) on v2.1, surpassing both OPSD and SFT in zero training steps.

[User Research Methodology & Standards]
- Research-Driven: Prioritize fundamental mechanism validation and real case inspection over engineering convenience.
- Real Case Walkthroughs: Always inspect concrete case-level outputs alongside aggregate metrics (since aggregate denominators can mask over-merging or over-splitting).
- Co-Pilot Iterative Rhythm: Advance one verifiable artifact per phase, report findings and anomalies transparently, and back every numerical claim with reproducible scripts.
```
