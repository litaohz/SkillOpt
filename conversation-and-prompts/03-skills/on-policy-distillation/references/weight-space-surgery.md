# Weight-Space Diagnostics & Zero-Training LoRA Surgery

When comparing On-Policy Self-Distillation (OPSD) against standard Cross-Entropy SFT trained on the same frontier reference labels $y^\star$, or when combining their complementary strengths without additional GPU training, use low-rank weight-space SVD and layerwise circuit grafting.

---

## 1. Fast Exact QR-SVD for Low-Rank LoRA Differences

Never materialize dense $(d_{\text{out}} \times d_{\text{in}})$ matrices to run full SVD across dozens of layers — full SVD on $2560 \times 9728$ matrices times out or bottlenecks CPU/GPU. Because each LoRA update $W = \frac{\alpha}{r} B A$ has rank $r$, the difference between two adapters $\Delta W = \frac{\alpha}{r}(B_1 A_1 - B_0 A_0)$ has rank at most $2r$ (e.g., $2r = 32$):

```python
import torch

def low_rank_svd(A1: torch.Tensor, B1: torch.Tensor, A0: torch.Tensor, B0: torch.Tensor, scale: float = 2.0):
    """Exact SVD of scale * (B1 @ A1 - B0 @ A0) in O((d_out + d_in) * r^2) via reduced QR."""
    B_cat = torch.cat([B1, -B0], dim=1) * scale  # (d_out, 2r)
    A_cat = torch.cat([A1, A0], dim=0)           # (2r, d_in)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    U_m, S, Vh_m = torch.linalg.svd(Rb @ Ra.T)   # (2r, 2r) tiny SVD
    U = Qb @ U_m                                 # (d_out, 2r) orthonormal columns
    Vh = Vh_m @ Qa.T                             # (2r, d_in) orthonormal rows
    return S, U, Vh

def frob_inner(A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor) -> float:
    """Exact Frobenius inner product <B1 A1, B2 A2>_F without materializing dense matrices."""
    return ((B1.T @ B2) * (A1 @ A2.T)).sum().item()
```

- **Compare effective matrices $\Delta W = W - W_{\text{init}}$, never raw $(A, B)$ factors**: LoRA factorizations are gauge-invariant under invertible $r \times r$ transformations $(B R^{-1})(R A)$, so cosine similarity on raw `lora_A` or `lora_B` tensors is meaningless. Always subtract the shared starting checkpoint $W_{\text{init}}$ and compute inner products/SVD on effective matrices.
- **Physical Meaning of the 6 Core LoRA Module Rows (for Heatmaps & Diagnosis)**:
  1. `mlp.gate_proj` ($d_{\text{model}} \to d_{\text{ff}}$): FFN nonlinear SiLU switch $\sigma(W_{\text{gate}} h)$ deciding which intermediate neurons fire.
  2. `mlp.up_proj` ($d_{\text{model}} \to d_{\text{ff}}$): FFN semantic feature extractor multiplied elementwise with the gate.
  3. `mlp.down_proj` ($d_{\text{ff}} \to d_{\text{model}}$, **Residual Writer**): Projects active FFN neurons into the residual stream; each column $j \in \{1,\dots,d_{\text{ff}}\}$ is an atomic vocabulary writer pushing specific token logits (e.g., `\n` vs `,`).
  4. `attn_in` (`self_attn.q_proj` or `linear_attn.in_proj_qkv`): Cross-entity search query projector governing what features a token searches for across other items.
  5. `attn_mid` (`self_attn.k_proj` or `linear_attn.in_proj_z`): Cross-entity matching key / recurrent output gate controlling which items bind into a cluster.
  6. `attn_out` (`self_attn.o_proj` or `linear_attn.out_proj`, **Residual Writer**): Writes aggregated cross-entity relational context back into the $d_{\text{model}}$ residual stream.

---

## 2. Unembedding Vocabulary Projection (Detecting Late-Layer Lexical Drift)

Standard teacher-forced SFT on frontier reference labels $y^\star$ forces the student to imitate every surface lexical convention of the teacher (e.g., compound `' & '` group names or stricter `Ungrouped` rejection thresholds), whereas OPSD evaluates loss on the student's own rollout prefixes $\hat{y}$ and only corrects structural split/merge boundaries.

To diagnose where lexical drift or threshold over-fitting enters the network:

1. Load the base model's output unembedding matrix $W_U$ (or `embed_tokens.weight` if tied) and final RMSNorm weight $\gamma_{\text{final}}$, forming the effective readout matrix:
   $$U_{\text{eff}} = W_U \odot \gamma_{\text{final}}$$
2. For each residual-stream writer module (`mlp.down_proj`, `self_attn.o_proj`, `linear_attn.out_proj`), project the top-1 left singular vector $u_1$ of $\Delta W$ onto the vocabulary:
   $$z_{u_1} = U_{\text{eff}} u_1$$
   and inspect both `topk(z_u1)` and `topk(-z_u1)` (singular vectors are sign-indeterminate up to simultaneous flip of $u_1, v_1$).
3. Measure the exact projection norm onto specific suspected drift tokens $v_t = U_{\text{eff}}[t] / \|U_{\text{eff}}[t]\|_2$:
   $$\|v_t^\top \Delta W\|_2 = \sqrt{(v_t^\top B_\Delta)(A_\Delta A_\Delta^\top)(B_\Delta^\top v_t)}$$

**Typical Layerwise Signature (32-Layer Hybrid/Transformer SLMs):**
- **Early-to-Middle Layers (e.g., L08–L17)**: $\Delta W_{\text{OPSD}}$ and $\Delta W_{\text{SFT}}$ exhibit their highest alignment ($\cos \approx 0.30\text{–}0.37$). These layers encode task-level semantic clustering and fine-grained sub-task boundary detection. Because SFT applies hard cross-entropy across all tokens, its update norm $\|\Delta W_{\text{SFT}}\|_F$ is typically $1.4\times\text{–}1.7\times$ larger than OPSD's, often giving SFT stronger raw feature separation in middle layers.
- **Late Readout Layers (e.g., L18–L31)**: $\Delta W_{\text{OPSD}}$ and $\Delta W_{\text{SFT}}$ rapidly decorrelate ($\cos \to 0.05\text{–}0.14$). SFT's late writer singular vectors heavily concentrate on teacher-specific surface tokens (e.g., $8\times\text{–}11\times$ higher projection onto `' &'` or `Ungrouped` subwords), causing recall collapse on standard benchmarks while OPSD preserves clean readout calibration.

---

## 3. Zero-Training Weight-Space Surgical Recipes

When SFT excels on hard fine-grained splitting (via stronger middle-layer feature updates) while OPSD excels on recall preservation and readout calibration (via clean late layers), combine them in weight space with zero additional training:

### Recipe A · Layerwise Circuit Grafting (`Graft-L*`)
Locate the crossover layer $L^\star$ where $\cos(\Delta W_{\text{OPSD}}, \Delta W_{\text{SFT}})$ drops sharply and SFT's lexical drift projection spikes (e.g., $L^\star = 18$ in a 32-layer model). Construct a hybrid adapter by taking:
- **Layers $0 \dots L^\star - 1$ from `SFT-on-Frontier-Labels`** (stronger middle-layer fine-grained feature splitting).
- **Layers $L^\star \dots L_{\max}$ from `OPSD`** (clean readout policy without surface lexical drift or over-rejection).
- **Always run the reverse graft (`L0..L*-1 OPSD + L*..Lmax SFT`) as a causal control** to verify that the recall degradation indeed transfers with the late SFT layers.

### Recipe B · Exact Rank-$r$ QR-SVD Subspace Blending (`SVD-Blend`)
Never average `lora_A` and `lora_B` tensors directly ($\frac{1}{2}(B_1 + B_2)\frac{1}{2}(A_1 + A_2)$ introduces unwanted cross-terms $B_1 A_2 + B_2 A_1$). Instead, blend the effective matrices $W_{\text{blend}} = w_1 B_1 A_1 + w_2 B_2 A_2$ and project back to rank $r$ via reduced QR-SVD:

```python
def svd_blend_lora(A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor, w1: float = 0.5, w2: float = 0.5, r: int = 16):
    """Exact rank-r SVD refactorization of (w1 * B1 @ A1 + w2 * B2 @ A2)."""
    B_cat = torch.cat([B1 * w1, B2 * w2], dim=1)  # (d_out, 2r)
    A_cat = torch.cat([A1, A2], dim=0)            # (2r, d_in)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    sqrt_S = torch.sqrt(S[:r])
    B_new = ((Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)).contiguous()
    A_new = (sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)).contiguous()
    return A_new, B_new
```
Why it works: aligned middle-layer directions reinforce coherently ($w_1 u + w_2 u = u$), while orthogonal late-layer lexical drift directions are damped by $w_2 = 0.5$ (reducing drift energy by $w_2^2 = 0.25$) and trailing singular noise modes are truncated by $\operatorname{SVD}_r$.

---

## 4. Single-Model Value & Neuron Surgery (No Second Adapter Needed)

When you only have a single model or a single SFT adapter (without an OPSD adapter to graft from), apply direct mechanistic edits in weight space:

### Recipe C · Contrastive Unembedding Neuron Valve Scaling (Fixing Over-Merging on Base SFT)
In structured grouping/listing tasks, over-merging is governed at readout time by a race between emitting a group-termination token (e.g., `\n`, token $t_{\text{split}}$) versus a continuation token (e.g., `,`, token $t_{\text{merge}}$). Each column $j \in \{0,\dots,d_{\text{ff}}-1\}$ of `mlp.down_proj.weight` ($\in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$) acts as an atomic residual-stream writer neuron:

```python
# 1. Compute normalized contrastive readout direction in residual space
v_split = U_eff[t_split] / torch.norm(U_eff[t_split])
v_merge = U_eff[t_merge] / torch.norm(U_eff[t_merge])
v_diff = (v_split - v_merge) / torch.norm(v_split - v_merge)

# 2. Score all d_ff columns in late MLP down_proj layers (e.g., L26..L31)
for l in range(26, 32):
    W = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.float()
    scores = v_diff @ W  # (d_ff,)
    top_split_cols = torch.topk(scores, 64).indices
    top_merge_cols = torch.topk(-scores, 64).indices
    W[:, top_split_cols] *= 2.0  # Boost split-promoting valve neurons
    W[:, top_merge_cols] *= 0.0  # Silence inertial merge-continuation neurons
```
- **Scope & Boundary**: Editing $64 / 9216$ columns ($0.69\%$) in the final 6 layers unlocks latent splitting capability and breaks catch-all giant buckets at zero training cost, though it cannot create new middle-layer semantic representations that the base model never learned.

### Recipe D · Single-Scalar Singular Value Excision ($\sigma_1, \sigma_2 \to 0$ on Off-Policy SFT)
When an off-policy SFT adapter $\Delta W_{\text{SFT}} = W_{\text{SFT}} - W_0$ overfits surface teacher conventions or overly strict rejection thresholds in late layers ($l \ge L^\star$), excise only the top $k \in \{1, 2\}$ singular scalars of the displacement while keeping modes $k+1 \dots r$ intact:

```python
# Decompose delta W_SFT = B_s A_s - B_0 A_0 via reduced QR-SVD
B_d = torch.cat([B_s, -B_0], dim=1)
A_d = torch.cat([A_s, A_0], dim=0)
Qb, Rb = torch.linalg.qr(B_d, mode='reduced')
Qa, Ra = torch.linalg.qr(A_d.T, mode='reduced')
U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)

# Zero ONLY the top-k scalar singular values of the late-layer update
S_d[:k] = 0.0

# Add filtered update back to W_0 and re-factorize to exact rank r=16
B_cat = torch.cat([B_0, (Qb @ U_d) * S_d.unsqueeze(0)], dim=1)
A_cat = torch.cat([A_0, Vh_d @ Qa.T], dim=0)
Qb2, Rb2 = torch.linalg.qr(B_cat, mode='reduced')
Qa2, Ra2 = torch.linalg.qr(A_cat.T, mode='reduced')
U2, S2, Vh2 = torch.linalg.svd(Rb2 @ Ra2.T)
sqrt_S2 = torch.sqrt(S2[:r])
B_new = ((Qb2 @ U2[:, :r]) * sqrt_S2.unsqueeze(0)).contiguous()
A_new = (sqrt_S2.unsqueeze(1) * (Vh2[:r, :] @ Qa2.T)).contiguous()
```

### Recipe E · Full-Attention QK-Norm Temperature Scaling
In hybrid Linear/Full-Attention architectures (e.g., Qwen3.5 with Full-Attention at layers $3, 7, 11, 15, 19, 23, 27, 31$), scaling `self_attn.q_norm.weight` by a single scalar $\gamma \in [1.25, 1.50]$ across middle relational layers ($L \in \{7, 11, 15, 19\}$) directly sharpens cross-entity attention logits without modifying any projection matrix.
