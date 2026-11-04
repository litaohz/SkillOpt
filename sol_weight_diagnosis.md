## First: the table supports a hypothesis, not an exact causal mechanism

The strongest interpretation is **a partially shared middle-layer update, followed by different late-layer lexical and grouping policies**. But weight cosines and vocabulary projections cannot establish that the late SFT directions *cause* either v1 recall loss or v2.1 splitting.

I will assume:

- \(W_0\) is the **common starting checkpoint**, including its starting adapter.
- \(d_O=W_{\mathrm{OPSD}}-W_0\), \(d_S=W_{\mathrm{SFT}}-W_0\).
- These are differences of **effective weight matrices**, not differences of LoRA \(A/B\) factors. Factor-space cosines are gauge-dependent.
- The missing token names in your question refer to `&`, `learning`, and `Learning`.

If your table uses whole adapters relative to the frozen backbone, subtract the common starting adapter before interpreting the *training-induced* directions.

# 1. What the layer pattern actually says

### L08–L15: a common component, but not predominantly the same update

A cosine of \(0.36\) is meaningful relative to your late layers, but it is not strong alignment.

For each module, write

\[
d_S=a\,d_O+d_{S,\perp},
\qquad
a=\frac{\langle d_S,d_O\rangle_F}{\|d_O\|_F^2}.
\]

The fraction of SFT update energy along the **single flattened OPSD direction** is \(c^2\).

- **L13 gate:** \(c=0.3616\), so approximately **13.1%** of SFT energy lies along that direction.
- **L26 down:** \(c=0.0517\), so approximately **0.27%** does.

Thus the distinction is not “same middle-layer program versus different late-layer program.” It is:

> **A larger shared component in the middle, followed by updates that are almost orthogonal in some late modules.**

A plausible explanation is that both procedures improve common task computations—interpreting tab relationships, formatting assignments, or calibrating grouping decisions—but receive different training signals for the final policy. The table does **not** identify which middle-layer computation is shared.

### Why the objectives can produce this pattern

For ordinary teacher-forced SFT,

\[
\frac{\partial \mathcal L_{\mathrm{CE}}}{\partial z_t}
=
p_\theta(\cdot\mid x,y_{<t})-\mathrm{onehot}(y_t).
\]

Training therefore reinforces the Astra answers at **Astra-answer prefixes**.

On-policy distillation instead evaluates its loss at student-generated prefixes, with the exact gradient depending on your implementation. This changes:

1. Which prefixes receive training.
2. Which errors the model encounters.
3. Whether targets are hard tokens or distributions.
4. How much pressure exists to imitate the reference’s particular output conventions.

The same reference answers do not make those optimization problems equivalent. Also, 150 steps does not equalize effective update magnitude: SFT has substantially larger update norms in many modules.

### L18–L31: evidence for systematic lexical specialization—and potentially policy changes

Your late SFT update repeatedly exposes:

- `&` and conjunction-like tokens;
- `learning` / `Learning`;
- group-label vocabulary.

This is consistent with the training-label shift:

\[
13.03/1.43 \approx 9.11.
\]

The incidence of group names containing `&` is approximately nine times the original SFT rate. Increased Ungrouped labeling supplies a separate pressure toward leaving tabs out of groups.

Those two pressures could contribute differently:

- More Ungrouped decisions can reduce v1 recall.
- Different partition boundaries can improve v2.1.
- Compound names can accompany broader categories—but `&` itself does not prove over-merging.

OPSD is not “nonlexical,” either: its late singular vectors expose `History`, `YouTube`, `Programming`, etc. The contrast is **which lexical directions become systematic**, not lexical versus semantic learning.

### Why “\(u_1\) writes `&`” is not yet a mechanism

For a residual-output matrix,

\[
dW=\sum_i \sigma_i u_i v_i^\top,
\]

and its leading-mode contribution is

\[
\delta r_t=\sigma_1 u_1\bigl(v_1^\top h_t\bigr).
\]

Therefore:

- \(u_1\) specifies an output direction.
- \(v_1^\top h_t\) determines whether that direction is actually used, with what magnitude and sign.
- The downstream network determines its effect on logits and assignments.

The first-order logit effect is

\[
\delta z_t \approx J_{\ell,t}\,\delta r_t,
\]

not simply \(W_Uu_1\). Here \(J_{\ell,t}\) includes subsequent layers and normalization.

Also, the SVD pair can flip sign simultaneously: “top positive tokens of \(u_1\)” is not sign-invariant evidence of positive token promotion.

**Consequently, a late SFT direction can be:**

1. A naming/renderer direction.
2. A grouping-policy direction.
3. A direction serving both.
4. A mostly inactive direction on your evaluation prompts.

Your table does not distinguish these.

**The decisive zero-training diagnostic:** patch the candidate singular-mode contribution between OPSD and SFT on actual examples. Measure changes in tab assignments separately from changes in group names. If only names change, deleting that mode will not recover splitting.

---

# 2. Three exact zero-training surgical candidates

These are **three alternative candidates**, not three steps that must all be stacked. Their formulas are exact; achieving both recall targets is an empirical question.

Define

\[
C_m=d_{S,m}-d_{O,m}
    =W_{S,m}-W_{O,m}.
\]

All unmentioned modules remain OPSD.

## Edit 1 — Late full-attention module graft

**Purpose:** test whether SFT’s useful v2.1 change can be transferred through a small subset of attention-output pathways without replacing its late MLPs.

Replace only:

\[
\mathcal M_1=
\{
(19,\texttt{self\_attn.o\_proj}),
(23,\texttt{self\_attn.o\_proj}),
(27,\texttt{self\_attn.o\_proj}),
(31,\texttt{self\_attn.o\_proj})
\}.
\]

Use

\[
W^{(1)}_m=
\begin{cases}
W_{S,m}, & m\in\mathcal M_1,\\
W_{O,m}, & \text{otherwise}.
\end{cases}
\]

### Implementation

Copy the complete SFT LoRA \(A_m,B_m\) pair for those four modules into the OPSD adapter.

### Why this candidate

- Keeps all OPSD MLPs, including its late gates.
- Avoids transplanting the numerous SFT `down_proj` directions associated with `&` / `learning`.
- Tests a localized attention-pathway hypothesis.
- **Preserves rank 16 exactly.**

This is a clean causal control, not a claim that full attention uniquely implements splitting. Linear-attention blocks can also mediate global relationships.

**Failure mode:** the useful partitioning changes may reside in MLPs, linear-attention blocks, or interactions among several modules. These four output maps also are not lexically clean.

---

## Edit 2 — Unembedding-null, norm-capped late graft

**Purpose:** import late SFT innovations while excluding their component along specified lexical output directions.

### A. Construct a token-direction projector

Let \(W_U\) be the actual output unembedding matrix. Select vocabulary IDs whose individually decoded strings exactly match:

```text
"&", " &",
"learning", " learning",
"Learning", " Learning"
```

Ignore strings without a corresponding individual token.

Construct

\[
Q_T=\operatorname{orth}\left(W_U[T,:]^\top\right),
\qquad
P_T=Q_TQ_T^\top.
\]

Use the output unembedding, not input embeddings unless they are tied.

### B. Select late residual-output modules

\[
\mathcal M_2=
\left\{
(\ell,\texttt{mlp.down\_proj}),
(\ell,\texttt{attention-output})
:\ell=18,\ldots,31
\right\},
\]

where attention-output means the appropriate `linear_attn.out_proj` or `self_attn.o_proj`.

Leave late gates unchanged.

### C. Remove the lexical component and cap the transplant

For each selected module:

\[
X_m=(I-P_T)C_m.
\]

Define

\[
\operatorname{cap}_m(X)
=
X\min\left(
1,\frac{0.5\|d_{O,m}\|_F}{\|X\|_F}
\right),
\]

with a zero result when \(X=0\). Then set

\[
\boxed{
W^{(2)}_m=W_{O,m}+\operatorname{cap}_m(X_m)
}.
\]

The \(0.5\) cap is a conservative starting choice, **not a value implied by your table**.

### Exact property

Before any rank recompression,

\[
Q_T^\top\bigl(W^{(2)}_m-W_{O,m}\bigr)=0.
\]

Thus, for a fixed module input, the added residual vector has zero projection onto those unembedding directions.

### Important limitation

This is **unembedding-null**, not causally “lexical-null” through the whole network. Downstream transformations and normalization can still change those token logits.

It also cannot determine whether the removed component was useful for grouping. Its advantage over deleting every leading SFT singular mode is that it removes a narrowly specified output subspace rather than indiscriminately removing high-energy computation.

**Interpretation of the result:** if `&` frequency falls but group sizes and Ungrouped rates do not improve, you primarily edited naming.

---

## Edit 3 — v1-input-protected SFT innovation graft

**Purpose:** transfer SFT changes preferentially outside activation directions heavily used by OPSD on v1.

This requires **forward-pass calibration**, but no labels, gradients, or parameter optimization.

### A. Collect OPSD module inputs on a separate v1 calibration set

For each edited module, collect its actual input vectors during OPSD generation:

\[
H_m=[h_{m,1},\ldots,h_{m,N}].
\]

Use **uncentered** activations.

Let \(Z_m\) contain the leading

\[
k_m=\min(32,\operatorname{rank}(H_m))
\]

left singular vectors of \(H_m\), and define

\[
P_{1,m}=Z_mZ_m^\top.
\]

### B. Use this explicit module mask

Edit:

- All displayed module types—gate, down, attention-output—at **L08–L15**.
- The late residual-output modules \(\mathcal M_2\) at **L18–L31**.

Keep L00–L07, L16–L17, and late L18–L31 gates at OPSD.

This tests a combination of the relatively shared middle-band update and the divergent late output update, while retaining OPSD’s late gating.

### C. Project on both sides

For residual-output modules, set \(P_{\text{out},m}=P_T\).

For `gate_proj`, set \(P_{\text{out},m}=0\): gate outputs are not residual-stream coordinates, so unembedding projection is inappropriate there.

Now compute

\[
X_m=
(I-P_{\text{out},m})\,C_m\,(I-P_{1,m}),
\]

and apply

\[
\boxed{
W^{(3)}_m=W_{O,m}+\operatorname{cap}_m(X_m)
}.
\]

### Exact property

The added update does not affect inputs in the protected subspace:

\[
\bigl(W^{(3)}_m-W_{O,m}\bigr)Z_m=0.
\]

This is a meaningful preservation constraint. By contrast, making a weight update Frobenius-orthogonal to \(d_O\) would **not** preserve OPSD’s function.

### What is and is not preserved

Protecting the top 32 activation directions does not guarantee unchanged v1 behavior. It protects only that subspace.

If instead you protect the **entire column span of \(H_m\)** at every edited module, all collected OPSD traces are unchanged in exact arithmetic. However:

- The span can become full rank, making the edit zero.
- Protection on calibration traces does not guarantee protection on unseen prompts.

This candidate is most promising when v1 and v2.1 exercise meaningfully different activation directions.

---

# 3. LoRA implementation constraints

Apply the equations to **effective matrices**:

\[
L_m=\frac{\alpha}{r}B_mA_m=2B_mA_m.
\]

Do not average \(A\) and \(B\) independently. That introduces cross terms and is not matrix interpolation.

### Rank consequences

- **Edit 1:** remains rank 16 exactly.
- **Edits 2–3:** generally cannot be represented exactly by the original rank-16 adapter.

For exact surgery, either merge into dense weights or retain an expanded-rank representation.

If rank 16 is mandatory, recompress the **entire resulting adapter relative to the frozen backbone**:

\[
L_m^* = W_m^*-W_{\mathrm{backbone},m},
\]

\[
\widehat L_m=\operatorname{SVD}_{16}(L_m^*).
\]

For \(\widehat L=U\Sigma V^\top\), one valid factorization at scaling \(2\) is

\[
B=U\sqrt{\Sigma/2},
\qquad
A=\sqrt{\Sigma/2}V^\top.
\]

**Recompression can destroy the lexical-null and activation-protection constraints.** Evaluate both the exact merged candidate and its rank-16 approximation.

# 4. How I would run the experiment

| Candidate | Main question |
|---|---|
| Edit 1 | Can a small late attention graft transfer the useful behavior? |
| Edit 2 | Can late SFT innovation help after removing the specified lexical projection? |
| Edit 3 | Can activation protection reduce interference with OPSD’s v1 behavior? |

Measure, separately:

1. v1 and v2.1 precision/recall.
2. Tab-membership accuracy independent of names.
3. Ungrouped rate.
4. Largest-group size and group-size distribution.
5. `&` and `learning` naming frequencies.
6. Session-level uncertainty, preferably bootstrap intervals.

One metric caveat: under ordinary **pairwise** scoring, pure merging cannot reduce recall. If giant-bucket over-merging explains low reported recall, your scorer must involve cluster matching, exclusions, accompanying ungrouping, or another definition. Verify that before treating recall as a direct splitting measure.

**My priority would be Edit 2, followed by Edit 3; Edit 1 is the cleanest low-cost control.** The attainable target is the measured joint frontier—not an assumed additive combination of OPSD’s 70.20% and SFT’s 57.86%. The table motivates these interventions, but it cannot guarantee that frontier exists.