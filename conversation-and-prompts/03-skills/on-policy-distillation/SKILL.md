---
name: on-policy-distillation
description: "Use when distilling LLMs via GKD, OPD, or OPSD rollouts."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [distillation, opd, opsd, gkd, lora, slm]
    related_skills: [huggingface-hub]
---

# On-Policy Distillation (GKD / OPD / OPSD)

## When to Use
Use when compressing or improving a fine-tuned Small Language Model (SLM) toward a stronger teacher model, especially when standard SFT plateaus on recall or suffers from coarse over-merging / exposure bias on complex multi-entity or structured generation tasks.

## Choosing Between GKD (White-Box OPD) and OPSD (Privileged-Context Self-Distillation)

| Paradigm | Teacher Setup | Best When | Key Limitation |
|---|---|---|---|
| **Standard White-Box OPD / GKD** (e.g., `9B-SFT ➔ 4B-SFT`) | A larger white-box model sharing the exact same tokenizer and vocabulary (`vocab.json`, `merges.txt`, `tokenizer.json`, `chat_template.jinja`) conditioned on the standard prompt $x$. | You already have a significantly stronger white-box teacher in the same model family and want pure model-size compression. | If the larger white-box teacher was trained on the same weak/noisy SFT labels as the student, it shares the same structural biases (e.g., giant-bucket over-merging) and yields minimal recall gain. |
| **On-Policy Self-Distillation (OPSD)** (e.g., `4B-SFT + Frontier Answer ➔ 4B-SFT`) | A frozen copy of the student (or same-family white-box model) conditioned on **Privileged Context** `[Prompt x + Frontier Reference Solution y*]` while the student only sees `[Prompt x]`. | The true quality ceiling is a closed-source frontier API (which cannot provide token-level echo logprobs on student rollouts and uses an incompatible tokenizer). | Requires an offline pass to label a subset of training prompts (e.g., 1,000–2,000 deterministic samples) with the frontier model's text responses $y^\star$. |

**Why Closed-Source APIs Cannot Be Called Directly Inside a Token-Level OPD Loop:**
1. Token-level OPD requires evaluating the teacher's conditional distribution $p_T(\cdot \mid x, \hat{y}_{<t})$ on the **student's own sampled trajectory $\hat{y}$** via teacher forcing. Standard Chat/Responses APIs only return logprobs for tokens the API generated itself, not for caller-supplied assistant continuations.
2. Closed-source frontier models have incompatible tokenizers and vocabularies.
3. **OPSD solves both**: the frontier model only provides offline text solutions $y^\star$; during GPU training, the open-book white-box teacher reads `[x + y*]` and performs a single local forward pass over the student's rollout $\hat{y}$ to produce exact, vocabulary-aligned token-level logits.

---

## Core Procedure

### Step 1 · Verify Tokenizer Identity (for Cross-Size OPD)
Before running white-box OPD between two model sizes (e.g., 9B and 4B), verify that `vocab.json`, `merges.txt`, `tokenizer.json`, and `chat_template.jinja` have identical SHA256 digests so token IDs and logit indices align 1-to-1.

### Step 2 · Prepare Privileged Reference Labels (for OPSD)
1. Sample a deterministic subset of training prompts (e.g., 1,200–2,000 cases using a fixed shuffle seed).
2. Query the frontier model API concurrently (with atomic `.tmp`-then-rename per-case checkpoints so interrupted runs resume cleanly) to produce `astra_chosen` ($y^\star$).
3. Consolidate into a deterministic JSONL file containing `prompt` and the frontier reference solution.

### Step 3 · On-Policy Rollout + Single-Pass Teacher Forcing
For each training sample in a gradient accumulation step:

1. **Student On-Policy Rollout ($\hat{y} \sim p_S(\cdot \mid x)$)**:
   - Format the blind student prompt `s_prompt_ids` (length $L_s$).
   - Switch student to `eval()` with `use_cache=True` under `torch.inference_mode()` and sample a continuation (`do_sample=True, temperature=0.8, top_p=0.95, top_k=50`).
   - Extract generated token IDs `gen_ids = gen_out[0, L_s:]` (length $L_c$). Skip if $L_c \le 0$.

2. **Teacher Single Forward Pass (No Autoregressive Generation Needed)**:
   - Because causal language models apply a lower-triangular attention mask, feeding a full sequence of length $L_p + L_c$ into `teacher(input_ids=seq)` computes all $L_c$ next-token distributions in **one parallel forward pass**.
   - For **GKD**, `t_seq = s_prompt_ids + gen_ids` (prompt length $L_t = L_s$).
   - For **OPSD**, construct the privileged prompt by appending the frontier reference solution $y^\star$ to $x$:
     ```text
     {original_prompt}

     === Privileged Expert Reference Solution (for teacher guidance only) ===
     {frontier_answer}
     === End Reference Solution ===

     Using the privileged expert reference solution above, output the coherent response in the exact required format:
     ```
     Encode as `t_prompt_ids` (length $L_t$) and concatenate `t_seq = t_prompt_ids + gen_ids`.
   - Slice the teacher logits at the exact positions predicting `gen_ids`:
     ```python
     with torch.no_grad():
         t_out = teacher(input_ids=t_seq, attention_mask=torch.ones_like(t_seq), use_cache=False)
         # Index L_t - 1 predicts gen_ids[0]; index -2 predicts gen_ids[-1]
         t_logits = t_out.logits[:, L_t - 1 : -1, :].to(torch.float32).detach()
     ```

3. **Student Forward Pass (With Gradients)**:
   - Switch student to `train()` with `use_cache=False`.
   - Run forward pass on `s_seq = s_prompt_ids + gen_ids` and slice `s_logits = s_out.logits[:, L_s - 1 : -1, :]`.

4. **Top-$k$ Generalized JSD Loss**:
   - Compute Generalized Jensen-Shannon Divergence ($\beta=0.5$, temperature $1.0$) restricted to the teacher's `top_k=64` logit indices at each token position, with per-token clipping (`token_clip=0.1`) to prevent gradient spikes on outlier tokens:
     ```python
     def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1):
         s_logits = student_logits / temperature
         t_logits = teacher_logits / temperature
         if top_k is not None and top_k > 0:
             _, top_k_indices = torch.topk(t_logits, k=top_k, dim=-1)
             s_logits = torch.gather(s_logits, dim=-1, index=top_k_indices)
             t_logits = torch.gather(t_logits, dim=-1, index=top_k_indices)
         s_log_probs = F.log_softmax(s_logits, dim=-1)
         t_log_probs = F.log_softmax(t_logits, dim=-1)
         b = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
         mix_log_probs = torch.logsumexp(
             torch.stack([s_log_probs + torch.log1p(-b), t_log_probs + torch.log(b)]), dim=0
         )
         kl_t = F.kl_div(mix_log_probs, t_log_probs, reduction='none', log_target=True).sum(dim=-1)
         kl_s = F.kl_div(mix_log_probs, s_log_probs, reduction='none', log_target=True).sum(dim=-1)
         jsd = beta * kl_t + (1.0 - beta) * kl_s
         if token_clip is not None:
             jsd = jsd.clamp(max=token_clip)
         return jsd.mean()
     ```

### Step 4 · Memory-Efficient Single-Model OPSD via LoRA State Swapping
When running OPSD where both teacher and student start from an SFT LoRA checkpoint (rather than the raw base model):
- Do **not** call `disable_adapter()` if the raw base model has poor format compliance; the privileged teacher should retain its SFT LoRA weights so its output distribution stays strictly in format.
- Instead of loading two copies of the base model, clone the initial SFT LoRA tensors (`teacher_lora_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if 'lora_' in n}`), swap `teacher_lora_state` in during the `torch.no_grad()` teacher forward pass, and restore the student's current LoRA tensors immediately afterward.

### Step 5 · Post-Distillation Weight-Space Diagnosis & Zero-Training Surgery
When comparing OPSD against standard Cross-Entropy SFT on the same frontier reference labels $y^\star$ (or when fixing over-merging / recall collapse directly in weight space without retraining):
- Compute layerwise effective weight deltas $\Delta W = \frac{\alpha}{r}(B A - B_0 A_0)$ via reduced QR-SVD (`lora_A` is the $r \times d_{\text{in}}$ reader; `lora_B` is the $d_{\text{out}} \times r$ writer) and project late residual-writer singular vectors onto the base model's unembedding matrix $U_{\text{eff}} = W_U \odot \gamma_{\text{final}}$.
- **Dual-Adapter Surgery**: Apply **Layerwise Circuit Grafting** (taking early/middle clustering layers from Frontier-SFT and late readout layers from OPSD) or **Rank-$r$ QR-SVD Spectral Consensus Blending** (`0.5 OPSD + 0.5 SFT` compressed to rank $r$) to amplify shared middle-layer clustering directions while halving and truncating orthogonal late-layer lexical drift.
- **Single-Model Value/Neuron Surgery (No Second Adapter Needed)**:
  1. *Readout Valve Neuron Scaling*: Project late-layer `mlp.down_proj` columns ($[d_{\text{model}}, d_{\text{ff}}]$) onto the contrastive unembedding vector $v_{\text{split}} = \hat{u}_{\text{newline}} - \hat{u}_{\text{comma}}$, then scale the top-$K$ (e.g., 64) split-promoting columns by $2.0\times$ and zero the top-$K$ merge-promoting columns to break giant-bucket over-merging in zero training steps.
  2. *Late Singular Value Excision ($\sigma_1, \sigma_2 \to 0$)*: In an off-policy SFT adapter suffering from late-layer lexical or rejection-threshold drift, zero only the top 1–2 singular values of $\Delta W_{\text{SFT}} = W_{\text{SFT}} - W_0$ in the late readout layers while keeping modes $3\dots r$ intact.
  3. *Full-Attention QK-Norm Temperature Scaling*: Scale `q_norm.weight` in middle full-attention layers by a scalar factor (e.g., $1.35\times$) to sharpen cross-entity attention clustering.
- See `references/weight-space-surgery.md` for exact QR-SVD formulas, 6-module physical anatomy, single-neuron polarity probes, singular-value excision, and surgical blending code.

---

## Pitfalls
- **Slice logits at `[L_prompt - 1 : -1]`, never `[L_prompt : ]`** — in causal LMs, the logit at index $L_{\text{prompt}} - 1$ predicts the first generated token $\hat{y}_0$, while the final index $-1$ predicts the token *after* the sequence ends.
- **Toggle `use_cache` explicitly between rollout and training** — enable `model.eval(); model.config.use_cache = True` before `model.generate()`, and reset `model.train(); model.config.use_cache = False` before the gradient forward pass, because gradient checkpointing conflicts with KV caching.
- **Pass positional arguments `(optimizer, num_warmup_steps, num_training_steps)` to `get_cosine_schedule_with_warmup`** — newer `transformers` versions rename keyword arguments (`num_warmup_steps` vs `warmup_steps`), so positional arguments avoid `TypeError` crashes across environments.
- **Inspect real case outputs alongside aggregate Precision/Recall** — verify on concrete multi-topic examples that OPSD actually breaks up coarse catch-all groups into fine-grained topic clusters rather than gaming the denominator by emitting fewer groups.
