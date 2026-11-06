import json, statistics, sys, time, torch
from pathlib import Path
from safetensors.torch import load_file
from peft import PeftModel

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from model_runtime import load_base
from adapter_runtime import logits_option

EVAL_DIR = Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch')
sys.path.insert(0, str(EVAL_DIR / '_engine'))
from scoring import score

refs = {}
inputs = {}
for ds_dir in (EVAL_DIR / 'datasets').iterdir():
    if not ds_dir.is_dir(): continue
    ds = ds_dir.name
    refs[ds] = {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()}
    inputs[ds] = [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()]

cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}
sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}

base_4b, tok = load_base(path_4b, spec_4b, device='cuda:0')
ckpt_sft2000 = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter')
model = PeftModel.from_pretrained(base_4b, ckpt_sft2000, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
model.eval()
model.config.use_cache = True

eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

U_vocab = model.base_model.model.model.language_model.embed_tokens.weight.detach().float()
final_norm = model.base_model.model.model.language_model.norm.weight.detach().float()
U_eff = U_vocab * final_norm.unsqueeze(0)

v_nl = U_eff[198] / torch.norm(U_eff[198])   # '\n'
v_cm = U_eff[11] / torch.norm(U_eff[11])     # ','
v_diff_nl_cm = (v_nl - v_cm) / torch.norm(v_nl - v_cm)

diag_ids = ('JH02', 'JH11', 'JH12', 'JH16', 'service-4-f43f114595c2', 'service-5-b48adde126b3', 'service-7-1c3a2d56655d')
diag_cases = []
for ds in sorted(refs.keys()):
    for inp in sorted(inputs[ds], key=lambda x: x['case_id']):
        if inp['run'] == 1 and inp['case_id'] in diag_ids:
            diag_cases.append((ds, inp))

def load_adapter_sd(sd_dict):
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))

def eval_cases(label, cases_to_run):
    print(f'\n==================== {label} ====================', flush=True)
    for ds, inp in cases_to_run:
        messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': inp['prompt']}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device='cuda:0')
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg['eval_max_new_tokens'], do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
        pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
        sc = score(pred, refs[ds][inp['case_id']], {int(k): v for k, v in inp['mapping'].items()})
        lines = [l for l in pred.splitlines() if ':' in l]
        print(f"[{ds}/{inp['case_id'][:10]}] P={sc['precision']*100:5.1f}% R={sc['recall']*100:5.1f}% ({len(lines)} lines): " + ' | '.join(lines), flush=True)

# =========================================================================
# IDEA B: Single-Scalar Singular Value Edit (sigma_1 = 0 in L18..L31 of sft_astra)
# =========================================================================
# In sft_astra, for each module in L18..L31, decompose (W_sft_astra - W_sft2000) = U S V^T,
# set ONLY the single scalar S[0] = 0.0 (killing the top-1 drift mode while keeping modes 2..16 intact!),
# and add back to W_sft2000!
sd_idea_b_top1_zero = {}
sd_idea_b_top2_zero = {}
for k in sds['sft2000']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])
    if l >= 18:
        A0, B0 = sds['sft2000'][k].float(), sds['sft2000'][bk].float()
        As, Bs = sds['sft_astra'][k].float(), sds['sft_astra'][bk].float()
        # d_SFT = Bs As - B0 A0
        B_d = torch.cat([Bs, -B0], dim=1)
        A_d = torch.cat([As, A0], dim=0)
        Qb, Rb = torch.linalg.qr(B_d, mode='reduced')
        Qa, Ra = torch.linalg.qr(A_d.T, mode='reduced')
        U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)

        for target_sd, k_kill in [(sd_idea_b_top1_zero, 1), (sd_idea_b_top2_zero, 2)]:
            S_mod = S_d.clone()
            S_mod[:k_kill] = 0.0 # <--- LITERALLY ZEROING 1 (or 2) SCALAR VALUE(S) PER MODULE!
            # Reconstruct W_new = B0 A0 + (Qb U_d) diag(S_mod) (Vh_d Qa^T) and compress to rank 16
            B_new_cat = torch.cat([B0, (Qb @ U_d) * S_mod.unsqueeze(0)], dim=1)
            A_new_cat = torch.cat([A0, Vh_d @ Qa.T], dim=0)
            Qb2, Rb2 = torch.linalg.qr(B_new_cat, mode='reduced')
            Qa2, Ra2 = torch.linalg.qr(A_new_cat.T, mode='reduced')
            U2, S2, Vh2 = torch.linalg.svd(Rb2 @ Ra2.T)
            r = 16
            sqrt_S2 = torch.sqrt(S2[:r])
            target_sd[bk] = ((Qb2 @ U2[:, :r]) * sqrt_S2.unsqueeze(0)).contiguous()
            target_sd[k] = (sqrt_S2.unsqueeze(1) * (Vh2[:r, :] @ Qa2.T)).contiguous()
    else:
        sd_idea_b_top1_zero[k] = sds['sft_astra'][k].clone()
        sd_idea_b_top1_zero[bk] = sds['sft_astra'][bk].clone()
        sd_idea_b_top2_zero[k] = sds['sft_astra'][k].clone()
        sd_idea_b_top2_zero[bk] = sds['sft_astra'][bk].clone()

load_adapter_sd(sd_idea_b_top1_zero)
eval_cases('IDEA B-1: Zero ONLY 1 Scalar (sigma_1 = 0) of d_SFT in L18-31', diag_cases)

load_adapter_sd(sd_idea_b_top2_zero)
eval_cases('IDEA B-2: Zero Top-2 Scalars (sigma_1 = sigma_2 = 0) of d_SFT in L18-31', diag_cases)

# =========================================================================
# IDEA A: Active Neuron Localization & Single/Few-Neuron Edit on sft2000 (Blog Exp 8 style)
# =========================================================================
load_adapter_sd(sds['sft2000'])
# Let's hook L16..L31 mlp.down_proj on JH02 and JH12 at the exact token positions where sft2000 outputs ', 2' (merging Grok into Soccer / Stocks into Travel)
# In JH02, sft2000 outputs: "Soccer: 1, 2, 3..." where tab 1 is Soccer Ball and tab 2 is Grok!
# At the token position right after "Soccer: 1", sft2000 predicts "," followed by " 2" instead of ", 6" or "\n".
# Wait: even simpler! In L24..L31 mlp.down_proj, let's test scaling the top-K columns of base W_down that project onto (v_newline - v_comma)!
orig_down = {}
for l in range(24, 32):
    m = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer
    orig_down[l] = m.weight.data.clone()

for mult_nl, mult_cm, tag in [(1.5, 0.5, 'Boost top-32 newline cols 1.5x & suppress top-32 comma cols 0.5x in L28-31'), (2.0, 0.0, 'Boost top-64 newline cols 2.0x & zero top-64 comma cols in L26-31')]:
    layers_to_edit = range(28, 32) if mult_nl == 1.5 else range(26, 32)
    k_cols = 32 if mult_nl == 1.5 else 64
    for l in layers_to_edit:
        W = orig_down[l].float()
        scores = v_diff_nl_cm @ W # (9216,)
        top_nl_idx = torch.topk(scores, k_cols).indices
        top_cm_idx = torch.topk(-scores, k_cols).indices
        W_mod = W.clone()
        W_mod[:, top_nl_idx] *= mult_nl
        W_mod[:, top_cm_idx] *= mult_cm
        model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(W_mod.to(orig_down[l].dtype))
    eval_cases(f'IDEA A ({tag})', diag_cases)
    for l in layers_to_edit:
        model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(orig_down[l])

# =========================================================================
# IDEA C: Full-Attention QK-Norm Temperature Scaling in L07, L11, L15, L19 on sft2000
# =========================================================================
orig_qnorm = {}
for l in [7, 11, 15, 19]:
    qn = model.base_model.model.model.language_model.layers[l].self_attn.q_norm
    orig_qnorm[l] = qn.weight.data.clone()

for scale_q in [1.35, 1.75, 0.70]:
    for l in [7, 11, 15, 19]:
        qn = model.base_model.model.model.language_model.layers[l].self_attn.q_norm
        qn.weight.data.copy_((orig_qnorm[l].float() * scale_q).to(orig_qnorm[l].dtype))
    eval_cases(f'IDEA C: Scale Full-Attn q_norm by {scale_q}x at L07, L11, L15, L19 on sft2000', diag_cases)
    for l in [7, 11, 15, 19]:
        model.base_model.model.model.language_model.layers[l].self_attn.q_norm.weight.data.copy_(orig_qnorm[l])
