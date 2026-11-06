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

def load_adapter_sd(sd_dict):
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))

def run_full_r1(name):
    t0 = time.monotonic()
    res = {}
    for ds in sorted(refs.keys()):
        r1_inputs = sorted([i for i in inputs[ds] if i['run'] == 1], key=lambda x: x['case_id'])
        sc_list = []
        per_case = {}
        for inp in r1_inputs:
            messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': inp['prompt']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device='cuda:0')
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg['eval_max_new_tokens'], do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
            pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
            mapping = {int(k): v for k, v in inp['mapping'].items()}
            sc = score(pred, refs[ds][inp['case_id']], mapping)
            sc_list.append(sc)
            per_case[inp['case_id']] = {'precision': round(sc['precision']*100, 2), 'recall': round(sc['recall']*100, 2), 'prediction': pred.strip()}
        init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100
        rec = statistics.mean(x['recall'] for x in sc_list) * 100
        res[ds] = {'init_p': round(init_p, 2), 'recall': round(rec, 2), 'cases': per_case}
        print(f'{name:42s} | {ds:6s} (n={len(sc_list):2d}) | Init P = {init_p:6.2f}% | Recall = {rec:6.2f}% | ({time.monotonic()-t0:.1f}s)', flush=True)
    return res

all_results = {}

# 1. IDEA B-1: Zero ONLY 1 Scalar (sigma_1 = 0) of d_SFT in L18-31
sd_b1 = {}
for k in sds['sft2000']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])
    if l >= 18:
        A0, B0 = sds['sft2000'][k].float(), sds['sft2000'][bk].float()
        As, Bs = sds['sft_astra'][k].float(), sds['sft_astra'][bk].float()
        B_d = torch.cat([Bs, -B0], dim=1)
        A_d = torch.cat([As, A0], dim=0)
        Qb, Rb = torch.linalg.qr(B_d, mode='reduced')
        Qa, Ra = torch.linalg.qr(A_d.T, mode='reduced')
        U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)
        S_mod = S_d.clone()
        S_mod[:1] = 0.0
        B_new_cat = torch.cat([B0, (Qb @ U_d) * S_mod.unsqueeze(0)], dim=1)
        A_new_cat = torch.cat([A0, Vh_d @ Qa.T], dim=0)
        Qb2, Rb2 = torch.linalg.qr(B_new_cat, mode='reduced')
        Qa2, Ra2 = torch.linalg.qr(A_new_cat.T, mode='reduced')
        U2, S2, Vh2 = torch.linalg.svd(Rb2 @ Ra2.T)
        r = 16
        sqrt_S2 = torch.sqrt(S2[:r])
        sd_b1[bk] = ((Qb2 @ U2[:, :r]) * sqrt_S2.unsqueeze(0)).contiguous()
        sd_b1[k] = (sqrt_S2.unsqueeze(1) * (Vh2[:r, :] @ Qa2.T)).contiguous()
    else:
        sd_b1[k] = sds['sft_astra'][k].clone()
        sd_b1[bk] = sds['sft_astra'][bk].clone()

load_adapter_sd(sd_b1)
all_results['idea_b1_sigma1_zero'] = run_full_r1('Idea B-1 (sft_astra L18-31 sigma_1=0)')

# 2. IDEA A: Top-64 newline/comma neuron surgery in L26-31 mlp.down_proj on sft2000
load_adapter_sd(sds['sft2000'])
orig_down = {}
for l in range(26, 32):
    m = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer
    orig_down[l] = m.weight.data.clone()
    W = orig_down[l].float()
    scores = v_diff_nl_cm @ W
    top_nl_idx = torch.topk(scores, 64).indices
    top_cm_idx = torch.topk(-scores, 64).indices
    W_mod = W.clone()
    W_mod[:, top_nl_idx] *= 2.0
    W_mod[:, top_cm_idx] *= 0.0
    m.weight.data.copy_(W_mod.to(orig_down[l].dtype))

all_results['idea_a_64neurons_sft2000'] = run_full_r1('Idea A (sft2000 L26-31 64-neuron edit)')

for l in range(26, 32):
    model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(orig_down[l])

# 3. IDEA C: Full-Attention q_norm * 1.35x at L07, L11, L15, L19 on sft2000
orig_qnorm = {}
for l in [7, 11, 15, 19]:
    qn = model.base_model.model.model.language_model.layers[l].self_attn.q_norm
    orig_qnorm[l] = qn.weight.data.clone()
    qn.weight.data.copy_((orig_qnorm[l].float() * 1.35).to(orig_qnorm[l].dtype))

all_results['idea_c_qnorm135_sft2000'] = run_full_r1('Idea C (sft2000 Full-Attn q_norm * 1.35x)')

for l in [7, 11, 15, 19]:
    model.base_model.model.model.language_model.layers[l].self_attn.q_norm.weight.data.copy_(orig_qnorm[l])

Path('/home/dlisuser/work/abc_full_r1_results.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
print('ALL ABC FULL RUN-1 EVALUATIONS COMPLETED!', flush=True)
