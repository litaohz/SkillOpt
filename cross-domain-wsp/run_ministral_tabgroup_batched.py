"""Batched evaluation of Zero-Training Weight Surgeries on Ministral-3-8B-sft2000 for Browser TabGroup (v2.1 & v1)."""
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import load_file
from peft import PeftModel

MIN_DIR = Path('/home/dlisuser/work/tabgroup-ministral8b-eval-20261026')
sys.path.insert(0, str(MIN_DIR))
from model_runtime import load_base
from adapter_runtime import logits_option

EVAL_DIR = Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch')
sys.path.insert(0, str(EVAL_DIR / '_engine'))
from scoring import score

refs = {}
inputs = {}
for ds_dir in (EVAL_DIR / 'datasets').iterdir():
    if not ds_dir.is_dir():
        continue
    ds = ds_dir.name
    refs[ds] = {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()}
    inputs[ds] = [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()]

cfg = json.loads((MIN_DIR / 'training-config.json').read_text(encoding='utf8'))
MODEL_ID = 'ministral-3-8b-instruct-2512-bf16'
spec = next(m for m in cfg['models'] if m['id'] == MODEL_ID)
folder = Path('/home/dlisuser/cache/sft-models/ministral-3-8b-instruct-2512-bf16/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734')
checkpoint = MIN_DIR / 'checkpoint/adapter'

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Loading Ministral-3-8B base + step-002000 LoRA adapter...', flush=True)
base, tok = load_base(folder, spec, device='cuda:0')
base.config._attn_implementation = 'sdpa'
if hasattr(base, 'language_model'):
    base.language_model.config._attn_implementation = 'sdpa'
if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
    base.model.language_model.config._attn_implementation = 'sdpa'
model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
for name, p in model.named_parameters():
    if 'lora_' in name:
        p.data = p.data.to(torch.float32)
model.eval()
model.config.use_cache = True

sd_orig = load_file(str(checkpoint / 'adapter_model.safetensors'), device='cuda:0')

eos = model.generation_config.eos_token_id
eos = list(eos) if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
if tok.eos_token_id is not None:
    eos.append(tok.eos_token_id)
eos = sorted(set(eos))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

lm = model.base_model.model.model.language_model
U_vocab = lm.embed_tokens.weight.detach().float()
final_norm = lm.norm.weight.detach().float()
U_eff = U_vocab * final_norm.unsqueeze(0)

nl_ids = tok.encode('\n', add_special_tokens=False)
cm_ids = tok.encode(',', add_special_tokens=False)
ug_ids = tok.encode('Ungrouped', add_special_tokens=False)

v_nl = U_eff[nl_ids[-1]] / torch.norm(U_eff[nl_ids[-1]])
v_cm = U_eff[cm_ids[-1]] / torch.norm(U_eff[cm_ids[-1]])
v_ug = U_eff[ug_ids[0]] / torch.norm(U_eff[ug_ids[0]])
v_split = (v_nl - v_cm) / torch.norm(v_nl - v_cm)
v_anti_ug = (0.5 * (v_nl + v_cm) - v_ug)
v_anti_ug = v_anti_ug / torch.norm(v_anti_ug)


def load_adapter_sd(sd_dict):
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))


def eval_batched(ds_list, name, batch_size=4):
    t0 = time.monotonic()
    res = {}
    for ds in ds_list:
        r1_inputs = sorted([i for i in inputs[ds] if i['run'] == 1], key=lambda x: x['case_id'])
        enc_list = []
        for inp in r1_inputs:
            messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': inp['prompt']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec['chat_template_kwargs'])
            enc_list.append(tok.encode(text, add_special_tokens=False))
        sc_list = []
        for b_start in range(0, len(r1_inputs), batch_size):
            b_rows = r1_inputs[b_start:b_start + batch_size]
            b_encs = enc_list[b_start:b_start + batch_size]
            max_l = max(len(e) for e in b_encs)
            padded_ids = [[pad_id] * (max_l - len(e)) + e for e in b_encs]
            attn_mask = [[0] * (max_l - len(e)) + [1] * len(e) for e in b_encs]
            ids_t = torch.tensor(padded_ids, device='cuda:0', dtype=torch.long)
            mask_t = torch.tensor(attn_mask, device='cuda:0', dtype=torch.long)
            with torch.inference_mode():
                gen = model.generate(
                    input_ids=ids_t,
                    attention_mask=mask_t,
                    max_new_tokens=300,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos,
                    pad_token_id=pad_id,
                    **logits_option(spec),
                )
            for idx_b, inp in enumerate(b_rows):
                pred = tok.decode(gen[idx_b, max_l:].tolist(), skip_special_tokens=True)
                mapping = {int(k): v for k, v in inp['mapping'].items()}
                sc = score(pred, refs[ds][inp['case_id']], mapping)
                sc_list.append(sc)
        init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100
        rec = statistics.mean(x['recall'] for x in sc_list) * 100
        f1 = 2 * init_p * rec / max(init_p + rec, 1e-6)
        res[ds] = {'init_p': round(init_p, 2), 'recall': round(rec, 2), 'f1': round(f1, 2)}
        print(f'{name:48s} | {ds:5s} (n={len(sc_list):2d}) | P={init_p:6.2f}% R={rec:6.2f}% F1={f1:6.2f}% | ({time.monotonic()-t0:.1f}s)', flush=True)
    return res


def make_idea_b(sd_in, cut_l=24, k_zero=1):
    sd_out = {}
    for k in sd_in:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        As, Bs = sd_in[k].float(), sd_in[bk].float()
        if l >= cut_l:
            Qb, Rb = torch.linalg.qr(Bs, mode='reduced')
            Qa, Ra = torch.linalg.qr(As.T, mode='reduced')
            U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)
            S_mod = S_d.clone()
            S_mod[:k_zero] = 0.0
            sqrt_S = torch.sqrt(S_mod)
            sd_out[bk] = ((Qb @ U_d) * sqrt_S.unsqueeze(0)).contiguous()
            sd_out[k] = (sqrt_S.unsqueeze(1) * (Vh_d @ Qa.T)).contiguous()
        else:
            sd_out[k], sd_out[bk] = As.clone(), Bs.clone()
    return sd_out


def make_idea_c_qproj(sd_in, start_l=8, end_l=20, scale=1.35):
    sd_out = {}
    for k, v in sd_in.items():
        if 'self_attn.q_proj.lora_B' in k:
            parts = k.split('.')
            l = int(parts[parts.index('layers') + 1])
            if start_l <= l <= end_l:
                sd_out[k] = (v.float() * scale).to(v.dtype)
                continue
        sd_out[k] = v.clone()
    return sd_out


results = {
    'Ministral-8B Base sft2000': {
        'v1': {'init_p': 82.72, 'recall': 75.44, 'f1': 78.92},
        'v2.1': {'init_p': 67.86, 'recall': 56.89, 'f1': 61.89}
    },
    'Ministral-8B + Idea_B1 (L24-33 sigma1=0)': {
        'v1': {'init_p': 84.21, 'recall': 76.15, 'f1': 79.98},
        'v2.1': {'init_p': 62.71, 'recall': 54.91, 'f1': 58.55}
    }
}

# 3. Idea C (L08-20 q_proj * 1.30x)
load_adapter_sd(make_idea_c_qproj(sd_orig, 8, 20, 1.30))
results['Ministral-8B + Idea_C (L08-20 q_proj*1.30x)'] = eval_batched(['v2.1', 'v1'], 'Ministral-8B + Idea_C (L08-20 q_proj*1.30x)')

# 4. Idea A (Anti-Ungrouped / Valve edit on L28-33 mlp.down_proj)
load_adapter_sd(sd_orig)
orig_down = {}
for l in range(28, 34):
    m = lm.layers[l].mlp.down_proj.base_layer
    orig_down[l] = m.weight.data.clone()
    W = orig_down[l].float()
    scores = v_anti_ug @ W
    top_pos = torch.topk(scores, 64).indices
    top_neg = torch.topk(-scores, 64).indices
    W_mod = W.clone()
    W_mod[:, top_pos] *= 1.8
    W_mod[:, top_neg] *= 0.2
    m.weight.data.copy_(W_mod.to(orig_down[l].dtype))

results['Ministral-8B + Idea_A (64-Col Anti-Ungrouped Valve)'] = eval_batched(['v2.1', 'v1'], 'Ministral-8B + Idea_A (Anti-Ungrouped Valve)')

for l in range(28, 34):
    lm.layers[l].mlp.down_proj.base_layer.weight.data.copy_(orig_down[l])

Path('/home/dlisuser/work/public-two-stage-20261104/ministral_tabgroup_surgery.json').write_text(json.dumps(results, indent=2))
print('=== MINISTRAL TABGROUP SURGERY COMPLETED! ===', flush=True)
