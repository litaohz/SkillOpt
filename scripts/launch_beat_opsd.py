"""Test applying Weight-Space Surgery directly ON TOP OF opsd_astra to beat OPSD on BOTH v1 and v2.1!"""
import pathlib
import paramiko

REMOTE_SCRIPT = r"""import json, statistics, sys, time, torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from peft import PeftModel

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from model_runtime import load_base
from adapter_runtime import logits_option
from training_common import digest

EVAL_DIR = Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch')
sys.path.insert(0, str(EVAL_DIR / '_engine'))
from scoring import score

refs = {ds_dir.name: {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()} for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}
inputs = {ds_dir.name: [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()] for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}

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
ckpt_opsd = Path('/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter')
model = PeftModel.from_pretrained(base_4b, ckpt_opsd, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
model.eval()
model.config.use_cache = True

eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

U_vocab = model.base_model.model.model.language_model.embed_tokens.weight.detach().float()
final_norm = model.base_model.model.model.language_model.norm.weight.detach().float()
U_eff = U_vocab * final_norm.unsqueeze(0)
v_nl = U_eff[198] / torch.norm(U_eff[198])
v_cm = U_eff[11] / torch.norm(U_eff[11])
v_diff_nl_cm = (v_nl - v_cm) / torch.norm(v_nl - v_cm)

def load_adapter_sd(sd_dict):
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))

def run_full_r1(name):
    t0 = time.monotonic()
    for ds in sorted(refs.keys()):
        r1_inputs = sorted([i for i in inputs[ds] if i['run'] == 1], key=lambda x: x['case_id'])
        sc_list = []
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
        init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100
        rec = statistics.mean(x['recall'] for x in sc_list) * 100
        print(f'{name:48s} | {ds:6s} (n={len(sc_list):2d}) | Init P = {init_p:6.2f}% | Recall = {rec:6.2f}% | ({time.monotonic()-t0:.1f}s)', flush=True)

# Why did OPSD have lower Recall on v2.1 (59.36% vs Graft-18 64.52%)?
# Finding 1 showed: in L08-L15 (the middle clustering layers), OPSD's update direction is aligned with SFT (cos=0.37),
# BUT OPSD's update magnitude ||d_OPSD|| is 1.5x smaller than SFT's because of lazy on-policy gradients!
# What if we take ONLY the single model `opsd_astra` (NO sft_astra used!) and amplify `d_OPSD = W_opsd - W_sft2000` by 1.5x in the middle clustering layers L08-L15?!
def scale_opsd_mid(mult=1.5, l_start=8, l_end=16):
    sd_out = {}
    for k in sds['opsd_astra']:
        if '.lora_A.' not in k: continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        if l_start <= l < l_end:
            A0, B0 = sds['sft2000'][k].float(), sds['sft2000'][bk].float()
            Ao, Bo = sds['opsd_astra'][k].float(), sds['opsd_astra'][bk].float()
            # W_new = W_0 + mult * (W_opsd - W_0) = mult * W_opsd + (1 - mult) * W_0
            B_cat = torch.cat([Bo * mult, B0 * (1.0 - mult)], dim=1)
            A_cat = torch.cat([Ao, A0], dim=0)
            Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
            Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
            U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
            r = 16
            sqrt_S = torch.sqrt(S[:r])
            sd_out[bk] = ((Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)).contiguous()
            sd_out[k] = (sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)).contiguous()
        else:
            sd_out[k] = sds['opsd_astra'][k].clone()
            sd_out[bk] = sds['opsd_astra'][bk].clone()
    return sd_out

# Candidate 1: Pure OPSD with L08-L15 update amplified by 1.5x (compensating OPSD's lazy gradient step size in middle layers!)
sd_opsd_boost15 = scale_opsd_mid(1.5, 8, 16)
load_adapter_sd(sd_opsd_boost15)
run_full_r1('1. OPSD + L08-15 1.5x Gain Boost (Pure OPSD!)')

# Candidate 2: Surgical Hybrid L08-15 (50% OPSD + 50% SFT ONLY in L08-15, 100% OPSD in L00-07 and L16-31!)
sd_mid_graft = {}
for k in sds['opsd_astra']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])
    if 8 <= l <= 15:
        Ao, Bo = sds['opsd_astra'][k].float(), sds['opsd_astra'][bk].float()
        As, Bs = sds['sft_astra'][k].float(), sds['sft_astra'][bk].float()
        B_cat = torch.cat([Bo * 0.65, Bs * 0.35], dim=1)
        A_cat = torch.cat([Ao, As], dim=0)
        Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
        Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
        U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
        r = 16
        sqrt_S = torch.sqrt(S[:r])
        sd_mid_graft[bk] = ((Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)).contiguous()
        sd_mid_graft[k] = (sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)).contiguous()
    else:
        sd_mid_graft[k] = sds['opsd_astra'][k].clone()
        sd_mid_graft[bk] = sds['opsd_astra'][bk].clone()

load_adapter_sd(sd_mid_graft)
run_full_r1('2. OPSD + L08-15 Consensus (0.65 OPSD + 0.35 SFT)')

# Candidate 3: OPSD + Idea A (mild 1.25x newline / 0.75x comma top-32 neurons in L28-31 on top of OPSD!)
load_adapter_sd(sds['opsd_astra'])
orig_down = {}
for l in range(28, 32):
    m = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer
    orig_down[l] = m.weight.data.clone()
    W = orig_down[l].float()
    scores = v_diff_nl_cm @ W
    top_nl_idx = torch.topk(scores, 32).indices
    top_cm_idx = torch.topk(-scores, 32).indices
    W_mod = W.clone()
    W_mod[:, top_nl_idx] *= 1.25
    W_mod[:, top_cm_idx] *= 0.75
    m.weight.data.copy_(W_mod.to(orig_down[l].dtype))

run_full_r1('3. OPSD + Idea A (L28-31 32-neuron valve 1.25x)')
for l in range(28, 32):
    model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(orig_down[l])
"""

c = paramiko.SSHClient()
c.load_host_keys(str(pathlib.Path.home() / '.ssh/known_hosts'))
c.connect(
    '<A100_DEVBOX_IP>',
    username='dlisuser',
    key_filename=str(pathlib.Path.home() / '.ssh/codex-a100-sft'),
    allow_agent=False,
    look_for_keys=False,
    timeout=20,
)
sftp = c.open_sftp()
with sftp.open('/home/dlisuser/work/eval_beat_opsd.py', 'w') as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
_, o, e = c.exec_command('nohup /home/dlisuser/envs/sft-modern/bin/python -u /home/dlisuser/work/eval_beat_opsd.py > /home/dlisuser/work/eval_beat_opsd.log 2>&1 < /dev/null &')
c.close()
print('Launched eval_beat_opsd.py on A100!')
