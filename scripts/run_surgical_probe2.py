"""Upload and run v1 + v2.1 probe on surgical adapters on A100."""
import pathlib
import paramiko

REMOTE_SCRIPT = r"""import json, sys, torch
from pathlib import Path
from safetensors.torch import load_file
from peft import PeftModel

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from model_runtime import load_base
from adapter_runtime import logits_option

cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}
sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}

# Surgical candidates:
# 1. Graft-18: L0-17 Astra-SFT + L18-31 OPSD
# 2. Reverse-Graft-18: L0-17 OPSD + L18-31 Astra-SFT (to test if v1 collapse lives in L0-17 or L18-31!)
# 3. Magnitude-Matched SFT: L0-31 direction of d_SFT = W_sft - W_0, but rescaled layerwise to match ||d_OPSD||!
#    Wait! Earlier we saw ||d_SFT|| is 1.5x-1.7x larger than ||d_OPSD|| across layers! What if SFT's v1 collapse is simply step-size / norm overshoot relative to OPSD?
sd_graft_18 = {}
sd_rev_graft_18 = {}
sd_norm_matched = {}

for k in sds['sft2000']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])

    src18 = 'sft_astra' if l < 18 else 'opsd_astra'
    sd_graft_18[k] = sds[src18][k].clone()
    sd_graft_18[bk] = sds[src18][bk].clone()

    rev18 = 'opsd_astra' if l < 18 else 'sft_astra'
    sd_rev_graft_18[k] = sds[rev18][k].clone()
    sd_rev_graft_18[bk] = sds[rev18][bk].clone()

    # Norm-matched: W_new = W_0 + (||d_OPSD|| / ||d_SFT||) * (W_sft - W_0)
    A0, B0 = sds['sft2000'][k].float(), sds['sft2000'][bk].float()
    Ao, Bo = sds['opsd_astra'][k].float(), sds['opsd_astra'][bk].float()
    As, Bs = sds['sft_astra'][k].float(), sds['sft_astra'][bk].float()

    # QR-SVD for exact norms and rank-16 refactorization
    def qr_svd(B_cat, A_cat):
        Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
        Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
        U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
        return Qb, U, S, Vh, Qa

    _, _, s_opsd, _, _ = qr_svd(torch.cat([Bo, -B0], dim=1), torch.cat([Ao, A0], dim=0))
    _, _, s_sft, _, _ = qr_svd(torch.cat([Bs, -B0], dim=1), torch.cat([As, A0], dim=0))
    alpha = (torch.norm(s_opsd) / (torch.norm(s_sft) + 1e-12)).item()

    # W_new = (1 - alpha) W_0 + alpha W_sft
    Qb, U, S, Vh, Qa = qr_svd(torch.cat([B0 * (1.0 - alpha), Bs * alpha], dim=1), torch.cat([A0, As], dim=0))
    r = 16
    sqrt_S = torch.sqrt(S[:r])
    sd_norm_matched[bk] = ((Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)).contiguous()
    sd_norm_matched[k] = (sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)).contiguous()

base_4b, tok = load_base(path_4b, spec_4b, device='cuda:0')
ckpt_opsd = Path('/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter')
model = PeftModel.from_pretrained(base_4b, ckpt_opsd, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
model.eval()
model.config.use_cache = True

with (JOB_4B / 'eval-inputs.jsonl').open(encoding='utf8') as f:
    eval_rows = [json.loads(line) for line in f]

target_prefixes = ('service-4-', 'service-7-', 'service-18-', 'service-5-', 'JH02', 'JH11', 'JH12', 'JH16')
probe_cases = [r for r in eval_rows if r['run'] == 1 and any(r['case_id'].startswith(p) for p in target_prefixes)]

eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

def run_probe(sd_dict, name):
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))
    print(f'\n==================== {name} ====================', flush=True)
    for row in probe_cases:
        messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': row['prompt']}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
        ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device='cuda:0')
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg['eval_max_new_tokens'], do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
        pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
        lines = [l for l in pred.splitlines() if ':' in l]
        print(f"[{row['dataset']}/{row['case_id']}] ({len(lines)} lines): " + ' | '.join(lines), flush=True)

run_probe(sds['opsd_astra'], '1. Original OPSD-Astra')
run_probe(sds['sft_astra'], '2. Original Astra-SFT')
run_probe(sd_graft_18, '3. Graft-18 (L0-17 Astra-SFT + L18-31 OPSD)')
run_probe(sd_rev_graft_18, '4. Reverse-Graft-18 (L0-17 OPSD + L18-31 Astra-SFT)')
run_probe(sd_norm_matched, '5. Norm-Matched Astra-SFT (direction of d_SFT scaled to ||d_OPSD||)')
"""

c = paramiko.SSHClient()
c.load_host_keys(str(pathlib.Path.home() / '.ssh/known_hosts'))
c.connect(
    '40.75.2.87',
    username='dlisuser',
    key_filename=str(pathlib.Path.home() / '.ssh/codex-a100-sft'),
    allow_agent=False,
    look_for_keys=False,
    timeout=20,
)
sftp = c.open_sftp()
with sftp.open('/home/dlisuser/work/test_surgical_probes2.py', 'w') as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
_, o, e = c.exec_command('/home/dlisuser/envs/sft-modern/bin/python /home/dlisuser/work/test_surgical_probes2.py')
print(o.read().decode('utf-8', errors='replace'))
c.close()
