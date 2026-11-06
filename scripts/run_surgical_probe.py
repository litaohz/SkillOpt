"""Upload and run surgical weight-space probe on A100."""
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

def ablate_sft_top1(A_sft, B_sft, A_opsd, B_opsd):
    B_diff = torch.cat([B_sft, -B_opsd], dim=1)
    A_diff = torch.cat([A_sft, A_opsd], dim=0)
    Qb, Rb = torch.linalg.qr(B_diff, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_diff.T, mode='reduced')
    U_m, S, Vh_m = torch.linalg.svd(Rb @ Ra.T)
    u1 = Qb @ U_m[:, 0]
    B_cat = torch.cat([B_sft, -u1.unsqueeze(1) * (u1 @ B_sft), u1.unsqueeze(1) * (u1 @ B_opsd)], dim=1)
    A_cat = torch.cat([A_sft, A_sft, A_opsd], dim=0)
    Qb2, Rb2 = torch.linalg.qr(B_cat, mode='reduced')
    Qa2, Ra2 = torch.linalg.qr(A_cat.T, mode='reduced')
    U2, S2, Vh2 = torch.linalg.svd(Rb2 @ Ra2.T)
    r = 16
    sqrt_S = torch.sqrt(S2[:r])
    B_new = (Qb2 @ U2[:, :r]) * sqrt_S.unsqueeze(0)
    A_new = sqrt_S.unsqueeze(1) * (Vh2[:r, :] @ Qa2.T)
    return A_new.contiguous(), B_new.contiguous()

sd_graft_18 = {}
sd_graft_16 = {}
sd_ablate = {}

for k in sds['sft2000']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])
    mod = '.'.join(parts[parts.index('layers') + 2 : -2])

    src18 = 'sft_astra' if l < 18 else 'opsd_astra'
    sd_graft_18[k] = sds[src18][k].clone()
    sd_graft_18[bk] = sds[src18][bk].clone()

    src16 = 'sft_astra' if l < 16 else 'opsd_astra'
    sd_graft_16[k] = sds[src16][k].clone()
    sd_graft_16[bk] = sds[src16][bk].clone()

    if l >= 18 and mod in ('mlp.down_proj', 'self_attn.o_proj', 'linear_attn.out_proj'):
        An, Bn = ablate_sft_top1(sds['sft_astra'][k].float(), sds['sft_astra'][bk].float(), sds['opsd_astra'][k].float(), sds['opsd_astra'][bk].float())
        sd_ablate[k] = An
        sd_ablate[bk] = Bn
    else:
        sd_ablate[k] = sds['sft_astra'][k].clone()
        sd_ablate[bk] = sds['sft_astra'][bk].clone()

print('Constructed 3 surgical adapters in memory. Testing on key diagnostic cases...', flush=True)
base_4b, tok = load_base(path_4b, spec_4b, device='cuda:0')
ckpt_opsd = Path('/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter')
model = PeftModel.from_pretrained(base_4b, ckpt_opsd, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
model.eval()
model.config.use_cache = True

with (JOB_4B / 'eval-inputs.jsonl').open(encoding='utf8') as f:
    eval_rows = [json.loads(line) for line in f]

probe_cases = [r for r in eval_rows if r['run'] == 1 and r['case_id'] in ('service-4', 'JH02', 'JH11', 'JH12', 'JH16')]

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
run_probe(sd_graft_18, '3. Surgical Graft-18 (L0-17 Astra-SFT + L18-31 OPSD)')
run_probe(sd_graft_16, '4. Surgical Graft-16 (L0-15 Astra-SFT + L16-31 OPSD)')
run_probe(sd_ablate, '5. Surgical Rank-1 Drift Ablation (L18-31 writer top-1 direction replaced)')
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
with sftp.open('/home/dlisuser/work/test_surgical_probes.py', 'w') as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
_, o, e = c.exec_command('/home/dlisuser/envs/sft-modern/bin/python /home/dlisuser/work/test_surgical_probes.py')
print(o.read().decode('utf-8', errors='replace'))
err = e.read().decode('utf-8', errors='replace')
if err:
    print('STDERR:', err)
c.close()
