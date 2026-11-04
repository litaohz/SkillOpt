"""Run Run-1 (45 v1 + 16 v2.1 = 61 cases) exact P/R benchmark across surgical weight-space adapters on A100."""
import pathlib
import paramiko

REMOTE_SCRIPT = r"""import json, re, statistics, sys, time, torch
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

# Load references and inputs for v1 and v2.1
refs = {}
inputs = {}
for ds_dir in (EVAL_DIR / 'datasets').iterdir():
    if not ds_dir.is_dir(): continue
    ds = ds_dir.name
    refs[ds] = {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()}
    inputs[ds] = [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()]

print('Loaded datasets:', {k: len(v) for k, v in refs.items()}, flush=True)

cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}
sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}

# Build surgical candidates:
# 1. Graft-16: L0-15 Astra-SFT + L16-31 OPSD
# 2. Graft-18: L0-17 Astra-SFT + L18-31 OPSD
# 3. Graft-20: L0-19 Astra-SFT + L20-31 OPSD
# 4. Reverse-Graft-18: L0-17 OPSD + L18-31 Astra-SFT
# 5. Subspace-Ortho-Merge: In every module, take d_OPSD + the component of d_SFT orthogonal to d_OPSD in layers 0..17, or 50/50 SVD merge of OPSD + Astra-SFT!
def svd_blend(A1, B1, A2, B2, w1=0.5, w2=0.5):
    B_cat = torch.cat([B1 * w1, B2 * w2], dim=1)
    A_cat = torch.cat([A1, A2], dim=0)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    r = 16
    sqrt_S = torch.sqrt(S[:r])
    B_new = (Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)
    A_new = sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)
    return A_new.contiguous(), B_new.contiguous()

candidates = {
    'graft_14 (L0-13 SFT + L14-31 OPSD)': {},
    'graft_18 (L0-17 SFT + L18-31 OPSD)': {},
    'graft_22 (L0-21 SFT + L22-31 OPSD)': {},
    'rev_graft_18 (L0-17 OPSD + L18-31 SFT)': {},
    'svd_blend_50_50 (0.5 OPSD + 0.5 SFT)': {},
}

for k in sds['sft2000']:
    if '.lora_A.' not in k: continue
    bk = k.replace('.lora_A.', '.lora_B.')
    parts = k.split('.')
    l = int(parts[parts.index('layers') + 1])

    for cut, cname in [(14, 'graft_14 (L0-13 SFT + L14-31 OPSD)'), (18, 'graft_18 (L0-17 SFT + L18-31 OPSD)'), (22, 'graft_22 (L0-21 SFT + L22-31 OPSD)')]:
        src = 'sft_astra' if l < cut else 'opsd_astra'
        candidates[cname][k] = sds[src][k].clone()
        candidates[cname][bk] = sds[src][bk].clone()

    rev_src = 'opsd_astra' if l < 18 else 'sft_astra'
    candidates['rev_graft_18 (L0-17 OPSD + L18-31 SFT)'][k] = sds[rev_src][k].clone()
    candidates['rev_graft_18 (L0-17 OPSD + L18-31 SFT)'][bk] = sds[rev_src][bk].clone()

    Ab, Bb = svd_blend(sds['opsd_astra'][k].float(), sds['opsd_astra'][bk].float(), sds['sft_astra'][k].float(), sds['sft_astra'][bk].float(), 0.5, 0.5)
    candidates['svd_blend_50_50 (0.5 OPSD + 0.5 SFT)'][k] = Ab
    candidates['svd_blend_50_50 (0.5 OPSD + 0.5 SFT)'][bk] = Bb

# First compute exact Run-1 scores of existing cached records for sft2000, opsd_astra, sft_astra
existing_dirs = {
    '0_sft2000': Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/eval/records'),
    '1_opsd_astra': Path('/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/eval/records'),
    '2_sft_astra': Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch/records'),
}

print('\n=== Run-1 Reference Baselines (from cached records) ===', flush=True)
for mname, rdir in existing_dirs.items():
    recs = [json.loads(p.read_text()) for p in rdir.glob('*.json')]
    for ds in sorted(refs.keys()):
        r1 = [r for r in recs if r['dataset'] == ds and r['run'] == 1]
        grid = {(i['case_id'], i['run']): i for i in inputs[ds]}
        sc_list = []
        for r in r1:
            inp = grid[r['case_id'], r['run']]
            mapping = {int(k): v for k, v in inp['mapping'].items()}
            sc = score(r['prediction'], refs[ds][r['case_id']], mapping)
            sc_list.append(sc)
        init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100
        rec = statistics.mean(x['recall'] for x in sc_list) * 100
        print(f'{mname:38s} | {ds:8s} (n={len(sc_list):2d}) | Init P = {init_p:6.2f}% | Recall = {rec:6.2f}%', flush=True)

base_4b, tok = load_base(path_4b, spec_4b, device='cuda:0')
ckpt_opsd = Path('/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter')
model = PeftModel.from_pretrained(base_4b, ckpt_opsd, is_trainable=False, autocast_adapter_dtype=False, local_files_only=True)
model.eval()
model.config.use_cache = True

eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

print('\n=== Evaluating 5 Zero-Training Surgical Candidates on Run-1 (45 v1 + 16 v2.1 = 61 cases) ===', flush=True)
for cname, sd_dict in candidates.items():
    t0 = time.monotonic()
    p_map = dict(model.named_parameters())
    for k, v in sd_dict.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))

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
        print(f'{cname:38s} | {ds:8s} (n={len(sc_list):2d}) | Init P = {init_p:6.2f}% | Recall = {rec:6.2f}% | ({time.monotonic()-t0:.1f}s)', flush=True)
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
with sftp.open('/home/dlisuser/work/eval_surgical_r1.py', 'w') as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
_, o, e = c.exec_command('nohup /home/dlisuser/envs/sft-modern/bin/python -u /home/dlisuser/work/eval_surgical_r1.py > /home/dlisuser/work/eval_surgical_r1.log 2>&1 < /dev/null &')
c.close()
print('Launched eval_surgical_r1.py on A100!')
