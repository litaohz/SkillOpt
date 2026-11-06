"""Test Ideas A, B, and C (pure single-neuron / single-singular-value / QK-norm scalar edits) on A100."""
import pathlib
import paramiko

REMOTE_SCRIPT = r"""import json, statistics, sys, time, torch
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

# Effective unembedding U_eff = embed_tokens * final_norm
U_vocab = model.base_model.model.model.language_model.embed_tokens.weight.detach().float()
final_norm = model.base_model.model.model.language_model.norm.weight.detach().float()
U_eff = U_vocab * final_norm.unsqueeze(0)

v_nl = U_eff[198] / torch.norm(U_eff[198])   # '\n'
v_cm = U_eff[11] / torch.norm(U_eff[11])     # ','
v_diff_nl_cm = (v_nl - v_cm) / torch.norm(v_nl - v_cm)

# Diagnostic cases where sft2000 over-merges or sft_astra drops v1 groups:
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

# 0. Baseline sft2000 on diagnostic cases
load_adapter_sd(sds['sft2000'])
eval_cases('0. Baseline sft2000 (unchanged)', diag_cases)

# =========================================================================
# IDEA A: Single-Neuron / Top-K Neuron Column Scaling in L28..L31 mlp.down_proj on sft2000
# =========================================================================
# In each layer 28..31, find the neurons in base mlp.down_proj most aligned with (v_newline - v_comma)
# Wait: during forward pass of sft2000 on JH02 & JH12 (at the exact comma positions where it over-merges),
# which specific neurons in L24..L31 actually FIRE and write toward comma (',') vs newline ('\n')?
# Just like MIT Blog Experiment 8 (where Astra looked at activations on looping contexts to find the exact 64 neurons in L15 & L19)!
print('\n--- Probing active neurons in sft2000 on JH02 & JH12 (like Blog Exp 8) ---', flush=True)
# Let's run a single forward pass on JH02 and JH12 with sft2000's own generated output to see which MLP neurons write the comma-continuation direction!
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
with sftp.open('/home/dlisuser/work/probe_abc_step1.py', 'w') as f:
    f.write(REMOTE_SCRIPT)
sftp.close()
_, o, e = c.exec_command('/home/dlisuser/envs/sft-modern/bin/python /home/dlisuser/work/probe_abc_step1.py')
print(o.read().decode('utf-8', errors='replace'))
c.close()
