"""Build complete visualization data bundle locally from cached records + A100 abc_full_r1_results.json."""
import json
import pathlib
import paramiko

ROOT = pathlib.Path('Q:/wiki/misc/tabgroup-opd-20261026')
EVAL_DIR = pathlib.Path('Q:/wiki/misc/tabgroup-astra-sft-control-20261102/eval-batch')

import sys
sys.path.insert(0, str(EVAL_DIR / '_engine'))
from scoring import score

refs = {}
inputs = {}
for ds_dir in (EVAL_DIR / 'datasets').iterdir():
    if not ds_dir.is_dir():
        continue
    ds = ds_dir.name
    refs[ds] = {
        json.loads(l)['case_id']: json.loads(l)
        for l in (ds_dir / 'references.jsonl').read_text(encoding='utf-8').splitlines()
        if l.strip()
    }
    inputs[ds] = [
        json.loads(l)
        for l in (ds_dir / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()
        if l.strip()
    ]

all_data = json.loads((ROOT / 'abc_full_r1_results.json').read_text(encoding='utf-8'))

existing_dirs = {
    'sft2000': pathlib.Path('Q:/wiki/misc/tabgroup-nine-sft-devbox-20261016/downloaded/jobs/qwen3_5-4b/outputs/qwen3_5-4b/eval/records'),
    'opd9b': ROOT / 'downloaded_arm1/outputs/qwen3_5-4b-opd-9b/eval/records',
    'opsd_astra': ROOT / 'downloaded_arm2/outputs/qwen3_5-4b-opsd-astra/eval/records',
    'sft_astra': pathlib.Path('Q:/wiki/misc/tabgroup-astra-sft-control-20261102/downloaded/eval-batch/records'),
}

for mname, rdir in existing_dirs.items():
    recs = [json.loads(p.read_text(encoding='utf-8')) for p in rdir.glob('*.json')]
    res = {}
    for ds in sorted(refs.keys()):
        r1 = sorted([r for r in recs if r['dataset'] == ds and r['run'] == 1], key=lambda x: x['case_id'])
        grid = {(i['case_id'], i['run']): i for i in inputs[ds]}
        sc_list = []
        per_case = {}
        for r in r1:
            inp = grid[r['case_id'], r['run']]
            mapping = {int(k): v for k, v in inp['mapping'].items()}
            sc = score(r['prediction'], refs[ds][r['case_id']], mapping)
            sc_list.append(sc)
            per_case[r['case_id']] = {
                'precision': round(sc['precision'] * 100, 2),
                'recall': round(sc['recall'] * 100, 2),
                'prediction': r['prediction'].strip(),
            }
        init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100
        rec = sum(x['recall'] for x in sc_list) / len(sc_list) * 100
        res[ds] = {'init_p': round(init_p, 2), 'recall': round(rec, 2), 'cases': per_case}
    all_data[mname] = res

# Also fetch the 64 neuron indices and scores in L26..L31 from A100 so we can draw the exact neuron bar chart
REMOTE_NEURON_SCRIPT = r"""import json, sys, torch
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

refs = {ds_dir.name: {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()} for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}
inputs = {ds_dir.name: [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()] for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}

cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

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

v_nl = U_eff[198] / torch.norm(U_eff[198])
v_cm = U_eff[11] / torch.norm(U_eff[11])
v_diff = (v_nl - v_cm) / torch.norm(v_nl - v_cm)

neuron_spectrum = {}
for l in range(26, 32):
    W = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.detach().float()
    sc = v_diff @ W
    top_nl = torch.topk(sc, 16)
    top_cm = torch.topk(-sc, 16)
    neuron_spectrum[f'L{l}'] = {
        'top_newline_cols': [{'col': int(i), 'score': round(float(v), 4)} for i, v in zip(top_nl.indices, top_nl.values)],
        'top_comma_cols': [{'col': int(i), 'score': round(-float(v), 4)} for i, v in zip(top_cm.indices, top_cm.values)],
    }

# Also run the 7 showcase cases on graft_18, svd_blend_50_50, idea_b2_sft_top2_zero
paths = {
    'graft_18': '/home/dlisuser/work/tabgroup-weight-space-20261103/graft_18/adapter/adapter_model.safetensors',
    'svd_blend_50_50': '/home/dlisuser/work/tabgroup-weight-space-20261103/svd_blend_50_50/adapter/adapter_model.safetensors',
    'idea_b2_sft_top2_zero': '/home/dlisuser/work/tabgroup-weight-space-20261103/idea_b2_sft_top2_zero/adapter/adapter_model.safetensors',
}
showcase_ids = {'JH02', 'JH11', 'JH12', 'JH16', 'service-4-f43f114595c2', 'service-5-b48adde126b3', 'service-7-1c3a2d56655d', 'service-18-53ff1cc26ac3'}
showcase_runs = {}
p_map = dict(model.named_parameters())

for mname, spath in paths.items():
    sd = load_file(spath, device='cuda:0')
    for k, v in sd.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))
    showcase_runs[mname] = {}
    for ds in sorted(refs.keys()):
        for inp in inputs[ds]:
            if inp['run'] == 1 and inp['case_id'] in showcase_ids:
                messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': inp['prompt']}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
                ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device='cuda:0')
                with torch.inference_mode():
                    gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=cfg['eval_max_new_tokens'], do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
                pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
                mapping = {int(k): v for k, v in inp['mapping'].items()}
                sc = score(pred, refs[ds][inp['case_id']], mapping)
                showcase_runs[mname][inp['case_id']] = {
                    'precision': round(sc['precision']*100, 2),
                    'recall': round(sc['recall']*100, 2),
                    'prediction': pred.strip()
                }

Path('/home/dlisuser/work/showcase_extra.json').write_text(json.dumps({'neuron_spectrum': neuron_spectrum, 'showcase_runs': showcase_runs}, ensure_ascii=False, indent=2))
print('showcase_extra.json written!')
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
with sftp.open('/home/dlisuser/work/build_showcase_extra.py', 'w') as f:
    f.write(REMOTE_NEURON_SCRIPT)
sftp.close()
_, o, e = c.exec_command('/home/dlisuser/envs/sft-modern/bin/python /home/dlisuser/work/build_showcase_extra.py')
print(o.read().decode('utf-8', errors='replace'))
sftp = c.open_sftp()
sftp.get('/home/dlisuser/work/showcase_extra.json', str(ROOT / 'showcase_extra.json'))
sftp.close()
c.close()

extra = json.loads((ROOT / 'showcase_extra.json').read_text(encoding='utf-8'))
all_data['_extra'] = extra

showcase_ids = {'JH02', 'JH11', 'JH12', 'JH16', 'service-4-f43f114595c2', 'service-5-b48adde126b3', 'service-7-1c3a2d56655d', 'service-18-53ff1cc26ac3'}
showcase_meta = {}
for ds in sorted(refs.keys()):
    r1_map = {i['case_id']: i for i in inputs[ds] if i['run'] == 1}
    for cid, ref_obj in refs[ds].items():
        if cid in showcase_ids:
            showcase_meta[cid] = {
                'dataset': ds,
                'case_id': cid,
                'prompt': r1_map[cid]['prompt'],
                'mapping': r1_map[cid]['mapping'],
                'tabs': ref_obj['tabs'],
                'reference_alternatives': ref_obj['reference_alternatives'],
            }
all_data['_showcase_meta'] = showcase_meta

(ROOT / 'complete_viz_bundle.json').write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved complete_viz_bundle.json successfully!')
