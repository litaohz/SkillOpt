"""Exact script that queried gpt-6.1-sol (effort=xhigh) and wrote sol_weight_diagnosis.md."""
import json, pathlib, urllib.request, time

ROOT = pathlib.Path('Q:/wiki/misc/tabgroup-opd-20261026')
env_path = pathlib.Path.home() / '.env'
token = [l.split('=',1)[1].strip().strip('"').strip("'") for l in env_path.read_text(encoding='utf-8').splitlines() if l.startswith('EDGE_PROXY_TOKEN=')][0]
dossier = json.loads((ROOT / 'weight_space_dossier.json').read_text(encoding='utf-8'))

lines = ['Layer | Module | cos(d_OPSD,d_SFT) | ||d_OPSD|| | ||d_SFT|| | top1_ratio(OPSD/SFT) | vocab_proj(& OPSD/SFT) | u1_SFT_top_pos | u1_OPSD_top_pos']
for l_obj in dossier['layers']:
    l = l_obj['layer']
    for mod in ['mlp.down_proj', 'self_attn.o_proj', 'linear_attn.out_proj', 'mlp.gate_proj']:
        if mod not in l_obj['modules']: continue
        m = l_obj['modules'][mod]
        if 'vocab_proj_opsd' in m:
            po, ps = m['vocab_proj_opsd'], m['vocab_proj_sft']
            lines.append(f"L{l:02d} | {mod:20s} | {m['cos_opsd_sft']:6.4f} | {m['norm_opsd']:6.4f} | {m['norm_sft']:6.4f} | {m['top1_ratio_opsd']:.3f}/{m['top1_ratio_sft']:.3f} | {po['amp_space']:.4f}/{ps['amp_space']:.4f} | {m['u1_sft_top_pos'][:3]} | {m['u1_opsd_top_pos'][:3]}")
        else:
            lines.append(f"L{l:02d} | {mod:20s} | {m['cos_opsd_sft']:6.4f} | {m['norm_opsd']:6.4f} | {m['norm_sft']:6.4f} | {m['top1_ratio_opsd']:.3f}/{m['top1_ratio_sft']:.3f} | - | - | -")

compact_table = '\n'.join(lines)
prompt = f'''We are testing Weight-Space Programming on Qwen/Qwen3.5-4B (32 layers, hybrid Linear-Attention + Full-Attention at layers 3,7,11,15,19,23,27,31; LoRA r=16, alpha=32) for Browser Tab Grouping.
We have 4 adapters starting from `sft2000`:
- `sft2000`: v1 93.85% P / 68.68% R; v2.1 90.14% P / 48.00% R (giant-bucket over-merging).
- `opsd_astra` (150 steps on-policy self-distillation with Astra reference): v1 97.99% P / 70.20% R; v2.1 92.36% P / 54.60% R.
- `sft_astra` (150 steps standard SFT on same 1,200 Astra answers): v1 97.13% P / 60.30% R (-9.90pp Recall drop!); v2.1 94.21% P / 57.86% R (+3.26pp higher Recall).
In the 1,200 Astra training labels, 13.03% of group names contain `' & '` (vs 1.43% in original SFT) and Ungrouped tabs are 15.3% higher.

Weight-space SVD & vocabulary projection table (`d_OPSD = W_opsd - W_0`, `d_SFT = W_sft - W_0`):
{compact_table}

1. Explain the exact mechanistic difference across layers (specifically why L08-L15 align up to cos=0.36 while L18-L31 diverge down to cos=0.05 with `sft_astra` top-1 singular vector writing `' &'`, `'学习与'`, `'Learning'` into the residual stream).
2. Propose 3 exact Zero-Training Weight-Space Surgical Edits to combine OPSD's v1 Recall (~70%+) and Astra-SFT's v2.1 splitting (~58%+).'''

headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
t0 = time.monotonic()
for eff in ['xhigh', 'high']:
    body = {'model': 'gpt-6.1-sol', 'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}], 'max_output_tokens': 16384, 'stream': True, 'reasoning': {'effort': eff}}
    req = urllib.request.Request('https://<INTERNAL_LLM_PROXY>/v1/responses', data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
    deltas = []
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            for raw_line in r:
                line = raw_line.decode('utf-8', errors='replace').strip()
                if line.startswith('data:'):
                    p = line[5:].strip()
                    if p and p != '[DONE]':
                        evt = json.loads(p)
                        if evt.get('type') == 'response.output_text.delta':
                            deltas.append(evt.get('delta', ''))
        ans = ''.join(deltas)
        if ans:
            (ROOT / 'sol_weight_diagnosis.md').write_text(ans, encoding='utf-8')
            print(f'gpt-6.1-sol ({eff}) succeeded in {time.monotonic()-t0:.1f}s! Chars:', len(ans))
            print(ans[:3000])
            break
    except Exception as e:
        print(f'gpt-6.1-sol ({eff}) failed:', e)
