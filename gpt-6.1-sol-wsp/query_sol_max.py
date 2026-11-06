"""Query gpt-6.1-sol (effort=max) for Weight-Space Programming diagnosis and surgical recipes."""
import json
import pathlib
import urllib.request

ROOT = pathlib.Path('Q:/wiki/misc/tabgroup-opd-20261026')
env_path = pathlib.Path.home() / '.env'
token = [
    l.split('=', 1)[1].strip().strip('"').strip("'")
    for l in env_path.read_text(encoding='utf-8').splitlines()
    if l.startswith('EDGE_PROXY_TOKEN=')
][0]
dossier_text = (ROOT / 'weight_space_dossier.json').read_text(encoding='utf-8')

prompt = f"""You are performing a Weight-Space Programming & Mechanistic Analysis on Qwen/Qwen3.5-4B (32 layers, hybrid Linear-Attention + Full-Attention at layers 3,7,11,15,19,23,27,31; d_model=2560, LoRA r=16, alpha=32) for the Browser Tab Grouping task.

We have 4 exact LoRA adapters starting from the same 2000-step SFT checkpoint (`sft2000`):
1. `sft2000` (Base SFT): v1 Review P = 93.85%, Recall = 68.68%; v2.1 (hard multi-task) Review P = 90.14%, Recall = 48.00% (suffers from giant-bucket over-merging).
2. `opd9b` (Standard white-box OPD from 9B-SFT, 150 steps): v1 94.06% / 68.73%, v2.1 89.87% / 45.95% (no fix).
3. `opsd_astra` (On-Policy Self-Distillation with GPT-6-astra privileged context, 150 steps on 1,200 prompts):
   - v1: Review P = 97.99% (+4.14pp), Recall = 70.20% (+1.52pp)
   - v2.1: Review P = 92.36% (+2.22pp), Recall = 54.60% (+6.60pp)
   - Grouped/Ungrouped tab counts: v1 (1564 grouped, 238 ungrouped, 0.7% &-groups); v2.1 (1017 grouped, 104 ungrouped, 0% &-groups).
4. `sft_astra` (Standard Cross-Entropy SFT on the exact same 1,200 Astra reference solutions, 150 steps):
   - v1: Review P = 97.13%, Recall = 60.30% (-9.90pp collapse vs OPSD!)
   - v2.1: Review P = 94.21%, Recall = 57.86% (+3.26pp higher than OPSD!)
   - Grouped/Ungrouped tab counts: v1 (1457 grouped, 324 ungrouped, 2.5% &-groups); v2.1 (991 grouped, 121 ungrouped, 1.6% &-groups).
   - Note: In the 1,200 Astra training labels, 13.03% of group names contain `' & '` (vs 1.43% in original SFT labels) and Ungrouped tabs are 15.3% higher.

Below is the complete 32-layer weight-space SVD and vocabulary-projection dossier (`weight_space_dossier.json`) comparing `d_OPSD = W_opsd - W_sft2000` and `d_SFT = W_sft_astra - W_sft2000`:

{dossier_text}

Based on this weight-space evidence:
1. Diagnose the exact layer-by-layer mechanistic difference between `opsd_astra` and `sft_astra`:
   - Where in the 32 layers do `opsd_astra` and `sft_astra` share a common circuit (that fixes giant-bucket over-merging on v2.1)?
   - Where do they diverge (Layers 16-31), and what is `sft_astra` writing into the residual stream that causes the -9.90pp Recall collapse on `v1`?
2. Propose 3 concrete, mathematically precise **Zero-Training Weight-Space Surgical Edits** (producing a new LoRA adapter from the existing adapters / SVD directions / layerwise grafting or direction projection removal) designed to combine OPSD's v1 Recall preservation (~70%+) with Astra-SFT's v2.1 fine-grained splitting (~58%+). Give exact layer numbers, module names, and formulas."""

headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
body = {
    'model': 'gpt-6.1-sol',
    'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}]}],
    'max_output_tokens': 16384,
    'stream': True,
    'reasoning': {'effort': 'max'},
}
req = urllib.request.Request(
    'https://<INTERNAL_LLM_PROXY>/v1/responses',
    data=json.dumps(body).encode('utf-8'),
    headers=headers,
    method='POST',
)
deltas = []
completed_data = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw_line in r:
        line = raw_line.decode('utf-8', errors='replace').strip()
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if not payload or payload == '[DONE]':
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        etype = evt.get('type', '')
        if etype == 'response.output_text.delta':
            deltas.append(evt.get('delta', ''))
        elif etype == 'response.completed':
            completed_data = evt.get('response', evt)

ans = ''.join(deltas)
if not ans and completed_data:
    out_text = []
    for o in completed_data.get('output', []):
        if o.get('type') == 'message':
            for c in o.get('content', []):
                if c.get('type') == 'output_text':
                    out_text.append(c['text'])
    ans = '\n'.join(out_text)
(ROOT / 'sol_max_weight_diagnosis.md').write_text(ans, encoding='utf-8')
print('Saved sol_max_weight_diagnosis.md, chars:', len(ans))
print(ans[:4000])
