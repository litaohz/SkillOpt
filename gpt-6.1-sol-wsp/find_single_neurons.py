
import json, torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text().strip()
tok = AutoTokenizer.from_pretrained(path_4b, local_files_only=True)
wmap = json.loads((Path(path_4b) / 'model.safetensors.index.json').read_text())['weight_map']

# Load base down_proj of Layer 27, 29, 30, 31 and U_eff
U_vocab = load_file(str(Path(path_4b) / wmap['model.language_model.embed_tokens.weight']), device='cuda:0')['model.language_model.embed_tokens.weight'].float()
final_norm = load_file(str(Path(path_4b) / wmap['model.language_model.norm.weight']), device='cuda:0')['model.language_model.norm.weight'].float()
U_eff = U_vocab * final_norm.unsqueeze(0) # (248320, 2560)

# Direction of newline (\n, id=198) minus comma (',', id=11)
v_newline = U_eff[198] / torch.norm(U_eff[198])
v_comma = U_eff[11] / torch.norm(U_eff[11])
v_ung = U_eff[938] / torch.norm(U_eff[938]) # 'ng' in Ungrouped
v_amp = U_eff[594] / torch.norm(U_eff[594]) # ' &'

# Check which single intermediate MLP columns in L30 and L31 down_proj align most with newline vs comma
for l in [28, 30, 31]:
    k = f'model.language_model.layers.{l}.mlp.down_proj.weight'
    W_down = load_file(str(Path(path_4b) / wmap[k]), device='cuda:0')[k].float() # (2560, d_ff)
    scores_nl_vs_comma = (v_newline - v_comma) @ W_down # (d_ff,)
    top_nl = torch.topk(scores_nl_vs_comma, 3)
    top_cm = torch.topk(-scores_nl_vs_comma, 3)
    print(f'L{l:02d} mlp.down_proj (shape {list(W_down.shape)}):')
    print('  Top newline-promoting neurons (cols):', list(zip(top_nl.indices.tolist(), [round(x,4) for x in top_nl.values.tolist()])))
    print('  Top comma-promoting (merge) neurons:', list(zip(top_cm.indices.tolist(), [round(x,4) for x in top_cm.values.tolist()])))
