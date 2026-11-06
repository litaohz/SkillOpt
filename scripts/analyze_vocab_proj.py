
import json, torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text().strip()
tok = AutoTokenizer.from_pretrained(path_4b, local_files_only=True)

# Load lm_head / embed_tokens from base model safetensors
index_file = Path(path_4b) / 'model.safetensors.index.json'
idx_data = json.loads(index_file.read_text())
wmap = idx_data['weight_map']
print('Sample base keys:', [k for k in wmap if 'head' in k or 'embed' in k or 'norm' in k])

lm_key = 'lm_head.weight' if 'lm_head.weight' in wmap else 'model.language_model.embed_tokens.weight'
norm_key = [k for k in wmap if 'language_model.norm.weight' in k][0]
lm_shard = load_file(str(Path(path_4b) / wmap[lm_key]), device='cuda:0')
U_vocab = lm_shard[lm_key].float() # (vocab_size, d_model)
norm_shard = load_file(str(Path(path_4b) / wmap[norm_key]), device='cuda:0')
final_norm = norm_shard[norm_key].float() # (d_model,)
# Effective unembedding with final RMSNorm weights
U_eff = U_vocab * final_norm.unsqueeze(0)

print('U_eff shape:', U_eff.shape)

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opd9b': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opd-9b/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}
sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}

# Residual-stream writer modules: mlp.down_proj, self_attn.o_proj, linear_attn.out_proj
scale = 2.0
def low_rank_svd(A1, B1, A0, B0):
    B_cat = torch.cat([B1, -B0], dim=1) * scale
    A_cat = torch.cat([A1, A0], dim=0)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    M = Rb @ Ra.T
    U_m, S, Vh_m = torch.linalg.svd(M)
    return S, Qb @ U_m, Vh_m @ Qa.T

# Key tokens to track in TabGroup:
key_strs = ['\n', ',', ', ', ':', ': ', 'Ungrouped', ' Ungrouped', '1', '2', '3']
key_ids = {}
for s in key_strs:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        key_ids[repr(s)] = ids[0]
print('Key token IDs:', key_ids)

for l in range(20, 32):
    for mod in ['mlp.down_proj', 'self_attn.o_proj', 'linear_attn.out_proj']:
        ak = f'base_model.model.model.language_model.layers.{l}.{mod}.lora_A.weight'
        if ak not in sds['sft2000']:
            continue
        bk = ak.replace('.lora_A.', '.lora_B.')
        A0, B0 = sds['sft2000'][ak].float(), sds['sft2000'][bk].float()
        A_opsd, B_opsd = sds['opsd_astra'][ak].float(), sds['opsd_astra'][bk].float()
        A_sft, B_sft = sds['sft_astra'][ak].float(), sds['sft_astra'][bk].float()

        s_opsd, u_opsd, _ = low_rank_svd(A_opsd, B_opsd, A0, B0)
        s_sft, u_sft, _ = low_rank_svd(A_sft, B_sft, A0, B0)
        # Difference between SFT_Astra and OPSD_Astra directly:
        s_diff, u_diff, _ = low_rank_svd(A_sft, B_sft, A_opsd, B_opsd)

        # Project top-1 left singular vector of (SFT - OPSD) onto vocabulary
        logits_u1 = U_eff @ u_diff[:, 0]
        top_pos = torch.topk(logits_u1, 6)
        top_neg = torch.topk(-logits_u1, 6)
        pos_toks = [repr(tok.decode([i])) for i in top_pos.indices.tolist()]
        neg_toks = [repr(tok.decode([i])) for i in top_neg.indices.tolist()]
        print(f'L{l:02d} {mod:20s} | ||SFT-OPSD||={torch.norm(s_diff):.4f} (top1={s_diff[0]:.4f}) | +dir: {pos_toks[:4]} | -dir: {neg_toks[:4]}')
