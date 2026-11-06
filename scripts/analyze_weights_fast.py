
import json, torch
from collections import defaultdict
from pathlib import Path
from safetensors.torch import load_file

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opd9b': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opd-9b/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}

sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}
modules = {}
for k in sds['sft2000']:
    if '.lora_A.' in k:
        prefix = k.split('.lora_A.')[0]
        b_key = k.replace('.lora_A.', '.lora_B.')
        modules[prefix] = (k, b_key)

scale = 2.0
def get_AB(sd, ak, bk):
    return sd[ak].float(), sd[bk].float()

# Fast exact Frobenius norm and inner product of low-rank difference:
# Delta W = scale * (B1 A1 - B0 A0) = scale * [B1, -B0] @ [A1; A0] (rank <= 32)
def low_rank_svd(A1, B1, A0, B0):
    B_cat = torch.cat([B1, -B0], dim=1) * scale # (d_out, 2r)
    A_cat = torch.cat([A1, A0], dim=0)          # (2r, d_in)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    M = Rb @ Ra.T # (2r, 2r) = 32x32!
    U_m, S, Vh_m = torch.linalg.svd(M)
    return S, Qb @ U_m, Vh_m @ Qa.T

def frob_inner(A1, B1, A2, B2):
    # trace((B1 A1)^T (B2 A2)) = trace(A1^T (B1^T B2) A2) = sum((B1^T B2) * (A1 A2^T))
    return ((B1.T @ B2) * (A1 @ A2.T)).sum().item()

stats = []
for prefix, (ak, bk) in sorted(modules.items()):
    parts = prefix.split('.')
    layer_idx = int(parts[parts.index('layers') + 1])
    mod_name = '.'.join(parts[parts.index('layers') + 2:])
    A0, B0 = get_AB(sds['sft2000'], ak, bk)
    A_opd, B_opd = get_AB(sds['opd9b'], ak, bk)
    A_opsd, B_opsd = get_AB(sds['opsd_astra'], ak, bk)
    A_sft, B_sft = get_AB(sds['sft_astra'], ak, bk)

    s0, _, _ = low_rank_svd(A0, B0, torch.zeros_like(A0), torch.zeros_like(B0))
    s_opd, _, _ = low_rank_svd(A_opd, B_opd, A0, B0)
    s_opsd, u_opsd, vh_opsd = low_rank_svd(A_opsd, B_opsd, A0, B0)
    s_sft, u_sft, vh_sft = low_rank_svd(A_sft, B_sft, A0, B0)

    n0 = torch.norm(s0).item()
    n_opd = torch.norm(s_opd).item()
    n_opsd = torch.norm(s_opsd).item()
    n_sft = torch.norm(s_sft).item()

    # Cosine between d_opsd and d_sft using low-rank factors
    B_d1 = torch.cat([B_opsd, -B0], dim=1) * scale
    A_d1 = torch.cat([A_opsd, A0], dim=0)
    B_d2 = torch.cat([B_sft, -B0], dim=1) * scale
    A_d2 = torch.cat([A_sft, A0], dim=0)
    inner = frob_inner(A_d1, B_d1, A_d2, B_d2)
    cos_opsd_sft = inner / (n_opsd * n_sft + 1e-12)

    top1_opsd = (s_opsd[0]**2 / (s_opsd**2).sum()).item()
    top1_sft = (s_sft[0]**2 / (s_sft**2).sum()).item()
    # Cosine of top-1 left singular vector (output direction)
    u1_cos = abs(torch.dot(u_opsd[:, 0], u_sft[:, 0]).item())

    stats.append({
        'layer': layer_idx,
        'mod': mod_name,
        'n0': round(n0, 4),
        'n_opd': round(n_opd, 4),
        'n_opsd': round(n_opsd, 4),
        'n_sft': round(n_sft, 4),
        'cos': round(cos_opsd_sft, 4),
        'u1_cos': round(u1_cos, 4),
        'top1_opsd': round(top1_opsd, 4),
        'top1_sft': round(top1_sft, 4),
    })

by_mod = defaultdict(list)
for x in stats:
    by_mod[x['mod']].append(x)

print('=== Summary by Module Type ===')
for m, items in sorted(by_mod.items()):
    avg_opsd = sum(x['n_opsd'] for x in items)/len(items)
    avg_sft = sum(x['n_sft'] for x in items)/len(items)
    avg_opd = sum(x['n_opd'] for x in items)/len(items)
    avg_cos = sum(x['cos'] for x in items)/len(items)
    avg_top1_opsd = sum(x['top1_opsd'] for x in items)/len(items)
    avg_top1_sft = sum(x['top1_sft'] for x in items)/len(items)
    print(f'{m:25s} | cnt={len(items):2d} | d_OPD={avg_opd:.4f} | d_OPSD={avg_opsd:.4f} | d_SFT={avg_sft:.4f} | cos(OPSD,SFT)={avg_cos:.4f} | top1(OPSD/SFT)={avg_top1_opsd:.3f}/{avg_top1_sft:.3f}')

print('\n=== Layer-by-Layer Total Update Norm (d_OPSD vs d_SFT) ===')
by_layer = defaultdict(list)
for x in stats:
    by_layer[x['layer']].append(x)
for l in sorted(by_layer):
    items = by_layer[l]
    l_opsd = (sum(x['n_opsd']**2 for x in items))**0.5
    l_sft = (sum(x['n_sft']**2 for x in items))**0.5
    l_opd = (sum(x['n_opd']**2 for x in items))**0.5
    avg_cos = sum(x['cos'] for x in items)/len(items)
    attn_type = 'FullAttn' if any('self_attn' in x['mod'] for x in items) else 'LinAttn '
    print(f'L{l:02d} ({attn_type}) | ||d_OPD||={l_opd:.4f} | ||d_OPSD||={l_opsd:.4f} | ||d_SFT||={l_sft:.4f} | ratio(SFT/OPSD)={l_sft/l_opsd:.2f}x | avg_cos={avg_cos:.4f}')
