
import json, torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text().strip()
tok = AutoTokenizer.from_pretrained(path_4b, local_files_only=True)

index_file = Path(path_4b) / 'model.safetensors.index.json'
wmap = json.loads(index_file.read_text())['weight_map']
lm_key = 'model.language_model.embed_tokens.weight'
norm_key = 'model.language_model.norm.weight'
U_vocab = load_file(str(Path(path_4b) / wmap[lm_key]), device='cuda:0')[lm_key].float()
final_norm = load_file(str(Path(path_4b) / wmap[norm_key]), device='cuda:0')[norm_key].float()
U_eff = U_vocab * final_norm.unsqueeze(0)

paths = {
    'sft2000': '/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b/outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter/adapter_model.safetensors',
    'opd9b': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opd-9b/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'opsd_astra': '/home/dlisuser/work/tabgroup-opd-20261026/outputs/qwen3_5-4b-opsd-astra/checkpoints/step-000150/adapter/adapter_model.safetensors',
    'sft_astra': '/home/dlisuser/work/tabgroup-astra-sft-control-20261102/outputs/checkpoints/step-000150/adapter/adapter_model.safetensors',
}
sds = {k: load_file(v, device='cuda:0') for k, v in paths.items()}

scale = 2.0
def low_rank_svd(A1, B1, A0, B0):
    B_cat = torch.cat([B1, -B0], dim=1) * scale
    A_cat = torch.cat([A1, A0], dim=0)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    M = Rb @ Ra.T
    U_m, S, Vh_m = torch.linalg.svd(M)
    return S, Qb @ U_m, Vh_m @ Qa.T

def frob_inner(A1, B1, A2, B2):
    return ((B1.T @ B2) * (A1 @ A2.T)).sum().item()

# Key vocabulary directions (unit normalized)
target_toks = {'newline': 198, 'comma': 11, 'colon': 25, 'amp_space': 594, 'U_ungrouped': 52, 'ng_ungrouped': 938, 'roup_ungrouped': 858}
V_dirs = {}
for k, tid in target_toks.items():
    v = U_eff[tid]
    V_dirs[k] = v / torch.norm(v)

report = {'layers': []}
for l in range(32):
    layer_entry = {'layer': l, 'modules': {}}
    for mod in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'linear_attn.in_proj_qkv', 'linear_attn.in_proj_z', 'linear_attn.out_proj']:
        ak = f'base_model.model.model.language_model.layers.{l}.{mod}.lora_A.weight'
        if ak not in sds['sft2000']:
            continue
        bk = ak.replace('.lora_A.', '.lora_B.')
        A0, B0 = sds['sft2000'][ak].float(), sds['sft2000'][bk].float()
        A_opd, B_opd = sds['opd9b'][ak].float(), sds['opd9b'][bk].float()
        A_opsd, B_opsd = sds['opsd_astra'][ak].float(), sds['opsd_astra'][bk].float()
        A_sft, B_sft = sds['sft_astra'][ak].float(), sds['sft_astra'][bk].float()

        s_opsd, u_opsd, _ = low_rank_svd(A_opsd, B_opsd, A0, B0)
        s_sft, u_sft, _ = low_rank_svd(A_sft, B_sft, A0, B0)
        s_diff, u_diff, _ = low_rank_svd(A_sft, B_sft, A_opsd, B_opsd)

        n_opsd = torch.norm(s_opsd).item()
        n_sft = torch.norm(s_sft).item()
        n_diff = torch.norm(s_diff).item()

        B_d1 = torch.cat([B_opsd, -B0], dim=1) * scale
        A_d1 = torch.cat([A_opsd, A0], dim=0)
        B_d2 = torch.cat([B_sft, -B0], dim=1) * scale
        A_d2 = torch.cat([A_sft, A0], dim=0)
        cos_opsd_sft = frob_inner(A_d1, B_d1, A_d2, B_d2) / (n_opsd * n_sft + 1e-12)

        mod_info = {
            'norm_opsd': round(n_opsd, 4),
            'norm_sft': round(n_sft, 4),
            'norm_diff': round(n_diff, 4),
            'cos_opsd_sft': round(cos_opsd_sft, 4),
            'top1_ratio_opsd': round((s_opsd[0]**2 / (s_opsd**2).sum()).item(), 4),
            'top1_ratio_sft': round((s_sft[0]**2 / (s_sft**2).sum()).item(), 4),
        }
        # If residual stream writer (d_out == 2560), compute vocab projections
        if u_opsd.shape[0] == 2560:
            # Project d_OPSD and d_SFT onto key token directions: || v^T dW ||_2
            proj_opsd = {}
            proj_sft = {}
            for tk_name, v_dir in V_dirs.items():
                # || v^T (B_d A_d) ||_2 = sqrt( (v^T B_d) (A_d A_d^T) (B_d^T v) )
                vb1 = v_dir @ B_d1
                p1 = torch.sqrt(torch.clamp(vb1 @ (A_d1 @ A_d1.T) @ vb1, min=0.0)).item()
                vb2 = v_dir @ B_d2
                p2 = torch.sqrt(torch.clamp(vb2 @ (A_d2 @ A_d2.T) @ vb2, min=0.0)).item()
                proj_opsd[tk_name] = round(p1, 5)
                proj_sft[tk_name] = round(p2, 5)
            mod_info['vocab_proj_opsd'] = proj_opsd
            mod_info['vocab_proj_sft'] = proj_sft

            logits_opsd = U_eff @ u_opsd[:, 0]
            logits_sft = U_eff @ u_sft[:, 0]
            mod_info['u1_opsd_top_pos'] = [tok.decode([i]) for i in torch.topk(logits_opsd, 5).indices.tolist()]
            mod_info['u1_opsd_top_neg'] = [tok.decode([i]) for i in torch.topk(-logits_opsd, 5).indices.tolist()]
            mod_info['u1_sft_top_pos'] = [tok.decode([i]) for i in torch.topk(logits_sft, 5).indices.tolist()]
            mod_info['u1_sft_top_neg'] = [tok.decode([i]) for i in torch.topk(-logits_sft, 5).indices.tolist()]

        layer_entry['modules'][mod] = mod_info
    report['layers'].append(layer_entry)

Path('/home/dlisuser/work/weight_space_dossier.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('Saved weight_space_dossier.json successfully! Size:', Path('/home/dlisuser/work/weight_space_dossier.json').stat().st_size)
