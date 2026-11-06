"""End-to-End Weight-Space Anatomy & Zero-Training LoRA Surgery on Public Re-DocRED vs Orig-DocRED Benchmark."""
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import get_cosine_schedule_with_warmup

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from model_runtime import load_base, attach_lora
from adapter_runtime import logits_option
from training_common import seed_all
from prepare_tokens import encode_record

BASE_DIR = Path('/home/dlisuser/work/public-two-stage-20261104')
RAW_DIR = BASE_DIR / 'raw'
DATA_DIR = BASE_DIR / 'data'
OUT_DIR = BASE_DIR / 'docred_exp'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYS_PROMPT = 'You extract document-level entity relations.'

rel_info = json.loads((RAW_DIR / 'docred_rel_info.json').read_text(encoding='utf-8'))
name_to_pid = {v.lower(): k for k, v in rel_info.items()}


def parse_docred_triples(pred_text, num_ents):
    triples = set()
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        if line.lower().startswith('unrelated') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rname = parts[0].strip()
        if '&' in rname:
            rname = rname.split('&', 1)[1].strip()
        pid = name_to_pid.get(rname.lower())
        if not pid:
            continue
        for m in re.finditer(r'E(\d+)\s*->\s*E(\d+)', parts[1]):
            h, t = int(m.group(1)) - 1, int(m.group(2)) - 1
            if 0 <= h < num_ents and 0 <= t < num_ents and h != t:
                triples.add((h, pid, t))
    return triples


def score_docred(pred_text, row):
    pred_triples = parse_docred_triples(pred_text, row['num_entities'])
    gold_triples = {tuple(x) for x in row['re_triples']}
    hits = len(pred_triples & gold_triples)
    p = hits / len(pred_triples) if pred_triples else 0.0
    r = hits / len(gold_triples) if gold_triples else 0.0
    return {
        'emitted': len(pred_triples),
        'hits': hits,
        'gold': len(gold_triples),
        'precision': p,
        'recall': r,
    }


def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1):
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    if top_k is not None and top_k > 0:
        _, top_k_indices = torch.topk(t_logits, k=top_k, dim=-1)
        s_logits = torch.gather(s_logits, dim=-1, index=top_k_indices)
        t_logits = torch.gather(t_logits, dim=-1, index=top_k_indices)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    b = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
    mix_log_probs = torch.logsumexp(
        torch.stack([s_log_probs + torch.log1p(-b), t_log_probs + torch.log(b)]), dim=0
    )
    kl_t = F.kl_div(mix_log_probs, t_log_probs, reduction='none', log_target=True).sum(dim=-1)
    kl_s = F.kl_div(mix_log_probs, s_log_probs, reduction='none', log_target=True).sum(dim=-1)
    jsd = beta * kl_t + (1.0 - beta) * kl_s
    if token_clip is not None:
        jsd = jsd.clamp(max=token_clip)
    return jsd.mean()


def build_privileged_prompt(original_prompt: str, ref_answer: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"=== Privileged Expert Reference Solution (for teacher guidance only) ===\n"
        f"{ref_answer.strip()}\n"
        f"=== End Reference Solution ===\n\n"
        f"Using the privileged expert reference solution above, output the extracted relations in the exact required format:"
    )


def frob_inner(dA1, dB1, dA2, dB2):
    return ((dB1.T @ dB2) * (dA1 @ dA2.T)).sum().item()


def svd_blend(A1, B1, A2, B2, w1=0.5, w2=0.5, r=16):
    B_cat = torch.cat([B1 * w1, B2 * w2], dim=1)
    A_cat = torch.cat([A1, A2], dim=0)
    Qb, Rb = torch.linalg.qr(B_cat, mode='reduced')
    Qa, Ra = torch.linalg.qr(A_cat.T, mode='reduced')
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    sqrt_S = torch.sqrt(S[:r])
    B_new = ((Qb @ U[:, :r]) * sqrt_S.unsqueeze(0)).contiguous()
    A_new = (sqrt_S.unsqueeze(1) * (Vh[:r, :] @ Qa.T)).contiguous()
    return A_new, B_new


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
    path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

    train_rows = [json.loads(l) for l in (DATA_DIR / 'docred_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    eval_rows = [json.loads(l) for l in (DATA_DIR / 'docred_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:35]

    print(f'Loaded Re-DocRED: train={len(train_rows)}, eval={len(eval_rows)}', flush=True)

    base_4b, tok = load_base(path_4b, spec_4b, device=device)
    model, _ = attach_lora(base_4b, spec_4b, cfg)
    params = [p for p in model.parameters() if p.requires_grad]

    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    eos = sorted(set(eos))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    def get_lora_sd():
        sd = {}
        for n, p in model.named_parameters():
            if 'lora_' in n:
                clean_k = n.replace('.lora_A.default.weight', '.lora_A.weight').replace('.lora_B.default.weight', '.lora_B.weight')
                sd[clean_k] = p.data.detach().clone()
        return sd

    def set_lora_sd(sd):
        p_map = dict(model.named_parameters())
        for k, v in sd.items():
            pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
            p_map[pk].data.copy_(v.to(p_map[pk].dtype))

    # =========================================================================
    # STAGE 1: Train Base Incomplete-DocRED LoRA (0_sft_orig_docred)
    # =========================================================================
    ckpt_base = OUT_DIR / 'adapters/0_sft_orig_docred/adapter_model.safetensors'
    if not ckpt_base.exists():
        print('\n=== [1/4] Training 0_sft_orig_docred (Standard LoRA SFT on Incomplete Orig-DocRED, 60 steps x 4 accum = 240 docs) ===', flush=True)
        seed_all(20261104)
        steps_base = 60
        accum = 4
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 6, steps_base)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, steps_base + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for row in train_rows[(step - 1) * accum : step * accum]:
                rec, _ = encode_record(tok, spec_4b, SYS_PROMPT, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['weak_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 15 == 0 or step == 1:
                print(f'  [0_sft_orig_docred] step {step:3d}/{steps_base} | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_base.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_base))
    sd_base = load_file(str(ckpt_base), device=device)

    # =========================================================================
    # STAGE 2: Train Off-Policy Re-DocRED SFT (2_sft_redocred)
    # =========================================================================
    ckpt_sft = OUT_DIR / 'adapters/2_sft_redocred/adapter_model.safetensors'
    if not ckpt_sft.exists():
        print('\n=== [2/4] Training 2_sft_redocred (Off-Policy Re-DocRED SFT, 50 steps x 4 accum = 200 docs) ===', flush=True)
        set_lora_sd(sd_base)
        seed_all(20261105)
        steps_sft = 50
        accum = 4
        opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 5, steps_sft)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, steps_sft + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for row in train_rows[240 + (step - 1) * accum : 240 + step * accum]:
                rec, _ = encode_record(tok, spec_4b, SYS_PROMPT, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['teacher_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 10 == 0 or step == 1:
                print(f'  [2_sft_redocred] step {step:3d}/{steps_sft} | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_sft.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_sft))
    sd_sft = load_file(str(ckpt_sft), device=device)

    # =========================================================================
    # STAGE 3: Train On-Policy Self-Distillation (1_opsd_redocred)
    # =========================================================================
    ckpt_opsd = OUT_DIR / 'adapters/1_opsd_redocred/adapter_model.safetensors'
    if not ckpt_opsd.exists():
        print('\n=== [3/4] Training 1_opsd_redocred (On-Policy Self-Distillation on Re-DocRED, 50 steps x 4 accum = 200 docs) ===', flush=True)
        set_lora_sd(sd_base)
        seed_all(20261105)
        teacher_lora_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if 'lora_' in n}
        steps_opsd = 50
        accum = 4
        opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 5, steps_opsd)
        t0 = time.monotonic()
        for step in range(1, steps_opsd + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for row in train_rows[240 + (step - 1) * accum : 240 + step * accum]:
                s_messages = [{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': row['prompt']}]
                s_prompt_text = tok.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
                s_prompt_ids = tok.encode(s_prompt_text, add_special_tokens=False)
                L_s = len(s_prompt_ids)
                s_p_tensor = torch.tensor([s_prompt_ids], device=device, dtype=torch.long)

                model.eval()
                model.config.use_cache = True
                with torch.inference_mode():
                    gen_out = model.generate(
                        input_ids=s_p_tensor,
                        attention_mask=torch.ones_like(s_p_tensor),
                        max_new_tokens=320,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        top_k=50,
                        use_cache=True,
                        eos_token_id=eos,
                        pad_token_id=pad_id,
                        **logits_option(spec_4b),
                    )
                gen_ids = gen_out[0, L_s:].tolist()
                if len(gen_ids) <= 0:
                    continue

                t_user = build_privileged_prompt(row['prompt'], row['gold_clean_target'])
                t_messages = [{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': t_user}]
                t_prompt_text = tok.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
                t_prompt_ids = tok.encode(t_prompt_text, add_special_tokens=False)
                L_t = len(t_prompt_ids)

                t_seq = torch.tensor([t_prompt_ids + gen_ids], device=device, dtype=torch.long)
                s_seq = torch.tensor([s_prompt_ids + gen_ids], device=device, dtype=torch.long)

                saved_student = {}
                for n, p in model.named_parameters():
                    if n in teacher_lora_state:
                        saved_student[n] = p.data.clone()
                        p.data.copy_(teacher_lora_state[n])
                with torch.no_grad():
                    t_out = model(input_ids=t_seq, attention_mask=torch.ones_like(t_seq), use_cache=False)
                    t_logits = t_out.logits[:, L_t - 1 : -1, :].to(torch.float32).detach()
                    del t_out
                for n, p in model.named_parameters():
                    if n in saved_student:
                        p.data.copy_(saved_student[n])
                del saved_student

                model.train()
                model.config.use_cache = False
                s_out = model(input_ids=s_seq, attention_mask=torch.ones_like(s_seq), use_cache=False)
                s_logits = s_out.logits[:, L_s - 1 : -1, :]

                loss = generalized_jsd_loss(s_logits, t_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1)
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
                del s_out, s_logits, t_logits

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 10 == 0 or step == 1:
                print(f'  [1_opsd_redocred] step {step:3d}/{steps_opsd} | loss={sloss:.5f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_opsd.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_opsd))
    sd_opsd = load_file(str(ckpt_opsd), device=device)

    # =========================================================================
    # STAGE 4: 32-Layer Weight-Space Anatomy & Surgical Candidates on Re-DocRED
    # =========================================================================
    print('\n=== [4/4] Computing 32-Layer QR-SVD Anatomy & Evaluating Surgical Models on Re-DocRED ===', flush=True)
    U_vocab = model.base_model.model.model.language_model.embed_tokens.weight.detach().float()
    final_norm = model.base_model.model.model.language_model.norm.weight.detach().float()
    U_eff = U_vocab * final_norm.unsqueeze(0)

    v_amp = U_eff[594] / torch.norm(U_eff[594])     # ' &'
    v_nl = U_eff[198] / torch.norm(U_eff[198])      # '\n'
    v_cm = U_eff[11] / torch.norm(U_eff[11])        # ','
    # In Re-DocRED, continuing to emit more pairs/relations uses ',' and '\n', whereas stopping early emits 'Unrelated' (token 'Un')!
    un_id = tok.encode('Unrelated', add_special_tokens=False)[0]
    v_un = U_eff[un_id] / torch.norm(U_eff[un_id])
    v_emit_vs_stop = ((v_nl + v_cm) / 2.0 - v_un)
    v_emit_vs_stop = v_emit_vs_stop / torch.norm(v_emit_vs_stop)

    layer_stats = defaultdict(dict)
    for k in sorted(sd_base.keys()):
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        mod = '.'.join(parts[parts.index('layers') + 2 : parts.index('lora_A')])

        A0, B0 = sd_base[k].float(), sd_base[bk].float()
        Ao, Bo = sd_opsd[k].float(), sd_opsd[bk].float()
        As, Bs = sd_sft[k].float(), sd_sft[bk].float()

        dBo = torch.cat([Bo, -B0], dim=1) * 2.0
        dAo = torch.cat([Ao, A0], dim=0)
        dBs = torch.cat([Bs, -B0], dim=1) * 2.0
        dAs = torch.cat([As, A0], dim=0)

        no = (frob_inner(dAo, dBo, dAo, dBo)) ** 0.5
        ns = (frob_inner(dAs, dBs, dAs, dBs)) ** 0.5
        cos = frob_inner(dAo, dBo, dAs, dBs) / max(no * ns, 1e-12)

        entry = {'norm_opsd': round(no, 5), 'norm_sft': round(ns, 5), 'ratio_sft_opsd': round(ns / max(no, 1e-9), 3), 'cos': round(cos, 4)}
        if mod in ('mlp.down_proj', 'self_attn.o_proj', 'linear_attn.out_proj'):
            po = torch.norm((v_amp @ dBo) @ dAo).item()
            ps = torch.norm((v_amp @ dBs) @ dAs).item()
            entry['amp_ratio'] = round(ps / max(po, 1e-9), 2)
        layer_stats[l][mod] = entry

    (OUT_DIR / 'docred_weight_dossier.json').write_text(json.dumps(layer_stats, indent=2), encoding='utf-8')
    print('Re-DocRED Layerwise summary (mlp.down_proj & attn_in):')
    for l in range(32):
        dp = layer_stats[l].get('mlp.down_proj', {})
        ai = layer_stats[l].get('self_attn.q_proj') or layer_stats[l].get('linear_attn.in_proj_qkv', {})
        print(f"  L{l:02d} | attn_in cos={ai.get('cos',0):+.4f} | down_proj cos={dp.get('cos',0):+.4f}, norm_ratio={dp.get('ratio_sft_opsd',0):.2f}x, '&'_ratio={dp.get('amp_ratio',0):.2f}x", flush=True)

    sd_graft18 = {}
    sd_rev_graft18 = {}
    sd_svd_blend = {}
    sd_idea_b2 = {}

    for k in sd_base:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])

        src_g = sd_sft if l < 18 else sd_opsd
        src_r = sd_opsd if l < 18 else sd_sft
        sd_graft18[k], sd_graft18[bk] = src_g[k].clone(), src_g[bk].clone()
        sd_rev_graft18[k], sd_rev_graft18[bk] = src_r[k].clone(), src_r[bk].clone()

        Ab, Bb = svd_blend(sd_opsd[k].float(), sd_opsd[bk].float(), sd_sft[k].float(), sd_sft[bk].float(), 0.5, 0.5)
        sd_svd_blend[k], sd_svd_blend[bk] = Ab, Bb

        if l >= 18:
            A0, B0 = sd_base[k].float(), sd_base[bk].float()
            As, Bs = sd_sft[k].float(), sd_sft[bk].float()
            B_d = torch.cat([Bs, -B0], dim=1)
            A_d = torch.cat([As, A0], dim=0)
            Qb, Rb = torch.linalg.qr(B_d, mode='reduced')
            Qa, Ra = torch.linalg.qr(A_d.T, mode='reduced')
            U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)
            S_d[:2] = 0.0
            B_new_cat = torch.cat([B0, (Qb @ U_d) * S_d.unsqueeze(0)], dim=1)
            A_new_cat = torch.cat([A0, Vh_d @ Qa.T], dim=0)
            Qb2, Rb2 = torch.linalg.qr(B_new_cat, mode='reduced')
            Qa2, Ra2 = torch.linalg.qr(A_new_cat.T, mode='reduced')
            U2, S2, Vh2 = torch.linalg.svd(Rb2 @ Ra2.T)
            sqrt_S2 = torch.sqrt(S2[:16])
            sd_idea_b2[bk] = ((Qb2 @ U2[:, :16]) * sqrt_S2.unsqueeze(0)).contiguous()
            sd_idea_b2[k] = (sqrt_S2.unsqueeze(1) * (Vh2[:16, :] @ Qa2.T)).contiguous()
        else:
            sd_idea_b2[k], sd_idea_b2[bk] = sd_sft[k].clone(), sd_sft[bk].clone()

    def evaluate_docred(name, sd_to_load, apply_idea_a=False, apply_idea_c=False):
        set_lora_sd(sd_to_load)
        orig_down = {}
        orig_qnorm = {}
        if apply_idea_a:
            for l in range(26, 32):
                m = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer
                orig_down[l] = m.weight.data.clone()
                W = orig_down[l].float()
                scores = v_emit_vs_stop @ W
                top_emit = torch.topk(scores, 64).indices
                top_stop = torch.topk(-scores, 64).indices
                W[:, top_emit] *= 2.0
                W[:, top_stop] *= 0.0
                m.weight.data.copy_(W.to(orig_down[l].dtype))
        if apply_idea_c:
            for l in [7, 11, 15, 19]:
                qn = model.base_model.model.model.language_model.layers[l].self_attn.q_norm
                orig_qnorm[l] = qn.weight.data.clone()
                qn.weight.data.copy_((orig_qnorm[l].float() * 1.35).to(orig_qnorm[l].dtype))

        model.eval()
        model.config.use_cache = True
        sc_list = []
        preds = {}
        for inp in eval_rows:
            messages = [{'role': 'system', 'content': SYS_PROMPT}, {'role': 'user', 'content': inp['prompt']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=device)
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=384, do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
            pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            sc = score_docred(pred, inp)
            sc_list.append(sc)
            preds[inp['case_id']] = pred

        if apply_idea_a:
            for l, w in orig_down.items():
                model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
        if apply_idea_c:
            for l, w in orig_qnorm.items():
                model.base_model.model.model.language_model.layers[l].self_attn.q_norm.weight.data.copy_(w)

        micro_p = sum(x['hits'] for x in sc_list) / max(sum(x['emitted'] for x in sc_list), 1) * 100
        micro_r = sum(x['hits'] for x in sc_list) / max(sum(x['gold'] for x in sc_list), 1) * 100
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
        print(f"{name:34s} | Re-DocRED: P={micro_p:6.2f}% | R={micro_r:6.2f}% | F1={micro_f1:6.2f}% (Emitted={sum(x['emitted'] for x in sc_list)}, Hits={sum(x['hits'] for x in sc_list)})", flush=True)
        return {'P': round(micro_p, 2), 'R': round(micro_r, 2), 'F1': round(micro_f1, 2), 'preds': preds}

    all_eval = {}
    for m_name, sd_m, ia, ic in [
        ('0_sft_orig_docred (Incomplete SFT)', sd_base, False, False),
        ('Idea_C (q_norm * 1.35x)', sd_base, False, True),
        ('Idea_A (L26-31 64-Col Valve)', sd_base, True, False),
        ('2_sft_redocred (Off-Policy SFT)', sd_sft, False, False),
        ('Idea_B2 (L18-31 sigma1,2=0)', sd_idea_b2, False, False),
        ('1_opsd_redocred (On-Policy OPSD)', sd_opsd, False, False),
        ('Rev_Graft_18 (L0-17 OPSD+SFT)', sd_rev_graft18, False, False),
        ('Graft_18 (L0-17 SFT+L18-31 OPSD)', sd_graft18, False, False),
        ('SVD_Blend_50_50 (Spectral Cons.)', sd_svd_blend, False, False),
    ]:
        all_eval[m_name] = evaluate_docred(m_name, sd_m, apply_idea_a=ia, apply_idea_c=ic)

    (OUT_DIR / 'docred_eval_results.json').write_text(json.dumps(all_eval, indent=2), encoding='utf-8')
    print('\nALL RE-DOCRED EXPERIMENTS COMPLETED SUCCESSFULLY!', flush=True)


if __name__ == '__main__':
    main()
