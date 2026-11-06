"""Ultra-Fast Batched Comprehensive Pure Single-Stage Suite Across Public Benchmarks (Re-DocRED, BioRED, PHEE):
1. SFT Step-Count Scaling (50 -> 100 -> 150 steps) with TRUE single continuous AdamW trajectory
   & Step-Count Paradox under Zero-Training Surgery (Idea A / B-1 / B-2 / C)
2. Fair Post-SFT Comparison Starting from Trained SFT (step 100):
   - Continued SFT (step 100 -> 150)
   - SFT -> On-Policy Self-Distillation (OPSD, +25 batched steps with 4x rollout batching)
   - SFT -> White-Box On-Policy Distillation from 9B-SFT Teacher (OPD 9B->4B, +25 batched steps)
   - Dual-Adapter Zero-Training Surgeries (Graft-18, SVD-Blend 50/50)
3. Cross-Scale & Cross-Architecture Verification:
   - Qwen/Qwen3.5-2B (24-layer hybrid Gated-DeltaNet)
   - mistralai/Ministral-3-8B-Instruct-2512-BF16 (34-layer pure full-attention Transformer)
"""
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer
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
SUITE_DIR = BASE_DIR / 'comprehensive_suite_v2'
SUITE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    'qwen3_5-2b': '/home/dlisuser/cache/sft-models/qwen3_5-2b/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc',
    'qwen3_5-4b': '/home/dlisuser/cache/sft-models/qwen3_5-4b/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'qwen3_5-9b': '/home/dlisuser/cache/sft-models/qwen3_5-9b/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a',
    'ministral-3-8b-instruct-2512-bf16': '/home/dlisuser/cache/sft-models/ministral-3-8b-instruct-2512-bf16/models--mistralai--Ministral-3-8B-Instruct-2512-BF16/snapshots/f6fae9795746f63c9be8344932f01275f3c63734',
}

rel_info = json.loads((RAW_DIR / 'docred_rel_info.json').read_text(encoding='utf-8'))
name_to_pid = {v.lower(): k for k, v in rel_info.items()}
BIORED_RELS = ['association', 'positive correlation', 'negative correlation', 'bind', 'comparison', 'cotreatment', 'drug interaction', 'conversion']
allowed_br = {x.lower(): x for x in BIORED_RELS}
PHEE_FINE_ROLES = [
    'Subject', 'Subject.Age', 'Subject.Gender', 'Subject.Population', 'Subject.Disorder', 'Subject.Race',
    'Treatment', 'Treatment.Drug', 'Combination.Drug', 'Treatment.Disorder', 'Treatment.Route',
    'Treatment.Dosage', 'Treatment.Time_elapsed', 'Treatment.Duration', 'Treatment.Freq', 'Effect',
]
allowed_ph = {x.lower(): x for x in PHEE_FINE_ROLES}


def parse_docred(pred_text, num_ents):
    triples = set()
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```') or line.lower().startswith('unrelated') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rname = parts[0].strip()
        pid = name_to_pid.get(rname.lower())
        if not pid:
            continue
        for m in re.finditer(r'E(\d+)\s*->\s*E(\d+)', parts[1]):
            h, t = int(m.group(1)) - 1, int(m.group(2)) - 1
            if 0 <= h < num_ents and 0 <= t < num_ents and h != t:
                triples.add((h, pid, t))
    return triples


def parse_biored_undir(pred_text, num_ents):
    triples = set()
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```') or line.lower().startswith('unrelated') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rt = allowed_br.get(parts[0].strip().lower())
        if not rt:
            continue
        for m in re.finditer(r'E(\d+)\s*->\s*E(\d+)', parts[1]):
            h, t = int(m.group(1)), int(m.group(2))
            if 1 <= h <= num_ents and 1 <= t <= num_ents and h != t:
                triples.add((min(h, t), rt, max(h, t)))
    return triples


def parse_phee(pred_text, num_spans):
    pairs = set()
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```') or line.lower().startswith('unassigned') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rl = allowed_ph.get(parts[0].strip().lower())
        if not rl:
            continue
        for m in re.finditer(r'S(\d+)', parts[1]):
            sid = int(m.group(1))
            if 1 <= sid <= num_spans:
                pairs.add((rl, sid))
    return pairs


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
        f"Using the privileged expert reference solution above, output the extracted structures in the exact required format:"
    )


def get_lora_sd(model):
    sd = {}
    for n, p in model.named_parameters():
        if 'lora_' in n:
            clean_k = n.replace('.lora_A.default.weight', '.lora_A.weight').replace('.lora_B.default.weight', '.lora_B.weight')
            sd[clean_k] = p.data.detach().clone()
    return sd


def set_lora_sd(model, sd):
    p_map = dict(model.named_parameters())
    for k, v in sd.items():
        pk = k.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        p_map[pk].data.copy_(v.to(p_map[pk].dtype))


def reset_lora_zero(model):
    seed_all(20261104)
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters('default', init_lora_weights=True)
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.data = p.data.to(torch.float32)


def make_idea_b(sd_in, cut_layer=18, k_kill=1):
    sd_out = {}
    for k in sd_in:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        if l >= cut_layer:
            As, Bs = sd_in[k].float(), sd_in[bk].float()
            Qb, Rb = torch.linalg.qr(Bs, mode='reduced')
            Qa, Ra = torch.linalg.qr(As.T, mode='reduced')
            U_d, S_d, Vh_d = torch.linalg.svd(Rb @ Ra.T)
            S_mod = S_d.clone()
            S_mod[:k_kill] = 0.0
            sqrt_S = torch.sqrt(S_mod)
            sd_out[bk] = ((Qb @ U_d) * sqrt_S.unsqueeze(0)).contiguous()
            sd_out[k] = (sqrt_S.unsqueeze(1) * (Vh_d @ Qa.T)).contiguous()
        else:
            sd_out[k], sd_out[bk] = sd_in[k].clone(), sd_in[bk].clone()
    return sd_out


def reset_rope_deltas(model):
    for mod in model.modules():
        if hasattr(mod, 'rope_deltas'):
            mod.rope_deltas = None


def eval_model_batched(
    model, tok, spec, sys_prompt, eval_rows, parse_fn, gold_fn, count_key,
    sd_to_load, v_valve=None, apply_idea_a=False, apply_idea_c=False,
    eos=None, pad_id=None, name='', is_phee=False, batch_size=25
):
    device = 'cuda:0'
    set_lora_sd(model, sd_to_load)
    lm_layers = model.base_model.model.model.language_model.layers
    num_layers = len(lm_layers)
    orig_down = {}
    orig_qnorm = {}

    if apply_idea_a and v_valve is not None:
        start_l = max(0, num_layers - 6)
        for l in range(start_l, num_layers):
            m = lm_layers[l].mlp.down_proj.base_layer
            orig_down[l] = m.weight.data.clone()
            W = orig_down[l].float().clone()
            scores = v_valve @ W
            top_emit = torch.topk(scores, 64).indices
            top_stop = torch.topk(-scores, 64).indices
            W[:, top_emit] *= 1.8
            W[:, top_stop] *= 0.0
            m.weight.data.copy_(W.to(orig_down[l].dtype))

    if apply_idea_c:
        for l in range(num_layers):
            sa = getattr(lm_layers[l], 'self_attn', None)
            if sa is not None and hasattr(sa, 'q_norm') and int(num_layers * 0.2) <= l <= int(num_layers * 0.65):
                qn = sa.q_norm
                orig_qnorm[l] = qn.weight.data.clone()
                qn.weight.data.copy_((orig_qnorm[l].float().clone() * 1.35).to(orig_qnorm[l].dtype))

    model.eval()
    model.config.use_cache = True
    tok.padding_side = 'left'
    tok.pad_token_id = pad_id

    texts = [
        tok.apply_chat_template(
            [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': inp['prompt']}],
            tokenize=False, add_generation_prompt=True, **spec['chat_template_kwargs']
        )
        for inp in eval_rows
    ]

    tot_emit, tot_hits, tot_gold = 0, 0, 0
    sub_emit, sub_hits, sub_gold = 0, 0, 0
    preds = {}
    t0 = time.monotonic()
    for i in range(0, len(eval_rows), batch_size):
        b_rows = eval_rows[i : i + batch_size]
        b_texts = texts[i : i + batch_size]
        enc = tok(b_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            reset_rope_deltas(model)
            gen = model.generate(
                **enc, max_new_tokens=220, do_sample=False, use_cache=True,
                eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec)
            )
            reset_rope_deltas(model)
        in_len = enc['input_ids'].shape[1]
        for j, inp in enumerate(b_rows):
            pred = tok.decode(gen[j, in_len:].tolist(), skip_special_tokens=True).strip()
            pred_set = parse_fn(pred, inp[count_key])
            gold_set = gold_fn(inp)
            tot_emit += len(pred_set)
            tot_hits += len(pred_set & gold_set)
            tot_gold += len(gold_set)
            if is_phee:
                ps = {x for x in pred_set if '.' in x[0]}
                gs = {x for x in gold_set if '.' in x[0]}
                sub_emit += len(ps)
                sub_hits += len(ps & gs)
                sub_gold += len(gs)
            preds[inp['case_id']] = pred

    for l, w in orig_down.items():
        lm_layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
    for l, w in orig_qnorm.items():
        lm_layers[l].self_attn.q_norm.weight.data.copy_(w)

    p = tot_hits / max(tot_emit, 1) * 100
    r = tot_hits / max(tot_gold, 1) * 100
    f1 = 2 * p * r / max(p + r, 1e-9)
    res = {'P': round(p, 2), 'R': round(r, 2), 'F1': round(f1, 2), 'emitted': tot_emit, 'hits': tot_hits, 'gold': tot_gold, 'preds': preds}
    if is_phee:
        sp = sub_hits / max(sub_emit, 1) * 100
        sr = sub_hits / max(sub_gold, 1) * 100
        sf1 = 2 * sp * sr / max(sp + sr, 1e-9)
        res.update({'Sub_P': round(sp, 2), 'Sub_R': round(sr, 2), 'Sub_F1': round(sf1, 2)})
        print(f"  {name:44s} | P={p:6.2f}% R={r:6.2f}% F1={f1:6.2f}% | SubF1={sf1:6.2f}% ({tot_hits}/{tot_gold}) [{time.monotonic()-t0:.1f}s]", flush=True)
    else:
        print(f"  {name:44s} | P={p:6.2f}% R={r:6.2f}% F1={f1:6.2f}% ({tot_hits}/{tot_gold}, emit={tot_emit}) [{time.monotonic()-t0:.1f}s]", flush=True)
    return res


def compute_valve_vector(model, tok, stop_word='Unrelated'):
    lm = model.base_model.model.model.language_model
    U_vocab = lm.embed_tokens.weight.detach().float()
    final_norm = lm.norm.weight.detach().float()
    U_eff = U_vocab * final_norm.unsqueeze(0)
    nl_ids = tok.encode('\n', add_special_tokens=False)
    cm_ids = tok.encode(',', add_special_tokens=False)
    st_ids = tok.encode(stop_word, add_special_tokens=False)
    v_nl = U_eff[nl_ids[-1]] / torch.norm(U_eff[nl_ids[-1]])
    v_cm = U_eff[cm_ids[-1]] / torch.norm(U_eff[cm_ids[-1]])
    v_stop = U_eff[st_ids[0]] / torch.norm(U_eff[st_ids[0]])
    v_valve = (v_nl + v_cm) / 2.0 - v_stop
    return v_valve / torch.norm(v_valve), v_stop


def run_4b_benchmark_suite(bench_name, sys_prompt, train_rows, eval_rows, out_dir, parse_fn, gold_fn, count_key, stop_word, is_phee, model, tok, spec_4b, eos, pad_id):
    device = 'cuda:0'
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    v_valve, v_stop = compute_valve_vector(model, tok, stop_word)

    ckpt_50 = out_dir / 'adapters/sft_step50.safetensors'
    ckpt_100 = out_dir / 'adapters/sft_step100.safetensors'
    ckpt_150 = out_dir / 'adapters/sft_step150.safetensors'
    ckpt_50.parent.mkdir(parents=True, exist_ok=True)

    # Train 1-to-150 continuous SFT ladder with continuous AdamW momentum!
    if not ckpt_150.exists() or not (out_dir / 'adapters/sft100_clean.flag').exists():
        print(f'\n=== [{bench_name}] 1. Training Continuous 150-Step SFT Ladder (Saving @ Step 50, 100, 150) ===', flush=True)
        reset_lora_zero(model)
        accum = 4
        total_steps = 150
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 8, total_steps)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = train_rows[((step - 1) * accum + j) % len(train_rows)]
                rec, _ = encode_record(tok, spec_4b, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == 1:
                print(f'  [{bench_name} SFT] step {step:3d}/150 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            if step == 50:
                save_file(get_lora_sd(model), str(ckpt_50))
            elif step == 100:
                save_file(get_lora_sd(model), str(ckpt_100))
            elif step == 150:
                save_file(get_lora_sd(model), str(ckpt_150))

    sd_50 = load_file(str(ckpt_50), device=device)
    sd_100 = load_file(str(ckpt_100), device=device)
    sd_150 = load_file(str(ckpt_150), device=device)

    # 2. Train Batched SFT_step100 -> OPSD (+25 steps, 4 rollouts generated in parallel per step!)
    ckpt_opsd = out_dir / 'adapters/sft100_plus_opsd25.safetensors'
    if not ckpt_opsd.exists():
        print(f'\n=== [{bench_name}] 2. Training Batched SFT_step100 -> OPSD (+25 steps from SFT_step100, lr=1.5e-5, temp=0.3) ===', flush=True)
        set_lora_sd(model, sd_100)
        teacher_lora_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if 'lora_' in n}
        accum = 4
        opsd_steps = 25
        opt = torch.optim.AdamW(params, lr=1.5e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 3, opsd_steps)
        tok.padding_side = 'left'
        tok.pad_token_id = pad_id
        t0 = time.monotonic()

        for step in range(1, opsd_steps + 1):
            opt.zero_grad(set_to_none=True)
            b_rows = [train_rows[((100 + step - 1) * accum + j) % len(train_rows)] for j in range(accum)]
            s_texts = [
                tok.apply_chat_template(
                    [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': r['prompt']}],
                    tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs']
                )
                for r in b_rows
            ]
            enc = tok(s_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
            in_len = enc['input_ids'].shape[1]

            model.eval()
            model.config.use_cache = True
            with torch.inference_mode():
                reset_rope_deltas(model)
                gen_out = model.generate(
                    **enc, max_new_tokens=160, do_sample=True, temperature=0.3, top_p=0.9, top_k=30,
                    use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b)
                )
                reset_rope_deltas(model)

            sloss = 0.0
            for j, row in enumerate(b_rows):
                # Extract non-pad generated tokens for sample j
                raw_gen = gen_out[j, in_len:].tolist()
                # Trim at first eos or pad
                gen_ids = []
                for tid in raw_gen:
                    gen_ids.append(tid)
                    if tid in eos:
                        break
                if len(gen_ids) <= 0:
                    continue

                s_prompt_ids = tok.encode(s_texts[j], add_special_tokens=False)
                L_s = len(s_prompt_ids)
                t_user = build_privileged_prompt(row['prompt'], row['gold_clean_target'])
                t_text = tok.apply_chat_template(
                    [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': t_user}],
                    tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs']
                )
                t_prompt_ids = tok.encode(t_text, add_special_tokens=False)
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
            if step % 5 == 0 or step == 1:
                print(f'  [{bench_name} SFT100->OPSD] step {step:2d}/{opsd_steps} | loss={sloss:.5f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        save_file(get_lora_sd(model), str(ckpt_opsd))
    sd_opsd = load_file(str(ckpt_opsd), device=device)
    (out_dir / 'adapters/sft100_clean.flag').write_text('ok')

    # 3. Compute 32-Layer Anatomy
    print(f'\n=== [{bench_name}] 3. Computing 32-Layer Anatomy (Step 50 vs 100 vs 150 & Post-SFT100 OPSD vs ContSFT) ===', flush=True)
    layer_stats = defaultdict(dict)
    for k in sorted(sd_100.keys()):
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        mod = '.'.join(parts[parts.index('layers') + 2 : parts.index('lora_A')])

        A50, B50 = sd_50[k].float(), sd_50[bk].float()
        A100, B100 = sd_100[k].float(), sd_100[bk].float()
        A150, B150 = sd_150[k].float(), sd_150[bk].float()
        Ao, Bo = sd_opsd[k].float(), sd_opsd[bk].float()

        _, Rb50 = torch.linalg.qr(B50 * 2.0, mode='reduced')
        _, Ra50 = torch.linalg.qr(A50.T, mode='reduced')
        S50 = torch.linalg.svdvals(Rb50 @ Ra50.T)

        _, Rb100 = torch.linalg.qr(B100 * 2.0, mode='reduced')
        _, Ra100 = torch.linalg.qr(A100.T, mode='reduced')
        S100 = torch.linalg.svdvals(Rb100 @ Ra100.T)

        _, Rb150 = torch.linalg.qr(B150 * 2.0, mode='reduced')
        _, Ra150 = torch.linalg.qr(A150.T, mode='reduced')
        S150 = torch.linalg.svdvals(Rb150 @ Ra150.T)

        dAo = torch.cat([Ao, A100], dim=0)
        dBo = torch.cat([Bo * 2.0, -B100 * 2.0], dim=1)
        dAs = torch.cat([A150, A100], dim=0)
        dBs = torch.cat([B150 * 2.0, -B100 * 2.0], dim=1)

        no = (frob_inner(dAo, dBo, dAo, dBo)) ** 0.5
        ns = (frob_inner(dAs, dBs, dAs, dBs)) ** 0.5
        cos = frob_inner(dAo, dBo, dAs, dBs) / max(no * ns, 1e-12)

        entry = {
            'sigma1_step50': round(S50[0].item(), 4),
            'sigma1_step100': round(S100[0].item(), 4),
            'sigma1_step150': round(S150[0].item(), 4),
            'norm_opsd_from100': round(no, 5),
            'norm_sft150_from100': round(ns, 5),
            'ratio_sft_opsd': round(ns / max(no, 1e-9), 3),
            'cos_post100': round(cos, 4),
        }
        if mod in ('mlp.down_proj', 'self_attn.o_proj', 'linear_attn.out_proj'):
            po_stop = torch.norm((v_stop @ dBo) @ dAo).item()
            ps_stop = torch.norm((v_stop @ dBs) @ dAs).item()
            entry['stop_proj_ratio'] = round(ps_stop / max(po_stop, 1e-9), 2)
        layer_stats[l][mod] = entry

    (out_dir / 'suite_weight_dossier.json').write_text(json.dumps(layer_stats, indent=2), encoding='utf-8')

    # Construct Graft-18 (L00-17 SFT150 + L18-31 OPSD) and SVD-Blend 50/50
    sd_graft18 = {}
    sd_svd_blend = {}
    for k in sd_150:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        src = sd_150 if l < 18 else sd_opsd
        sd_graft18[k], sd_graft18[bk] = src[k].clone(), src[bk].clone()
        Ab, Bb = svd_blend(sd_opsd[k].float(), sd_opsd[bk].float(), sd_150[k].float(), sd_150[bk].float(), 0.5, 0.5)
        sd_svd_blend[k], sd_svd_blend[bk] = Ab, Bb

    print(f'\n=== [{bench_name}] 4. Batched Evaluation of Step Ladder (50, 100, 150) + Surgeries + Post-SFT OPSD ===', flush=True)
    results = {}
    eval_list = [
        ('SFT_step50 (Base 50 steps)', sd_50, False, False),
        ('SFT_step50 + Idea_A (64-Col Valve)', sd_50, True, False),
        ('SFT_step50 + Idea_C (q_norm * 1.35x)', sd_50, False, True),
        ('SFT_step50 + Idea_B1 (L18-31 sigma1=0)', make_idea_b(sd_50, 18, 1), False, False),
        ('SFT_step100 (Converged 100 steps)', sd_100, False, False),
        ('SFT_step100 + Idea_A (64-Col Valve)', sd_100, True, False),
        ('SFT_step100 + Idea_C (q_norm * 1.35x)', sd_100, False, True),
        ('SFT_step100 + Idea_B1 (L18-31 sigma1=0)', make_idea_b(sd_100, 18, 1), False, False),
        ('SFT_step150 (Extended 150 steps)', sd_150, False, False),
        ('SFT_step150 + Idea_A (64-Col Valve)', sd_150, True, False),
        ('SFT_step150 + Idea_C (q_norm * 1.35x)', sd_150, False, True),
        ('SFT_step150 + Idea_B1 (L18-31 sigma1=0)', make_idea_b(sd_150, 18, 1), False, False),
        ('SFT_step150 + Idea_B2 (L18-31 sigma1,2=0)', make_idea_b(sd_150, 18, 2), False, False),
        ('SFT_step100 -> OPSD (+25 steps)', sd_opsd, False, False),
        ('Graft_18 (L0-17 SFT150 + L18-31 OPSD)', sd_graft18, False, False),
        ('SVD_Blend_50_50 (SFT150 + OPSD)', sd_svd_blend, False, False),
    ]
    for m_name, sd_m, ia, ic in eval_list:
        results[m_name] = eval_model_batched(
            model, tok, spec_4b, sys_prompt, eval_rows, parse_fn, gold_fn, count_key,
            sd_m, v_valve=v_valve, apply_idea_a=ia, apply_idea_c=ic,
            eos=eos, pad_id=pad_id, name=m_name, is_phee=is_phee, batch_size=len(eval_rows)
        )
    (out_dir / 'suite_eval_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def run_whitebox_opd_9b_to_4b(cfg, train_rows, eval_rows, sys_prompt, out_dir):
    device = 'cuda:0'
    ckpt_9b = out_dir / 'adapters/teacher_9b_sft100.safetensors'
    ckpt_opd = out_dir / 'adapters/sft100_plus_opd9b_25.safetensors'
    if (out_dir / 'opd_9b_to_4b_results.json').exists():
        return

    spec_9b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-9b')
    spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')

    if not ckpt_9b.exists():
        print('\n=== [BioRED White-Box OPD] 1. Loading Qwen3.5-9B & Training 100-Step Teacher SFT ===', flush=True)
        base_9b, tok_9b = load_base(MODEL_PATHS['qwen3_5-9b'], spec_9b, device=device)
        model_9b, _ = attach_lora(base_9b, spec_9b, cfg)
        reset_lora_zero(model_9b)
        params_9b = [p for p in model_9b.parameters() if p.requires_grad]
        accum = 4
        opt = torch.optim.AdamW(params_9b, lr=5e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 5, 100)
        model_9b.train()
        model_9b.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, 101):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = train_rows[((step - 1) * accum + j) % len(train_rows)]
                rec, _ = encode_record(tok_9b, spec_9b, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model_9b(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params_9b, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == 1:
                print(f'  [9B Teacher SFT] step {step:3d}/100 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        save_file(get_lora_sd(model_9b), str(ckpt_9b))
        eos_9b = sorted(set((list(model_9b.generation_config.eos_token_id) if isinstance(model_9b.generation_config.eos_token_id, (list, tuple)) else [model_9b.generation_config.eos_token_id]) + [tok_9b.eos_token_id]))
        pad_9b = tok_9b.pad_token_id or eos_9b[0]
        v_valve_9b, _ = compute_valve_vector(model_9b, tok_9b, 'Unrelated')
        res_9b = {}
        for m_name, sd_m, ia, ic in [
            ('Qwen3.5-9B Standard SFT (100 steps)', get_lora_sd(model_9b), False, False),
            ('Qwen3.5-9B + Idea_A (64-Col Valve, 0-Train)', get_lora_sd(model_9b), True, False),
            ('Qwen3.5-9B + Idea_C (q_norm * 1.35x, 0-Train)', get_lora_sd(model_9b), False, True),
            ('Qwen3.5-9B + Idea_B1 (L18-31 sigma1=0, 0-Train)', make_idea_b(get_lora_sd(model_9b), 18, 1), False, False),
        ]:
            res_9b[m_name] = eval_model_batched(
                model_9b, tok_9b, spec_9b, sys_prompt, eval_rows, parse_biored_undir,
                lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}, 'num_entities',
                sd_m, v_valve=v_valve_9b, apply_idea_a=ia, apply_idea_c=ic,
                eos=eos_9b, pad_id=pad_9b, name=m_name, batch_size=25
            )
        (out_dir / 'teacher_9b_eval.json').write_text(json.dumps(res_9b, indent=2), encoding='utf-8')
        del model_9b, base_9b
        gc.collect()
        torch.cuda.empty_cache()

    print('\n=== [BioRED White-Box OPD] 2. Co-loading 9B-SFT Teacher + 4B-SFT100 Student on 80GB A100 for Online White-Box OPD (9B -> 4B) ===', flush=True)
    base_9b, tok_9b = load_base(MODEL_PATHS['qwen3_5-9b'], spec_9b, device=device)
    teacher_9b, _ = attach_lora(base_9b, spec_9b, cfg)
    set_lora_sd(teacher_9b, load_file(str(ckpt_9b), device=device))
    teacher_9b.eval()
    for p in teacher_9b.parameters():
        p.requires_grad_(False)

    base_4b, tok_4b = load_base(MODEL_PATHS['qwen3_5-4b'], spec_4b, device=device)
    student_4b, _ = attach_lora(base_4b, spec_4b, cfg)
    sd_100 = load_file(str(out_dir / 'adapters/sft_step100.safetensors'), device=device)
    set_lora_sd(student_4b, sd_100)
    params_4b = [p for p in student_4b.parameters() if p.requires_grad]

    eos_4b = sorted(set((list(student_4b.generation_config.eos_token_id) if isinstance(student_4b.generation_config.eos_token_id, (list, tuple)) else [student_4b.generation_config.eos_token_id]) + [tok_4b.eos_token_id]))
    pad_4b = tok_4b.pad_token_id or eos_4b[0]
    tok_4b.padding_side = 'left'
    tok_4b.pad_token_id = pad_4b

    accum = 4
    opd_steps = 25
    opt = torch.optim.AdamW(params_4b, lr=1.5e-5, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, 3, opd_steps)
    t0 = time.monotonic()

    for step in range(1, opd_steps + 1):
        opt.zero_grad(set_to_none=True)
        b_rows = [train_rows[((100 + step - 1) * accum + j) % len(train_rows)] for j in range(accum)]
        s_texts = [
            tok_4b.apply_chat_template(
                [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': r['prompt']}],
                tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs']
            )
            for r in b_rows
        ]
        enc = tok_4b(s_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        in_len = enc['input_ids'].shape[1]

        student_4b.eval()
        student_4b.config.use_cache = True
        with torch.inference_mode():
            reset_rope_deltas(student_4b)
            gen_out = student_4b.generate(
                **enc, max_new_tokens=160, do_sample=True, temperature=0.3, top_p=0.9, top_k=30,
                use_cache=True, eos_token_id=eos_4b, pad_token_id=pad_4b, **logits_option(spec_4b)
            )
            reset_rope_deltas(student_4b)

        sloss = 0.0
        for j, row in enumerate(b_rows):
            raw_gen = gen_out[j, in_len:].tolist()
            gen_ids = []
            for tid in raw_gen:
                gen_ids.append(tid)
                if tid in eos_4b:
                    break
            if len(gen_ids) <= 0:
                continue

            s_prompt_ids = tok_4b.encode(s_texts[j], add_special_tokens=False)
            L_s = len(s_prompt_ids)
            full_seq = torch.tensor([s_prompt_ids + gen_ids], device=device, dtype=torch.long)

            with torch.no_grad():
                t_out = teacher_9b(input_ids=full_seq, attention_mask=torch.ones_like(full_seq), use_cache=False)
                t_logits = t_out.logits[:, L_s - 1 : -1, :].to(torch.float32).detach()
                del t_out

            student_4b.train()
            student_4b.config.use_cache = False
            s_out = student_4b(input_ids=full_seq, attention_mask=torch.ones_like(full_seq), use_cache=False)
            s_logits = s_out.logits[:, L_s - 1 : -1, :]

            loss = generalized_jsd_loss(s_logits, t_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1)
            (loss / accum).backward()
            sloss += loss.detach().item() / accum
            del s_out, s_logits, t_logits

        torch.nn.utils.clip_grad_norm_(params_4b, 1.0)
        opt.step()
        sched.step()
        if step % 5 == 0 or step == 1:
            print(f'  [BioRED White-Box OPD 9B->4B] step {step:2d}/{opd_steps} | loss={sloss:.5f} | ({time.monotonic()-t0:.1f}s)', flush=True)

    sd_opd = get_lora_sd(student_4b)
    save_file(sd_opd, str(ckpt_opd))

    del teacher_9b, base_9b
    gc.collect()
    torch.cuda.empty_cache()

    v_valve_4b, _ = compute_valve_vector(student_4b, tok_4b, 'Unrelated')
    sd_150 = load_file(str(out_dir / 'adapters/sft_step150.safetensors'), device=device)
    sd_graft_opd = {}
    sd_blend_opd = {}
    for k in sd_150:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        src = sd_150 if l < 18 else sd_opd
        sd_graft_opd[k], sd_graft_opd[bk] = src[k].clone(), src[bk].clone()
        Ab, Bb = svd_blend(sd_opd[k].float(), sd_opd[bk].float(), sd_150[k].float(), sd_150[bk].float(), 0.5, 0.5)
        sd_blend_opd[k], sd_blend_opd[bk] = Ab, Bb

    opd_res = {}
    for m_name, sd_m in [
        ('SFT_step100 -> WhiteBox_OPD_9B (+25 steps)', sd_opd),
        ('WhiteBox_OPD_9B + Idea_B1 (L18-31 sigma1=0)', make_idea_b(sd_opd, 18, 1)),
        ('Graft_18_OPD (L0-17 SFT150 + L18-31 OPD_9B)', sd_graft_opd),
        ('SVD_Blend_OPD (50/50 SFT150 + OPD_9B)', sd_blend_opd),
    ]:
        opd_res[m_name] = eval_model_batched(
            student_4b, tok_4b, spec_4b, sys_prompt, eval_rows, parse_biored_undir,
            lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}, 'num_entities',
            sd_m, v_valve=v_valve_4b, eos=eos_4b, pad_id=pad_4b, name=m_name, batch_size=25
        )
    (out_dir / 'opd_9b_to_4b_results.json').write_text(json.dumps(opd_res, indent=2), encoding='utf-8')
    del student_4b, base_4b
    gc.collect()
    torch.cuda.empty_cache()


def run_cross_model_verification(cfg, train_rows, eval_rows, sys_prompt, out_dir):
    device = 'cuda:0'
    cross_res = {}
    for mid, cut_l, steps in [
        ('qwen3_5-2b', 14, 100),
        ('ministral-3-8b-instruct-2512-bf16', 19, 100),
    ]:
        ckpt_m = out_dir / f'adapters/{mid}_sft{steps}.safetensors'
        spec_m = next(m for m in cfg['models'] if m['id'] == mid)
        print(f'\n=== [Cross-Model Verification] Loading {mid} (cut_layer=L{cut_l}) ===', flush=True)
        base_m, tok_m = load_base(MODEL_PATHS[mid], spec_m, device=device)
        model_m, _ = attach_lora(base_m, spec_m, cfg)
        params_m = [p for p in model_m.parameters() if p.requires_grad]

        if not ckpt_m.exists():
            reset_lora_zero(model_m)
            accum = 4
            opt = torch.optim.AdamW(params_m, lr=5e-5, weight_decay=0.01)
            sched = get_cosine_schedule_with_warmup(opt, 5, steps)
            model_m.train()
            model_m.config.use_cache = False
            t0 = time.monotonic()
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                sloss = 0.0
                for j in range(accum):
                    row = train_rows[((step - 1) * accum + j) % len(train_rows)]
                    rec, _ = encode_record(tok_m, spec_m, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
                    ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                    lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                    loss = model_m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                    (loss / accum).backward()
                    sloss += loss.detach().item() / accum
                torch.nn.utils.clip_grad_norm_(params_m, 1.0)
                opt.step()
                sched.step()
                if step % 25 == 0 or step == 1:
                    print(f'  [{mid} SFT] step {step:3d}/{steps} | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            save_file(get_lora_sd(model_m), str(ckpt_m))

        sd_m = load_file(str(ckpt_m), device=device)
        eos_m = model_m.generation_config.eos_token_id
        eos_m = list(eos_m) if isinstance(eos_m, (list, tuple)) else [eos_m] if eos_m is not None else []
        if tok_m.eos_token_id is not None:
            eos_m.append(tok_m.eos_token_id)
        eos_m = sorted(set(eos_m))
        pad_m = tok_m.pad_token_id if tok_m.pad_token_id is not None else eos_m[0]
        v_valve_m, _ = compute_valve_vector(model_m, tok_m, 'Unrelated')

        cross_res[mid] = {}
        for v_name, sd_v, ia, ic in [
            (f'{mid} Standard SFT ({steps} steps)', sd_m, False, False),
            (f'{mid} + Idea_A (64-Col Valve, 0-Train)', sd_m, True, False),
            (f'{mid} + Idea_C (q_norm * 1.35x, 0-Train)', sd_m, False, True),
            (f'{mid} + Idea_B1 (L{cut_l}+ sigma1=0, 0-Train)', make_idea_b(sd_m, cut_l, 1), False, False),
            (f'{mid} + Idea_B2 (L{cut_l}+ sigma1,2=0, 0-Train)', make_idea_b(sd_m, cut_l, 2), False, False),
        ]:
            cross_res[mid][v_name] = eval_model_batched(
                model_m, tok_m, spec_m, sys_prompt, eval_rows, parse_biored_undir,
                lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}, 'num_entities',
                sd_v, v_valve=v_valve_m, apply_idea_a=ia, apply_idea_c=ic,
                eos=eos_m, pad_id=pad_m, name=v_name, batch_size=25
            )
        del model_m, base_m
        gc.collect()
        torch.cuda.empty_cache()

    (out_dir / 'cross_model_results.json').write_text(json.dumps(cross_res, indent=2), encoding='utf-8')


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')

    base_4b, tok = load_base(MODEL_PATHS['qwen3_5-4b'], spec_4b, device=device)
    model, _ = attach_lora(base_4b, spec_4b, cfg)
    eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    biored_train = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    biored_eval = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:25]

    docred_train = [json.loads(l) for l in (DATA_DIR / 'docred_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    docred_eval = [json.loads(l) for l in (DATA_DIR / 'docred_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:20]

    phee_train = [json.loads(l) for l in (DATA_DIR / 'phee_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    phee_eval = [json.loads(l) for l in (DATA_DIR / 'phee_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:30]

    # 1. BioRED 4B Full Suite
    run_4b_benchmark_suite(
        bench_name='BioRED',
        sys_prompt='You extract biomedical entity relations from scientific abstracts.',
        train_rows=biored_train,
        eval_rows=biored_eval,
        out_dir=SUITE_DIR / 'biored',
        parse_fn=parse_biored_undir,
        gold_fn=lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']},
        count_key='num_entities',
        stop_word='Unrelated',
        is_phee=False,
        model=model, tok=tok, spec_4b=spec_4b, eos=eos, pad_id=pad_id,
    )

    # 2. Re-DocRED 4B Full Suite
    run_4b_benchmark_suite(
        bench_name='Re-DocRED',
        sys_prompt='You extract document-level entity relations.',
        train_rows=docred_train,
        eval_rows=docred_eval,
        out_dir=SUITE_DIR / 'docred',
        parse_fn=parse_docred,
        gold_fn=lambda inp: {tuple(x) for x in inp['re_triples']},
        count_key='num_entities',
        stop_word='Unrelated',
        is_phee=False,
        model=model, tok=tok, spec_4b=spec_4b, eos=eos, pad_id=pad_id,
    )

    # 3. PHEE 4B Full Suite
    run_4b_benchmark_suite(
        bench_name='PHEE',
        sys_prompt='You extract fine-grained pharmacovigilance event arguments from clinical text.',
        train_rows=phee_train,
        eval_rows=phee_eval,
        out_dir=SUITE_DIR / 'phee',
        parse_fn=parse_phee,
        gold_fn=lambda inp: {tuple(x) for x in inp['gold_pairs']},
        count_key='num_spans',
        stop_word='Unassigned',
        is_phee=True,
        model=model, tok=tok, spec_4b=spec_4b, eos=eos, pad_id=pad_id,
    )

    del model, base_4b
    gc.collect()
    torch.cuda.empty_cache()

    # 4. White-Box OPD (9B Teacher -> 4B Student) on BioRED
    run_whitebox_opd_9b_to_4b(
        cfg=cfg,
        train_rows=biored_train,
        eval_rows=biored_eval,
        sys_prompt='You extract biomedical entity relations from scientific abstracts.',
        out_dir=SUITE_DIR / 'biored',
    )

    # 5. Cross-Model Verification (Qwen3.5-2B & Ministral-3-8B pure full-attention Transformer) on BioRED
    run_cross_model_verification(
        cfg=cfg,
        train_rows=biored_train,
        eval_rows=biored_eval,
        sys_prompt='You extract biomedical entity relations from scientific abstracts.',
        out_dir=SUITE_DIR / 'biored',
    )

    print('\n=== ALL V2 COMPREHENSIVE EXPERIMENTS COMPLETED SUCCESSFULLY! ===', flush=True)


if __name__ == '__main__':
    main()
