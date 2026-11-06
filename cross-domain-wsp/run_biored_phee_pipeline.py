"""End-to-End Weight-Space Anatomy & Zero-Training LoRA Surgery on Public BioRED & PHEE Benchmarks."""
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
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================================
# DATASET 2: BioRED (Biomedical Document-Level Relation Extraction)
# ============================================================================
BIORED_RELS = [
    'association',
    'positive correlation',
    'negative correlation',
    'bind',
    'comparison',
    'cotreatment',
    'drug interaction',
    'conversion',
]
BIORED_CODE_MAP = {
    'association': 'B01 & association',
    'positive correlation': 'B02 & positive correlation',
    'negative correlation': 'B03 & negative correlation',
    'bind': 'B04 & bind',
    'comparison': 'B05 & comparison',
    'cotreatment': 'B06 & cotreatment',
    'drug interaction': 'B07 & drug interaction',
    'conversion': 'B08 & conversion',
}


def build_biored_datasets():
    for split_name, out_name in [('train', 'biored_parsed_train.jsonl'), ('test', 'biored_parsed_test.jsonl')]:
        raw_rows = [json.loads(l) for l in (RAW_DIR / f'biored_{split_name}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        parsed = []
        for idx, r in enumerate(raw_rows):
            text = r['input']
            sents = text.split(' . ')
            sent_bounds = []
            cur = 0
            for s in sents:
                sent_bounds.append((cur, cur + len(s) + 3))
                cur += len(s) + 3

            def get_sent_id(pos):
                for s_i, (st, en) in enumerate(sent_bounds):
                    if st <= pos < en:
                        return s_i
                return len(sent_bounds) - 1

            ent_map = {}
            ent_list = []
            for etype, elist in sorted(r['output']['entities'].items()):
                for e in sorted(elist, key=lambda x: x['start']):
                    key = e['text'].strip().lower()
                    if key not in ent_map:
                        eid = len(ent_map) + 1
                        ent_map[key] = (eid, e['text'].strip(), etype)
                        ent_list.append(f'E{eid}: {e["text"].strip()} ({etype})')

            if len(ent_map) < 6:
                continue

            full_rels = defaultdict(set)
            intra_rels = defaultdict(set)
            full_triples = set()
            intra_triples = set()

            for rtype, rlist in r['output']['relations'].items():
                if rtype not in BIORED_CODE_MAP:
                    continue
                for rel in rlist:
                    hk = rel['head']['text'].strip().lower()
                    tk = rel['tail']['text'].strip().lower()
                    if hk in ent_map and tk in ent_map and hk != tk:
                        h_id = ent_map[hk][0]
                        t_id = ent_map[tk][0]
                        full_rels[rtype].add((h_id, t_id))
                        full_triples.add((h_id, rtype, t_id))
                        if get_sent_id(rel['head']['start']) == get_sent_id(rel['tail']['start']):
                            intra_rels[rtype].add((h_id, t_id))
                            intra_triples.add((h_id, rtype, t_id))

            if len(full_triples) < 6 or len(intra_triples) < 2:
                continue

            prompt = (
                'Extract all biomedical relations between the numbered entities from the abstract.\n'
                'Use only relation names from the allowed relation list.\n'
                'Group extracted entity pairs by relation name, outputting one line per relation in the exact form:\n'
                '"Relation name: E_head->E_tail, E_head->E_tail"\n'
                'Finally output "Unrelated: high-confidence NA". No explanations or code fences.\n\n'
                f'Allowed relations: {", ".join(BIORED_RELS)}\n\n'
                f'Abstract:\n{text}\n\n'
                'Entities:\n' + '\n'.join(ent_list)
            )

            def fmt_target(rel_dict, use_code=False):
                lines = []
                for rt in sorted(rel_dict.keys()):
                    label = BIORED_CODE_MAP[rt] if use_code else rt
                    pairs = [f'E{h}->E{t}' for h, t in sorted(rel_dict[rt])]
                    lines.append(f'{label}: ' + ', '.join(pairs))
                lines.append('Unrelated: high-confidence NA')
                return '\n'.join(lines)

            parsed.append({
                'case_id': f'BR_{split_name}_{idx:03d}',
                'prompt': prompt,
                'num_entities': len(ent_map),
                'weak_target': fmt_target(intra_rels, use_code=False),
                'teacher_target': fmt_target(full_rels, use_code=True),
                'gold_clean_target': fmt_target(full_rels, use_code=False),
                'intra_triples': [list(x) for x in sorted(intra_triples)],
                'full_triples': [list(x) for x in sorted(full_triples)],
            })
        (DATA_DIR / out_name).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in parsed) + '\n', encoding='utf-8')
        print(f'Built BioRED {split_name}: {len(parsed)} abstracts', flush=True)


def parse_biored_triples(pred_text, num_ents):
    triples = set()
    allowed_set = {x.lower(): x for x in BIORED_RELS}
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```') or line.lower().startswith('unrelated') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rname = parts[0].strip()
        if '&' in rname:
            rname = rname.split('&', 1)[1].strip()
        rt = allowed_set.get(rname.lower())
        if not rt:
            continue
        for m in re.finditer(r'E(\d+)\s*->\s*E(\d+)', parts[1]):
            h, t = int(m.group(1)), int(m.group(2))
            if 1 <= h <= num_ents and 1 <= t <= num_ents and h != t:
                triples.add((h, rt, t))
    return triples


# ============================================================================
# DATASET 3: PHEE (Pharmacovigilance Hierarchical Event Argument Extraction)
# ============================================================================
PHEE_COARSE_ROLES = ['Subject', 'Treatment', 'Effect']
PHEE_FINE_ROLES = [
    'Subject',
    'Subject.Age',
    'Subject.Gender',
    'Subject.Population',
    'Subject.Disorder',
    'Subject.Race',
    'Treatment',
    'Treatment.Drug',
    'Combination.Drug',
    'Treatment.Disorder',
    'Treatment.Route',
    'Treatment.Dosage',
    'Treatment.Time_elapsed',
    'Treatment.Duration',
    'Treatment.Freq',
    'Effect',
]
PHEE_CODE_MAP = {r: f'R{i+1:02d} & {r}' for i, r in enumerate(PHEE_FINE_ROLES)}


def build_phee_datasets():
    for split_name, out_name in [('train', 'phee_parsed_train.jsonl'), ('test', 'phee_parsed_test.jsonl')]:
        raw_rows = [json.loads(l) for l in (RAW_DIR / f'phee_{split_name}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
        parsed = []
        for idx, r in enumerate(raw_rows):
            text = r['input']
            evs = r['output'].get('events', {})
            # Collect all unique spans and fine-grained roles across events
            span_map = {}
            span_list = []
            fine_role_to_ids = defaultdict(set)
            coarse_role_to_ids = defaultdict(set)

            for etype, elist in sorted(evs.items()):
                for ev in elist:
                    for arg in sorted(ev.get('arguments', []), key=lambda x: x['start']):
                        role = arg['role']
                        if role not in PHEE_CODE_MAP:
                            continue
                        stext = arg['text'].strip()
                        skey = stext.lower()
                        if skey not in span_map:
                            sid = len(span_map) + 1
                            span_map[skey] = (sid, stext)
                            span_list.append(f'S{sid}: {stext}')
                        sid = span_map[skey][0]
                        fine_role_to_ids[role].add(sid)
                        c_role = role.split('.')[0]
                        if c_role == 'Combination':
                            c_role = 'Treatment'
                        coarse_role_to_ids[c_role].add(sid)

            # Filter for rich multi-argument cases (at least 4 distinct spans and at least 4 fine roles)
            if len(span_map) < 4 or len(fine_role_to_ids) < 4:
                continue

            prompt = (
                'Assign the numbered clinical spans to their exact fine-grained pharmacovigilance event argument roles.\n'
                'Use only role names from the allowed role list.\n'
                'Output one line per role in the exact form:\n'
                '"Role name: S_id, S_id"\n'
                'Finally output "Unassigned: high-confidence NA". No explanations or code fences.\n\n'
                f'Allowed roles: {", ".join(PHEE_FINE_ROLES)}\n\n'
                f'Clinical Text:\n{text}\n\n'
                'Candidate Spans:\n' + '\n'.join(span_list)
            )

            def fmt_phee(rdict, use_code=False):
                lines = []
                for rl in sorted(rdict.keys()):
                    label = PHEE_CODE_MAP[rl] if use_code else rl
                    sids = [f'S{i}' for i in sorted(rdict[rl])]
                    lines.append(f'{label}: ' + ', '.join(sids))
                lines.append('Unassigned: high-confidence NA')
                return '\n'.join(lines)

            fine_pairs = [(rl, i) for rl, sids in fine_role_to_ids.items() for i in sorted(sids)]
            parsed.append({
                'case_id': f'PH_{split_name}_{idx:04d}',
                'prompt': prompt,
                'num_spans': len(span_map),
                'weak_target': fmt_phee(coarse_role_to_ids, use_code=False),
                'teacher_target': fmt_phee(fine_role_to_ids, use_code=True),
                'gold_clean_target': fmt_phee(fine_role_to_ids, use_code=False),
                'gold_pairs': [list(x) for x in sorted(fine_pairs)],
            })
        (DATA_DIR / out_name).write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in parsed) + '\n', encoding='utf-8')
        print(f'Built PHEE {split_name}: {len(parsed)} cases', flush=True)


def parse_phee_pairs(pred_text, num_spans):
    pairs = set()
    allowed_set = {x.lower(): x for x in PHEE_FINE_ROLES}
    for line in pred_text.splitlines():
        line = line.strip()
        if not line or line.startswith('```') or line.lower().startswith('unassigned') or line.lower() == 'none':
            continue
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        rname = parts[0].strip()
        if '&' in rname:
            rname = rname.split('&', 1)[1].strip()
        rl = allowed_set.get(rname.lower())
        if not rl:
            continue
        for m in re.finditer(r'S(\d+)', parts[1]):
            sid = int(m.group(1))
            if 1 <= sid <= num_spans:
                pairs.add((rl, sid))
    return pairs


def run_benchmark_experiment(
    bench_name,
    sys_prompt,
    train_rows,
    eval_rows,
    out_dir,
    parse_fn,
    gold_key,
    count_key,
    stop_word,
    steps_base=40,
    steps_sft=30,
    steps_opsd=25,
    model=None,
    tok=None,
    spec_4b=None,
    eos=None,
    pad_id=None,
    U_eff=None,
):
    device = 'cuda:0'
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]

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

    # Re-initialize LoRA to clean Kaiming/zero state before Stage 1
    from peft.tuners.lora import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters('default', init_lora_weights=True)
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.data = p.data.to(torch.float32)

    # 1. Stage 1: Base Weak SFT
    ckpt_base = out_dir / 'adapters/0_sft_base/adapter_model.safetensors'
    if not ckpt_base.exists():
        print(f'\n=== [{bench_name} 1/4] Training 0_sft_base ({steps_base} steps x 4 accum) ===', flush=True)
        seed_all(20261104)
        accum = 4
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 5, steps_base)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, steps_base + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for row in train_rows[(step - 1) * accum : step * accum]:
                rec, _ = encode_record(tok, spec_4b, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['weak_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 10 == 0 or step == 1:
                print(f'  [{bench_name} 0_sft_base] step {step:3d}/{steps_base} | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_base.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_base))
    sd_base = load_file(str(ckpt_base), device=device)

    # 2. Stage 2: Off-Policy Teacher SFT
    ckpt_sft = out_dir / 'adapters/2_sft_teacher/adapter_model.safetensors'
    if not ckpt_sft.exists():
        print(f'\n=== [{bench_name} 2/4] Training 2_sft_teacher ({steps_sft} steps x 4 accum) ===', flush=True)
        set_lora_sd(sd_base)
        seed_all(20261105)
        accum = 4
        opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 4, steps_sft)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        offset = steps_base * accum
        for step in range(1, steps_sft + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            batch_rows = [train_rows[(offset + (step - 1) * accum + j) % len(train_rows)] for j in range(accum)]
            for row in batch_rows:
                rec, _ = encode_record(tok, spec_4b, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['teacher_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 10 == 0 or step == 1:
                print(f'  [{bench_name} 2_sft_teacher] step {step:3d}/{steps_sft} | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_sft.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_sft))
    sd_sft = load_file(str(ckpt_sft), device=device)

    # 3. Stage 3: On-Policy Self-Distillation (OPSD)
    ckpt_opsd = out_dir / 'adapters/1_opsd_teacher/adapter_model.safetensors'
    if not ckpt_opsd.exists():
        print(f'\n=== [{bench_name} 3/4] Training 1_opsd_teacher ({steps_opsd} steps x 4 accum) ===', flush=True)
        set_lora_sd(sd_base)
        seed_all(20261105)
        teacher_lora_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if 'lora_' in n}
        accum = 4
        opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 4, steps_opsd)
        t0 = time.monotonic()
        offset = steps_base * accum
        for step in range(1, steps_opsd + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            batch_rows = [train_rows[(offset + (step - 1) * accum + j) % len(train_rows)] for j in range(accum)]
            for row in batch_rows:
                s_messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': row['prompt']}]
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
                        max_new_tokens=180,
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
                t_messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': t_user}]
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
            if step % 5 == 0 or step == 1:
                print(f'  [{bench_name} 1_opsd_teacher] step {step:3d}/{steps_opsd} | loss={sloss:.5f} | ({time.monotonic()-t0:.1f}s)', flush=True)
        ckpt_opsd.parent.mkdir(parents=True, exist_ok=True)
        save_file(get_lora_sd(), str(ckpt_opsd))
    sd_opsd = load_file(str(ckpt_opsd), device=device)

    # 4. Stage 4: 32-Layer Anatomy & Surgical Evaluation
    print(f'\n=== [{bench_name} 4/4] Computing 32-Layer QR-SVD Anatomy & Evaluating 9 Models ===', flush=True)
    v_amp = U_eff[594] / torch.norm(U_eff[594])
    v_nl = U_eff[198] / torch.norm(U_eff[198])
    v_cm = U_eff[11] / torch.norm(U_eff[11])
    stop_id = tok.encode(stop_word, add_special_tokens=False)[0]
    v_stop = U_eff[stop_id] / torch.norm(U_eff[stop_id])
    v_valve = ((v_nl + v_cm) / 2.0 - v_stop)
    v_valve = v_valve / torch.norm(v_valve)

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

    (out_dir / 'weight_dossier.json').write_text(json.dumps(layer_stats, indent=2), encoding='utf-8')
    for l in [0, 4, 8, 11, 12, 15, 18, 21, 24, 27, 30, 31]:
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

    def eval_candidate(name, sd_to_load, apply_idea_a=False, apply_idea_c=False):
        set_lora_sd(sd_to_load)
        orig_down = {}
        orig_qnorm = {}
        if apply_idea_a:
            for l in range(26, 32):
                m = model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer
                orig_down[l] = m.weight.data.clone()
                W = orig_down[l].float()
                scores = v_valve @ W
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
        tot_emit, tot_hits, tot_gold = 0, 0, 0
        preds = {}
        for inp in eval_rows:
            messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': inp['prompt']}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=device)
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=220, do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec_4b))
            pred = tok.decode(gen[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()
            pred_set = parse_fn(pred, inp[count_key])
            gold_set = {tuple(x) for x in inp[gold_key]}
            tot_emit += len(pred_set)
            tot_hits += len(pred_set & gold_set)
            tot_gold += len(gold_set)
            preds[inp['case_id']] = pred

        if apply_idea_a:
            for l, w in orig_down.items():
                model.base_model.model.model.language_model.layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
        if apply_idea_c:
            for l, w in orig_qnorm.items():
                model.base_model.model.model.language_model.layers[l].self_attn.q_norm.weight.data.copy_(w)

        p = tot_hits / max(tot_emit, 1) * 100
        r = tot_hits / max(tot_gold, 1) * 100
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"  {name:34s} | P={p:6.2f}% | R={r:6.2f}% | F1={f1:6.2f}% (Emitted={tot_emit}, Hits={tot_hits}/{tot_gold})", flush=True)
        return {'P': round(p, 2), 'R': round(r, 2), 'F1': round(f1, 2), 'emitted': tot_emit, 'hits': tot_hits, 'gold': tot_gold, 'preds': preds}

    all_res = {}
    for m_name, sd_m, ia, ic in [
        ('0_sft_base', sd_base, False, False),
        ('Idea_C (q_norm * 1.35x)', sd_base, False, True),
        ('Idea_A (L26-31 64-Col Valve)', sd_base, True, False),
        ('2_sft_teacher (Off-Policy SFT)', sd_sft, False, False),
        ('Idea_B2 (L18-31 sigma1,2=0)', sd_idea_b2, False, False),
        ('1_opsd_teacher (On-Policy OPSD)', sd_opsd, False, False),
        ('Rev_Graft_18 (L0-17 OPSD+SFT)', sd_rev_graft18, False, False),
        ('Graft_18 (L0-17 SFT+L18-31 OPSD)', sd_graft18, False, False),
        ('SVD_Blend_50_50 (Spectral Cons.)', sd_svd_blend, False, False),
    ]:
        all_res[m_name] = eval_candidate(m_name, sd_m, apply_idea_a=ia, apply_idea_c=ic)

    (out_dir / 'eval_results.json').write_text(json.dumps(all_res, indent=2), encoding='utf-8')
    print(f'=== {bench_name} COMPLETED! ===\n', flush=True)


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'

    build_biored_datasets()
    build_phee_datasets()

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')
    path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()

    base_4b, tok = load_base(path_4b, spec_4b, device=device)
    model, _ = attach_lora(base_4b, spec_4b, cfg)

    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    eos = sorted(set(eos))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    U_vocab = model.base_model.model.model.language_model.embed_tokens.weight.detach().float()
    final_norm = model.base_model.model.model.language_model.norm.weight.detach().float()
    U_eff = U_vocab * final_norm.unsqueeze(0)

    # Run BioRED
    biored_train = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    biored_eval = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:25]
    run_benchmark_experiment(
        bench_name='BioRED',
        sys_prompt='You extract biomedical entity relations from scientific abstracts.',
        train_rows=biored_train,
        eval_rows=biored_eval,
        out_dir=BASE_DIR / 'biored_exp',
        parse_fn=parse_biored_triples,
        gold_key='full_triples',
        count_key='num_entities',
        stop_word='Unrelated',
        steps_base=40,
        steps_sft=30,
        steps_opsd=25,
        model=model,
        tok=tok,
        spec_4b=spec_4b,
        eos=eos,
        pad_id=pad_id,
        U_eff=U_eff,
    )

    # Run PHEE
    phee_train = [json.loads(l) for l in (DATA_DIR / 'phee_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    phee_eval = [json.loads(l) for l in (DATA_DIR / 'phee_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:30]
    run_benchmark_experiment(
        bench_name='PHEE',
        sys_prompt='You extract fine-grained pharmacovigilance event arguments from clinical text.',
        train_rows=phee_train,
        eval_rows=phee_eval,
        out_dir=BASE_DIR / 'phee_exp',
        parse_fn=parse_phee_pairs,
        gold_key='gold_pairs',
        count_key='num_spans',
        stop_word='Unassigned',
        steps_base=40,
        steps_sft=30,
        steps_opsd=25,
        model=model,
        tok=tok,
        spec_4b=spec_4b,
        eos=eos,
        pad_id=pad_id,
        U_eff=U_eff,
    )


if __name__ == '__main__':
    main()
