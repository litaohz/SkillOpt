"""Phase 2 Weight-Space Programming (WSP) Suite on Larger Base Model:
mistralai/Ministral-3-14B-Instruct-2512-BF16 (14B parameters, 40 pure full-attention layers, d_model=5120, d_ff=16384)
Evaluated across ALL 4 Heterogeneous Task Domains:
1. Mathematical Reasoning: GSM8K
2. Scientific Knowledge & Reasoning: SciKnowEval (Compound-Disease & Drug-Drug Interaction)
3. Biomedical Document Relation Extraction: BioRED
4. Real-World Industrial Agent: Browser TabGroup (v1 & v2.1, 61 sessions, including ARI/NMI clustering metrics)
"""
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from adapter_runtime import logits_option
from training_common import seed_all, target_paths
from prepare_tokens import encode_record

EVAL_DIR = Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch')
sys.path.insert(0, str(EVAL_DIR / '_engine'))
from scoring import score as tab_score

sys.path.insert(0, '/home/dlisuser/work/public-two-stage-20261104')
from run_comprehensive_suite_v2 import (
    DATA_DIR, parse_biored_undir, compute_valve_vector,
    get_lora_sd, set_lora_sd, reset_lora_zero, make_idea_b, reset_rope_deltas,
    svd_blend, generalized_jsd_loss
)
from run_wsp_phase1 import (
    WSP_DIR, make_idea_c_qproj, anatomize_adapter,
    train_and_run_wsp_pipeline, compute_target_nll
)
from audit_wsp_full_metrics import compute_ari_nmi, paired_bootstrap_f1

FOLDER_14B = Path('/home/dlisuser/cache/sft-models/ministral-3-14b-instruct-2512-bf16/models--mistralai--Ministral-3-14B-Instruct-2512-BF16/snapshots/3cea74c1ebaf5ce5f5a2553de470e2ceab825142')
MODEL_ID_14B = 'ministral-3-14b-instruct-2512-bf16'
SPEC_14B = {
    'id': MODEL_ID_14B,
    'model': 'mistralai/Ministral-3-14B-Instruct-2512-BF16',
    'kind': 'ministral',
    'dtype': 'bfloat16',
    'attention': 'sdpa',
    'chat_template_kwargs': {},
}


def load_14b_with_lora(cfg, device='cuda:0'):
    print(f'=== Loading 40-Layer 14B Base Model: {MODEL_ID_14B} (BF16 + SDPA) ===', flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        FOLDER_14B, local_files_only=True, attn_implementation='sdpa',
        dtype=torch.bfloat16, device_map={'': device}
    )
    if hasattr(base, 'language_model'):
        base.language_model.config._attn_implementation = 'sdpa'
    if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
        base.model.language_model.config._attn_implementation = 'sdpa'
    tok = AutoTokenizer.from_pretrained(FOLDER_14B, local_files_only=True, trust_remote_code=False)

    targets = target_paths(base, 'ministral')
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
            lora_dropout=cfg['lora_dropout'], target_modules=targets,
            task_type='CAUSAL_LM', bias='none'
        ),
        autocast_adapter_dtype=False
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    print(f'  Attached LoRA to {len(targets)} modules across 40 layers. VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB', flush=True)
    return model, tok


def eval_biored_14b_comprehensive(
    model, tok, spec, sys_prompt, eval_rows, sd_to_load,
    v_valve=None, apply_idea_a=False, eos=None, pad_id=None, name='', batch_size=25
):
    device = 'cuda:0'
    t0 = time.monotonic()
    set_lora_sd(model, sd_to_load)
    lm_layers = model.base_model.model.model.language_model.layers
    num_layers = len(lm_layers)
    orig_down = {}

    if apply_idea_a and v_valve is not None:
        start_l = max(0, num_layers - 6)  # L34-39 for 40L
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

    model.eval()
    model.config.use_cache = True
    tok.padding_side = 'left'
    tok.pad_token_id = pad_id

    nll, ppl = compute_target_nll(model, tok, spec, sys_prompt, eval_rows, max_samples=25)
    texts = [
        tok.apply_chat_template(
            [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': inp['prompt']}],
            tokenize=False, add_generation_prompt=True, **spec['chat_template_kwargs']
        )
        for inp in eval_rows
    ]

    tot_emit, tot_hits, tot_gold = 0, 0, 0
    rel_tp, rel_fp, rel_fn = defaultdict(int), defaultdict(int), defaultdict(int)
    case_stats = []
    preds = {}

    for i in range(0, len(eval_rows), batch_size):
        b_rows = eval_rows[i:i + batch_size]
        b_texts = texts[i:i + batch_size]
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
            out_ids = gen[j, in_len:].tolist()
            gen_toks = 0
            for tid in out_ids:
                gen_toks += 1
                if tid in eos:
                    break
            pred_txt = tok.decode(out_ids, skip_special_tokens=True).strip()
            pred_set = parse_biored_undir(pred_txt, inp['num_entities'])
            gold_set = {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}

            h = len(pred_set & gold_set)
            e = len(pred_set)
            g = len(gold_set)
            tot_hits += h
            tot_emit += e
            tot_gold += g

            for item in pred_set & gold_set:
                rel_tp[item[1]] += 1
            for item in pred_set - gold_set:
                rel_fp[item[1]] += 1
            for item in gold_set - pred_set:
                rel_fn[item[1]] += 1

            p_i = h / max(e, 1) * 100.0 if e > 0 else 0.0
            r_i = h / max(g, 1) * 100.0 if g > 0 else 100.0
            f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-9)
            case_stats.append({
                'case_id': inp['case_id'],
                'hits': h, 'emitted': e, 'gold': g,
                'sample_f1': f1_i, 'gen_tokens': gen_toks,
                'valid_format': int(e > 0 or 'unrelated' in pred_txt.lower()),
            })
            preds[inp['case_id']] = pred_txt

    for l, w in orig_down.items():
        lm_layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)

    p = tot_hits / max(tot_emit, 1) * 100.0
    r = tot_hits / max(tot_gold, 1) * 100.0
    f1 = 2 * p * r / max(p + r, 1e-9)

    all_rels = set(rel_tp.keys()) | set(rel_fp.keys()) | set(rel_fn.keys())
    macro_list = []
    for rt in all_rels:
        tp, fp, fn = rel_tp[rt], rel_fp[rt], rel_fn[rt]
        if tp + fn == 0:
            continue
        pr = tp / max(tp + fp, 1) * 100.0
        rc = tp / max(tp + fn, 1) * 100.0
        macro_list.append(2 * pr * rc / max(pr + rc, 1e-9))
    macro_f1 = statistics.mean(macro_list) if macro_list else 0.0
    sample_f1 = statistics.mean(x['sample_f1'] for x in case_stats)
    avg_toks = statistics.mean(x['gen_tokens'] for x in case_stats)

    print(
        f"  {name:48s} | P={p:5.2f}% R={r:5.2f}% MicroF1={f1:5.2f}% MacroF1={macro_f1:5.2f}% "
        f"SampleF1={sample_f1:5.2f}% | Toks={avg_toks:4.1f} NLL={nll:.3f} ({tot_hits}/{tot_gold}, emit={tot_emit}) [{time.monotonic()-t0:.1f}s]",
        flush=True
    )
    return {
        'Micro_P': round(p, 2), 'Micro_R': round(r, 2), 'Micro_F1': round(f1, 2),
        'Macro_F1': round(macro_f1, 2), 'Sample_Avg_F1': round(sample_f1, 2),
        'Avg_Gen_Tokens': round(avg_toks, 1), 'Gold_NLL': nll, 'Gold_PPL': ppl,
        'Total_Hits': tot_hits, 'Total_Emitted': tot_emit, 'Total_Gold': tot_gold,
        'cases': case_stats, 'preds': preds,
    }


def run_biored_14b(model, tok, eos, pad_id):
    device = 'cuda:0'
    out_dir = WSP_DIR / 'biored_14b'
    out_dir.mkdir(parents=True, exist_ok=True)
    adap_dir = out_dir / 'adapters'
    adap_dir.mkdir(parents=True, exist_ok=True)

    biored_train = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    biored_eval = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:25]
    sys_prompt = 'You extract biomedical entity relations from scientific abstracts.'

    ckpt_100 = adap_dir / f'{MODEL_ID_14B}_sft100.safetensors'
    ckpt_150 = adap_dir / f'{MODEL_ID_14B}_sft150.safetensors'
    ckpt_200 = adap_dir / f'{MODEL_ID_14B}_sft200.safetensors'
    params = [p for p in model.parameters() if p.requires_grad]

    if not ckpt_200.exists():
        print(f'\n=== [BioRED | {MODEL_ID_14B}] Training 200-Step SFT Ladder (Saving @ 100, 150, 200) ===', flush=True)
        reset_lora_zero(model)
        accum = 4
        total_steps = 200
        opt = torch.optim.AdamW(params, lr=8e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 10, total_steps)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = biored_train[((step - 1) * accum + j) % len(biored_train)]
                rec, _ = encode_record(tok, SPEC_14B, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == 1:
                print(f'  [BioRED | {MODEL_ID_14B} SFT] step {step:3d}/200 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            if step == 100:
                save_file(get_lora_sd(model), str(ckpt_100))
            elif step == 150:
                save_file(get_lora_sd(model), str(ckpt_150))
            elif step == 200:
                save_file(get_lora_sd(model), str(ckpt_200))

    sd_100 = load_file(str(ckpt_100), device=device)
    sd_150 = load_file(str(ckpt_150), device=device)
    sd_200 = load_file(str(ckpt_200), device=device)

    v_valve, _ = compute_valve_vector(model, tok, 'Unrelated')
    variants = {}
    # For 40 layers: cut_l = 22 (55% depth), late_l = 28 (70% depth), mid_l = L09-L24
    for tag, sd_cur in [('Step100', sd_100), ('Step150', sd_150), ('Step200', sd_200)]:
        variants[f'{tag} (Standard)'] = eval_biored_14b_comprehensive(
            model, tok, SPEC_14B, sys_prompt, biored_eval, sd_cur, v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'14B {tag} (Standard)'
        )
        variants[f'{tag} + WSP_Valve (Idea_A, L34-39)'] = eval_biored_14b_comprehensive(
            model, tok, SPEC_14B, sys_prompt, biored_eval, sd_cur, v_valve=v_valve, apply_idea_a=True, eos=eos, pad_id=pad_id, name=f'14B {tag} + WSP_Valve (Idea_A)'
        )
        variants[f'{tag} + WSP_MidAttn (Idea_C, L09-24 q_proj*1.25x)'] = eval_biored_14b_comprehensive(
            model, tok, SPEC_14B, sys_prompt, biored_eval, make_idea_c_qproj(sd_cur, 9, 24, 1.25), v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'14B {tag} + WSP_MidAttn (Idea_C)'
        )
        variants[f'{tag} + WSP_SpecNull (L22+ sigma1=0)'] = eval_biored_14b_comprehensive(
            model, tok, SPEC_14B, sys_prompt, biored_eval, make_idea_b(sd_cur, 22, 1), v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'14B {tag} + WSP_SpecNull (L22+ s1=0)'
        )
        variants[f'{tag} + WSP_SpecNull_Late (L28+ sigma1=0)'] = eval_biored_14b_comprehensive(
            model, tok, SPEC_14B, sys_prompt, biored_eval, make_idea_b(sd_cur, 28, 1), v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'14B {tag} + WSP_SpecNull_Late (L28+ s1=0)'
        )

    from run_wsp_phase1 import add_significance_rel
    add_significance_rel(variants, 'Step100 (Standard)')
    add_significance_rel(variants, 'Step150 (Standard)')
    add_significance_rel(variants, 'Step200 (Standard)')

    (out_dir / f'{MODEL_ID_14B}_biored_wsp_results.json').write_text(json.dumps(variants, indent=2), encoding='utf-8')


def run_tabgroup_14b(model, tok, eos, pad_id):
    device = 'cuda:0'
    out_dir = WSP_DIR / 'tabgroup_14b'
    out_dir.mkdir(parents=True, exist_ok=True)
    adap_dir = out_dir / 'adapters'
    adap_dir.mkdir(parents=True, exist_ok=True)

    sys_prompt = 'You group browser tabs by topic.'
    astra_rows = [json.loads(l) for l in Path('/home/dlisuser/work/tabgroup-opd-20261026/astra-train-1200.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    refs = {ds_dir.name: {json.loads(l)['case_id']: json.loads(l) for l in (ds_dir / 'references.jsonl').read_text().splitlines() if l.strip()} for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}
    inputs = {ds_dir.name: [json.loads(l) for l in (ds_dir / 'inputs.jsonl').read_text().splitlines() if l.strip()] for ds_dir in (EVAL_DIR / 'datasets').iterdir() if ds_dir.is_dir()}

    ckpt_100 = adap_dir / f'{MODEL_ID_14B}_tabgroup_sft100.safetensors'
    ckpt_150 = adap_dir / f'{MODEL_ID_14B}_tabgroup_sft150.safetensors'
    params = [p for p in model.parameters() if p.requires_grad]

    if not ckpt_150.exists():
        print(f'\n=== [TabGroup | {MODEL_ID_14B}] Training 150-Step SFT Ladder (Saving @ 100, 150) ===', flush=True)
        reset_lora_zero(model)
        accum = 4
        total_steps = 150
        opt = torch.optim.AdamW(params, lr=8e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 8, total_steps)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = astra_rows[((step - 1) * accum + j) % len(astra_rows)]
                rec, _ = encode_record(tok, SPEC_14B, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['astra_chosen']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == 1:
                print(f'  [TabGroup | {MODEL_ID_14B} SFT] step {step:3d}/150 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            if step == 100:
                save_file(get_lora_sd(model), str(ckpt_100))
            elif step == 150:
                save_file(get_lora_sd(model), str(ckpt_150))

    sd_100 = load_file(str(ckpt_100), device=device)
    sd_150 = load_file(str(ckpt_150), device=device)

    lm = model.base_model.model.model.language_model
    U_vocab = lm.embed_tokens.weight.detach().float()
    final_norm = lm.norm.weight.detach().float()
    U_eff = U_vocab * final_norm.unsqueeze(0)
    nl_ids = tok.encode('\n', add_special_tokens=False)
    cm_ids = tok.encode(',', add_special_tokens=False)
    ug_ids = tok.encode('Ungrouped', add_special_tokens=False)
    v_nl = U_eff[nl_ids[-1]] / torch.norm(U_eff[nl_ids[-1]])
    v_cm = U_eff[cm_ids[-1]] / torch.norm(U_eff[cm_ids[-1]])
    v_ug = U_eff[ug_ids[0]] / torch.norm(U_eff[ug_ids[0]])
    v_anti_ug = (0.5 * (v_nl + v_cm) - v_ug)
    v_anti_ug = v_anti_ug / torch.norm(v_anti_ug)

    def eval_tabgroup_variant(sd_to_load, name, apply_idea_a=False, batch_size=4):
        set_lora_sd(model, sd_to_load)
        orig_down = {}
        if apply_idea_a:
            for l in range(34, 40):
                m = lm.layers[l].mlp.down_proj.base_layer
                orig_down[l] = m.weight.data.clone()
                W = orig_down[l].float().clone()
                scores = v_anti_ug @ W
                top_pos = torch.topk(scores, 64).indices
                top_neg = torch.topk(-scores, 64).indices
                W[:, top_pos] *= 1.5
                W[:, top_neg] *= 0.4
                m.weight.data.copy_(W.to(orig_down[l].dtype))

        model.eval()
        model.config.use_cache = True
        res = {}
        for ds in ['v1', 'v2.1']:
            t0 = time.monotonic()
            r1_inputs = sorted([i for i in inputs[ds] if i['run'] == 1], key=lambda x: x['case_id'])
            enc_list = []
            for inp in r1_inputs:
                messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': inp['prompt']}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                enc_list.append(tok.encode(text, add_special_tokens=False))
            sc_list = []
            aris, nmis, gen_lens = [], [], []
            for b_start in range(0, len(r1_inputs), batch_size):
                b_rows = r1_inputs[b_start:b_start + batch_size]
                b_encs = enc_list[b_start:b_start + batch_size]
                max_l = max(len(e) for e in b_encs)
                padded_ids = [[pad_id] * (max_l - len(e)) + e for e in b_encs]
                attn_mask = [[0] * (max_l - len(e)) + [1] * len(e) for e in b_encs]
                ids_t = torch.tensor(padded_ids, device=device, dtype=torch.long)
                mask_t = torch.tensor(attn_mask, device=device, dtype=torch.long)
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=ids_t, attention_mask=mask_t, max_new_tokens=300,
                        do_sample=False, use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(SPEC_14B)
                    )
                for idx_b, inp in enumerate(b_rows):
                    out_ids = gen[idx_b, max_l:].tolist()
                    gtoks = 0
                    for tid in out_ids:
                        gtoks += 1
                        if tid in eos:
                            break
                    gen_lens.append(gtoks)
                    pred = tok.decode(out_ids, skip_special_tokens=True)
                    mapping = {int(k): v for k, v in inp['mapping'].items()}
                    ref = refs[ds][inp['case_id']]
                    sc = tab_score(pred, ref, mapping)
                    sc_list.append(sc)
                    # Parse groups for ARI/NMI
                    pred_groups = []
                    for line in pred.splitlines():
                        line = line.strip()
                        if not line or line.startswith('```'):
                            continue
                        parts = line.rsplit(':', 1)
                        if len(parts) == 2 and parts[0].strip().casefold() != 'ungrouped' and re.fullmatch(r'\d+(?:\s*,\s*\d+)*', parts[1].strip()):
                            ids = [int(v) for v in parts[1].split(',')]
                            if len(ids) >= 2 and all(i in mapping for i in ids):
                                pred_groups.append({mapping[i] for i in ids})
                    gold_groups = [{x for x in g} for g in ref['reference_alternatives'][0] if len(g) >= 2]
                    ari, nmi = compute_ari_nmi(pred_groups, gold_groups, set(mapping.values()))
                    aris.append(ari)
                    nmis.append(nmi)

            init_p = sum(x['matched_groups'] for x in sc_list) / max(sum(x['emitted_groups'] for x in sc_list), 1) * 100.0
            rec = statistics.mean(x['recall'] for x in sc_list) * 100.0
            f1 = 2 * init_p * rec / max(init_p + rec, 1e-6)
            ari_m = statistics.mean(aris) * 100.0
            nmi_m = statistics.mean(nmis) * 100.0
            tok_m = statistics.mean(gen_lens)
            res[ds] = {
                'init_p': round(init_p, 2), 'recall': round(rec, 2), 'f1': round(f1, 2),
                'ARI': round(ari_m, 2), 'NMI': round(nmi_m, 2), 'avg_gen_tokens': round(tok_m, 1)
            }
            print(f'  {name:48s} | {ds:5s} | P={init_p:5.2f}% R={rec:5.2f}% F1={f1:5.2f}% | ARI={ari_m:5.2f}% NMI={nmi_m:5.2f}% Toks={tok_m:4.1f} [{time.monotonic()-t0:.1f}s]', flush=True)

        for l, w in orig_down.items():
            lm.layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
        return res

    tab_res = {}
    for tag, sd_cur in [('Step100', sd_100), ('Step150', sd_150)]:
        tab_res[f'{tag} (Standard)'] = eval_tabgroup_variant(sd_cur, f'14B TabGroup {tag} (Standard)')
        tab_res[f'{tag} + WSP_Valve (Idea_A, L34-39)'] = eval_tabgroup_variant(sd_cur, f'14B TabGroup {tag} + WSP_Valve (Idea_A)', apply_idea_a=True)
        tab_res[f'{tag} + WSP_MidAttn (Idea_C, L09-24 q_proj*1.25x)'] = eval_tabgroup_variant(make_idea_c_qproj(sd_cur, 9, 24, 1.25), f'14B TabGroup {tag} + WSP_MidAttn (Idea_C)')
        tab_res[f'{tag} + WSP_SpecNull (L22+ sigma1=0)'] = eval_tabgroup_variant(make_idea_b(sd_cur, 22, 1), f'14B TabGroup {tag} + WSP_SpecNull (L22+ s1=0)')
        tab_res[f'{tag} + WSP_SpecNull_Late (L28+ sigma1=0)'] = eval_tabgroup_variant(make_idea_b(sd_cur, 28, 1), f'14B TabGroup {tag} + WSP_SpecNull_Late (L28+ s1=0)')

    (out_dir / f'{MODEL_ID_14B}_tabgroup_wsp_results.json').write_text(json.dumps(tab_res, indent=2), encoding='utf-8')


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    model, tok = load_14b_with_lora(cfg, device='cuda:0')
    eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    gsm_train = [json.loads(l) for l in (DATA_DIR / 'gsm8k_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    gsm_eval = [json.loads(l) for l in (DATA_DIR / 'gsm8k_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    sci_train = [json.loads(l) for l in (DATA_DIR / 'sciknow_rel_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    sci_eval = [json.loads(l) for l in (DATA_DIR / 'sciknow_rel_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    # 1. GSM8K Math Reasoning on 14B
    train_and_run_wsp_pipeline(
        task_name='GSM8K_Math_14B',
        sys_prompt='You are a precise mathematical reasoning assistant. Solve problems step by step and end with #### <number>.',
        train_rows=gsm_train, eval_rows=gsm_eval, out_dir=WSP_DIR / 'gsm8k_14b',
        model_id=MODEL_ID_14B, spec=SPEC_14B, model=model, tok=tok, eos=eos, pad_id=pad_id,
        is_gsm=True, run_opsd=True
    )

    # 2. SciKnowEval Scientific Knowledge & Relation Reasoning on 14B
    train_and_run_wsp_pipeline(
        task_name='SciKnowEval_SciRel_14B',
        sys_prompt='You are a scientific knowledge extraction assistant specializing in biomedical and pharmacological literature.',
        train_rows=sci_train, eval_rows=sci_eval, out_dir=WSP_DIR / 'sciknoweval_14b',
        model_id=MODEL_ID_14B, spec=SPEC_14B, model=model, tok=tok, eos=eos, pad_id=pad_id,
        is_gsm=False, run_opsd=True
    )

    # 3. BioRED Biomedical Relation Extraction on 14B
    run_biored_14b(model, tok, eos, pad_id)

    # 4. Browser TabGroup Industrial Benchmark on 14B
    run_tabgroup_14b(model, tok, eos, pad_id)

    print('\n=== PHASE 2 (MINISTRAL-3-14B ACROSS ALL 4 DOMAINS) COMPLETED SUCCESSFULLY! ===', flush=True)


if __name__ == '__main__':
    main()
