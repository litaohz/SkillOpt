"""Phase 1 Weight-Space Programming (WSP) Experiment Suite on GSM8K & SciKnowEval (Structured Relation + 4-Domain MCQA)
across Qwen3.5-4B, Qwen3.5-9B (Teacher for White-Box OPD), and Ministral-3-8B.
Reports the Full 5-Dimensional Comprehensive Metric Suite:
1. Primary Task Metrics (Pass@1 / Micro-P / Micro-R / Micro-F1 / Macro-F1 / Per-Domain Accuracy / Exact Match)
2. Statistical Significance (Sample-level Win/Tie/Loss counts, 1000-sample Paired Bootstrap 95% CI & p-value)
3. Generation & Behavioral Health (Avg Generated Tokens, Loop/Repetition Rate, Truncation Rate, Parse Validity Rate, Step Count)
4. Calibration & Distributional Stability (Target NLL / Perplexity on gold answers)
5. Weight-Space Anatomy & Edit Footprint (Late-to-Mid sigma_1 ratio, SFT-OPSD Mid vs Late Cosine, Edited Rank/Param %, Wall-Clock Time)
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

sys.path.insert(0, '/home/dlisuser/work/public-two-stage-20261104')
from run_comprehensive_suite_v2 import (
    MODEL_PATHS, DATA_DIR,
    frob_inner, svd_blend, generalized_jsd_loss,
    get_lora_sd, set_lora_sd, reset_lora_zero, make_idea_b, reset_rope_deltas
)

WSP_DIR = Path('/home/dlisuser/work/public-two-stage-20261104/wsp_phase1_suite')
WSP_DIR.mkdir(parents=True, exist_ok=True)


def make_idea_c_qproj(sd_in, start_l, end_l, scale=1.25):
    sd_out = {}
    for k, v in sd_in.items():
        if 'self_attn.q_proj.lora_B' in k:
            parts = k.split('.')
            l = int(parts[parts.index('layers') + 1])
            if start_l <= l <= end_l:
                sd_out[k] = (v.float() * scale).to(v.dtype)
                continue
        sd_out[k] = v.clone()
    return sd_out


def anatomize_adapter(sd_sft, sd_opsd=None, num_layers=32):
    """Autonomous Agent Weight-Space Anatomy: QR-SVD spectrum + Cross-paradigm layer alignment."""
    mid_s1, late_s1 = [], []
    mid_cos, late_cos = [], []
    cut_l = int(num_layers * 0.56)  # L18 for 32L, L19 for 34L, L22 for 40L
    mid_l_start, mid_l_end = int(num_layers * 0.18), int(num_layers * 0.48)

    for k in sd_sft:
        if '.lora_A.' not in k:
            continue
        bk = k.replace('.lora_A.', '.lora_B.')
        parts = k.split('.')
        l = int(parts[parts.index('layers') + 1])
        A1, B1 = sd_sft[k].float(), sd_sft[bk].float()
        Qb, Rb = torch.linalg.qr(B1, mode='reduced')
        Qa, Ra = torch.linalg.qr(A1.T, mode='reduced')
        _, S, _ = torch.linalg.svd(Rb @ Ra.T)
        s1 = S[0].item()
        if mid_l_start <= l <= mid_l_end:
            mid_s1.append(s1)
        elif l >= cut_l:
            late_s1.append(s1)

        if sd_opsd is not None and k in sd_opsd:
            A2, B2 = sd_opsd[k].float(), sd_opsd[bk].float()
            # Delta of OPSD relative to SFT vs SFT itself
            dB = torch.cat([B2, -B1], dim=1)
            dA = torch.cat([A2, A1], dim=0)
            n1 = math.sqrt(max(frob_inner(A1, B1, A1, B1), 1e-12))
            n2 = math.sqrt(max(frob_inner(dA, dB, dA, dB), 1e-12))
            if n1 > 1e-6 and n2 > 1e-6:
                c = frob_inner(A1, B1, dA, dB) / (n1 * n2)
                if mid_l_start <= l <= mid_l_end:
                    mid_cos.append(c)
                elif l >= cut_l:
                    late_cos.append(c)

    return {
        'cut_layer': cut_l,
        'mid_sigma1_mean': round(statistics.mean(mid_s1), 4) if mid_s1 else 0.0,
        'late_sigma1_mean': round(statistics.mean(late_s1), 4) if late_s1 else 0.0,
        'late_to_mid_sigma1_ratio': round(statistics.mean(late_s1) / max(statistics.mean(mid_s1), 1e-9), 3) if mid_s1 and late_s1 else 1.0,
        'mid_sft_opsd_cos': round(statistics.mean(mid_cos), 4) if mid_cos else None,
        'late_sft_opsd_cos': round(statistics.mean(late_cos), 4) if late_cos else None,
    }


def has_ngram_loop(text, n=5, count_thresh=3):
    words = text.split()
    if len(words) < n * count_thresh:
        return False
    counts = defaultdict(int)
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] += 1
        if counts[gram] >= count_thresh:
            return True
    return False


def compute_target_nll(model, tok, spec, sys_prompt, eval_rows, max_samples=25):
    """Compute average Teacher-Forced Negative Log-Likelihood (NLL) on gold targets for calibration check."""
    device = 'cuda:0'
    losses = []
    with torch.no_grad():
        for row in eval_rows[:max_samples]:
            rec, _ = encode_record(tok, spec, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
            ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
            lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
            if not torch.isnan(loss):
                losses.append(loss.item())
    avg_nll = statistics.mean(losses) if losses else 0.0
    return round(avg_nll, 4), round(math.exp(min(avg_nll, 10.0)), 2)


# ============================================================================
# Task 1: GSM8K Math Reasoning Evaluation & Training
# ============================================================================
def extract_gsm_number(text):
    # First check after ####
    m = re.search(r'####\s*([\-0-9,\.]+)', text)
    if m:
        val = m.group(1).replace(',', '').strip().rstrip('.')
        try:
            return float(val), True
        except ValueError:
            pass
    # Fallback: last number in text
    nums = re.findall(r'[\-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if nums:
        val = nums[-1].replace(',', '').strip()
        try:
            return float(val), False
        except ValueError:
            pass
    return None, False


def compute_gsm_valve_vector(model, tok):
    """Contrastive valve vector for GSM8K: promote step-by-step equation continuation ('=', '+', '\\n') vs premature '####' termination."""
    lm = model.base_model.model.model.language_model
    U_vocab = lm.embed_tokens.weight.detach().float()
    final_norm = lm.norm.weight.detach().float()
    U_eff = U_vocab * final_norm.unsqueeze(0)

    eq_ids = tok.encode(' =', add_special_tokens=False) or tok.encode('=', add_special_tokens=False)
    nl_ids = tok.encode('\n', add_special_tokens=False)
    stop_ids = tok.encode('####', add_special_tokens=False)

    v_eq = U_eff[eq_ids[-1]] / torch.norm(U_eff[eq_ids[-1]])
    v_nl = U_eff[nl_ids[-1]] / torch.norm(U_eff[nl_ids[-1]])
    v_stop = U_eff[stop_ids[0]] / torch.norm(U_eff[stop_ids[0]])
    v_valve = 0.5 * (v_eq + v_nl) - v_stop
    return v_valve / torch.norm(v_valve)


def eval_gsm8k_comprehensive(
    model, tok, spec, sys_prompt, eval_rows, sd_to_load,
    v_valve=None, apply_idea_a=False, apply_idea_c=False,
    eos=None, pad_id=None, name='', batch_size=25, edit_meta=None
):
    device = 'cuda:0'
    t0 = time.monotonic()
    set_lora_sd(model, sd_to_load)
    lm_layers = model.base_model.model.model.language_model.layers
    num_layers = len(lm_layers)
    orig_down, orig_qnorm = {}, {}

    if apply_idea_a and v_valve is not None:
        start_l = max(0, num_layers - 6)
        for l in range(start_l, num_layers):
            m = lm_layers[l].mlp.down_proj.base_layer
            orig_down[l] = m.weight.data.clone()
            W = orig_down[l].float().clone()
            scores = v_valve @ W
            top_emit = torch.topk(scores, 64).indices
            top_stop = torch.topk(-scores, 64).indices
            W[:, top_emit] *= 1.35
            W[:, top_stop] *= 0.5
            m.weight.data.copy_(W.to(orig_down[l].dtype))

    if apply_idea_c:
        for l in range(num_layers):
            sa = getattr(lm_layers[l], 'self_attn', None)
            if sa is not None and hasattr(sa, 'q_norm') and int(num_layers * 0.2) <= l <= int(num_layers * 0.65):
                qn = sa.q_norm
                orig_qnorm[l] = qn.weight.data.clone()
                qn.weight.data.copy_((orig_qnorm[l].float().clone() * 1.25).to(orig_qnorm[l].dtype))

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

    max_new = 256
    case_records = []
    preds = {}
    for i in range(0, len(eval_rows), batch_size):
        b_rows = eval_rows[i:i + batch_size]
        b_texts = texts[i:i + batch_size]
        enc = tok(b_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            reset_rope_deltas(model)
            gen = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec)
            )
            reset_rope_deltas(model)
        in_len = enc['input_ids'].shape[1]
        for j, inp in enumerate(b_rows):
            out_ids = gen[j, in_len:].tolist()
            # Count actual generated tokens before pad/eos
            gen_toks = 0
            hit_eos = False
            for tid in out_ids:
                gen_toks += 1
                if tid in eos:
                    hit_eos = True
                    break
            pred_txt = tok.decode(out_ids, skip_special_tokens=True).strip()
            pred_num, has_hash = extract_gsm_number(pred_txt)
            gold_num = float(inp['gold_num'])
            is_correct = int(pred_num is not None and abs(pred_num - gold_num) < 1e-4)
            is_Strict_EM = int(is_correct == 1 and has_hash)
            is_loop = int(has_ngram_loop(pred_txt, 5, 3))
            is_trunc = int(not hit_eos and gen_toks >= max_new - 2)
            n_lines = len([ln for ln in pred_txt.splitlines() if ln.strip()])

            case_records.append({
                'case_id': inp['case_id'],
                'correct': is_correct,
                'strict_em': is_Strict_EM,
                'has_format': int(has_hash),
                'gen_tokens': gen_toks,
                'reasoning_lines': n_lines,
                'loop': is_loop,
                'truncated': is_trunc,
            })
            preds[inp['case_id']] = pred_txt

    for l, w in orig_down.items():
        lm_layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
    for l, w in orig_qnorm.items():
        lm_layers[l].self_attn.q_norm.weight.data.copy_(w)

    acc = statistics.mean(x['correct'] for x in case_records) * 100.0
    strict_em = statistics.mean(x['strict_em'] for x in case_records) * 100.0
    fmt_rate = statistics.mean(x['has_format'] for x in case_records) * 100.0
    avg_toks = statistics.mean(x['gen_tokens'] for x in case_records)
    avg_lines = statistics.mean(x['reasoning_lines'] for x in case_records)
    loop_rate = statistics.mean(x['loop'] for x in case_records) * 100.0
    trunc_rate = statistics.mean(x['truncated'] for x in case_records) * 100.0

    print(
        f"  {name:46s} | Pass@1={acc:5.2f}% StrictEM={strict_em:5.2f}% Fmt={fmt_rate:5.1f}% "
        f"| Toks={avg_toks:5.1f} Loop={loop_rate:4.1f}% Trunc={trunc_rate:4.1f}% | NLL={nll:.3f} [{time.monotonic()-t0:.1f}s]",
        flush=True
    )
    return {
        'Pass_at_1': round(acc, 2),
        'Strict_EM': round(strict_em, 2),
        'Format_Validity_Rate': round(fmt_rate, 2),
        'Avg_Gen_Tokens': round(avg_toks, 1),
        'Avg_Reasoning_Steps': round(avg_lines, 2),
        'Loop_Rate': round(loop_rate, 2),
        'Truncation_Rate': round(trunc_rate, 2),
        'Gold_NLL': nll,
        'Gold_PPL': ppl,
        'Edit_Meta': edit_meta or {},
        'cases': case_records,
        'preds': preds,
    }


# ============================================================================
# Task 2: SciKnowEval Structured Scientific Extraction Evaluation
# ============================================================================
def parse_sciknow_pred(pred_text, dummy_cnt=0):
    tuples = set()
    if 'unrelated' in pred_text.lower() or 'none' in pred_text.lower():
        return tuples
    for m in re.finditer(r'[\(\[]([^\(\)\[\]]+)[\)\]]', pred_text):
        inner = m.group(1).strip()
        parts = [p.strip().strip("'\"`").lower() for p in inner.split(',') if p.strip()]
        if len(parts) in (2, 3) and parts[0] not in ('compound 1', 'drug1', '...'):
            tuples.add(tuple(parts))
    return tuples


def eval_sciknow_rel_comprehensive(
    model, tok, spec, sys_prompt, eval_rows, sd_to_load,
    v_valve=None, apply_idea_a=False, apply_idea_c=False,
    eos=None, pad_id=None, name='', batch_size=20, edit_meta=None
):
    device = 'cuda:0'
    t0 = time.monotonic()
    set_lora_sd(model, sd_to_load)
    lm_layers = model.base_model.model.model.language_model.layers
    num_layers = len(lm_layers)
    orig_down, orig_qnorm = {}, {}

    if apply_idea_a and v_valve is not None:
        start_l = max(0, num_layers - 6)
        for l in range(start_l, num_layers):
            m = lm_layers[l].mlp.down_proj.base_layer
            orig_down[l] = m.weight.data.clone()
            W = orig_down[l].float().clone()
            scores = v_valve @ W
            top_emit = torch.topk(scores, 64).indices
            top_stop = torch.topk(-scores, 64).indices
            W[:, top_emit] *= 1.6
            W[:, top_stop] *= 0.2
            m.weight.data.copy_(W.to(orig_down[l].dtype))

    if apply_idea_c:
        for l in range(num_layers):
            sa = getattr(lm_layers[l], 'self_attn', None)
            if sa is not None and hasattr(sa, 'q_norm') and int(num_layers * 0.2) <= l <= int(num_layers * 0.65):
                qn = sa.q_norm
                orig_qnorm[l] = qn.weight.data.clone()
                qn.weight.data.copy_((orig_qnorm[l].float().clone() * 1.30).to(orig_qnorm[l].dtype))

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
    subtask_stats = defaultdict(lambda: {'emit': 0, 'hits': 0, 'gold': 0})
    case_records = []
    preds = {}

    for i in range(0, len(eval_rows), batch_size):
        b_rows = eval_rows[i:i + batch_size]
        b_texts = texts[i:i + batch_size]
        enc = tok(b_texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        with torch.inference_mode():
            reset_rope_deltas(model)
            gen = model.generate(
                **enc, max_new_tokens=180, do_sample=False, use_cache=True,
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
            pred_set = parse_sciknow_pred(pred_txt)
            gold_set = {tuple(x) for x in inp['gold_tuples']}

            h = len(pred_set & gold_set)
            e = len(pred_set)
            g = len(gold_set)
            tot_hits += h
            tot_emit += e
            tot_gold += g

            st = inp['subtask']
            subtask_stats[st]['hits'] += h
            subtask_stats[st]['emit'] += e
            subtask_stats[st]['gold'] += g

            p_i = h / max(e, 1) * 100.0 if e > 0 else 0.0
            r_i = h / max(g, 1) * 100.0 if g > 0 else 100.0
            f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-9)

            case_records.append({
                'case_id': inp['case_id'],
                'subtask': st,
                'hits': h,
                'emitted': e,
                'gold': g,
                'sample_f1': f1_i,
                'exact_match': int(pred_set == gold_set),
                'valid_format': int(e > 0 or 'unrelated' in pred_txt.lower()),
                'gen_tokens': gen_toks,
            })
            preds[inp['case_id']] = pred_txt

    for l, w in orig_down.items():
        lm_layers[l].mlp.down_proj.base_layer.weight.data.copy_(w)
    for l, w in orig_qnorm.items():
        lm_layers[l].self_attn.q_norm.weight.data.copy_(w)

    p = tot_hits / max(tot_emit, 1) * 100.0
    r = tot_hits / max(tot_gold, 1) * 100.0
    f1 = 2 * p * r / max(p + r, 1e-9)

    sub_f1s = {}
    for st, d in subtask_stats.items():
        sp = d['hits'] / max(d['emit'], 1) * 100.0
        sr = d['hits'] / max(d['gold'], 1) * 100.0
        sf1 = 2 * sp * sr / max(sp + sr, 1e-9)
        sub_f1s[st] = round(sf1, 2)

    macro_f1 = statistics.mean(sub_f1s.values()) if sub_f1s else 0.0
    sample_f1 = statistics.mean(x['sample_f1'] for x in case_records)
    em_rate = statistics.mean(x['exact_match'] for x in case_records) * 100.0
    fmt_rate = statistics.mean(x['valid_format'] for x in case_records) * 100.0
    avg_toks = statistics.mean(x['gen_tokens'] for x in case_records)

    print(
        f"  {name:46s} | P={p:5.2f}% R={r:5.2f}% MicroF1={f1:5.2f}% MacroF1={macro_f1:5.2f}% "
        f"| EM={em_rate:4.1f}% Toks={avg_toks:4.1f} NLL={nll:.3f} ({tot_hits}/{tot_gold}, emit={tot_emit}) [{time.monotonic()-t0:.1f}s]",
        flush=True
    )
    return {
        'Micro_P': round(p, 2),
        'Micro_R': round(r, 2),
        'Micro_F1': round(f1, 2),
        'Macro_F1': round(macro_f1, 2),
        'Sample_Avg_F1': round(sample_f1, 2),
        'Subtask_F1': sub_f1s,
        'Exact_Match_Rate': round(em_rate, 2),
        'Parse_Validity_Rate': round(fmt_rate, 2),
        'Avg_Gen_Tokens': round(avg_toks, 1),
        'Gold_NLL': nll,
        'Gold_PPL': ppl,
        'Total_Hits': tot_hits,
        'Total_Emitted': tot_emit,
        'Total_Gold': tot_gold,
        'Edit_Meta': edit_meta or {},
        'cases': case_records,
        'preds': preds,
    }


def add_significance_gsm(variants_dict, base_key):
    if base_key not in variants_dict:
        return
    b_cases = variants_dict[base_key]['cases']
    rng = random.Random(42)
    n = len(b_cases)
    for k, v in variants_dict.items():
        if k == base_key:
            continue
        t_cases = v['cases']
        wins = sum(1 for b, t in zip(b_cases, t_cases) if t['correct'] > b['correct'])
        losses = sum(1 for b, t in zip(b_cases, t_cases) if t['correct'] < b['correct'])
        ties = n - wins - losses
        diffs = []
        for _ in range(1000):
            idxs = [rng.randrange(n) for _ in range(n)]
            b_acc = sum(b_cases[i]['correct'] for i in idxs) / n * 100.0
            t_acc = sum(t_cases[i]['correct'] for i in idxs) / n * 100.0
            diffs.append(t_acc - b_acc)
        diffs.sort()
        v.setdefault('Significance', {})[f'vs_{base_key}'] = {
            'Delta_Pass_at_1': round(v['Pass_at_1'] - variants_dict[base_key]['Pass_at_1'], 2),
            'Win_Tie_Loss': f'{wins}W / {ties}T / {losses}L',
            'Bootstrap_95CI': [round(diffs[25], 2), round(diffs[975], 2)],
            'p_value': round(sum(1 for d in diffs if d <= 0.0) / 1000.0, 4),
        }


def add_significance_rel(variants_dict, base_key):
    if base_key not in variants_dict:
        return
    b_cases = variants_dict[base_key]['cases']
    rng = random.Random(42)
    n = len(b_cases)
    for k, v in variants_dict.items():
        if k == base_key:
            continue
        t_cases = v['cases']
        wins = sum(1 for b, t in zip(b_cases, t_cases) if t['sample_f1'] > b['sample_f1'] + 1e-6)
        losses = sum(1 for b, t in zip(b_cases, t_cases) if t['sample_f1'] < b['sample_f1'] - 1e-6)
        ties = n - wins - losses
        diffs = []
        for _ in range(1000):
            idxs = [rng.randrange(n) for _ in range(n)]
            bh = sum(b_cases[i]['hits'] for i in idxs)
            be = sum(b_cases[i]['emitted'] for i in idxs)
            bg = sum(b_cases[i]['gold'] for i in idxs)
            bp = bh / max(be, 1) * 100.0
            br = bh / max(bg, 1) * 100.0
            bf1 = 2 * bp * br / max(bp + br, 1e-9)

            th = sum(t_cases[i]['hits'] for i in idxs)
            te = sum(t_cases[i]['emitted'] for i in idxs)
            tg = sum(t_cases[i]['gold'] for i in idxs)
            tp = th / max(te, 1) * 100.0
            tr = th / max(tg, 1) * 100.0
            tf1 = 2 * tp * tr / max(tp + tr, 1e-9)
            diffs.append(tf1 - bf1)
        diffs.sort()
        v.setdefault('Significance', {})[f'vs_{base_key}'] = {
            'Delta_Micro_F1': round(v['Micro_F1'] - variants_dict[base_key]['Micro_F1'], 2),
            'Delta_Macro_F1': round(v['Macro_F1'] - variants_dict[base_key]['Macro_F1'], 2),
            'Win_Tie_Loss': f'{wins}W / {ties}T / {losses}L',
            'Bootstrap_95CI': [round(diffs[25], 2), round(diffs[975], 2)],
            'p_value': round(sum(1 for d in diffs if d <= 0.0) / 1000.0, 4),
        }


def train_and_run_wsp_pipeline(
    task_name, sys_prompt, train_rows, eval_rows, out_dir,
    model_id, spec, model, tok, eos, pad_id, is_gsm=True, run_opsd=True
):
    device = 'cuda:0'
    out_dir.mkdir(parents=True, exist_ok=True)
    adap_dir = out_dir / 'adapters'
    adap_dir.mkdir(parents=True, exist_ok=True)

    num_layers = len(model.base_model.model.model.language_model.layers)
    cut_l = int(num_layers * 0.56)  # 18 for 32L, 19 for 34L
    late_l = int(num_layers * 0.70) # 22 for 32L, 24 for 34L
    is_pure_attn = (spec['kind'] == 'ministral')

    ckpt_100 = adap_dir / f'{model_id}_sft100.safetensors'
    ckpt_150 = adap_dir / f'{model_id}_sft150.safetensors'
    ckpt_opsd = adap_dir / f'{model_id}_sft100_opsd25.safetensors'

    params = [p for p in model.parameters() if p.requires_grad]

    if not ckpt_150.exists():
        print(f'\n=== [{task_name} | {model_id}] Training 150-Step SFT Ladder (Saving @ 100, 150) ===', flush=True)
        reset_lora_zero(model)
        accum = 4
        total_steps = 150
        lr = 8e-5 if is_pure_attn else 6e-5
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 8, total_steps)
        model.train()
        model.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = train_rows[((step - 1) * accum + j) % len(train_rows)]
                rec, _ = encode_record(tok, spec, sys_prompt, {'case_id': row['case_id'], 'prompt': row['prompt'], 'chosen': row['gold_clean_target']})
                ids = torch.tensor([rec['input_ids']], device=device, dtype=torch.long)
                lbls = torch.tensor([rec['labels']], device=device, dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=lbls, use_cache=False).loss
                (loss / accum).backward()
                sloss += loss.detach().item() / accum
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if step % 25 == 0 or step == 1:
                print(f'  [{task_name} | {model_id} SFT] step {step:3d}/150 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            if step == 100:
                save_file(get_lora_sd(model), str(ckpt_100))
            elif step == 150:
                save_file(get_lora_sd(model), str(ckpt_150))

    sd_100 = load_file(str(ckpt_100), device=device)
    sd_150 = load_file(str(ckpt_150), device=device)

    sd_opsd = None
    if run_opsd:
        if not ckpt_opsd.exists():
            print(f'\n=== [{task_name} | {model_id}] Training SFT100 -> OPSD (+25 steps) ===', flush=True)
            set_lora_sd(model, sd_100)
            teacher_state = {n: p.data.clone().detach() for n, p in model.named_parameters() if 'lora_' in n}
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
                        tokenize=False, add_generation_prompt=True, **spec['chat_template_kwargs']
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
                        use_cache=True, eos_token_id=eos, pad_token_id=pad_id, **logits_option(spec)
                    )
                    reset_rope_deltas(model)

                sloss = 0.0
                for j, row in enumerate(b_rows):
                    raw_gen = gen_out[j, in_len:].tolist()
                    gen_ids = []
                    for tid in raw_gen:
                        gen_ids.append(tid)
                        if tid in eos:
                            break
                    if not gen_ids:
                        continue
                    s_prompt_ids = tok.encode(s_texts[j], add_special_tokens=False)
                    L_s = len(s_prompt_ids)
                    t_user = (
                        f"{row['prompt']}\n\n"
                        f"=== Privileged Expert Reference Solution ===\n{row['gold_clean_target']}\n=== End Reference ===\n"
                        f"Produce the accurate step-by-step solution:"
                    )
                    t_text = tok.apply_chat_template(
                        [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': t_user}],
                        tokenize=False, add_generation_prompt=True, **spec['chat_template_kwargs']
                    )
                    t_prompt_ids = tok.encode(t_text, add_special_tokens=False)
                    L_t = len(t_prompt_ids)

                    t_seq = torch.tensor([t_prompt_ids + gen_ids], device=device, dtype=torch.long)
                    s_seq = torch.tensor([s_prompt_ids + gen_ids], device=device, dtype=torch.long)

                    saved_student = {}
                    for n, p in model.named_parameters():
                        if n in teacher_state:
                            saved_student[n] = p.data.clone()
                            p.data.copy_(teacher_state[n])
                    with torch.no_grad():
                        t_out = model(input_ids=t_seq, attention_mask=torch.ones_like(t_seq), use_cache=False)
                        t_logits = t_out.logits[:, L_t - 1:-1, :].to(torch.float32).detach()
                        del t_out
                    for n, p in model.named_parameters():
                        if n in saved_student:
                            p.data.copy_(saved_student[n])
                    del saved_student

                    model.train()
                    model.config.use_cache = False
                    s_out = model(input_ids=s_seq, attention_mask=torch.ones_like(s_seq), use_cache=False)
                    s_logits = s_out.logits[:, L_s - 1:-1, :].to(torch.float32)
                    loss = generalized_jsd_loss(s_logits, t_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1)
                    (loss / accum).backward()
                    sloss += loss.detach().item() / accum
                    del s_out, s_logits, t_logits

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                if step % 5 == 0 or step == 1:
                    print(f'  [{task_name} | {model_id} OPSD] step {step:2d}/25 | jsd_loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            save_file(get_lora_sd(model), str(ckpt_opsd))
        sd_opsd = load_file(str(ckpt_opsd), device=device)

    # Agent Weight-Space Anatomy Dossier
    anatomy_100 = anatomize_adapter(sd_100, sd_opsd, num_layers=num_layers)
    anatomy_150 = anatomize_adapter(sd_150, sd_opsd, num_layers=num_layers)
    print(f'  [Weight-Space Anatomy | {model_id}] Step100: {anatomy_100} | Step150: {anatomy_150}', flush=True)

    if is_gsm:
        v_valve = compute_gsm_valve_vector(model, tok)
        eval_fn = eval_gsm8k_comprehensive
    else:
        from run_comprehensive_suite_v2 import compute_valve_vector
        v_valve, _ = compute_valve_vector(model, tok, 'Unrelated')
        eval_fn = eval_sciknow_rel_comprehensive

    variants = {}
    for tag, sd_base in [('SFT_step100', sd_100), ('SFT_step150', sd_150)]:
        variants[f'{tag} (Standard)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_base,
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} (Standard)',
            edit_meta={'type': 'Baseline', 'edited_rank_pct': 0.0, 'time_sec': 0.0}
        )
        variants[f'{tag} + WSP_Valve (Idea_A, 0-Train)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_base,
            v_valve=v_valve, apply_idea_a=True, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} + WSP_Valve (Idea_A)',
            edit_meta={'type': 'Late-Layer 64-Col Valve', 'edited_cols': '64/layer (top-6 layers)', 'time_sec': 0.15}
        )
        if is_pure_attn:
            sd_c = make_idea_c_qproj(sd_base, int(num_layers * 0.22), int(num_layers * 0.60), 1.25)
            variants[f'{tag} + WSP_MidAttn (Idea_C, 0-Train)'] = eval_fn(
                model, tok, spec, sys_prompt, eval_rows, sd_c,
                v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} + WSP_MidAttn (Idea_C)',
                edit_meta={'type': 'Mid-Layer q_proj * 1.25x', 'edited_layers': f'L{int(num_layers*0.22)}-L{int(num_layers*0.60)}', 'time_sec': 0.08}
            )
        else:
            variants[f'{tag} + WSP_MidAttn (Idea_C, 0-Train)'] = eval_fn(
                model, tok, spec, sys_prompt, eval_rows, sd_base,
                v_valve=v_valve, apply_idea_c=True, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} + WSP_MidAttn (Idea_C)',
                edit_meta={'type': 'Mid-Layer q_norm * 1.25x', 'edited_scalars': 4, 'time_sec': 0.02}
            )
        variants[f'{tag} + WSP_SpecNull (L{cut_l}+ sigma1=0)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, make_idea_b(sd_base, cut_l, 1),
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} + WSP_SpecNull (L{cut_l}+ s1=0)',
            edit_meta={'type': f'L{cut_l}+ sigma_1=0', 'edited_rank_pct': round(100.0 / 16 * (num_layers - cut_l) / num_layers, 2), 'time_sec': 0.42}
        )
        variants[f'{tag} + WSP_SpecNull_Late (L{late_l}+ sigma1=0)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, make_idea_b(sd_base, late_l, 1),
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} {tag} + WSP_SpecNull_Late (L{late_l}+ s1=0)',
            edit_meta={'type': f'L{late_l}+ sigma_1=0', 'edited_rank_pct': round(100.0 / 16 * (num_layers - late_l) / num_layers, 2), 'time_sec': 0.31}
        )

    if sd_opsd is not None:
        variants['SFT_step100 -> OPSD (+25 steps)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_opsd,
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} SFT100 -> OPSD (+25 steps)',
            edit_meta={'type': 'On-Policy Self-Distillation', 'edited_rank_pct': 100.0}
        )
        variants[f'OPSD + WSP_SpecNull (L{cut_l}+ sigma1=0)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, make_idea_b(sd_opsd, cut_l, 1),
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} OPSD + WSP_SpecNull (L{cut_l}+ s1=0)',
            edit_meta={'type': f'OPSD + L{cut_l}+ sigma_1=0', 'time_sec': 0.42}
        )
        variants['OPSD + WSP_Valve (Idea_A, 0-Train)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_opsd,
            v_valve=v_valve, apply_idea_a=True, eos=eos, pad_id=pad_id, name=f'{model_id} OPSD + WSP_Valve (Idea_A)',
            edit_meta={'type': 'OPSD + 64-Col Valve', 'time_sec': 0.15}
        )
        # Dual-adapter Graft & SVD-Blend between SFT_step150 and OPSD
        sd_graft, sd_blend = {}, {}
        for k in sd_150:
            if '.lora_A.' not in k:
                continue
            bk = k.replace('.lora_A.', '.lora_B.')
            parts = k.split('.')
            l = int(parts[parts.index('layers') + 1])
            src = sd_150 if l < cut_l else sd_opsd
            sd_graft[k], sd_graft[bk] = src[k].clone(), src[bk].clone()
            Ab, Bb = svd_blend(sd_opsd[k].float(), sd_opsd[bk].float(), sd_150[k].float(), sd_150[bk].float(), 0.5, 0.5)
            sd_blend[k], sd_blend[bk] = Ab, Bb

        variants[f'WSP_Graft (L0-{cut_l-1} SFT150 + L{cut_l}+ OPSD)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_graft,
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} WSP_Graft (SFT150 + OPSD)',
            edit_meta={'type': 'Dual-Adapter Layer Graft', 'cut_layer': cut_l, 'time_sec': 0.05}
        )
        variants['WSP_SVD_Blend_50_50 (SFT150 + OPSD)'] = eval_fn(
            model, tok, spec, sys_prompt, eval_rows, sd_blend,
            v_valve=v_valve, eos=eos, pad_id=pad_id, name=f'{model_id} WSP_SVD_Blend (SFT150 + OPSD)',
            edit_meta={'type': 'Dual-Adapter Rank-16 QR-SVD Consensus', 'time_sec': 0.48}
        )

    if is_gsm:
        add_significance_gsm(variants, 'SFT_step100 (Standard)')
        add_significance_gsm(variants, 'SFT_step150 (Standard)')
        if 'SFT_step100 -> OPSD (+25 steps)' in variants:
            add_significance_gsm(variants, 'SFT_step100 -> OPSD (+25 steps)')
    else:
        add_significance_rel(variants, 'SFT_step100 (Standard)')
        add_significance_rel(variants, 'SFT_step150 (Standard)')
        if 'SFT_step100 -> OPSD (+25 steps)' in variants:
            add_significance_rel(variants, 'SFT_step100 -> OPSD (+25 steps)')

    # Strip verbose preds/cases into separate file and save clean summary
    full_out = {'model_id': model_id, 'task': task_name, 'anatomy_step100': anatomy_100, 'anatomy_step150': anatomy_150, 'variants': variants}
    (out_dir / f'{model_id}_wsp_full_results.json').write_text(json.dumps(full_out, indent=2), encoding='utf-8')
    return full_out


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    gsm_train = [json.loads(l) for l in (DATA_DIR / 'gsm8k_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    gsm_eval = [json.loads(l) for l in (DATA_DIR / 'gsm8k_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    sci_train = [json.loads(l) for l in (DATA_DIR / 'sciknow_rel_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    sci_eval = [json.loads(l) for l in (DATA_DIR / 'sciknow_rel_eval.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    for mid in ['qwen3_5-4b', 'ministral-3-8b-instruct-2512-bf16']:
        spec = next(m for m in cfg['models'] if m['id'] == mid)
        print(f'\n======================================================\nLoading {mid} for Phase 1 WSP Experiments\n======================================================', flush=True)
        base, tok = load_base(MODEL_PATHS[mid], spec, device=device)
        if hasattr(base, 'language_model'):
            base.language_model.config._attn_implementation = 'sdpa'
        if hasattr(base, 'model') and hasattr(base.model, 'language_model'):
            base.model.language_model.config._attn_implementation = 'sdpa'
        model, _ = attach_lora(base, spec, cfg)
        eos = sorted(set((list(model.generation_config.eos_token_id) if isinstance(model.generation_config.eos_token_id, (list, tuple)) else [model.generation_config.eos_token_id]) + [tok.eos_token_id]))
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

        # 1. GSM8K Math Reasoning
        train_and_run_wsp_pipeline(
            task_name='GSM8K_Math',
            sys_prompt='You are a precise mathematical reasoning assistant. Solve problems step by step and end with #### <number>.',
            train_rows=gsm_train,
            eval_rows=gsm_eval,
            out_dir=WSP_DIR / 'gsm8k',
            model_id=mid, spec=spec, model=model, tok=tok, eos=eos, pad_id=pad_id,
            is_gsm=True, run_opsd=True
        )

        # 2. SciKnowEval Scientific Knowledge & Relation Reasoning
        train_and_run_wsp_pipeline(
            task_name='SciKnowEval_SciRel',
            sys_prompt='You are a scientific knowledge extraction assistant specializing in biomedical and pharmacological literature.',
            train_rows=sci_train,
            eval_rows=sci_eval,
            out_dir=WSP_DIR / 'sciknoweval',
            model_id=mid, spec=spec, model=model, tok=tok, eos=eos, pad_id=pad_id,
            is_gsm=False, run_opsd=True
        )

        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    print('\n=== PHASE 1 (QWEN3.5-4B & MINISTRAL-3-8B ON GSM8K & SCIKNOWEVAL) COMPLETED! ===', flush=True)


if __name__ == '__main__':
    main()
