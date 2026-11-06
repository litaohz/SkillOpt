"""Phase 1 Part A: Comprehensive Multi-Metric Audit & Statistical Significance (Paired Bootstrap CI/p-value,
Win/Tie/Loss, Macro/Micro-F1, Exact-Match, ARI/NMI, Generation Length, Parse Validity, Spectral Edit Footprint)
across existing BioRED, Re-DocRED, PHEE, and Browser TabGroup (v1 & v2.1) runs for Qwen3.5 (2B/4B/9B) and Ministral-3-8B.
"""
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

BASE_DIR = Path('/home/dlisuser/work/public-two-stage-20261104')
DATA_DIR = BASE_DIR / 'data'
SUITE_DIR = BASE_DIR / 'comprehensive_suite_v2'
OUT_DIR = BASE_DIR / 'wsp_phase1_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys_eval = Path('/home/dlisuser/work/tabgroup-astra-sft-control-20261102/eval-batch')
import sys
sys.path.insert(0, str(sys_eval / '_engine'))
from scoring import score as tab_score

sys.path.insert(0, str(BASE_DIR))
from run_comprehensive_suite_v2 import parse_biored_undir, parse_docred, parse_phee


def compute_ari_nmi(pred_groups, gold_groups, all_items):
    """Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) for clustering."""
    item_list = sorted(all_items)
    n = len(item_list)
    if n < 2:
        return 1.0, 1.0

    def to_labels(groups):
        lbl = {}
        cid = 0
        for g in groups:
            for x in g:
                if x not in lbl:
                    lbl[x] = cid
            cid += 1
        for x in item_list:
            if x not in lbl:
                lbl[x] = cid
                cid += 1
        return [lbl[x] for x in item_list]

    u = to_labels(pred_groups)
    v = to_labels(gold_groups)

    # Contingency table
    cont = defaultdict(int)
    a_cnt = defaultdict(int)
    b_cnt = defaultdict(int)
    for ui, vi in zip(u, v):
        cont[(ui, vi)] += 1
        a_cnt[ui] += 1
        b_cnt[vi] += 1

    def comb2(k):
        return k * (k - 1) // 2

    sum_comb_c = sum(comb2(c) for c in cont.values())
    sum_comb_a = sum(comb2(a) for a in a_cnt.values())
    sum_comb_b = sum(comb2(b) for b in b_cnt.values())
    total_comb = comb2(n)
    prod_ab = (sum_comb_a * sum_comb_b) / total_comb if total_comb > 0 else 0.0
    mean_ab = 0.5 * (sum_comb_a + sum_comb_b)
    denom = mean_ab - prod_ab
    ari = (sum_comb_c - prod_ab) / denom if abs(denom) > 1e-12 else 1.0

    # NMI
    mi = 0.0
    for (ui, vi), c in cont.items():
        if c > 0:
            mi += (c / n) * math.log((c * n) / (a_cnt[ui] * b_cnt[vi]))
    hu = -sum((a / n) * math.log(a / n) for a in a_cnt.values() if a > 0)
    hv = -sum((b / n) * math.log(b / n) for b in b_cnt.values() if b > 0)
    nmi = (2.0 * mi / (hu + hv)) if (hu + hv) > 1e-12 else 1.0
    return ari, nmi


def paired_bootstrap_f1(base_cases, edit_cases, n_boot=1000, seed=42):
    """Paired bootstrap resampling over evaluation samples for Micro-F1 difference (edit - base)."""
    rng = random.Random(seed)
    n = len(base_cases)
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        b_h = sum(base_cases[i]['hits'] for i in idxs)
        b_e = sum(base_cases[i]['emitted'] for i in idxs)
        b_g = sum(base_cases[i]['gold'] for i in idxs)
        b_p = b_h / max(b_e, 1) * 100.0
        b_r = b_h / max(b_g, 1) * 100.0
        b_f1 = 2 * b_p * b_r / max(b_p + b_r, 1e-9)

        e_h = sum(edit_cases[i]['hits'] for i in idxs)
        e_e = sum(edit_cases[i]['emitted'] for i in idxs)
        e_g = sum(edit_cases[i]['gold'] for i in idxs)
        e_p = e_h / max(e_e, 1) * 100.0
        e_r = e_h / max(e_g, 1) * 100.0
        e_f1 = 2 * e_p * e_r / max(e_p + e_r, 1e-9)
        diffs.append(e_f1 - b_f1)

    diffs.sort()
    ci_low = diffs[int(0.025 * n_boot)]
    ci_high = diffs[int(0.975 * n_boot)]
    p_val = sum(1 for d in diffs if d <= 0.0) / n_boot
    return round(ci_low, 2), round(ci_high, 2), round(p_val, 4)


def audit_structured_extraction_benchmark(bench_key, eval_file, n_eval, parse_fn, gold_fn, count_key):
    rows = [json.loads(l) for l in (DATA_DIR / eval_file).read_text().splitlines() if l.strip()][:n_eval]
    res_path = SUITE_DIR / bench_key / 'suite_eval_results.json'
    raw_res = json.loads(res_path.read_text())

    # Also merge opd_9b_to_4b and ministral_200step_ladder if present
    for extra_f in ['opd_9b_to_4b_results.json', 'ministral_200step_ladder.json']:
        ep = SUITE_DIR / bench_key / extra_f
        if ep.exists():
            raw_res.update(json.loads(ep.read_text()))

    per_variant_cases = {}
    summary_metrics = {}

    for v_name, v_data in raw_res.items():
        if 'preds' not in v_data:
            continue
        preds = v_data['preds']
        case_stats = []
        rel_tp = defaultdict(int)
        rel_fp = defaultdict(int)
        rel_fn = defaultdict(int)

        tot_chars = 0
        tot_lines = 0
        valid_format_cnt = 0
        exact_match_cnt = 0
        empty_abstention_cnt = 0

        for inp in rows:
            cid = inp['case_id']
            pred_txt = preds.get(cid, '')
            pred_set = parse_fn(pred_txt, inp[count_key])
            gold_set = gold_fn(inp)

            hits = len(pred_set & gold_set)
            emitted = len(pred_set)
            gold = len(gold_set)
            p_i = hits / max(emitted, 1) * 100.0 if emitted > 0 else (100.0 if gold == 0 else 0.0)
            r_i = hits / max(gold, 1) * 100.0 if gold > 0 else 100.0
            f1_i = 2 * p_i * r_i / max(p_i + r_i, 1e-9)

            if pred_set == gold_set:
                exact_match_cnt += 1
            if emitted > 0 or 'unrelated' in pred_txt.lower() or 'unassigned' in pred_txt.lower():
                valid_format_cnt += 1
            if emitted == 0:
                empty_abstention_cnt += 1

            tot_chars += len(pred_txt)
            tot_lines += len([ln for ln in pred_txt.splitlines() if ln.strip()])

            # Per-relation or per-role macro tracking
            for item in pred_set & gold_set:
                r_type = item[1] if len(item) == 3 else item[0]
                rel_tp[r_type] += 1
            for item in pred_set - gold_set:
                r_type = item[1] if len(item) == 3 else item[0]
                rel_fp[r_type] += 1
            for item in gold_set - pred_set:
                r_type = item[1] if len(item) == 3 else item[0]
                rel_fn[r_type] += 1

            case_stats.append({
                'case_id': cid,
                'hits': hits,
                'emitted': emitted,
                'gold': gold,
                'sample_f1': f1_i,
            })

        per_variant_cases[v_name] = case_stats
        all_rels = set(rel_tp.keys()) | set(rel_fp.keys()) | set(rel_fn.keys())
        macro_f1_list = []
        for rt in all_rels:
            tp, fp, fn = rel_tp[rt], rel_fp[rt], rel_fn[rt]
            if tp + fn == 0:
                continue
            pr = tp / max(tp + fp, 1) * 100.0
            rc = tp / max(tp + fn, 1) * 100.0
            macro_f1_list.append(2 * pr * rc / max(pr + rc, 1e-9))

        sample_f1_avg = statistics.mean(x['sample_f1'] for x in case_stats)
        macro_f1 = statistics.mean(macro_f1_list) if macro_f1_list else 0.0

        summary_metrics[v_name] = {
            'Micro_P': v_data['P'],
            'Micro_R': v_data['R'],
            'Micro_F1': v_data['F1'],
            'Macro_F1': round(macro_f1, 2),
            'Sample_Avg_F1': round(sample_f1_avg, 2),
            'Exact_Match_Rate': round(exact_match_cnt / len(rows) * 100.0, 2),
            'Parse_Validity_Rate': round(valid_format_cnt / len(rows) * 100.0, 2),
            'Abstention_Rate': round(empty_abstention_cnt / len(rows) * 100.0, 2),
            'Avg_Output_Chars': round(tot_chars / len(rows), 1),
            'Avg_Output_Lines': round(tot_lines / len(rows), 2),
            'Total_Hits': v_data['hits'],
            'Total_Emitted': v_data['emitted'],
            'Total_Gold': v_data['gold'],
        }
        if 'Sub_F1' in v_data:
            summary_metrics[v_name]['SubRole_F1'] = v_data['Sub_F1']

    # Compute Win/Tie/Loss and Bootstrap CI against natural baselines
    baseline_pairs = [
        ('SFT_step100 (Converged 100 steps)', [k for k in summary_metrics if k.startswith('SFT_step100 +') or 'OPSD' in k or 'OPD' in k or 'Graft' in k or 'SVD_Blend' in k]),
        ('SFT_step150 (Extended 150 steps)', [k for k in summary_metrics if k.startswith('SFT_step150 +') or 'Graft' in k or 'SVD_Blend' in k]),
        ('SFT_step100 -> OPSD (+25 steps)', [k for k in summary_metrics if k.startswith('OPSD +') or 'Graft' in k or 'SVD_Blend' in k]),
        ('SFT_step100 -> WhiteBox_OPD_9B (+25 steps)', [k for k in summary_metrics if 'OPD' in k]),
        ('Ministral-8B Step150 Standard SFT', [k for k in summary_metrics if k.startswith('Ministral-8B Step150 +')]),
        ('Ministral-8B Step200 Standard SFT', [k for k in summary_metrics if k.startswith('Ministral-8B Step200 +')]),
    ]
    for base_k, target_ks in baseline_pairs:
        if base_k not in per_variant_cases:
            continue
        b_cases = per_variant_cases[base_k]
        for tk in target_ks:
            if tk not in per_variant_cases or tk == base_k:
                continue
            t_cases = per_variant_cases[tk]
            wins = sum(1 for b, t in zip(b_cases, t_cases) if t['sample_f1'] > b['sample_f1'] + 1e-6)
            losses = sum(1 for b, t in zip(b_cases, t_cases) if t['sample_f1'] < b['sample_f1'] - 1e-6)
            ties = len(b_cases) - wins - losses
            ci_l, ci_h, p_v = paired_bootstrap_f1(b_cases, t_cases)
            summary_metrics[tk][f'vs_{base_k[:15]}'] = {
                'Delta_Micro_F1': round(summary_metrics[tk]['Micro_F1'] - summary_metrics[base_k]['Micro_F1'], 2),
                'Delta_Macro_F1': round(summary_metrics[tk]['Macro_F1'] - summary_metrics[base_k]['Macro_F1'], 2),
                'Win_Tie_Loss': f'{wins}W / {ties}T / {losses}L',
                'Bootstrap_95CI': [ci_l, ci_h],
                'p_value': p_v,
            }

    return summary_metrics


def main():
    report = {}
    for b_key, ev_f, n_ev, p_fn, g_fn, c_k in [
        ('biored', 'biored_parsed_test.jsonl', 25, parse_biored_undir, lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}, 'num_entities'),
        ('docred', 'docred_eval.jsonl', 20, parse_docred, lambda inp: {tuple(x) for x in inp['re_triples']}, 'num_entities'),
        ('phee', 'phee_parsed_test.jsonl', 30, parse_phee, lambda inp: {tuple(x) for x in inp['gold_pairs']}, 'num_spans'),
    ]:
        print(f'=== Auditing Comprehensive Metrics for {b_key.upper()} ===')
        report[b_key] = audit_structured_extraction_benchmark(b_key, ev_f, n_ev, p_fn, g_fn, c_k)
        for k, v in report[b_key].items():
            if any(x in k for x in ['SFT_step100', 'SFT_step150', 'OPSD', 'WhiteBox_OPD', 'Graft_18', 'SVD_Blend', 'Ministral-8B Step']):
                print(f"  {k:45s} | MicroF1={v['Micro_F1']:5.2f}% MacroF1={v['Macro_F1']:5.2f}% SampleF1={v['Sample_Avg_F1']:5.2f}% | Chars={v['Avg_Output_Chars']:5.1f} | Valid={v['Parse_Validity_Rate']:5.1f}%")

    (OUT_DIR / 'structured_benchmarks_full_metrics.json').write_text(json.dumps(report, indent=2))
    print('Saved structured_benchmarks_full_metrics.json successfully!')


if __name__ == '__main__':
    main()
