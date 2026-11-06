"""Build GSM8K and SciKnowEval (Structured Relation + 4-Domain L2/L3 MCQA) datasets for Phase 1 Weight-Space Programming."""
import json
import random
import re
from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path('/home/dlisuser/work/public-two-stage-20261104')
RAW_DIR = BASE_DIR / 'raw'
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_gsm8k():
    rng = random.Random(20261106)
    tr_raw = [json.loads(l) for l in (RAW_DIR / 'gsm8k_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    te_raw = [json.loads(l) for l in (RAW_DIR / 'gsm8k_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    def format_gsm_row(r, idx, prefix):
        ans_raw = r['answer'].strip()
        # Clean <<...>> calculator annotations so the model learns clean step-by-step arithmetic + #### final_num
        ans_clean = re.sub(r'<<[^<>]*>>', '', ans_raw)
        m = re.search(r'####\s*([\-0-9,\.]+)', ans_clean)
        gold_num = m.group(1).replace(',', '').strip() if m else ''
        prompt = (
            f"Solve the following math word problem step by step. "
            f"End your response with the exact final numerical answer after '#### '.\n\n"
            f"Question: {r['question'].strip()}"
        )
        return {
            'case_id': f'{prefix}_{idx:04d}',
            'prompt': prompt,
            'gold_clean_target': ans_clean,
            'gold_num': gold_num,
            'num_steps': len([ln for ln in ans_clean.splitlines() if ln.strip()]) - 1,
        }

    tr_rows = [format_gsm_row(r, i, 'gsm_tr') for i, r in enumerate(tr_raw)]
    rng.shuffle(tr_rows)
    te_rows = [format_gsm_row(r, i, 'gsm_te') for i, r in enumerate(te_raw)]

    (DATA_DIR / 'gsm8k_train.jsonl').write_text('\n'.join(json.dumps(x) for x in tr_rows[:1200]), encoding='utf-8')
    (DATA_DIR / 'gsm8k_eval.jsonl').write_text('\n'.join(json.dumps(x) for x in te_rows[:100]), encoding='utf-8')
    print(f'Built GSM8K: {len(tr_rows[:1200])} train, 100 eval. Sample gold_num={te_rows[0]["gold_num"]}')


def parse_sciknow_tuples(ans_str):
    """Parse (a, b) or (a, b, c) tuples from SciKnowEval relation_extraction gold/pred strings."""
    tuples = set()
    # Match anything inside (...) or [...]
    for m in re.finditer(r'[\(\[]([^\(\)\[\]]+)[\)\]]', ans_str):
        inner = m.group(1).strip()
        parts = [p.strip().strip("'\"`").lower() for p in inner.split(',') if p.strip()]
        if len(parts) >= 2 and parts[0] != 'compound 1' and parts[0] != 'drug1':
            tuples.add(tuple(parts))
    return tuples


def build_sciknoweval():
    rng = random.Random(20261106)
    p_sci = hf_hub_download('hicai-zju/SciKnowEval', 'data/v2/sciknoweval_test_v2.jsonl', repo_type='dataset')
    with open(p_sci, encoding='utf-8') as f:
        all_rows = [json.loads(l) for l in f if l.strip()]

    # Track 1: SciKnowEval Structured Scientific Relation Extraction (Compound-Disease + Drug-Drug Interaction, 964 items)
    cd_rows = [r for r in all_rows if r.get('details', {}).get('subtask') == 'compound_disease_relation_extraction']
    dd_rows = [r for r in all_rows if r.get('details', {}).get('subtask') == 'drug_drug_relation_extraction']
    rng.shuffle(cd_rows)
    rng.shuffle(dd_rows)

    def format_rel_row(r, idx, prefix):
        st = r['details']['subtask']
        dom = r['domain']
        gold_tuples = sorted(parse_sciknow_tuples(r['answer']))
        clean_str = ', '.join('(' + ', '.join(t) + ')' for t in gold_tuples) if gold_tuples else 'Unrelated'
        if st == 'compound_disease_relation_extraction':
            task_desc = (
                "Identify all (compound, disease) pairs in the biomedical abstract below. "
                "Output in the format '(compound1, disease1), (compound2, disease2)'. "
                "If none exist, output 'Unrelated'."
            )
        else:
            task_desc = (
                "Identify all drug-drug interactions (drug1, interaction_type, drug2) in the pharmacological text below, "
                "where interaction_type is one of {effect, mechanism, advise, int}. "
                "Output in the format '(drug1, interaction_type, drug2), ...'. "
                "If none exist, output 'Unrelated'."
            )
        return {
            'case_id': f'{prefix}_{idx:04d}',
            'domain': dom,
            'subtask': st,
            'prompt': f"{task_desc}\n\nText: {r['question'].strip()}",
            'gold_clean_target': clean_str,
            'gold_tuples': [list(t) for t in gold_tuples],
            'num_entities': max(len(gold_tuples), 1),
        }

    # Split 800 train, 80 eval (40 Compound-Disease + 40 Drug-Drug)
    eval_raw = cd_rows[:40] + dd_rows[:40]
    train_raw = cd_rows[40:440] + dd_rows[40:440]
    rng.shuffle(train_raw)

    sci_rel_train = [format_rel_row(r, i, 'scirel_tr') for i, r in enumerate(train_raw)]
    sci_rel_eval = [format_rel_row(r, i, 'scirel_te') for i, r in enumerate(eval_raw)]

    (DATA_DIR / 'sciknow_rel_train.jsonl').write_text('\n'.join(json.dumps(x) for x in sci_rel_train), encoding='utf-8')
    (DATA_DIR / 'sciknow_rel_eval.jsonl').write_text('\n'.join(json.dumps(x) for x in sci_rel_eval), encoding='utf-8')
    print(f'Built SciKnowEval-Rel: {len(sci_rel_train)} train, {len(sci_rel_eval)} eval.')

    # Track 2: SciKnowEval 4-Domain L2/L3 Multi-Discipline Reasoning MCQA (Biology, Chemistry, Material, Physics)
    mcq_train = []
    mcq_eval = []
    for dom in ['Biology', 'Chemistry', 'Material', 'Physics']:
        sub = [
            r for r in all_rows
            if r.get('domain') == dom
            and r.get('type') == 'mcq-4-choices'
            and r.get('details', {}).get('level') in ('L2', 'L3')
            and r.get('answerKey') in ('A', 'B', 'C', 'D')
            and len(r.get('choices', {}).get('text', [])) == 4
        ]
        rng.shuffle(sub)
        for i, r in enumerate(sub[:30]):
            opts = '\n'.join(f"{lbl}. {txt}" for lbl, txt in zip(r['choices']['label'], r['choices']['text']))
            mcq_eval.append({
                'case_id': f'scimcq_te_{dom[:3].lower()}_{i:03d}',
                'domain': dom,
                'level': r['details']['level'],
                'subtask': r['details']['subtask'],
                'prompt': (
                    f"Domain: {dom} ({r['details']['subtask']})\n"
                    f"Question: {r['question'].strip()}\n\n"
                    f"Options:\n{opts}\n\n"
                    f"Select the single best option (A, B, C, or D). Output 'Answer: <letter>'."
                ),
                'gold_clean_target': f"Answer: {r['answerKey'].strip()}",
                'gold_letter': r['answerKey'].strip(),
            })
        for i, r in enumerate(sub[30:230]):
            opts = '\n'.join(f"{lbl}. {txt}" for lbl, txt in zip(r['choices']['label'], r['choices']['text']))
            mcq_train.append({
                'case_id': f'scimcq_tr_{dom[:3].lower()}_{i:03d}',
                'domain': dom,
                'level': r['details']['level'],
                'subtask': r['details']['subtask'],
                'prompt': (
                    f"Domain: {dom} ({r['details']['subtask']})\n"
                    f"Question: {r['question'].strip()}\n\n"
                    f"Options:\n{opts}\n\n"
                    f"Select the single best option (A, B, C, or D). Output 'Answer: <letter>'."
                ),
                'gold_clean_target': f"Answer: {r['answerKey'].strip()}",
                'gold_letter': r['answerKey'].strip(),
            })

    rng.shuffle(mcq_train)
    (DATA_DIR / 'sciknow_mcq_train.jsonl').write_text('\n'.join(json.dumps(x) for x in mcq_train), encoding='utf-8')
    (DATA_DIR / 'sciknow_mcq_eval.jsonl').write_text('\n'.join(json.dumps(x) for x in mcq_eval), encoding='utf-8')
    print(f'Built SciKnowEval-MCQ (4 domains): {len(mcq_train)} train, {len(mcq_eval)} eval (30 per domain).')


if __name__ == '__main__':
    build_gsm8k()
    build_sciknoweval()
