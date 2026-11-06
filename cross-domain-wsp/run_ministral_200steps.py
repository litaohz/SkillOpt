"""Continue Ministral-3-8B SFT from 100 -> 150 -> 200 steps on BioRED (and also Qwen3.5-9B 100 -> 150 -> 200 steps)
to test whether converged 8B/9B models exhibit the exact same late-layer valve bottleneck & zero-training surgery gains!
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import get_cosine_schedule_with_warmup

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))
from model_runtime import load_base, attach_lora
from prepare_tokens import encode_record

sys.path.insert(0, '/home/dlisuser/work/public-two-stage-20261104')
from run_comprehensive_suite_v2 import (
    MODEL_PATHS, DATA_DIR, SUITE_DIR,
    parse_biored_undir, get_lora_sd, set_lora_sd, reset_lora_zero,
    make_idea_b, eval_model_batched, compute_valve_vector
)


def make_idea_c_qproj(sd_in, start_l, end_l, scale=1.30):
    """For pure full-attention models without q_norm, sharpen middle-layer query projection (q_proj) by scaling its LoRA B."""
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


def main():
    os.environ.update(TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = 'cuda:0'

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    mid = 'ministral-3-8b-instruct-2512-bf16'
    spec_m = next(m for m in cfg['models'] if m['id'] == mid)

    biored_train = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    biored_eval = [json.loads(l) for l in (DATA_DIR / 'biored_parsed_test.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()][:25]
    sys_prompt = 'You extract biomedical entity relations from scientific abstracts.'
    gold_fn = lambda inp: {(min(h, t), rt, max(h, t)) for h, rt, t in inp['full_triples']}
    out_dir = SUITE_DIR / 'biored'

    print(f'=== Loading {mid} for Continuous 200-Step SFT Ladder (Saving @ 100, 150, 200) ===', flush=True)
    base_m, tok_m = load_base(MODEL_PATHS[mid], spec_m, device=device)
    model_m, _ = attach_lora(base_m, spec_m, cfg)
    params_m = [p for p in model_m.parameters() if p.requires_grad]

    ckpt_100 = out_dir / f'adapters/{mid}_step100_v2.safetensors'
    ckpt_150 = out_dir / f'adapters/{mid}_step150_v2.safetensors'
    ckpt_200 = out_dir / f'adapters/{mid}_step200_v2.safetensors'

    if not ckpt_200.exists():
        reset_lora_zero(model_m)
        accum = 4
        total_steps = 200
        opt = torch.optim.AdamW(params_m, lr=8e-5, weight_decay=0.01)
        sched = get_cosine_schedule_with_warmup(opt, 10, total_steps)
        model_m.train()
        model_m.config.use_cache = False
        t0 = time.monotonic()
        for step in range(1, total_steps + 1):
            opt.zero_grad(set_to_none=True)
            sloss = 0.0
            for j in range(accum):
                row = biored_train[((step - 1) * accum + j) % len(biored_train)]
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
                print(f'  [{mid} SFT] step {step:3d}/200 | loss={sloss:.4f} | ({time.monotonic()-t0:.1f}s)', flush=True)
            if step == 100:
                save_file(get_lora_sd(model_m), str(ckpt_100))
            elif step == 150:
                save_file(get_lora_sd(model_m), str(ckpt_150))
            elif step == 200:
                save_file(get_lora_sd(model_m), str(ckpt_200))

    sd_100 = load_file(str(ckpt_100), device=device)
    sd_150 = load_file(str(ckpt_150), device=device)
    sd_200 = load_file(str(ckpt_200), device=device)

    eos_m = sorted(set((list(model_m.generation_config.eos_token_id) if isinstance(model_m.generation_config.eos_token_id, (list, tuple)) else [model_m.generation_config.eos_token_id]) + [tok_m.eos_token_id]))
    pad_m = tok_m.pad_token_id or eos_m[0]
    v_valve_m, _ = compute_valve_vector(model_m, tok_m, 'Unrelated')

    print(f'\n=== Evaluating {mid} Across Step 100, Step 150, Step 200 + Surgeries ===', flush=True)
    res_m = {}
    for step_tag, sd_cur in [('Step100', sd_100), ('Step150', sd_150), ('Step200', sd_200)]:
        for v_name, sd_v, ia in [
            (f'Ministral-8B {step_tag} Standard SFT', sd_cur, False),
            (f'Ministral-8B {step_tag} + Idea_A (64-Col Valve)', sd_cur, True),
            (f'Ministral-8B {step_tag} + Idea_C (L08-20 q_proj*1.3x)', make_idea_c_qproj(sd_cur, 8, 20, 1.30), False),
            (f'Ministral-8B {step_tag} + Idea_B1 (L19+ sigma1=0)', make_idea_b(sd_cur, 19, 1), False),
            (f'Ministral-8B {step_tag} + Idea_B1_late (L24+ sigma1=0)', make_idea_b(sd_cur, 24, 1), False),
        ]:
            res_m[v_name] = eval_model_batched(
                model_m, tok_m, spec_m, sys_prompt, biored_eval, parse_biored_undir,
                gold_fn, 'num_entities', sd_v, v_valve=v_valve_m, apply_idea_a=ia,
                eos=eos_m, pad_id=pad_m, name=v_name, batch_size=25
            )

    (out_dir / 'ministral_200step_ladder.json').write_text(json.dumps(res_m, indent=2), encoding='utf-8')
    print('\n=== MINISTRAL 200-STEP LADDER COMPLETED! ===', flush=True)


if __name__ == '__main__':
    main()
