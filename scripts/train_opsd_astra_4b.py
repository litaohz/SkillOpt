"""On-Policy Self-Distillation (OPSD): Qwen3.5-4B with GPT-6-astra Privileged Context -> Qwen3.5-4B Student."""
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from peft import PeftModel

JOB_4B = Path('/home/dlisuser/work/tabgroup-sft-20261016/jobs/qwen3_5-4b')
sys.path.insert(0, str(JOB_4B))

from model_runtime import load_base
from adapter_runtime import logits_option
from training_common import digest, write_json, seed_all

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'outputs/qwen3_5-4b-opsd-astra'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def generalized_jsd_loss(student_logits, teacher_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1):
    s_logits = student_logits / temperature
    t_logits = teacher_logits / temperature
    if top_k is not None and top_k > 0:
        _, top_k_indices = torch.topk(t_logits, k=top_k, dim=-1)
        s_logits = torch.gather(s_logits, dim=-1, index=top_k_indices)
        t_logits = torch.gather(t_logits, dim=-1, index=top_k_indices)
    s_log_probs = F.log_softmax(s_logits, dim=-1)
    t_log_probs = F.log_softmax(t_logits, dim=-1)
    if beta == 1.0:
        jsd = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True).sum(dim=-1)
    elif beta == 0.0:
        jsd = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True).sum(dim=-1)
    else:
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


def build_privileged_prompt(original_prompt: str, astra_answer: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"=== Privileged Expert Reference Solution (for teacher guidance only) ===\n"
        f"{astra_answer.strip()}\n"
        f"=== End Reference Solution ===\n\n"
        f"Using the privileged expert reference solution above, output the coherent tab grouping in the exact required format:"
    )


def main():
    os.environ.update(
        TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',
    )
    assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    device = 'cuda:0'

    cfg = json.loads((JOB_4B / 'training-config.json').read_text(encoding='utf8'))
    spec_4b = next(m for m in cfg['models'] if m['id'] == 'qwen3_5-4b')

    path_4b = (JOB_4B / 'qwen3_5-4b.path').read_text(encoding='utf8').strip()
    ckpt_4b = JOB_4B / 'outputs/qwen3_5-4b/train/checkpoints/step-002000/adapter'

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])

    print('1. Loading Qwen3.5-4B + SFT step-002000 adapter for OPSD...', flush=True)
    base_4b, tok = load_base(path_4b, spec_4b, device=device)
    model = PeftModel.from_pretrained(base_4b, ckpt_4b, is_trainable=True, autocast_adapter_dtype=False, local_files_only=True)
    # Keep a frozen copy of initial SFT LoRA weights for the privileged Teacher
    teacher_lora_state = {}
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad_(True)
            p.data = p.data.to(torch.float32)
            teacher_lora_state[n] = p.data.clone().detach()
        else:
            p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and {p.dtype for p in params} == {torch.float32}

    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
    if tok.eos_token_id is not None:
        eos.append(tok.eos_token_id)
    eos = sorted(set(eos))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos[0]

    with (ROOT / 'astra-train-1200.jsonl').open(encoding='utf8') as f:
        astra_rows = [json.loads(line) for line in f if line.strip()]
    assert len(astra_rows) >= 1200, f'Expected 1200 Astra-labeled rows, got {len(astra_rows)}'

    max_steps = 150
    grad_accum = 8
    lr = 2e-5
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 15, max_steps)

    history = []
    started = time.monotonic()
    cursor = 0

    print(f'2. Starting OPSD training: {max_steps} steps x {grad_accum} accum = {max_steps*grad_accum} rollouts...', flush=True)
    for step in range(1, max_steps + 1):
        t_step = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_gen_tokens = 0

        for _ in range(grad_accum):
            row = astra_rows[cursor]
            cursor += 1

            # Student prompt (blind to Astra answer)
            s_messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': row['prompt']}]
            s_prompt_text = tok.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
            s_prompt_ids = tok.encode(s_prompt_text, add_special_tokens=False)
            L_s = len(s_prompt_ids)
            s_p_tensor = torch.tensor([s_prompt_ids], device=device, dtype=torch.long)

            # (a) On-policy rollout from Student
            model.eval()
            model.config.use_cache = True
            with torch.inference_mode():
                gen_out = model.generate(
                    input_ids=s_p_tensor,
                    attention_mask=torch.ones_like(s_p_tensor),
                    max_new_tokens=512,
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
            L_c = len(gen_ids)
            if L_c <= 0:
                continue
            step_gen_tokens += L_c

            # Teacher prompt (sees privileged Astra reference answer + Student's generated trajectory)
            t_user = build_privileged_prompt(row['prompt'], row['astra_chosen'])
            t_messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': t_user}]
            t_prompt_text = tok.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
            t_prompt_ids = tok.encode(t_prompt_text, add_special_tokens=False)
            L_t = len(t_prompt_ids)

            t_seq = torch.tensor([t_prompt_ids + gen_ids], device=device, dtype=torch.long)
            s_seq = torch.tensor([s_prompt_ids + gen_ids], device=device, dtype=torch.long)

            # (b) Privileged Teacher forward pass (using frozen initial SFT LoRA weights)
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

            # (c) Student forward pass (with grad)
            model.train()
            model.config.use_cache = False
            s_out = model(input_ids=s_seq, attention_mask=torch.ones_like(s_seq), use_cache=False)
            s_logits = s_out.logits[:, L_s - 1 : -1, :]

            loss = generalized_jsd_loss(s_logits, t_logits, beta=0.5, temperature=1.0, top_k=64, token_clip=0.1)
            (loss / grad_accum).backward()
            step_loss += loss.detach().item() / grad_accum
            del s_out, s_logits, t_logits

        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t_step
        rec = {
            'step': step,
            'loss': round(step_loss, 6),
            'grad_norm': round(norm.item(), 4),
            'gen_tokens': step_gen_tokens,
            'step_sec': round(elapsed, 2),
            'total_sec': round(time.monotonic() - started, 1),
        }
        history.append(rec)
        if step <= 5 or step % 5 == 0:
            print(json.dumps({'event': 'opsd_step', **rec}), flush=True)
            write_json(OUT_DIR / 'train-progress.json', {'step': step, 'target': max_steps, 'history': history})

        if step in (50, 100, 150):
            ckpt_dir = OUT_DIR / f'checkpoints/step-{step:06d}/adapter'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir, safe_serialization=True)
            print(json.dumps({'event': 'checkpoint_saved', 'step': step, 'path': str(ckpt_dir)}), flush=True)

    print('3. Running full 183-case evaluation (v1 + v2.1) on final OPSD step-000150 adapter...', flush=True)
    model.eval()
    model.config.use_cache = True
    final_adapter_file = OUT_DIR / 'checkpoints/step-000150/adapter/adapter_model.safetensors'
    adapter_hash = digest(final_adapter_file)
    logical_id = 'qwen3_5-4b-opsd-astra-step150'
    service_name = spec_4b['model'] + '@tabgroup-opsd-astra-step150'

    eval_out = OUT_DIR / 'eval'
    (eval_out / 'records').mkdir(parents=True, exist_ok=True)
    (eval_out / 'intents').mkdir(parents=True, exist_ok=True)

    model_entry = {
        'id': logical_id,
        'model': service_name,
        'api': 'hf_generate',
        'endpoint': 'local://A100-opsd-astra-4b',
        'parameters': {'max_tokens': cfg['eval_max_new_tokens'], 'temperature': 0},
        'runtime': {
            'repository': spec_4b['model'],
            'revision': spec_4b['revision'],
            'kind': spec_4b['kind'],
            'adapter_sha256': adapter_hash,
            'checkpoint_step': max_steps,
            'dtype': spec_4b['dtype'],
            'attention': spec_4b['attention'],
            'chat_template_kwargs': spec_4b['chat_template_kwargs'],
            'hardware': torch.cuda.get_device_name(0),
            'tf32': False,
            'quantization': None,
            'method': 'On-Policy Self-Distillation (OPSD JSD beta=0.5, top_k=64) with GPT-6-astra privileged answers on Qwen3.5-4B-SFT2000',
        },
    }
    write_json(eval_out / 'eval-model.json', model_entry)

    with (JOB_4B / 'eval-inputs.jsonl').open(encoding='utf8') as f:
        eval_rows = [json.loads(line) for line in f]
    for idx, row in enumerate(eval_rows, 1):
        messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': row['prompt']}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **spec_4b['chat_template_kwargs'])
        token_ids = tok.encode(text, add_special_tokens=False)
        key = sha(json.dumps([row['dataset'], logical_id, row['case_id'], row['run']]).encode())[:24]
        body = {'model': service_name, 'stream': False, 'messages': messages, 'max_tokens': cfg['eval_max_new_tokens'], 'temperature': 0}
        intent = {
            'dataset': row['dataset'],
            'model': logical_id,
            'case_id': row['case_id'],
            'run': row['run'],
            'input_sha256': row['input_sha256'],
            'request_sha256': sha(json.dumps(body, sort_keys=True).encode()),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'adapter_sha256': adapter_hash,
        }
        write_json(eval_out / 'intents' / (key + '.json'), intent)
        ids = torch.tensor([token_ids], device=device)
        t0 = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=cfg['eval_max_new_tokens'],
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                use_cache=True,
                eos_token_id=eos,
                pad_token_id=pad_id,
                **logits_option(spec_4b),
            )
        torch.cuda.synchronize()
        new_ids = generated[0, len(token_ids) :].tolist()
        prediction = tok.decode(new_ids, skip_special_tokens=True)
        hit = len(new_ids) >= cfg['eval_max_new_tokens'] and new_ids[-1] not in eos
        response = {
            'transport': 'hf_generate',
            'model': service_name,
            'revision': spec_4b['revision'],
            'adapter_sha256': adapter_hash,
            'choices': [{'message': {'role': 'assistant', 'content': prediction}, 'finish_reason': 'length' if hit else 'stop'}],
            'usage': {'prompt_tokens': len(token_ids), 'completion_tokens': len(new_ids)},
        }
        record = {
            'dataset': row['dataset'],
            'model': logical_id,
            'case_id': row['case_id'],
            'run': row['run'],
            'input_sha256': row['input_sha256'],
            'request_body': body,
            'http_status': None,
            'transport_success': True,
            'prediction': prediction,
            'raw_response': response,
            'created_utc': intent['created_utc'],
            'latency_seconds': time.monotonic() - t0,
            'rendered_prompt': text,
            'rendered_sha256': sha(text.encode()),
            'input_ids_sha256': sha(json.dumps(token_ids).encode()),
            'input_token_ids': token_ids,
            'generated_token_ids': new_ids,
            'hardware': torch.cuda.get_device_name(0),
            'gpus_per_worker': 1,
            'dtype': spec_4b['dtype'],
            'tf32': False,
            'adapter_sha256': adapter_hash,
            'intent': intent,
        }
        write_json(eval_out / 'records' / (key + '.json'), record)
        if idx % 20 == 0 or idx == len(eval_rows):
            print(json.dumps({'event': 'eval_progress', 'idx': idx, 'total': len(eval_rows)}), flush=True)

    write_json(eval_out / 'completion.json', {'status': 'completed', 'records': len(eval_rows), 'adapter_sha256': adapter_hash})
    print('OPSD ASTRA->4B TRAINING AND 183 EVALUATIONS COMPLETED!', flush=True)


if __name__ == '__main__':
    main()
