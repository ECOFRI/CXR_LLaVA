"""F0 bounded steps, batch capacity probes, local checkpoints and generation."""
import argparse
import gc
import json
import random
import time

from research.rrg_f0.core import ROOT, INDEX, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['lora64', 'bridge64', 'lora256'], required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from research.rrg_f0.data import FeatureDataset, collate
    from research.rrg_f0.model import RRG

    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    if not torch.cuda.is_available() or 'RTX 4000' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected authorized RTX 4000')
    base = ROOT / 'research/rrg_f0'
    downloads = json.loads((base / 'model_download.json').read_text())
    if downloads['status'] != 'PASS':
        raise RuntimeError('Model download gate is not PASS')
    model_path = downloads['models']['Qwen/Qwen3-1.7B']['path']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm, loading = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16,
                                                      attn_implementation='sdpa', output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('STOP_QWEN_WEIGHT_LOADING_MISMATCH')
    llm.requires_grad_(False)
    use_lora = args.arm.startswith('lora')
    if use_lora:
        llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules='all-linear', task_type='CAUSAL_LM'))
    if any(p.requires_grad and 'lora_' not in name for name, p in llm.named_parameters()):
        raise RuntimeError('Unexpected trainable base LLM parameter')
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    llm.config.use_cache = False
    model = RRG(llm).cuda().train()
    if any(p.dtype != torch.float16 for p in model.llm.parameters() if not p.requires_grad):
        raise RuntimeError('Frozen LLM parameters must stay FP16')
    tokens = 256 if args.arm == 'lora256' else 64
    dataset = FeatureDataset(tokenizer, tokens=tokens)
    # Keep the 2000-study cohort; choose only a bounded timing minibatch with Findings.
    rows = []
    missing = 0
    for index in range(128):
        try:
            rows.append(dataset[index])
        except ValueError as exc:
            if str(exc).startswith('Missing Findings;'):
                missing += 1
                continue
            raise
        if len(rows) == 32:
            break
    if len(rows) < 16:
        raise RuntimeError('Too few Findings rows for bounded batch probe')
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, eps=1e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    probes = []

    def make_batch(size):
        batch = collate(rows[:size], tokenizer.pad_token_id)
        # Fixed 320 text positions gives a reproducible, conservative capacity limit.
        for key, pad in [('ids', tokenizer.pad_token_id), ('labels', -100), ('mask', 0)]:
            batch[key] = torch.nn.functional.pad(batch[key], (0, 320 - batch[key].shape[1]), value=pad)
        return {k: v.cuda() for k, v in batch.items()}

    def measure(size, update=False):
        batch = make_batch(size)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, lm, bce = model(**batch)
        if not torch.isfinite(loss):
            raise RuntimeError('STOP_NONFINITE_LOSS')
        scaler.scale(loss).backward()
        grad_norm = None
        if update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        result = {'batch': size, 'seconds': time.perf_counter() - start, 'loss': loss.item(),
                  'lm_loss': lm.item(), 'bce': bce.item(), 'grad_norm': grad_norm.item() if grad_norm is not None else None, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                  'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
        del batch, loss, lm, bce
        return result

    if tokens == 64:
        low, high = 0, None
        candidate = 1
        while candidate <= 32:
            try:
                measured = measure(candidate)
                probes.append({'status': 'PASS', **measured})
                low = candidate
            except torch.cuda.OutOfMemoryError:
                probes.append({'status': 'OOM', 'batch': candidate})
                if candidate == 1:
                    raise RuntimeError('STOP_OOM_BATCH_1')
                high = candidate
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({'arm': args.arm, 'probe': probes[-1]}), flush=True)
            if high is not None:
                if high - low <= 1:
                    break
                candidate = (low + high) // 2
            else:
                candidate *= 2
        batch_size = low
    else:
        batch_size = 1
    if batch_size == 0:
        raise RuntimeError('STOP_NO_USABLE_BATCH')
    # Probes omit Adam state. Reserve that state before final capacity verification.
    for parameter in parameters:
        optimizer.state[parameter] = {'step': torch.tensor(0.), 'exp_avg': torch.zeros_like(parameter),
                                      'exp_avg_sq': torch.zeros_like(parameter)}
    # Reconfirm chosen batch with Adam state allocated; reduce only for batch>1.
    while True:
        try:
            measured = measure(batch_size)
            probes.append({'phase': 'with_adam_state', 'status': 'PASS', **measured})
            break
        except torch.cuda.OutOfMemoryError:
            probes.append({'phase': 'with_adam_state', 'status': 'OOM', 'batch': batch_size})
            if batch_size == 1:
                raise RuntimeError('STOP_OOM_BATCH_1')
            batch_size -= 1
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
    checkpoint = base / 'features' / f'{args.arm}_last.pt'
    contract = {'arm': args.arm, 'code_sha256': {name: sha256(base / name) for name in ('core.py', 'data.py', 'model.py', 'preflight.py')}, 'manifest_index_sha256': sha256(INDEX), 'config_sha256': sha256(base / 'config.json'),
                'qwen_revision': downloads['models']['Qwen/Qwen3-1.7B']['revision'], 'text_padding': 320,
                'max_target_including_eos': 256, 'batch_size': batch_size, 'seed': 42, 'lr': 1e-4, 'grad_scaler_initial_scale': 128.,
                'lora': {'rank': 16, 'alpha': 32, 'targets': 'all-linear', 'dropout': 0.} if use_lora else None,
                'auxiliary_labels': 'synthetic all-zero 14 labels; F0 execution only', 'bce_weight': .1,
                'sampling': 'repeat same bounded training minibatch; omit missing Findings; no validation training'}
    completed = 0
    steps = []
    if args.resume and tokens != 64:
        raise ValueError('Optional 256-token timing has no resume checkpoint')
    if args.resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['contract'] != contract:
            raise ValueError('Resume contract mismatch')
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(state['trainables']) != expected:
            raise ValueError('Resume trainable parameter set mismatch')
        model.load_state_dict(state['trainables'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        completed, steps = state['step'], state['steps']
        if not 0 <= completed <= 20 or len(steps) != completed:
            raise ValueError('Resume step cap or history mismatch')
    elif checkpoint.exists():
        raise FileExistsError('Preserve existing F0 checkpoint; use --resume explicitly')
    def save_checkpoint(step):
        trainables = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({'step': step, 'contract': contract, 'trainables': trainables, 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'steps': steps}, temporary)
        temporary.replace(checkpoint)

    cap = 3 if tokens == 256 else 20
    for step in range(completed, cap):
        try:
            measured = measure(batch_size, update=True)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError('STOP_OOM_DURING_OPTIMIZER_STEP')
        measured['step'] = step + 1
        steps.append(measured)
        print(json.dumps({'arm': args.arm, **measured}), flush=True)
        if tokens == 64 and (step + 1) % 5 == 0:
            save_checkpoint(step + 1)
    if tokens == 64:
        # Verify and restore actual saved step-20 values before generation.
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if restored['step'] != 20 or restored['contract'] != contract:
            raise RuntimeError('Checkpoint verification failed')
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data.copy_(restored['trainables'][name].to(parameter.device))
        del restored
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    generations = []
    if tokens == 64:
        validation = FeatureDataset(tokenizer, tokens=tokens, split='mimic_validate')
        model.eval()
        model.llm.gradient_checkpointing_disable()
        with torch.inference_mode():
            for index in range(8):
                patches, scores = validation.features(index)
                prefix = model.prefix(patches[None].cuda(), scores[None].cuda())
                ids = torch.tensor([validation.prompt_ids], device='cuda')
                embeddings = torch.cat((prefix, model.llm.get_input_embeddings()(ids)), 1)
                mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device='cuda')
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.llm.generate(inputs_embeds=embeddings, attention_mask=mask, max_new_tokens=128,
                                             do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                             eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                generations.append({'index': index, 'seconds': time.perf_counter() - start, 'tokens': output.shape[-1],
                                    'hit_cap': output.shape[-1] == 128, 'text': tokenizer.decode(output[0], skip_special_tokens=True)})
        generation_path = base / 'features' / f'{args.arm}_generated.json'
        if generation_path.exists():
            raise FileExistsError('Preserve previous generated reports')
        generation_path.write_text(json.dumps(generations, indent=2) + '\n')
    total_samples = sum(s['batch'] for s in steps)
    result = {'status': 'PASS', 'classification': 'technical preflight, not quality evaluation', 'contract': contract,
              'optimizer_steps': len(steps), 'seconds_per_step': sum(s['seconds'] for s in steps) / len(steps),
              'seconds_per_sample': sum(s['seconds'] for s in steps) / total_samples,
              'peak_allocated_mib': max(s['peak_allocated_mib'] for s in steps),
              'peak_reserved_mib': max(s['peak_reserved_mib'] for s in steps), 'first_loss': steps[0]['loss'], 'last_loss': steps[-1]['loss'],
              'loss_decreased': steps[-1]['loss'] < steps[0]['loss'], 'finite': True, 'probes': probes,
              'capacity_limit': 'fixed 320 text positions + 14 finding tokens + patch tokens; search up to batch32; optimizer state included',
              'missing_findings_skipped_in_bounded_sampler': missing, 'trainable_parameters': sum(p.numel() for p in parameters),
              'generation_count': len(generations),
              'seconds_per_report': sum(g['seconds'] for g in generations) / len(generations) if generations else None,
              'generation_tokens': [g['tokens'] for g in generations], 'generation_max_new_tokens': 128, 'steps': steps}
    (base / f'{args.arm}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('steps', 'probes')}), flush=True)


if __name__ == '__main__':
    main()
