"""One F1 epoch with matched order, masked auxiliary loss, collapse gates and resume."""
import argparse
import json
import math
import random
import time
import torch
from torch.utils.data import DataLoader, Subset
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json
from research.rrg_f1.data import TrainDataset, collate
from research.rrg_f1.runtime import build_model, protocol, restore_model, generate


def collapse_statistics(predictions):
    texts = [p['generated_report'].strip() for p in predictions]
    unique = len(set(texts))
    capped = sum(p['hit_max_length'] for p in predictions)
    repeats = []
    for text in texts:
        words = text.lower().split()
        grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
        repeats.append(1 - len(set(grams)) / len(grams) if grams else 0.)
    return {'unique_reports': unique, 'cap_count': capped, 'mean_repeated_4gram_fraction': sum(repeats) / len(repeats),
            'bad': unique < 4 or capped >= 12}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A2', 'A3', 'A4'], required=True)
    args = parser.parse_args()
    directory = PRIVATE / args.arm
    directory.mkdir(exist_ok=True)
    summary_path = BASE / f'{args.arm}_summary.json'
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if saved['status'] in ('COMPLETE', 'COLLAPSE'):
            print('ARM_ALREADY_TERMINAL', flush=True)
            return
    config = json.loads((BASE / 'train_config.json').read_text())
    model, tokenizer = build_model(args.arm)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], eps=config['adam_epsilon'], weight_decay=config['weight_decay'])
    from research.rrg_f1.bad_images import active_indices
    train_count = len(active_indices(30000))
    total_updates = (train_count // 1000) * math.ceil(1000 / 32) + math.ceil((train_count % 1000) / 32)
    warmup = max(1, int(total_updates * .05))

    def schedule(update):
        if update < warmup:
            return (update + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1., (update - warmup) / (total_updates - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    state = {'position': 0, 'updates': 0, 'checks': [], 'bad_streak': 0, 'status': 'RUNNING', 'attempt': 0,
             'contract': protocol(args.arm)}
    if (directory / 'last.pt').exists():
        restored = restore_model(model, args.arm)
        state = restored['state']
        optimizer.load_state_dict(restored['optimizer'])
        scheduler.load_state_dict(restored['scheduler'])
        scaler.load_state_dict(restored['scaler'])
        torch.set_rng_state(restored['torch_rng'])
        torch.cuda.set_rng_state_all(restored['cuda_rng'])
        del restored
    attempt_path = directory / 'attempt_counter.json'
    attempt = json.loads(attempt_path.read_text())['attempt'] + 1 if attempt_path.exists() else 1
    atomic_json(attempt_path, {'attempt': attempt})
    state['attempt'] = attempt
    dataset = TrainDataset()
    order = list(range(len(dataset)))
    random.Random(42).shuffle(order)
    if len(order) != train_count:
        raise ValueError('Training cohort must contain the effective eligible cohort')
    started = time.monotonic()
    step_log = (directory / f'updates_attempt_{attempt:03d}.jsonl').open('x')
    last_loss = None

    def checkpoint():
        payload = {'contract': protocol(args.arm), 'state': state,
                   'trainables': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                   'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        path = directory / 'last.tmp'
        torch.save(payload, path)
        path.replace(directory / 'last.pt')
        atomic_json(directory / 'heartbeat.json', {'status': state['status'], 'position': state['position'],
                    'updates': state['updates'], 'bad_streak': state['bad_streak'], 'unix_time': time.time(),
                    'seconds_this_attempt': time.monotonic() - started})

    if state['position'] == 0:
        checkpoint()
    while state['position'] < train_count and state['status'] not in ('COMPLETE', 'COLLAPSE'):
        segment_end = min(((state['position'] // 1000) + 1) * 1000, train_count)
        indices = order[state['position']:segment_end]
        loader = DataLoader(Subset(dataset, indices), batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
        iterator = iter(loader)
        remaining = len(indices)
        while remaining:
            group_samples = min(32, remaining)
            batches = math.ceil(group_samples / 4)
            optimizer.zero_grad(set_to_none=True)
            lm_sum, bce_sum = 0., 0.
            begin = time.monotonic()
            for _ in range(batches):
                batch = {k: v.cuda(non_blocking=True) for k, v in next(iterator).items()}
                loss, lm, bce = model(**batch)
                if not torch.isfinite(loss):
                    raise RuntimeError('STOP_NONFINITE_LOSS')
                fraction = batch['ids'].shape[0] / group_samples
                scaler.scale(loss * fraction).backward()
                lm_sum += lm.item() * fraction
                bce_sum += bce.item() * fraction
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            remaining -= group_samples
            state['position'] += group_samples
            state['updates'] += 1
            last_loss = {'position': state['position'], 'update': state['updates'], 'lm_loss': lm_sum, 'bce_loss': bce_sum,
                         'grad_norm': norm.item(), 'learning_rate': scheduler.get_last_lr()[0], 'samples': group_samples,
                         'seconds': time.monotonic() - begin}
            step_log.write(json.dumps(last_loss) + '\n')
            step_log.flush()
            atomic_json(directory / 'heartbeat.json', {**last_loss, 'status': 'TRAINING', 'unix_time': time.time()})
        del iterator, loader
        cases = [('mimic', i) for i in range(8)] + [('indiana', i) for i in range(8)]
        predictions = generate(model, tokenizer, cases)
        path = directory / f'dev_{state["position"]:05d}_attempt_{attempt:03d}.json'
        atomic_json(path, predictions)
        stats = collapse_statistics(predictions)
        state['bad_streak'] = state['bad_streak'] + 1 if stats['bad'] else 0
        state['checks'].append({'position': state['position'], **stats})
        if state['bad_streak'] >= 2:
            state['status'] = 'COLLAPSE'
        elif state['position'] == train_count:
            state['status'] = 'COMPLETE'
        checkpoint()  # includes each requested 2000-sample boundary and the dev gates
        print(json.dumps({'arm': args.arm, 'position': state['position'], 'status': state['status'], **stats}), flush=True)
        if state['status'] == 'COLLAPSE':
            break
    step_log.close()
    atomic_json(summary_path, {'status': state['status'], 'arm': args.arm, 'examples_seen': state['position'],
                'optimizer_updates': state['updates'], 'checks': state['checks'], 'last_training_step': last_loss,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                'wall_seconds_this_attempt': time.monotonic() - started, 'classification': 'training execution; dev checks only'})


if __name__ == '__main__':
    main()
