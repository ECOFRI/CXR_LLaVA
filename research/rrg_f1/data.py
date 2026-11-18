"""Private F1 rows, local feature cache and masked Findings targets."""
import hashlib
import json
import torch
from torch.utils.data import Dataset
from research.rrg_f0.core import ROOT, pool_grid, sha256
from research.rrg_f0.data import PROMPT, collate as f0_collate
from research.rrg_f1.bad_images import active_indices
from research.rrg_f1.cohort import BASE, PRIVATE, atomic_json, dev_rows


def rows_for(family):
    if family == 'train':
        status = json.loads((PRIVATE / 'cohort_state.json').read_text())
        path = PRIVATE / 'train_manifest.jsonl'
        if status['status'] != 'COMPLETE' or sha256(path) != status['manifest_sha256']:
            raise ValueError('Cohort is incomplete or changed')
        return [json.loads(s) for s in path.read_text().splitlines()]
    return dev_rows(family)


def binding(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def feature_path(family, index):
    if family == 'mimic':
        return ROOT / 'research/rrg_f0/features' / f'mimic_validate_{index:04d}.pt'
    return PRIVATE / 'features' / f'{family}_{index:05d}.pt'


def read_feature(family, index, row):
    payload = torch.load(feature_path(family, index), map_location='cpu', weights_only=True)
    if family != 'mimic' and payload['binding'] != binding(row):
        raise ValueError('Feature/cohort linkage mismatch')
    if family == 'mimic' and (payload['index'] != index or payload['split'] != 'mimic_validate'):
        raise ValueError('F0 validation cache row mismatch')
    patches, scores = payload['patch_tokens'], payload['scores']
    if patches.shape != (256, 768) or patches.dtype != torch.float16 or scores.shape != (14,):
        raise ValueError('Feature shape/dtype changed')
    return pool_grid(patches), scores


def bce_targets(raw):
    return (raw == 1).float(), (raw != -1).float()


class TrainDataset(Dataset):
    def __init__(self):
        self.rows = rows_for('train')
        state = torch.load(PRIVATE / 'encodings.pt', map_location='cpu', weights_only=True)
        if state['manifest_sha256'] != sha256(PRIVATE / 'train_manifest.jsonl'):
            raise ValueError('Encoding/cohort mismatch')
        self.encodings = state['rows']
        self.active = active_indices(len(self.rows))
        self.labels = json.loads((PRIVATE / 'train_labels.json').read_text())
        if len(self.rows) != len(self.labels) or len(self.rows) != len(self.encodings):
            raise ValueError('Training label/encoding count mismatch')

    def __len__(self):
        return len(self.active)

    def __getitem__(self, index):
        index = self.active[index]
        patches, scores = read_feature('train', index, self.rows[index])
        raw = torch.tensor(self.labels[index], dtype=torch.long)
        ids, labels = self.encodings[index]
        targets, valid = bce_targets(raw)
        return {'patches': patches, 'scores': scores, 'ids': ids, 'labels': labels,
                'auxiliary_targets': targets, 'auxiliary_mask': valid}


def collate(rows):
    pad_id = 151645
    result = f0_collate(rows, pad_id)
    result['auxiliary_mask'] = torch.stack([row['auxiliary_mask'] for row in rows])
    for key, pad in [('ids', pad_id), ('labels', -100), ('mask', 0)]:
        if result[key].shape[1] > 320:
            raise ValueError('Text exceeds locked 320 positions')
        result[key] = torch.nn.functional.pad(result[key], (0, 320 - result[key].shape[1]), value=pad)
    return result


def prepare():
    from transformers import AutoTokenizer
    from research.rrg_f0.encoder import checkpoint_path
    from research.rrg_f1.labeler import train_targets
    rows, texts = train_targets()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path('Qwen/Qwen3-1.7B'), local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise ValueError('Unexpected Qwen EOS')
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
    encoded, truncated = [], 0
    for target in texts:
        ids = tokenizer.encode(target, add_special_tokens=False)
        truncated += len(ids) > 255
        used = ids[:255] + [tokenizer.eos_token_id]
        encoded.append((torch.tensor(prompt + used), torch.tensor([-100] * len(prompt) + used)))
    path = PRIVATE / 'encodings.pt'
    state = {'manifest_sha256': sha256(PRIVATE / 'train_manifest.jsonl'), 'rows': encoded, 'prompt_ids': prompt}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    atomic_json(BASE / 'encoding_summary.json', {'status': 'PASS', 'rows': len(rows), 'truncated_targets': truncated,
                 'target_max_including_eos': 256, 'text_padding': 320, 'prompt_tokens': len(prompt)})


if __name__ == '__main__':
    prepare()
