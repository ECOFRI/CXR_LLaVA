"""Dataset reads only local frozen P4 references/features. No reference in prompt."""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from research.rrg_f0.core import ROOT, INDEX, EXPECTED_CLEAR_SHA, cohort, pool_grid, sha256

sys.path.insert(0, str(ROOT / 'research/blip2_cxr_v4/src'))
from vlm_medical.data.manifests import report_sections

PROMPT = 'Write the Findings section of a chest X-ray report.'


class FeatureDataset(Dataset):
    def __init__(self, tokenizer, tokens=64, split='mimic_train', max_target=256):
        if tokens not in (64, 256):
            raise ValueError('Only authorized 64/256 patch token counts')
        base = ROOT / 'research/rrg_f0'
        contract = json.loads((base / 'feature_contract.json').read_text())
        config = json.loads((base / 'config.json').read_text())
        if (contract['status'] != 'PASS' or contract['manifest_index_sha256'] != sha256(INDEX)
                or contract['clear_sha256'] != EXPECTED_CLEAR_SHA or contract['labels'] != config['finding_labels']):
            raise ValueError('Feature cache provenance mismatch')
        self.rows = cohort()[split]
        self.tokenizer, self.tokens, self.split, self.max_target = tokenizer, tokens, split, max_target
        self.prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=False)

    def __len__(self):
        return len(self.rows)

    def features(self, index):
        path = ROOT / 'research/rrg_f0/features' / f'{self.split}_{index:04d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if (payload['split'], payload['index']) != (self.split, index):
            raise ValueError('Feature row mismatch')
        patches = payload['patch_tokens']
        if patches.shape != (256, 768) or patches.dtype != torch.float16:
            raise ValueError('Expected FP16 256x768 disk features')
        if self.tokens == 64:
            patches = pool_grid(patches)
        return patches, payload['scores']

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['reference_path'])
        if sha256(path) != row['reference_file_sha256']:
            raise ValueError('Reference hash mismatch')
        findings = report_sections(path.read_text(errors='replace')).get('findings', '')
        if not findings.strip():
            raise ValueError('Missing Findings; STOP rather than substitute Impression')
        target_ids = self.tokenizer.encode(findings, add_special_tokens=False)[:self.max_target - 1]
        target_ids += [self.tokenizer.eos_token_id]
        ids = self.prompt_ids + target_ids
        labels = [-100] * len(self.prompt_ids) + target_ids
        patches, scores = self.features(index)
        return {'patches': patches, 'scores': scores, 'ids': torch.tensor(ids), 'labels': torch.tensor(labels),
                'auxiliary_targets': torch.zeros(14), 'target_tokens': len(target_ids)}


def collate(rows, pad_id):
    length = max(len(row['ids']) for row in rows)
    result = {key: torch.stack([row[key] for row in rows]) for key in ('patches', 'scores', 'auxiliary_targets')}
    result['ids'] = torch.full((len(rows), length), pad_id, dtype=torch.long)
    result['labels'] = torch.full((len(rows), length), -100, dtype=torch.long)
    result['mask'] = torch.zeros((len(rows), length), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row['ids'])
        result['ids'][i, :n] = row['ids']
        result['labels'][i, :n] = row['labels']
        result['mask'][i, :n] = 1
    return result
