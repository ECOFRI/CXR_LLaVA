"""Shared local-only Qwen loading and batched generation for F1."""
import json
import random
import time
import torch
from research.rrg_f0.core import sha256
from research.rrg_f0.encoder import checkpoint_path
from research.rrg_f0.data import PROMPT
from research.rrg_f1.cohort import BASE, PRIVATE
from research.rrg_f1.data import rows_for, read_feature
from research.rrg_f1.model import ArmModel


def protocol(arm):
    from research.rrg_f0.core import ROOT
    files = ['research/rrg_f0/core.py', 'research/rrg_f0/model.py', 'research/rrg_f1/model.py',
             'research/rrg_f1/data.py', 'research/rrg_f1/bad_images.py', 'research/rrg_f1/train.py', 'research/rrg_f1/runtime.py',
             'research/rrg_f1/config.json', 'research/rrg_f1/train_config.json', 'research/rrg_f1/labeler.py',
             'research/rrg_f1/train_label_summary.json']
    return {'arm': arm, 'code_and_config_sha256': {name: sha256(ROOT / name) for name in files},
            'manifest_sha256': sha256(PRIVATE / 'train_manifest.jsonl'),
            'labels_sha256': sha256(PRIVATE / 'train_labels.json'), 'exclusions_sha256': sha256(PRIVATE / 'bad_images.json') if (PRIVATE / 'bad_images.json').exists() else None}


def build_model(arm):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    path = checkpoint_path('Qwen/Qwen3-1.7B')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    llm, info = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16,
                                                   attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Qwen pretrained weight mismatch')
    llm.requires_grad_(False)
    llm = get_peft_model(llm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0., target_modules='all-linear', task_type='CAUSAL_LM'))
    if any(p.requires_grad and 'lora_' not in name for name, p in llm.named_parameters()):
        raise ValueError('Unexpected trainable base LLM parameter')
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    llm.config.use_cache = False
    from research.rrg_f1.bad_images import active_indices
    raw = torch.tensor(json.loads((PRIVATE / 'train_labels.json').read_text()))
    raw = raw[active_indices(len(raw))]
    prior = ((raw == 1).sum(0).float() + 1) / ((raw != -1).sum(0).float() + 2)
    model = ArmModel(llm, arm, prior).cuda()
    if any(p.dtype != torch.float16 for p in model.llm.parameters() if not p.requires_grad):
        raise ValueError('Frozen LLM dtype changed')
    return model, tokenizer


def restore_model(model, arm):
    checkpoint = torch.load(PRIVATE / arm / 'last.pt', map_location='cpu', weights_only=True)
    if checkpoint['contract'] != protocol(arm):
        raise ValueError('Checkpoint protocol changed')
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(checkpoint['trainables']):
        raise ValueError('Checkpoint trainable set changed')
    model.load_state_dict(checkpoint['trainables'], strict=False)
    return checkpoint


@torch.inference_mode()
def generate(model, tokenizer, cases, score_override=None):
    was_training = model.training
    model.eval()
    model.llm.gradient_checkpointing_disable()
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': PROMPT}], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    cohorts = {family: rows_for(family) for family in {family for family, _ in cases}}
    outputs = []
    for start in range(0, len(cases), 4):
        group = cases[start:start + 4]
        features = [read_feature(family, i, cohorts[family][i]) for family, i in group]
        patches = torch.stack([f[0] for f in features]).cuda()
        scores = torch.stack([f[1] for f in features]).cuda()
        if score_override is not None:
            scores = score_override[start:start + len(group)].cuda()
        prefix = model.prefix(patches, scores)
        ids = torch.tensor([prompt] * len(group), device='cuda')
        embeddings = torch.cat((prefix, model.llm.get_input_embeddings()(ids)), 1)
        mask = torch.ones(embeddings.shape[:2], dtype=torch.long, device='cuda')
        torch.cuda.synchronize()
        begin = time.monotonic()
        generated = model.llm.generate(inputs_embeds=embeddings, attention_mask=mask, max_new_tokens=256,
                                       do_sample=False, use_cache=True, num_beams=1, temperature=1., top_p=1., top_k=0,
                                       pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        latency = (time.monotonic() - begin) / len(group)
        for (family, index), sequence in zip(group, generated.cpu().tolist()):
            length = sequence.index(tokenizer.eos_token_id) + 1 if tokenizer.eos_token_id in sequence else len(sequence)
            outputs.append({'family': family, 'index': index, 'generated_report': tokenizer.decode(sequence[:length], skip_special_tokens=True),
                            'generated_tokens': length, 'hit_max_length': length == 256,
                            'latency_seconds': latency})
    if was_training:
        model.train()
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return outputs
