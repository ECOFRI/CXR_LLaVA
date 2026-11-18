"""CPU-only validation of the proposed 640-token diagnostic protocol."""

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import Blip2Processor, GenerationConfig


OUT = Path(__file__).resolve().parent
VR = OUT.parents[1]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / "scripts/train"))
sys.path.insert(0, str(VR / "scripts/experiments"))
sys.path.insert(0, str(VR / "scripts/p5"))
from p4_projection_gate import target_for, tokenize_training  # noqa: E402
from p4_paired_infer import SingleSequenceEOS, continuation_tokens, generate_single_sequence, paired_cohort, stable_key  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, name, details, results):
    if not condition:
        raise AssertionError(name + ": " + str(details))
    results.append({"name": name, "status": "PASS", "details": details})


class CaptureModel:
    def __init__(self, returned):
        self.language_model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=50118))
        self.returned = returned
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.returned


def causal_loss_test(tokenizer, config, results):
    ids, labels, _, _ = tokenize_training(tokenizer, config, "No acute cardiopulmonary disease.")
    ids, labels = ids[0], labels[0]
    prompt_end = int((labels != -100).nonzero()[0].item())
    check(torch.equal(ids[prompt_end:], labels[prompt_end:]) and bool((labels[:prompt_end] == -100).all())
          and labels[-1].item() == tokenizer.eos_token_id,
          "target_mask_and_eos", {"prompt_plus_bos_masked": prompt_end,
                                   "supervised_tokens": len(labels) - prompt_end}, results)
    # Same final-slice and one-position shift used by installed BLIP-2 forward.
    vocab = max(ids).item() + 1
    torch.manual_seed(42)
    logits = torch.randn((1, len(ids) + 32, vocab), dtype=torch.float32)
    sliced = logits[:, -len(labels):, :]
    shifted = sliced[:, :-1, :]
    targets = labels[1:].reshape(-1)
    base = torch.nn.functional.cross_entropy(shifted.reshape(-1, vocab), targets, ignore_index=-100)
    ignored = sliced.clone()
    ignored[:, :prompt_end - 1, :] += 1000
    ignored_loss = torch.nn.functional.cross_entropy(ignored[:, :-1, :].reshape(-1, vocab), targets,
                                                      ignore_index=-100)
    supervised = sliced.clone()
    supervised[:, prompt_end - 1, labels[prompt_end].item()] += 100
    supervised_loss = torch.nn.functional.cross_entropy(supervised[:, :-1, :].reshape(-1, vocab), targets,
                                                         ignore_index=-100)
    check(bool(torch.isfinite(base)) and math.isclose(base.item(), ignored_loss.item(), abs_tol=1e-6)
          and supervised_loss.item() < base.item(), "causal_shift_ignore_index",
          {"ignored_positions_do_not_change_loss": True,
           "supervised_target_position_changes_loss": True}, results)


def checkpoint_test(config, index, source, p5run, results):
    path = p5run / "P5.2_last.pt"
    status = json.loads((p5run / "P5.2_status.json").read_text())
    infer_status = json.loads((p5run / "P5.3_status.json").read_text())
    sha = digest(path)
    payload = torch.load(path, map_location="cpu")
    check(sha == status["checkpoint_sha256"] == infer_status["checkpoint_sha256"]
          and payload["config_sha256"] == digest(VR / "configs/experiments/P5_qformer_contract.approved_20261025.json")
          and payload["manifest_sha256"] == index["manifests"]["mimic_train"]["sha256"]
          and payload["stage"] == "P5.2" and payload["optimizer_updates"] == payload["next_position"] == 2000
          and payload["sampler_state"]["cursor"] == 2000,
          "checkpoint_identity_and_cursor", {"checkpoint_sha256": sha, "optimizer_updates": 2000}, results)
    weight_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    adapter = payload["adapter"]
    counts = {"qformer": 0, "query_tokens": 0, "projection": 0}
    handles = {}
    for name, value in adapter.items():
        if name.startswith("qformer.inner."):
            group = "qformer"
            original_name = "qformer." + name[len("qformer.inner."):]
        elif name == "query_tokens":
            group = "query_tokens"
            original_name = name
        elif name.startswith("language_projection.layer."):
            group = "projection"
            original_name = "language_projection." + name[len("language_projection.layer."):]
        else:
            raise AssertionError("Unexpected adapter key: " + name)
        file_name = weight_map[original_name]
        if file_name not in handles:
            handles[file_name] = safe_open(source / file_name, framework="pt", device="cpu")
        check_shape = tuple(handles[file_name].get_slice(original_name).get_shape())
        if tuple(value.shape) != check_shape:
            raise AssertionError("Adapter/source shape mismatch: " + name)
        counts[group] += value.numel()
    check(counts == {"qformer": 105137664, "query_tokens": 24576, "projection": 1968640}
          and sum(counts.values()) == config["model"]["trainable_parameter_count"],
          "checkpoint_adapter_loader_interface", {"group_parameters": counts,
                                                   "adapter_tensors": len(adapter)}, results)


def generation_test(p4config, p5config, source, tokenizer, results):
    saved_gen = GenerationConfig.from_pretrained(source, local_files_only=True)
    gen = dict(p5config["generation"])
    gen["max_new_tokens"] = 640
    check(saved_gen.max_length == 20 and saved_gen.max_new_tokens is None
          and saved_gen.eos_token_id == 50118 and tokenizer.eos_token_id == 2
          and p4config["generation"] == p5config["generation"]
          and gen["eos_token_id"] == tokenizer.eos_token_id
          and gen["bos_token_id"] == tokenizer.bos_token_id
          and gen["pad_token_id"] == tokenizer.pad_token_id,
          "pinned_tokenizer_and_eos_override",
          {"checkpoint_generation_max_length": saved_gen.max_length,
           "checkpoint_generation_max_new_tokens": saved_gen.max_new_tokens,
           "checkpoint_generation_eos": saved_gen.eos_token_id, "diagnostic_eos": gen["eos_token_id"],
           "bos": gen["bos_token_id"], "pad": gen["pad_token_id"]}, results)
    mock = CaptureModel(torch.tensor([[2, 14, 17, 2]]))
    returned = generate_single_sequence(mock, {"pixel_values": torch.zeros(1, 3, 2, 2)}, gen)
    kwargs = mock.kwargs
    criterion = kwargs["stopping_criteria"]
    check(kwargs["max_new_tokens"] == 640 and "max_length" not in kwargs
          and kwargs["eos_token_id"] is None and kwargs["bos_token_id"] == 2
          and kwargs["pad_token_id"] == 1 and kwargs["do_sample"] is False
          and kwargs["num_beams"] == 1 and kwargs["repetition_penalty"] == 1.0
          and kwargs["no_repeat_ngram_size"] == 0 and len(criterion) == 1
          and isinstance(criterion[0], SingleSequenceEOS)
          and mock.language_model.generation_config.eos_token_id == 50118,
          "runtime_generation_kwargs_640",
          {"max_new_tokens_forwarded": kwargs["max_new_tokens"], "max_length_forwarded": False,
           "eos_kwarg": None, "custom_stopping_eos": criterion[0].eos_token_id,
           "checkpoint_generation_eos_restored": 50118}, results)
    check(not criterion(torch.tensor([[2, 14, 17]]), None)
          and criterion(torch.tensor([[2, 14, 17, 2]]), None),
          "custom_eos_stopping", {"stops_on_2": True, "does_not_stop_on_other_token": True}, results)
    actual = continuation_tokens(returned, gen["bos_token_id"])
    full = continuation_tokens(torch.tensor([[2] + [17] * 640]), gen["bos_token_id"])
    check(actual.shape[1] == 3 and full.shape[1] == 640,
          "continuation_length_excludes_synthetic_bos", {"eos_sample_new_tokens": 3,
                                                           "capped_sample_new_tokens": 640}, results)


def main():
    results = []
    p4cfg_path = VR / "configs/experiments/P4_projection_contract.approved.json"
    p5cfg_path = VR / "configs/experiments/P5_qformer_contract.approved_20261025.json"
    p4cfg = json.loads(p4cfg_path.read_text())
    p5cfg = json.loads(p5cfg_path.read_text())
    p4run = VR / "runs/P4_projection_pilot_20261023"
    p5run = VR / "runs/P5_qformer_20261025"
    index_path = p4run / "manifest_index.approved.json"
    index = json.loads(index_path.read_text())
    source = ROOT / p5cfg["model"]["checkpoint_dir"]
    processor = Blip2Processor.from_pretrained(source, local_files_only=True, use_fast=False)
    tokenizer = processor.tokenizer
    causal_loss_test(tokenizer, p5cfg, results)
    checkpoint_test(p5cfg, index, source, p5run, results)
    generation_test(p4cfg, p5cfg, source, tokenizer, results)
    rows = paired_cohort(index)
    selected = [row for cohort in ("mimic", "indiana")
                for row in [r for r in rows if r["dataset"] == cohort][:20]]
    selected_keys = [stable_key(row) for row in selected]
    check(len(selected) == len(set(selected_keys)) == 40
          and {cohort: sum(row["dataset"] == cohort for row in selected)
               for cohort in ("mimic", "indiana")} == {"mimic": 20, "indiana": 20},
          "fixed_smoke_selection", {"rule": "first 20 in each approved ordered cohort",
                                    "mimic": 20, "indiana": 20}, results)
    output = {"classification": "CPU-only diagnostic; no model inference or training",
              "status": "PASS", "checks": results,
              "protocol": {"new_max_new_tokens": 640, "only_generation_delta": "max_new_tokens: 128 -> 640",
                           "source_p4_config_sha256": digest(p4cfg_path),
                           "source_p5_config_sha256": digest(p5cfg_path),
                           "manifest_index_sha256": digest(index_path),
                           "selected_stable_keys_sha256": hashlib.sha256(
                               json.dumps(selected_keys, separators=(",", ":")).encode()).hexdigest()},
              "limits": ["Mock generation verifies forwarded kwargs, not GPU runtime or output quality",
                         "No full BLIP-2 model loaded; adapter tensors and source shapes checked on CPU",
                         "No 640-token reports generated; historical 128-token outputs unchanged"]}
    path = OUT / "cpu_validation_results.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)
    print(json.dumps({"status": output["status"], "checks": len(results), "output": str(path)}))


if __name__ == "__main__":
    main()
