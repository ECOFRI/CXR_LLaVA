"""Read-only P4/P5 artifact diagnostics. Writes only beside this script."""

import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path


VR = Path(__file__).resolve().parents[2]
ROOT = VR.parents[1]
sys.path.insert(0, str(VR / "scripts/train"))
sys.path.insert(0, str(VR / "scripts/eval"))
sys.path.insert(0, str(VR / "src"))
from p4_projection_gate import target_for, tokenize_training  # noqa: E402
from p4_evaluate import read_inputs  # noqa: E402
from vlm_medical.eval.text_metrics import case_statistics, corpus_scores, tokenize  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def repeat_fourgrams(text):
    words = tokenize(text)
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.0


def summarize_steps(rows):
    segments = {"first100": rows[:100], "steps701_800": rows[700:800],
                "steps901_1000": rows[900:1000], "last100": rows[-100:]}
    return {name: {"n": len(part), "loss_mean": statistics.mean(x["loss"] for x in part),
                   "loss_median": statistics.median(x["loss"] for x in part),
                   "qformer_grad_median": statistics.median(x["gradient_norms"]["qformer"] for x in part),
                   "query_grad_median": statistics.median(x["gradient_norms"]["query"] for x in part),
                   "projection_grad_median": statistics.median(x["gradient_norms"]["projection"] for x in part),
                   "query_grad_below_1e_5": sum(x["gradient_norms"]["query"] < 1e-5 for x in part)}
            for name, part in segments.items()}


def checkpoint_diagnostics(p5run, p4run, source_dir):
    import torch
    from safetensors import safe_open

    p5path = p5run / "P5.2_last.pt"
    payload = torch.load(p5path, map_location="cpu")
    adapter = payload["adapter"]
    index = json.loads((source_dir / "model.safetensors.index.json").read_text())["weight_map"]
    groups = defaultdict(lambda: {"elements": 0, "tensors": 0, "unchanged_tensors": 0,
                                  "nonfinite_elements": 0, "source_sq": 0.0,
                                  "trained_sq": 0.0, "delta_sq": 0.0, "dot": 0.0,
                                  "max_abs_delta": 0.0})
    with ExitStack() as stack:
        handles = {name: stack.enter_context(safe_open(source_dir / name, framework="pt", device="cpu"))
                   for name in set(index.values())}
        for name, value in adapter.items():
            if name.startswith("qformer.inner."):
                group = "qformer"
                native_name = "qformer." + name[len("qformer.inner."):]
            elif name == "query_tokens":
                group = "query_tokens"
                native_name = name
            elif name.startswith("language_projection.layer."):
                group = "projection"
                native_name = "language_projection." + name[len("language_projection.layer."):]
            else:
                raise ValueError("Unexpected P5 adapter key: " + name)
            original = handles[index[native_name]].get_tensor(native_name).to(torch.float16).float()
            trained = value.float()
            if trained.shape != original.shape:
                raise ValueError("P5/source parameter shape differs: " + name)
            diff = trained - original
            item = groups[group]
            item["elements"] += trained.numel()
            item["tensors"] += 1
            item["unchanged_tensors"] += int(torch.equal(trained, original))
            item["nonfinite_elements"] += int((~torch.isfinite(trained)).sum().item())
            item["source_sq"] += float(original.square().sum().item())
            item["trained_sq"] += float(trained.square().sum().item())
            item["delta_sq"] += float(diff.square().sum().item())
            item["dot"] += float((original * trained).sum().item())
            item["max_abs_delta"] = max(item["max_abs_delta"], float(diff.abs().max().item()))
    output = {}
    for name, item in groups.items():
        output[name] = {k: item[k] for k in ("elements", "tensors", "unchanged_tensors",
                                               "nonfinite_elements", "max_abs_delta")}
        output[name]["source_rms"] = math.sqrt(item["source_sq"] / item["elements"])
        output[name]["trained_rms"] = math.sqrt(item["trained_sq"] / item["elements"])
        output[name]["delta_rms"] = math.sqrt(item["delta_sq"] / item["elements"])
        output[name]["relative_l2_change"] = math.sqrt(item["delta_sq"] / item["source_sq"])
        output[name]["cosine_similarity"] = item["dot"] / math.sqrt(item["source_sq"] * item["trained_sq"])
    p4payload = torch.load(p4run / "P4.2_last.pt", map_location="cpu")
    return {"p5_checkpoint_sha256": digest(p5path),
            "stage": payload["stage"], "next_position": payload["next_position"],
            "optimizer_updates": payload["optimizer_updates"], "epoch": payload["epoch"],
            "manifest_sha256": payload["manifest_sha256"], "config_sha256": payload["config_sha256"],
            "sampler_cursor": payload["sampler_state"]["cursor"],
            "optimizer_state_entries": len(payload["optimizer"]["state"]),
            "trainable_parameter_count": sum(item["elements"] for item in output.values()),
            "groups": output,
            "p4_checkpoint_sha256": digest(p4run / "P4.2_last.pt"),
            "p4_adapter_parameters": sum(value.numel() for value in p4payload["adapter"].values())}


def main():
    from transformers import Blip2Config, Blip2Processor

    p4run = VR / "runs/P4_projection_pilot_20261023"
    p5run = VR / "runs/P5_qformer_20261025"
    p4cfg_path = VR / "configs/experiments/P4_projection_contract.approved.json"
    p5cfg_path = VR / "configs/experiments/P5_qformer_contract.approved_20261025.json"
    p4cfg = json.loads(p4cfg_path.read_text())
    p5cfg = json.loads(p5cfg_path.read_text())
    index_path = p4run / "manifest_index.approved.json"
    index = json.loads(index_path.read_text())
    source_dir = ROOT / p5cfg["model"]["checkpoint_dir"]
    rows = {}
    manifests = {}
    for name, item in index["manifests"].items():
        path = ROOT / item["path"]
        records = jsonl(path)
        keys = [(r["dataset"], r["subject_id"], r["study_id"]) for r in records]
        images = [r["image_id"] for r in records]
        manifests[name] = {"path": item["path"], "sha256": digest(path),
                           "recorded_sha256": item["sha256"], "rows": len(records),
                           "unique_studies": len(set(keys)), "unique_images": len(set(images)),
                           "splits": dict(Counter(r["split"] for r in records))}
        rows[name] = records
    train_subjects = {r["subject_id"] for r in rows["mimic_train"]}
    val_subjects = {r["subject_id"] for r in rows["mimic_validate"]}
    source = Blip2Config.from_pretrained(source_dir, local_files_only=True)
    processor = Blip2Processor.from_pretrained(source_dir, local_files_only=True, use_fast=False)
    tok = processor.tokenizer
    prompt_ids = tok.encode(p5cfg["training_text"]["prompt"] + p5cfg["training_text"]["separator_before_target"], add_special_tokens=False)
    target_lengths = []
    used_lengths = []
    supervised_lengths = []
    reference_repeat = []
    first_target_pieces = Counter()
    for row in rows["mimic_train"]:
        target = target_for(row)
        ids, labels, before, used = tokenize_training(tok, p5cfg, target)
        ids, labels = ids[0].tolist(), labels[0].tolist()
        first = next(i for i, value in enumerate(labels) if value != -100)
        if (first != len(prompt_ids) + 1 or labels[first:-1] != ids[first:-1]
                or labels[-1] != tok.eos_token_id or any(value != -100 for value in labels[:first])):
            raise ValueError("Training label mask/target alignment failed")
        target_lengths.append(before)
        used_lengths.append(used)
        supervised_lengths.append(sum(value != -100 for value in labels))
        reference_repeat.append(repeat_fourgrams(target))
        first_target_pieces[tok.decode(ids[first:first + 3], clean_up_tokenization_spaces=False)] += 1
    p4pairs = jsonl(p4run / "P4.3_pairs.jsonl")
    p5pred = jsonl(p5run / "P5.3_predictions.jsonl")
    if len(p4pairs) != len(p5pred):
        raise ValueError("P4/P5 prediction count differs")
    for pair, pred in zip(p4pairs, p5pred):
        if (pair["stable_key"] != pred["stable_key"] or pair["study_id"] != pred["study_id"]
                or pair["image_id"] != pred["image_id"]):
            raise ValueError("P4/P5 output pairing differs")
    by_dataset = {}
    saved_metrics = json.loads((p5run / "P5.3_cpu_analysis/metrics.json").read_text())
    paired = read_inputs(p4cfg_path, index, p4run)
    metric_recheck = {}
    for dataset in ("mimic", "indiana"):
        outputs = [r for r in p5pred if r["stable_key"][0] == dataset]
        reports = [r["generated_report"] for r in outputs]
        counts = Counter(reports)
        by_dataset[dataset] = {"n": len(outputs), "unique_reports": len(counts),
            "pattern_counts": sorted(counts.values(), reverse=True),
            "pattern_prefixes": [text[:65] for text, _ in counts.most_common()],
            "generated_tokens": dict(Counter(r["generated_tokens"] for r in outputs)),
            "hit_max_length": sum(r["hit_max_length"] for r in outputs),
            "empty": sum(not text.strip() for text in reports),
            "repeat_fourgram_mean": statistics.mean(repeat_fourgrams(text) for text in reports)}
        positions = [i for i, row in enumerate(paired) if row["cohort"] == dataset]
        metric_recheck[dataset] = {}
        for model, text_at in (("P4_M1", lambda i: paired[i]["M1"]["generated_report"]),
                               ("P5_M3", lambda i: p5pred[i]["generated_report"])):
            scores = corpus_scores([case_statistics(paired[i]["reference"], text_at(i)) for i in positions])
            recorded = saved_metrics["datasets"][dataset]["models"][model]["text"]
            if any(abs(value - recorded[key]) >= 1e-12 for key, value in scores.items()):
                raise ValueError("Stored text metric mismatch: " + dataset + "/" + model)
            metric_recheck[dataset][model] = {"n": len(positions), "bleu1_to_4_and_rouge_l_match": True,
                                              "scores": scores}
    updates = jsonl(p5run / "P5.2_updates.jsonl")
    if [x["position"] for x in updates] != list(range(1, 2001)):
        raise ValueError("P5.2 update sequence differs from 1..2000")
    result = {"classification": "CPU-only artifact analysis, no model forward pass or inference",
              "source_artifacts": {"p4_config_sha256": digest(p4cfg_path),
                                   "p5_config_sha256": digest(p5cfg_path),
                                   "manifest_index_sha256": digest(index_path),
                                   "p4_pairs_sha256": digest(p4run / "P4.3_pairs.jsonl"),
                                   "p5_predictions_sha256": digest(p5run / "P5.3_predictions.jsonl")},
              "protocol_equal": {key: p4cfg[key] == p5cfg[key] for key in
                                 ("tokenizer", "image_preprocessing", "training_text", "optimization", "generation")},
              "manifests": manifests, "train_validation_subject_overlap": len(train_subjects & val_subjects),
              "model_metadata": {"num_query_tokens": source.num_query_tokens,
                                 "qformer_hidden_size": source.qformer_config.hidden_size,
                                 "vision_hidden_size": source.vision_config.hidden_size,
                                 "opt_hidden_size": source.text_config.hidden_size},
              "tokenizer": {"class": type(tok).__name__, "bos": tok.bos_token_id,
                            "eos": tok.eos_token_id, "pad": tok.pad_token_id,
                            "prompt_tokens": len(prompt_ids)},
              "training_targets": {"n": len(target_lengths),
                                   "target_tokens_median": statistics.median(target_lengths),
                                   "target_tokens_max": max(target_lengths),
                                   "truncated_count": sum(before > used for before, used in zip(target_lengths, used_lengths)),
                                   "supervised_tokens_min": min(supervised_lengths),
                                   "supervised_tokens_max": max(supervised_lengths),
                                   "repeat_fourgram_mean": statistics.mean(reference_repeat),
                                   "first_three_token_text_top": first_target_pieces.most_common(5)},
              "outputs": by_dataset, "metric_recheck": metric_recheck,
              "training_steps": summarize_steps(updates),
              "query_grad_below_1e_5_after_800": sum(x["gradient_norms"]["query"] < 1e-5 for x in updates[800:]),
              "checkpoint": checkpoint_diagnostics(p5run, p4run, source_dir)}
    output = Path(__file__).with_name("diagnostics.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output),
                      "trainable_parameters": result["checkpoint"]["trainable_parameter_count"],
                      "paired_outputs": len(p5pred)}))


if __name__ == "__main__":
    main()
