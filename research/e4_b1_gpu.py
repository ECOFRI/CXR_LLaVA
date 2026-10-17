"""Bounded E4-B1 greedy CXR-LLaVA smoke/full inference on four K80 devices."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
PROMPTS = OUT / "E4_B1_frozen_prompts.jsonl"
COHORT = OUT / "E4_B1_cohort.csv"
PROTOCOL_SHA = "a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269"
PREPARE_SHA = "86cf4d97482c2a63fe790c9ab24a6db2806b667de68e93973e2bbebdbaa115ae"
ARMS = ("C0", "N1", "F1-M", "F1-S", "F1-O", "F2-M", "F2-S", "F2-O", "F3-M", "F3-S", "F3-O")
GPU_CAP = 14_400.0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def preflight_env():
    for k in ("TMPDIR", "HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "XDG_CACHE_HOME"):
        if not os.environ.get(k, "").startswith("/storage/student4/hanhpm/"):
            raise RuntimeError(f"{k} must be under /storage/student4/hanhpm/")
    if sha(OUT / "E4_B1_protocol.md") != PROTOCOL_SHA or sha(PROMPTS) != PREPARE_SHA:
        raise RuntimeError("Frozen protocol/prompt hash mismatch")
    uuids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(uuids) != 4 or len(set(uuids)) != 4 or any(not x.startswith("GPU-") for x in uuids):
        raise RuntimeError("Exactly four K80 UUIDs must be selected")
    rows = list(csv.DictReader(COHORT.open(newline="", encoding="utf-8")))
    prompts = read_jsonl(PROMPTS)
    if len(rows) != 24 or len(prompts) != 264:
        raise RuntimeError("Cohort or prompt grid incomplete")
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    by_uuid = {}
    for line in gpu_rows.splitlines():
        fields = [x.strip() for x in line.split(",")]
        by_uuid[fields[0]] = {"name": fields[1], "used_mib": float(fields[2]), "free_mib": float(fields[3])}
    app_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = [line for line in app_rows.splitlines() if line.split(",")[0].strip() in uuids]
    if busy or any(u not in by_uuid or by_uuid[u]["name"] != "Tesla K80" or by_uuid[u]["used_mib"] > 0 or by_uuid[u]["free_mib"] < 10_000 for u in uuids):
        raise RuntimeError("Selected GPU is not free before model launch")
    atomic_json(OUT / "E4_B1_gpu_preflight.json", {"at_utc": now(), "selected_uuids": uuids,
                "gpu_state": {u: by_uuid[u] for u in uuids}, "selected_compute_processes": busy,
                "protocol_sha256": PROTOCOL_SHA, "prompts_sha256": PREPARE_SHA})
    return rows, prompts, uuids


def prior_wall():
    path = OUT / "E4_B1_gpu_windows.jsonl"
    if not path.exists():
        return 0.0
    lines = read_jsonl(path)
    starts = {x["window_id"]: x for x in lines if x["event"] == "start"}
    ends = {x["window_id"]: x for x in lines if x["event"] == "end"}
    if set(starts) != set(ends):
        raise RuntimeError("Unclosed previous GPU window; inspect interruption before resume")
    return sum(ends[k]["wall_sec"] for k in starts)


def append_window(value):
    path = OUT / "E4_B1_gpu_windows.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def output_stop_reason(tokens, eos_id, keyword_fired, max_tokens):
    if len(tokens) >= max_tokens:
        return "MAX_TOKENS"
    if keyword_fired:
        return "STOP_KEYWORD"
    if tokens and tokens[-1] == eos_id:
        return "EOS"
    return "UNDETERMINED"


def run_call(model, image_path, frozen, torch):
    from PIL import Image

    rendered = model.apply_chat_template(frozen["chat"])
    ids = model.tokenizer_image_token(rendered, model.tokenizer, return_tensors=None)
    if rendered != frozen["rendered_prompt"] or ids != frozen["input_ids"]:
        raise RuntimeError("Runtime rendered prompt/input_ids differ from frozen record")
    observed = {}
    native_generate = model.generate

    def capture_generate(*args, **kwargs):
        if kwargs.get("do_sample") is not False or kwargs.get("temperature") != 0:
            raise RuntimeError("Greedy decoding override detected")
        original = kwargs.get("stopping_criteria", [])
        if len(original) != 1:
            raise RuntimeError("Unexpected native stop criteria")
        crit = original[0]

        def tracked(input_ids, scores, **rest):
            fired = crit(input_ids, scores, **rest)
            if fired:
                observed["keyword_fired"] = True
            return fired

        kwargs["stopping_criteria"] = [tracked]
        observed["effective_generation_kwargs"] = {k: v for k, v in kwargs.items() if k in
            ("do_sample", "temperature", "top_p", "max_new_tokens", "use_cache")}
        result = native_generate(*args, **kwargs)
        sequences = result.sequences if hasattr(result, "sequences") else result
        observed["output_ids"] = sequences[0, len(ids):].detach().cpu().tolist()
        return result

    model.generate = capture_generate
    try:
        start = time.monotonic()
        with Image.open(image_path) as image, torch.inference_mode():
            text = model.generate_cxr_repsonse(chat=frozen["chat"], image=image, temperature=0, top_p=0.8)
        for j in range(4):
            torch.cuda.synchronize(j)
        elapsed = time.monotonic() - start
    finally:
        model.generate = native_generate
    if "output_ids" not in observed:
        raise RuntimeError("Native generate did not return captured output IDs")
    output_ids = observed["output_ids"]
    max_tokens = observed["effective_generation_kwargs"]["max_new_tokens"]
    return {"generated_report": text, "output_ids": output_ids, "output_token_count": len(output_ids),
            "stop_reason": output_stop_reason(output_ids, model.tokenizer.eos_token_id,
                                               observed.get("keyword_fired", False), max_tokens),
            "keyword_stop_fired": observed.get("keyword_fired", False),
            "effective_generation_kwargs": observed["effective_generation_kwargs"], "wall_sec": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "full"))
    args = parser.parse_args()
    cohort, prompts, uuids = preflight_env()
    previous = prior_wall()
    if previous >= GPU_CAP:
        raise RuntimeError("CXR-LLaVA cumulative wall cap reached")
    import torch
    import transformers

    if torch.__version__ != "1.12.1+cu102" or transformers.__version__ != "4.36.2" or torch.cuda.device_count() != 4:
        raise RuntimeError("Frozen K80 environment/device mismatch")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    bykey = {(r["image_id"], r["arm"]): r for r in prompts}
    byimage = {r["image_id"]: r for r in cohort}
    images = sorted(byimage)
    if args.phase == "smoke":
        if (OUT / "E4_B1_smoke_result.json").exists():
            raise RuntimeError("Smoke already completed")
        tasks = [(f"det{i}-{k}", i, "C0") for i in images[:2] for k in (1, 2)]
        tasks += [(f"grid-{i}-{arm}", i, arm) for i in images[:2] for arm in ARMS]
        folder = OUT / "smoke/commits"
    else:
        smoke = json.loads((OUT / "E4_B1_smoke_result.json").read_text())
        projection = json.loads((OUT / "E4_B1_runtime_projection.json").read_text())
        if smoke["status"] != "PASS" or projection["status"] != "GO_WITHIN_CAP":
            raise RuntimeError("Smoke/projection gate did not pass")
        tasks = [(f"{i}--{arm}", i, arm) for i in images for arm in ARMS]
        folder = OUT / "full/commits"
    folder.mkdir(parents=True, exist_ok=True)
    dm = {"vision_tower": 0, "mm_projector": 0, "lm_head": 0, "llama.embed_tokens": 0, "llama.norm": 3}
    dm.update({f"llama.layers.{i}": i // 8 for i in range(32)})
    wid = f"{args.phase}-{int(time.time())}-{os.getpid()}"
    began = time.monotonic()
    append_window({"event": "start", "window_id": wid, "phase": args.phase, "at_utc": now(),
                   "prior_gpu_wall_sec": previous, "selected_uuids": uuids})
    status = "FAILED"
    model = None
    try:
        load = time.monotonic()
        model = transformers.AutoModel.from_pretrained(
            str(ROOT / "model_cache/cxr-llava-v2-k80-fp32"), trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            device_map=dm, use_safetensors=True).eval().requires_grad_(False)
        for j in range(4):
            torch.cuda.synchronize(j)
        load_sec = time.monotonic() - load
        if model.generation_config.top_k != 50 or model.generation_config.do_sample:
            raise RuntimeError("Unexpected generation_config sampling defaults")
        atomic_json(OUT / f"E4_B1_{args.phase}_model_load.json", {"load_sec": load_sec,
                    "generation_config": model.generation_config.to_dict(), "device_map": dm,
                    "uuids": uuids, "torch": torch.__version__, "transformers": transformers.__version__})
        for index, (task_id, image, arm) in enumerate(tasks):
            if previous + time.monotonic() - began + 120 >= GPU_CAP:
                raise RuntimeError("Four-hour cumulative CXR wall cap admission stop")
            path = folder / f"{task_id}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if old["prompt_sha256"] != bykey[(image, arm)]["rendered_prompt_sha256"]:
                    raise RuntimeError("Resume prompt hash mismatch")
                continue
            row, frozen = byimage[image], bykey[(image, arm)]
            if sha(row["image_path"]) != row["image_sha256"]:
                raise RuntimeError("Image SHA-256 mismatch")
            for j in range(4):
                torch.cuda.reset_peak_memory_stats(j)
            record = {"task_id": task_id, "phase": args.phase, "image_id": image, "arm": arm,
                      "image_sha256": row["image_sha256"], "prompt_sha256": frozen["rendered_prompt_sha256"],
                      "input_ids_sha256": frozen["input_ids_sha256"], "rendered_prompt": frozen["rendered_prompt"],
                      "input_ids": frozen["input_ids"], "input_token_count": len(frozen["input_ids"]),
                      "chat": frozen["chat"], "started_utc": now()}
            record.update(run_call(model, row["image_path"], frozen, torch))
            record.update(completed_utc=now(), status="success",
                          peak_allocated_bytes=[torch.cuda.max_memory_allocated(j) for j in range(4)],
                          peak_reserved_bytes=[torch.cuda.max_memory_reserved(j) for j in range(4)])
            if not record["generated_report"].strip():
                raise RuntimeError("Empty generated report")
            atomic_json(path, record)
            print(args.phase.upper(), index + 1, "/", len(tasks), image, arm,
                  round(record["wall_sec"], 2), "s", flush=True)
        if args.phase == "smoke":
            det = {i: [json.loads((folder / f"det{i}-{k}.json").read_text()) for k in (1, 2)] for i in images[:2]}
            identical = all(v[0]["generated_report"].encode() == v[1]["generated_report"].encode() for v in det.values())
            if not identical:
                raise RuntimeError("Greedy C0 determinism failed")
            records = [json.loads((folder / f"{task_id}.json").read_text()) for task_id, _, _ in tasks]
            atomic_json(OUT / "E4_B1_smoke_result.json", {"status": "PASS", "images": images[:2],
                        "greedy_C0_byte_identical": identical, "reports": len(records),
                        "mean_report_wall_sec": sum(r["wall_sec"] for r in records) / len(records),
                        "peak_allocated_bytes": [max(r["peak_allocated_bytes"][j] for r in records) for j in range(4)],
                        "stop_reasons": sorted({r["stop_reason"] for r in records})})
        else:
            records = [json.loads((folder / f"{task_id}.json").read_text()) for task_id, _, _ in tasks]
            with (OUT / "E4_B1_reports.jsonl").open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = "PASS"
    except Exception as error:
        atomic_json(OUT / f"E4_B1_{args.phase}_error.json", {"status": "FAILED", "error": repr(error),
                    "traceback": traceback.format_exc(), "at_utc": now()})
        raise
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        wall = time.monotonic() - began
        append_window({"event": "end", "window_id": wid, "phase": args.phase, "at_utc": now(),
                       "wall_sec": wall, "status": status, "cumulative_gpu_wall_sec": previous + wall})
        print("WINDOW", args.phase, status, round(wall, 2), "sec", flush=True)


if __name__ == "__main__":
    main()
