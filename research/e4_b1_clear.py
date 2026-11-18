"""Score only the frozen E4-B1 Indiana development images with CLEAR on CPU."""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
COHORT = OUT / "E4_B1_cohort.csv"
PROTOCOL_SHA = "a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269"
CHECKPOINT_SHA = "da1617a7d47970667a7a158eabfa4c4f3798d5d03b7f98caec150d8ee6ab6a03"
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.__version__ == "2.3.1+cpu" and torch.version.cuda is None
    assert sha(OUT / "E4_B1_protocol.md") == PROTOCOL_SHA
    assert sha(COHORT) == "74f913413fbedca41c8c6592d79105946e2f72221d88be046fcfd9e5602fdce7"
    dest = OUT / "E4_B1_clear_scores.csv"
    if dest.exists():
        raise FileExistsError(dest)
    cohort = list(csv.DictReader(COHORT.open(newline="", encoding="utf-8")))
    assert len(cohort) == 24
    checkpoints = list(Path("/home/student4/.cache/huggingface/hub/models--peterhan91--CLEAR/snapshots").glob("*/best_model.pt"))
    assert len(checkpoints) == 1 and sha(checkpoints[0]) == CHECKPOINT_SHA
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(ROOT / "research/cache/torch_hub"))
    sys.path.insert(0, str(ROOT.parent / "CLEAR/src"))
    from clear import load_pretrained, tokenize

    start = time.monotonic()
    model, preprocess = load_pretrained(checkpoint_path=checkpoints[0], device="cpu", local_files_only=True, strict=True)
    model = model.float().eval().requires_grad_(False)
    prompts = [s for f in FINDINGS for s in (f.lower(), "no " + f.lower())]
    tokens = tokenize(prompts)
    with torch.inference_mode():
        text = F.normalize(model.encode_text(tokens), dim=-1)
    rows = []
    fields = ["image_id", "image_sha256", "finding", "positive_prompt", "negative_prompt",
              "positive_cosine_similarity", "negative_cosine_similarity", "paired_softmax_score",
              "direction", "checkpoint_sha256", "protocol_sha256", "forward_sec"]
    for k, item in enumerate(cohort):
        path = Path(item["image_path"])
        assert path.is_file() and sha(path) == item["image_sha256"]
        t = time.monotonic()
        with Image.open(path) as image:
            x = preprocess(image).unsqueeze(0).float()
        assert tuple(x.shape) == (1, 3, 448, 448)
        with torch.inference_mode():
            features = F.normalize(model.encode_image(x), dim=-1)
            cos = (features @ text.T).reshape(5, 2)
            paired = cos.softmax(-1)
        assert torch.isfinite(cos).all() and torch.isfinite(paired).all()
        sec = time.monotonic() - t
        for j, finding in enumerate(FINDINGS):
            positive = float(cos[j, 0])
            negative = float(cos[j, 1])
            rows.append({"image_id": item["image_id"], "image_sha256": item["image_sha256"],
                         "finding": finding, "positive_prompt": prompts[j * 2],
                         "negative_prompt": prompts[j * 2 + 1],
                         "positive_cosine_similarity": repr(positive),
                         "negative_cosine_similarity": repr(negative),
                         "paired_softmax_score": repr(float(paired[j, 0])),
                         "direction": "favors present" if positive > negative else "favors absent",
                         "checkpoint_sha256": CHECKPOINT_SHA, "protocol_sha256": PROTOCOL_SHA,
                         "forward_sec": repr(sec)})
        print("CLEAR", k + 1, "/24", item["image_id"], round(sec, 2), flush=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 120 and not torch.cuda.is_initialized()
    summary = {"status": "PASS", "images": 24, "rows": 120, "wall_sec": time.monotonic() - start,
               "checkpoint_sha256": CHECKPOINT_SHA, "cohort_sha256": sha(COHORT),
               "protocol_sha256": PROTOCOL_SHA, "scores_sha256": sha(dest),
               "torch": torch.__version__, "cuda_initialized": torch.cuda.is_initialized(),
               "python": sys.executable, "gpu_hours": 0}
    (OUT / "E4_B1_clear_compute.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
