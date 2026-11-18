"""Apply the installed offline rule-based CheXpert labeler to E4-B1 outputs."""

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
LABELER = ROOT / "model_cache/chexpert_tools/chexpert-labeler"
NEGBIO = ROOT / "model_cache/chexpert_tools/NegBio"
PYTHON = Path("/home/student4/anaconda3/envs/chexpert-label/bin/python")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    reports = [json.loads(x) for x in (OUT / "E4_B1_reports.jsonl").read_text().splitlines()]
    if len(reports) != 264 or any(r["status"] != "success" for r in reports):
        raise RuntimeError("Full 264-report grid is required")
    for k in ("TMPDIR", "HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "XDG_CACHE_HOME"):
        if not os.environ.get(k, "").startswith("/storage/student4/hanhpm/"):
            raise RuntimeError(f"{k} must be under /storage")
    dest = OUT / "E4_B1_labeler_output.csv"
    inp = OUT / "E4_B1_labeler_input.csv"
    if not inp.exists():
        with inp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for report in reports:
                w.writerow([report["generated_report"]])
    with inp.open(newline="", encoding="utf-8") as f:
        saved_input = [r[0] for r in csv.reader(f)]
    if saved_input != [r["generated_report"] for r in reports]:
        raise RuntimeError("Saved labeler input does not match frozen reports")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(NEGBIO)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if not dest.exists():
        with (OUT / "E4_B1_labeler.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(PYTHON), str(LABELER / "label.py"), "--reports_path", str(inp),
                            "--output_path", str(dest), "--verbose"], cwd=LABELER, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    with dest.open(newline="", encoding="utf-8") as f:
        labeled = list(csv.DictReader(f))
    if len(labeled) != 264 or [r["Reports"] for r in labeled] != [r["generated_report"] for r in reports]:
        raise RuntimeError("Labeler alignment failed")
    meta = {"status":"PASS", "reports":264, "input_sha256":sha(inp),"output_sha256":sha(dest),
            "labeler_revision":subprocess.check_output(["git","-C",str(LABELER),"rev-parse","HEAD"],universal_newlines=True).strip(),
            "negbio_revision":subprocess.check_output(["git","-C",str(NEGBIO),"rev-parse","HEAD"],universal_newlines=True).strip(),
            "cpu_only":True}
    (OUT / "E4_B1_labeler_manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
