"""Pinned checkpoint metadata preview; transfers require an explicit approval record."""
import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

VR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = VR / "configs/download/blip2_opt_2_7b.yaml"
DEFAULT_OUTPUT = VR / "runs/P1_blip2_candidate_dry_run"
SMALL_FILES = {
    "config.json", "generation_config.json", "preprocessor_config.json",
    "processor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "spiece.model", "tokenizer.model", "model.safetensors.index.json",
}


def within_vr(path):
    result = Path(path).expanduser().resolve()
    if result != VR and VR not in result.parents:
        raise ValueError("Output must stay within research/blip2_cxr_v4")
    return result


def fetch_metadata(model_id):
    url = "https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/") + "?blobs=true"
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def selected_files(metadata):
    return sorted(
        (s for s in metadata["siblings"]
         if s["rfilename"].endswith(".safetensors") or s["rfilename"] in SMALL_FILES),
        key=lambda s: s["rfilename"],
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def approval_matches(path, config):
    if not path:
        return False
    approval = json.loads(Path(path).read_text())
    return (
        approval.get("authorized") is True
        and approval.get("approved_by")
        and approval.get("model_id") == config["model_id"]
        and approval.get("revision") == config["revision"]
        and approval.get("max_gib") == config["max_gib"]
    )


def download_file(config, sibling, output_dir, resume):
    name = sibling["rfilename"]
    if Path(name).name != name:
        raise ValueError("Nested or unsafe checkpoint file name")
    target = output_dir / name
    expected = sibling.get("lfs", {}).get("sha256")
    if target.exists():
        if not resume:
            raise FileExistsError(f"Existing file: {name}; pass --resume")
        if expected and digest(target) == expected:
            return {"file": name, "sha256": expected, "status": "existing_verified"}
        raise ValueError(f"Existing file cannot be verified: {name}")
    url = "https://huggingface.co/" + config["model_id"] + "/resolve/" + config["revision"] + "/" + urllib.parse.quote(name)
    temporary = output_dir / (name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Partial file exists: {name}; manual review required")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
        if temporary.stat().st_size != sibling["size"]:
            raise ValueError(f"Size mismatch: {name}")
        actual = digest(temporary)
        if expected and actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        temporary.rename(target)
        return {"file": name, "sha256": actual, "status": "downloaded_verified"}
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    parser.add_argument("--resume", action="store_true", help="Verify existing files during approved execution")
    parser.add_argument("--execute", action="store_true", help="Requires matching approval file")
    parser.add_argument("--approval-file", type=Path, help="Owner and QA authorization record")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["model_id"] != "Salesforce/blip2-opt-2.7b" or config["format"] != "safetensors":
        raise ValueError("Only the compared official safetensors checkpoint is allowed")
    if config["max_gib"] > 20:
        raise ValueError("Budget exceeds 20 GiB")
    output_dir = within_vr(args.output_dir)
    metadata = fetch_metadata(config["model_id"])
    if metadata["sha"] != config["revision"]:
        raise ValueError("Pinned revision differs from current API revision; review required")
    files = selected_files(metadata)
    if not any(s["rfilename"].endswith(".safetensors") for s in files):
        raise ValueError("No safetensors weights found")
    total = sum(s.get("size") or 0 for s in files)
    if any(s.get("size") is None for s in files) or total > config["max_gib"] * 1024**3:
        raise ValueError("Unknown size or checkpoint exceeds budget")
    manifest = {
        "model_id": config["model_id"], "revision": config["revision"],
        "license": metadata.get("cardData", {}).get("license", config["license"]),
        "selected_bytes": total, "selected_gib": round(total / 1024**3, 4),
        "max_gib": config["max_gib"], "download_executed": False,
        "files": [{"file": s["rfilename"], "size": s["size"],
                   "sha256": s.get("lfs", {}).get("sha256", "UNKNOWN")} for s in files],
    }
    if args.execute:
        if args.dry_run or not approval_matches(args.approval_file, config):
            raise ValueError("Execution requires matching owner/QA approval file and no --dry-run")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest["verified_files"] = [download_file(config, s, output_dir, args.resume) for s in files]
        manifest["download_executed"] = True
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"status={'EXECUTED' if args.execute else 'DRY_RUN'} model={config['model_id']} size_gib={manifest['selected_gib']}")
    print(f"manifest={output_dir / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
