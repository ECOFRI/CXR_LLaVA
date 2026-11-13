"""Read-only catalog of run manifests (Python standard library only).

Usage: python3 -m research.project status [--format markdown|json]
No model import, inference, or artifact modification.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def scalar(data, keys):
    for key in keys:
        value = data.get(key)
        if isinstance(value, (str, int, float, bool)):
            return str(value)
    return ""


def catalog(root):
    """List raw manifest states; do not infer scientific success or rerun anything.

    Include nested phase manifests (e.g. G0, E13, PREFLIGHT, E5), exclude
    environment/input/labeler manifests and per-report checkpoints.
    """
    root = Path(root).resolve()
    runs = root / "research/runs"
    if not runs.is_dir():
        raise FileNotFoundError(str(runs))
    records = []
    for run in sorted(p for p in runs.iterdir() if p.is_dir() and not p.is_symlink()):
        found = False
        folders = [run] + sorted(p for p in run.iterdir() if p.is_dir() and not p.is_symlink())
        for folder in folders:
            candidates = sorted(p for p in folder.glob("*manifest.json")
                                if p.name in ("bundle_manifest.json", "run_manifest.json")
                                or p.name.endswith("_run_manifest.json")
                                or p.name == "E4_R2_manifest.json")
            for path in candidates:
                row = {"run": run.name, "component": str(folder.relative_to(run)),
                       "status": "UNSPECIFIED", "decision": "", "manifest": str(path.relative_to(root)),
                       "error": ""}
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("Manifest must be a JSON object")
                    row["status"] = scalar(data, ("status", "state", "run_status")) or "UNSPECIFIED"
                    row["decision"] = scalar(data, ("decision", "research_decision", "verdict"))
                except (OSError, ValueError) as error:
                    row["status"] = "INVALID_MANIFEST"
                    row["error"] = str(error)
                records.append(row)
                found = True
        if not found:
            records.append({"run": run.name, "component": ".", "status": "NO_RUN_MANIFEST",
                            "decision": "", "manifest": "", "error": ""})
    return records


def markdown(records):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Run catalog", "", "Raw recorded states; COMPLETE is not clinical/scientific PASS.",
             "Includes historical failed/blocked attempts; no inference or metric recalculation.", "",
             "| Run | Component | Status | Decision | Manifest | Error |",
             "|---|---|---|---|---|---|"]
    for row in records:
        lines.append("| " + " | ".join(cell(row[key]) for key in
                     ("run", "component", "status", "decision", "manifest", "error")) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Read all local run/phase manifests")
    status.add_argument("--format", choices=("markdown", "json"), default="markdown")
    status.add_argument("--output", type=Path, help="New output file; refuses overwrite")
    args = parser.parse_args(argv)
    records = catalog(args.root)
    rendered = markdown(records) if args.format == "markdown" else json.dumps(records, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
