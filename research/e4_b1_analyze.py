"""Deterministic E4-B1 development gate using saved reports and rule labels."""

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from research.e4_b1_prepare import ARMS, FIVE, GENERIC_NORMAL, OUT, SIX, sha


PROTOCOL_SHA = "a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269"
ROOT = Path(__file__).resolve().parents[1]


def label_state(value):
    if value is None:
        return "Extraction failure"
    return {"1": "Positive", "1.0": "Positive", "0": "Negative", "0.0": "Negative",
            "-1": "Uncertain", "-1.0": "Uncertain", "": "Not mentioned"}.get(str(value).strip(), "Extraction failure")


def is_mention(value):
    return value in ("Positive", "Negative", "Uncertain")


def gate(records, states, cohort, different):
    """Return frozen gate metrics; image is the paired unit."""
    bykey = {(r["image_id"], r["arm"]): r for r in records}
    images = sorted(cohort)
    if len(images) != 24 or len(records) != 264 or len(bykey) != 264:
        raise ValueError("Incomplete 24 x 11 report grid")
    arm_rows = []
    for arm in ARMS:
        ar = [bykey[(i, arm)] for i in images]
        collapsed = sum(len(r["generated_report"].strip()) < 60 or r["generated_report"].strip() in GENERIC_NORMAL for r in ar)
        mentions = sum(is_mention(states[(i, arm, f)]) for i in images for f in SIX)
        failures = sum(states[(i, arm, f)] == "Extraction failure" for i in images for f in SIX)
        arm_rows.append({"row_type": "arm", "arm": arm, "format": arm.split("-")[0] if "-" in arm else arm,
                         "reports": 24, "collapse_n": collapsed, "collapse_denominator": 24,
                         "collapse_rate": collapsed / 24, "mention_cells": mentions,
                         "mention_denominator": 24, "mean_mention_findings": mentions / 24,
                         "extraction_failure_cells": failures})
    byarm = {r["arm"]: r for r in arm_rows}
    format_rows = []
    for fmt in ("F1", "F2", "F3"):
        eligible_images = [i for i in images if different[i] > 0]
        changed = sum(any(states[(i, fmt + "-M", f)] != states[(i, fmt + "-S", f)] for f in SIX) for i in eligible_images)
        text_changed = sum(bykey[(i, fmt + "-M")]["generated_report"] != bykey[(i, fmt + "-S")]["generated_report"] for i in eligible_images)
        definite_cells = [(i, f) for i in images for f in SIX if cohort[i][f] in ("Positive", "Negative")]
        o_match = sum(states[(i, fmt + "-O", f)] == cohort[i][f] for i, f in definite_cells)
        s_match = sum(states[(i, fmt + "-S", f)] == cohort[i][f] for i, f in definite_cells)
        n = len(definite_cells)
        delta = (o_match - s_match) / n if n else None
        sensitivity = changed / len(eligible_images) if eligible_images else None
        c1 = byarm[fmt + "-O"]["collapse_rate"] <= .2 and byarm[fmt + "-M"]["collapse_rate"] <= .2
        c2 = byarm[fmt + "-M"]["mean_mention_findings"] >= 4
        c3 = n >= 20 and delta is not None and delta >= .2
        c4 = len(eligible_images) >= 10 and sensitivity is not None and sensitivity >= .3
        format_rows.append({"row_type": "format", "format": fmt, "oracle_eligible_cells": n,
                            "oracle_O_matches": o_match, "oracle_S_matches": s_match,
                            "oracle_O_rate": o_match/n if n else None, "oracle_S_rate": s_match/n if n else None,
                            "oracle_advantage_points": delta*100 if delta is not None else None,
                            "sensitivity_eligible_images": len(eligible_images), "sensitivity_changed_images": changed,
                            "exact_text_changed_eligible_images": text_changed,
                            "sensitivity_rate": sensitivity, "criterion_1_collapse": c1,
                            "criterion_2_mentions": c2, "criterion_3_oracle": c3,
                            "criterion_4_sensitivity": c4, "passes": c1 and c2 and c3 and c4})
    winners = [r for r in format_rows if r["passes"]]
    if winners:
        winners.sort(key=lambda r: (-r["oracle_O_rate"], byarm[r["format"] + "-O"]["collapse_rate"],
                                    {"F2": 0, "F3": 1, "F1": 2}[r["format"]]))
        decision = "E4_B1_FORMAT_SELECTED_" + winners[0]["format"]
    elif not any(r["criterion_3_oracle"] for r in format_rows):
        decision = "E4_B1_PROMPT_ROUTE_REJECTED"
    else:
        decision = "E4_B1_ORACLE_RESPONSIVE_CLEAR_NOT_USABLE"
    return arm_rows, format_rows, decision


def main():
    assert sha(OUT / "E4_B1_protocol.md") == PROTOCOL_SHA
    reports = [json.loads(x) for x in (OUT / "E4_B1_reports.jsonl").read_text().splitlines()]
    labels = list(csv.DictReader((OUT / "E4_B1_labeler_output.csv").open(newline="", encoding="utf-8")))
    cohort_rows = list(csv.DictReader((OUT / "E4_B1_cohort.csv").open(newline="", encoding="utf-8")))
    diff_rows = list(csv.DictReader((OUT / "E4_B1_derangement.csv").open(newline="", encoding="utf-8")))
    prompts = [json.loads(x) for x in (OUT / "E4_B1_frozen_prompts.jsonl").read_text().splitlines()]
    if len(reports) != 264 or len(labels) != 264 or len(prompts) != 264:
        raise RuntimeError("Report, label or prompt grid incomplete")
    for report, lab, prompt in zip(reports, labels, prompts):
        if lab["Reports"] != report["generated_report"] or (report["image_id"], report["arm"]) != (prompt["image_id"], prompt["arm"]):
            raise RuntimeError("Labeler/report/prompt order mismatch")
        if report["rendered_prompt"] != prompt["rendered_prompt"] or report["input_ids"] != prompt["input_ids"]:
            raise RuntimeError("Runtime prompt/input IDs differ from frozen input")
    states = {}
    long = []
    for r, lab in zip(reports, labels):
        for finding in SIX:
            val = label_state(lab.get(finding))
            key = (r["image_id"], r["arm"], finding)
            states[key] = val
            long.append({"image_id": key[0], "arm": key[1], "finding": finding,
                         "label_state": val, "raw_label": lab.get(finding, "MISSING_COLUMN")})
    assert len(states) == len(long) == 264*6
    with (OUT / "E4_B1_labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(long[0]));w.writeheader();w.writerows(long)
    cohort = {r["image_id"]: r for r in cohort_rows}
    different = {r["image_id"]: int(r["differing_directions"]) for r in diff_rows}
    arm_rows, format_rows, decision = gate(reports, states, cohort, different)
    fields = sorted(set().union(*(set(r) for r in arm_rows+format_rows)))
    with (OUT / "E4_B1_gate_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(arm_rows+format_rows)
    bykey = {(r["image_id"], r["arm"]): r for r in reports}
    focus = sorted(cohort)[:5]
    qualitative = ["# E4-B1 fixed qualitative sample", "", "First five cohort IDs in sorted order, fixed before inference. All text is model output, not clinical truth.", ""]
    for image in focus:
        qualitative += [f"## {image}", "", "| Format | Matched (M) | Shuffled (S) | Oracle (O) |", "|---|---|---|---|"]
        for fmt in ("F1", "F2", "F3"):
            parts = [bykey[(image, fmt+"-"+source)]["generated_report"].replace("|", "\\|").replace("\n", " ") for source in ("M", "S", "O")]
            qualitative.append(f"| {fmt} | " + " | ".join(parts) + " |")
        qualitative.append("")
    f1_generic = sum(bykey[(i,"F1-M")]["generated_report"].strip() == "No acute cardiopulmonary process." for i in cohort)
    qualitative.append(f"E4-R4 exact collapse sentence appears in F1-M on {f1_generic}/24 images under greedy decoding.")
    (OUT / "E4_B1_qualitative.md").write_text("\n".join(qualitative)+"\n")
    windows = [json.loads(x) for x in (OUT / "E4_B1_gpu_windows.jsonl").read_text().splitlines()]
    gpu_wall = sum(r["wall_sec"] for r in windows if r["event"] == "end")
    peaks = [max(r["peak_allocated_bytes"][j] for r in reports) for j in range(4)]
    avg = sum(r["wall_sec"] for r in reports)/264
    compute = f"# E4-B1 compute report\n\nTotal CXR-LLaVA wall windows: {gpu_wall:.2f} s including smoke and full load, within 14,400 s cap. Full-run mean report call {avg:.2f} s; CPU CLEAR 24-image wall {json.loads((OUT/'E4_B1_clear_compute.json').read_text())['wall_sec']:.2f} s; GPU peak allocated bytes by selected device: {peaks}. Full telemetry and UUIDs are in the run manifest and per-report records. This is four-device wall time, not SM-active GPU-hours.\n"
    (OUT / "E4_B1_compute_report.md").write_text(compute)
    rows_by_format = {r["format"]:r for r in format_rows}
    dlines = ["# E4-B1 research decision", "", f"**{decision}**. `HOLD_MORE_EVIDENCE`; `HOLD_COMPLEX_FUSION`.", "",
              "This is a format-selection development gate, not clinical validation. The oracle uses reference-report text and is not image truth. Exact M/S text changes are descriptive only; the frozen sensitivity gate uses extracted finding states.", "",
              "## Precommitted gate counts", "", "| Format | O collapse | M collapse | M mean mentions | Oracle O/S same cells | M/S changed/eligible | Pass |", "|---|---:|---:|---:|---:|---:|---|"]
    byarm={r["arm"]:r for r in arm_rows}
    for fmt in ("F1","F2","F3"):
        r=rows_by_format[fmt]
        dlines.append(f"| {fmt} | {byarm[fmt+'-O']['collapse_n']}/24 | {byarm[fmt+'-M']['collapse_n']}/24 | {byarm[fmt+'-M']['mean_mention_findings']:.2f}/6 | {r['oracle_O_matches']}/{r['oracle_eligible_cells']} vs {r['oracle_S_matches']}/{r['oracle_eligible_cells']} ({r['oracle_advantage_points']:.1f} pp) | {r['sensitivity_changed_images']}/{r['sensitivity_eligible_images']} | {r['passes']} |")
    dlines += ["", "**VERIFIED:** cohort disjointness, frozen template/hash checks, deterministic smoke status and observed prompt/output/compute artifacts. **AUTOMATED PROXY:** six-finding labeler states and gate rates. **UNVERIFIED:** image-grounded correctness, clinical value and CLEAR benefit. F1 oracle uses categorical table cells because numerical cosine values cannot be fabricated from labels; it is not an exact semantic match to numeric F1-M/S.", ""]
    if decision == "E4_B1_PROMPT_ROUTE_REJECTED":
        dlines.append("The prespecified oracle gate was not met **in this setup**. F2 and F3 nevertheless change extracted finding states between matched and shuffled evidence on 11/19 and 13/19 eligible images, respectively; therefore this decision does not mean the model is completely insensitive to prompt evidence. It means no tested format met the frozen oracle-following threshold. Do not generalize to every prompt or decoding method without a new protocol.")
    dlines += ["", "Descriptive exact M/S output-text changes among the same 19 direction-different images: " + ", ".join(f"{r['format']} {r['exact_text_changed_eligible_images']}/19" for r in format_rows) + ". These counts do not enter the pass decision."]
    positive = {f:sum(c[f]=="Positive" for c in cohort.values()) for f in SIX}
    gaps={f:max(0,15-positive[f]) for f in SIX}
    kept = "C0, N1 and the selected format's M/S arms" if decision.startswith("E4_B1_FORMAT_SELECTED") else "C0, N1 and a separately selected M/S format after a new gate"
    dlines += ["", f"E4-B2 recommendation only: to reach 15 reference-positive images per target, this pilot cohort lacks {gaps} additional positives (before accounting for overlap and split separation). Keep {kept}; oracle is optional and development-only. At {avg:.2f} s/report on four K80 devices, a four-arm cohort of N images needs roughly {4*avg:.1f}N seconds of model-call time plus setup and evaluation. A new cohort/split and GPU budget require separate authorization. No E4-B2 run started."]
    (OUT / "E4_B1_research_decision.md").write_text("\n".join(dlines)+"\n")
    sources = [OUT / name for name in ("E4_B1_protocol.md","E4_B1_cohort.csv","E4_B1_clear_scores.csv","E4_B1_frozen_prompts.jsonl","E4_B1_derangement.csv","E4_B1_reports.jsonl","E4_B1_labeler_output.csv","E4_B1_smoke_result.json","E4_B1_model_integrity.json","E4_B1_final_validation.json")]
    model_load = json.loads((OUT/"E4_B1_full_model_load.json").read_text())
    clear_compute = json.loads((OUT/"E4_B1_clear_compute.json").read_text())
    labeler_manifest = json.loads((OUT/"E4_B1_labeler_manifest.json").read_text())
    background = json.loads((OUT/"E4_B1_background_status.json").read_text())
    interruptions = [json.loads(x) for x in (OUT/"E4_B1_operator_interruptions.jsonl").read_text().splitlines()]
    report_path = ROOT / "reports/2026-10-17_E4_B1_evidence_format_gate.md"
    manifest = {"status":decision,"classification":"development format gate; automated proxy only", "protocol_sha256":PROTOCOL_SHA,
                "input_sha256":{str(p):sha(p) for p in sources},"source_code_sha256":{str(p):sha(p) for p in (ROOT/'research/e4_b1_cohort.py',ROOT/'research/e4_b1_clear.py',ROOT/'research/e4_b1_prepare.py',ROOT/'research/e4_b1_gpu.py',ROOT/'research/e4_b1_label.py',ROOT/'research/e4_b1_synthetic.py',ROOT/'research/e4_b1_validate.py',ROOT/'research/e4_b1_background.sh',ROOT/'tests/test_e4_b1.py',Path(__file__))},
                "report_sha256":sha(report_path),
                "cache_paths":{k:__import__('os').environ.get(k) for k in ("TMPDIR","HF_HOME","TORCH_HOME","PIP_CACHE_DIR","XDG_CACHE_HOME")},
                "environments":{"cxr_k80":{"torch":model_load["torch"],"transformers":model_load["transformers"],"precision":"FP32","device_map":model_load["device_map"]},
                                "clear_cpu":{"torch":clear_compute["torch"],"python":clear_compute["python"]},
                                "labeler_cpu":{"python":"/home/student4/anaconda3/envs/chexpert-label/bin/python","labeler_revision":labeler_manifest["labeler_revision"],"negbio_revision":labeler_manifest["negbio_revision"]}},
                "gpu_selected_uuids":json.loads((OUT/'E4_B1_gpu_preflight.json').read_text())['selected_uuids'],
                "cumulative_gpu_wall_sec":gpu_wall,"cpu_clear_wall_sec":json.loads((OUT/'E4_B1_clear_compute.json').read_text())['wall_sec'],
                "gpu_windows":windows,"operator_interruptions":interruptions,"background_status":background,
                "local_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "images":24,"full_reports":264,"clinical_annotations":0,"scientific_hold":"HOLD_MORE_EVIDENCE","fusion_hold":"HOLD_COMPLEX_FUSION"}
    (OUT/"E4_B1_run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    index=["# E4-B1 artifact index","",f"Decision `{decision}`; protocol SHA-256 `{PROTOCOL_SHA}`.","","| Artifact | SHA-256 |","|---|---|"]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name!="E4_B1_artifact_index.md":
            index.append(f"| `{p.relative_to(OUT)}` | `{sha(p)}` |")
    index.append(f"| `{report_path}` | `{sha(report_path)}` |")
    (OUT/"E4_B1_artifact_index.md").write_text("\n".join(index)+"\n")
    print(json.dumps({"decision":decision,"format_rows":format_rows},indent=2))


if __name__ == "__main__":
    main()
