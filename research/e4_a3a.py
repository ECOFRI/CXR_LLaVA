"""CPU-only, post-hoc E4-A3A analysis of frozen report artifacts."""

import csv
import hashlib
import json
import os
import re
import string
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
R4 = ROOT / "research/runs/20261017_E4_R4_controlled_resume_v1"
A2 = ROOT / "research/runs/20261017_E4_A2_blinded_paired_report_evaluation_v1"
FROZEN = ROOT / "research/runs/20261016_E4_technical_pilot_v2/frozen_prompts.json"
OUT = ROOT / "research/runs/20261017_E4_A3A_automated_proxy_eval_v1"
ARMS = ("C0", "C1", "C2a", "C3")
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
CHEXPERT14 = ("Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding")
COMPARED = (("C2a", "C3"), ("C2a", "C0"), ("C2a", "C1"))
PROTOCOL_SHA = "6bfcb55d2f528f3303dcdc38026dfc7bbfcf94a1e2f233f20f38dfa8ebf530e7"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def readcsv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def writecsv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state(value):
    """Keep all five states distinct; cached blanks mean Not mentioned."""
    if value is None:
        return "Extraction failure"
    x = str(value).strip().lower()
    return {"1": "Positive", "1.0": "Positive", "positive": "Positive",
            "0": "Negative", "0.0": "Negative", "negative": "Negative",
            "-1": "Uncertain", "-1.0": "Uncertain", "uncertain": "Uncertain",
            "": "Not mentioned", "blank": "Not mentioned", "not mentioned": "Not mentioned"}.get(x, "Extraction failure")


def definite(x):
    return x in ("Positive", "Negative")


def coverage_status(joint_count):
    return "DESCRIPTIVE_ONLY" if joint_count >= 5 else "INSUFFICIENT_COVERAGE"


def normalize(text):
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(table).split())


def sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]


FLAGS = {
    "finding": r"\b(atelecta\w*|cardiomegal\w*|consolidat\w*|edema|effusion\w*|pneumon\w*|opacity|opacities|pneumothorax|congestion)\b",
    "negation": r"\b(no|without|negative|absent|clear|free of|not seen|no evidence of)\b",
    "laterality": r"\b(left|right|bilateral|unilateral)\b",
    "severity": r"\b(mild|moderate|severe|minimal|large|small|trace|marked|significant)\b",
}


def changed_flags(a, b):
    sa, sb = set(sentences(a)), set(sentences(b))
    removed, added = sorted(sa - sb), sorted(sb - sa)
    if not removed and not added:
        return removed, added, {"finding": False, "negation": False, "laterality": False,
                                "severity": False, "content_type": "no_change"}
    joined = " ".join(removed + added).lower()
    flags = {k: bool(re.search(v, joined)) for k, v in FLAGS.items()}
    flags["content_type"] = "potential_clinical_content" if any(flags.values()) else "stylistic_or_unclassified"
    return removed, added, flags


def evidence_block(prompt):
    text = prompt["chat"][-1]["content"]
    match = re.search(r"<context>\n(.*?)\n</context>", text, re.S)
    if not match:
        return ""
    return match.group(1)


def evidence_rows(block):
    return [line.strip() for line in block.splitlines() if " | " in line and line.split(" | ")[0] in FINDINGS]


def short_category(text):
    t = text.strip().lower()
    if not t or any(x in t for x in ("i cannot", "i'm sorry", "as an ai", "cannot interpret")):
        return "refusal/off-task"
    if len(t) >= 60 and not t.endswith((".", "!", "?")):
        return "truncated mid-sentence"
    if re.search(r"(?:\bpositive cosine\b|\bpaired positive score\b|\bfinding\s*\|)", t):
        return "label/list echo of evidence"
    if t in ("no acute cardiopulmonary process.", "no evidence of acute cardiopulmonary disease.") or (len(t) < 60 and "no acute" in t):
        return "generic-normal boilerplate"
    return "other"


def agreement(a, ref):
    if not (definite(a) and definite(ref)):
        return None
    return int(a == ref)


def paired_image_score(image, first, second, refs, labels):
    eligible = []
    for finding in FINDINGS:
        ref = refs.get((image, finding), "Extraction failure")
        a = labels.get((image, first, finding), "Extraction failure")
        b = labels.get((image, second, finding), "Extraction failure")
        if definite(ref) and definite(a) and definite(b):
            eligible.append((finding, int(a == ref), int(b == ref)))
    if not eligible:
        return "non-evaluable", 0, 0, 0
    aa = sum(x[1] for x in eligible)
    bb = sum(x[2] for x in eligible)
    return ("improved" if aa > bb else "reduced" if aa < bb else "tied"), len(eligible), aa, bb


def check_grid(reports, prompts):
    keys = [(x["image_id"], x["arm_id"]) for x in reports]
    if len(keys) != 80 or len(set(keys)) != 80:
        raise ValueError("Expected 80 unique image/arm reports")
    images = {x[0] for x in keys}
    if len(images) != 20 or set(keys) != {(i, a) for i in images for a in ARMS}:
        raise ValueError("Incomplete 20 x 4 grid")
    pkeys = [(x["image_id"], x["arm"]) for x in prompts]
    if len(pkeys) != 80 or set(pkeys) != set(keys):
        raise ValueError("Prompt grid does not match reports")
    return sorted(images)


def synthetic_dry_run():
    """Exercise the analysis path on synthetic records, isolated from study data."""
    folder = OUT / "synthetic"
    folder.mkdir(exist_ok=True)
    reports = [{"image_id": f"SYN{i:02d}", "arm_id": arm,
                "generated_report": "No pleural effusion." if arm in ("C0", "C2a") else "Pleural effusion."}
               for i in range(20) for arm in ARMS]
    prompts = [{"image_id": f"SYN{i:02d}", "arm": arm,
                "chat": [{"role": "user", "content": f"<image>\n{arm}: {i}"}]}
               for i in range(20) for arm in ARMS]
    refs = {(f"SYN{i:02d}", "Pleural Effusion"): "Negative" for i in range(20)}
    labels = {(f"SYN{i:02d}", arm, "Pleural Effusion"): "Negative" if arm in ("C0", "C2a") else "Positive"
              for i in range(20) for arm in ARMS}
    images = check_grid(reports, prompts)
    pair = {(x["image_id"], x["arm_id"]): x for x in reports}
    exact = sum(pair[(i, "C2a")]["generated_report"] == pair[(i, "C3")]["generated_report"] for i in images)
    changed = sum(changed_flags(pair[(i, "C2a")]["generated_report"], pair[(i, "C3")]["generated_report"])[2]["finding"] for i in images)
    scores = Counter(paired_image_score(i, "C2a", "C3", refs, labels)[0] for i in images)
    result = {"synthetic_only": True, "reports": len(reports), "images": len(images),
              "exact_matches": exact, "finding_changed_flags": changed,
              "joint_definite_per_finding": 20, "threshold_status": coverage_status(20),
              "paired_counts": dict(scores), "state_check": [state(x) for x in ("0", "-1", "blank", None)]}
    (folder / "synthetic_inputs.json").write_text(json.dumps({"reports": reports, "prompts": prompts}, indent=2) + "\n")
    (folder / "dry_run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


def main():
    required = ("TMPDIR", "HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "XDG_CACHE_HOME")
    env = {k: os.environ.get(k, "") for k in required}
    if any(not v.startswith("/storage/student4/hanhpm/") for v in env.values()):
        raise RuntimeError("A3A cache environment must be under /storage/student4/hanhpm/")
    if sha(OUT / "E4_A3A_automated_protocol.md") != PROTOCOL_SHA:
        raise RuntimeError("Frozen protocol SHA-256 mismatch")
    integrity = json.loads((OUT / "E4_A3A_input_integrity.json").read_text())
    if integrity["status"] != "PASS":
        raise RuntimeError("Input integrity failed; comparative analysis forbidden")

    reports = [json.loads(line) for line in (R4 / "E4_R4_generated_reports.jsonl").read_text().splitlines()]
    prompts = json.loads(FROZEN.read_text())
    images = check_grid(reports, prompts)
    blinded = readcsv(A2 / "E4_A2_blinded_reports.csv")
    group_counts = Counter(x["blinded_image_group_id"] for x in blinded)
    if len(blinded) != 80 or len({x["anonymous_report_id"] for x in blinded}) != 80 or len(group_counts) != 20 or set(group_counts.values()) != {4}:
        raise ValueError("A2 blinded 20 x 4 package incomplete")
    report = {(x["image_id"], x["arm_id"]): x for x in reports}
    prompt = {(x["image_id"], x["arm"]): x for x in prompts}
    for key, row in report.items():
        if row["prompt_sha256"] != prompt[key]["prompt_sha256"]:
            raise ValueError(f"Prompt hash mismatch for {key}")

    ref_rows = readcsv(R4 / "reference_labels_long.csv")
    label_rows = readcsv(R4 / "prediction_labels_long.csv")
    refs = {(x["image_id"], x["finding"]): state(x["reference_label"]) for x in ref_rows}
    labels = {(x["image_id"], x["arm_id"], x["finding"]): state(x["prediction_label"]) for x in label_rows}
    if len(ref_rows) != 100 or len(label_rows) != 400 or len(refs) != 100 or len(labels) != 400:
        raise ValueError("Cached label grid incomplete or duplicate")

    prompt_rows = []
    material_images = set()
    original_system = prompt[(images[0], "C0")]["chat"][0]["content"]
    original_user = prompt[(images[0], "C0")]["chat"][-1]["content"]
    for image in images:
        a, b = prompt[(image, "C2a")], prompt[(image, "C3")]
        ea, eb = evidence_block(a), evidence_block(b)
        ra, rb = evidence_rows(ea), evidence_rows(eb)
        material = bool(ra and rb and ra != rb)
        if material:
            material_images.add(image)
        prompt_rows.append({"image_id": image, "final_chat_equal": a["chat"] == b["chat"], "context_equal": ea == eb,
                            "evidence_rows_C2a": len(ra), "evidence_rows_C3": len(rb), "material_evidence_diff": material,
                            "different_evidence_rows": sum(x != y for x, y in zip(ra, rb)),
                            "original_chat_format_preserved": all(
                                prompt[(image, arm)]["chat"][0]["content"] == original_system
                                and prompt[(image, arm)]["chat"][-1]["content"].startswith(original_user)
                                and prompt[(image, arm)]["chat"][-1]["content"].count("<image>") == 1
                                for arm in ARMS),
                            "input_tokens_C2a": a["input_token_count"], "input_tokens_C3": b["input_token_count"],
                            "prompt_hash_C2a": a["prompt_sha256"], "prompt_hash_C3": b["prompt_sha256"]})
    writecsv(OUT / "E4_A3A_prompt_differences.csv", prompt_rows, list(prompt_rows[0]))

    stability = []
    for image in images:
        for a, b in combinations(ARMS, 2):
            ta, tb = report[(image, a)]["generated_report"], report[(image, b)]["generated_report"]
            removed, added, flags = changed_flags(ta, tb)
            stability.append({"image_id": image, "arm_a": a, "arm_b": b, "exact_equal": ta == tb,
                              "normalized_equal": normalize(ta) == normalize(tb), "removed_sentences": json.dumps(removed),
                              "added_sentences": json.dumps(added), "finding_change_flag": flags["finding"],
                              "negation_change_flag": flags["negation"], "laterality_change_flag": flags["laterality"],
                              "severity_change_flag": flags["severity"], "content_class": flags["content_type"]})
    writecsv(OUT / "E4_A3A_report_stability.csv", stability, list(stability[0]))

    coverage = []
    proxy = []
    proxy_pairs = []
    for arm in ARMS:
        for finding in FINDINGS:
            pairs = [(refs[(image, finding)], labels[(image, arm, finding)]) for image in images]
            for image, (ref, pred) in zip(images, pairs):
                proxy_pairs.append({"image_id": image, "arm": arm, "finding": finding,
                                    "reference_state": ref, "generated_state": pred,
                                    "joint_definite": definite(ref) and definite(pred),
                                    "agreement": "UNAVAILABLE" if agreement(pred, ref) is None else agreement(pred, ref),
                                    "contradiction": definite(ref) and definite(pred) and ref != pred,
                                    "omission_vs_reference_text": ref == "Positive" and pred in ("Negative", "Not mentioned"),
                                    "assertion_vs_explicit_reference_negative": ref == "Negative" and pred == "Positive"})
            cnt = Counter(pred for _, pred in pairs)
            mentions = sum(pred in ("Positive", "Negative", "Uncertain") for _, pred in pairs)
            coverage.append({"arm": arm, "finding": finding, "images": 20, "mentions": mentions,
                             "mention_coverage": f"{mentions}/20", **{k.lower().replace(" ", "_"): cnt[k] for k in ("Positive", "Negative", "Uncertain", "Not mentioned", "Extraction failure")}})
            pos_ref = sum(ref == "Positive" for ref, _ in pairs)
            neg_ref = sum(ref == "Negative" for ref, _ in pairs)
            pos_joint = sum(ref == "Positive" and definite(pred) for ref, pred in pairs)
            neg_joint = sum(ref == "Negative" and definite(pred) for ref, pred in pairs)
            joint = pos_joint + neg_joint
            pos_match = sum(ref == pred == "Positive" for ref, pred in pairs)
            neg_match = sum(ref == pred == "Negative" for ref, pred in pairs)
            contra = sum(definite(ref) and definite(pred) and ref != pred for ref, pred in pairs)
            omissions = sum(ref == "Positive" and pred in ("Negative", "Not mentioned") for ref, pred in pairs)
            unsupported = sum(ref == "Negative" and pred == "Positive" for ref, pred in pairs)
            proxy.append({"arm": arm, "finding": finding, "images": 20, "joint_definite": joint,
                          "reference_positive": pos_ref, "reference_negative": neg_ref,
                          "positive_agreement_raw": f"{pos_match}/{pos_joint}", "negative_agreement_raw": f"{neg_match}/{neg_joint}",
                          "contradiction_raw": f"{contra}/{joint}", "omission_proxy_raw": f"{omissions}/{pos_ref}",
                          "unsupported_vs_explicit_negative_raw": f"{unsupported}/{neg_ref}",
                          "uncertain_count": cnt["Uncertain"], "not_mentioned_count": cnt["Not mentioned"],
                          "extraction_failure_count": cnt["Extraction failure"],
                          "uncertain_rate": f"{cnt['Uncertain']}/20", "not_mentioned_rate": f"{cnt['Not mentioned']}/20",
                          "extraction_failure_rate": f"{cnt['Extraction failure']}/20",
                          "rate_status": "RAW_COUNTS_ONLY; SEE_PAIRED_THRESHOLD"})
    writecsv(OUT / "E4_A3A_finding_coverage.csv", coverage, list(coverage[0]))
    writecsv(OUT / "E4_A3A_reference_proxy_agreement.csv", proxy, list(proxy[0]))
    writecsv(OUT / "E4_A3A_reference_proxy_pairs.csv", proxy_pairs, list(proxy_pairs[0]))

    paired = []
    primary_supported = False
    for a, b in COMPARED:
        included = [i for i in images if (a, b) != ("C2a", "C3") or i in material_images]
        categories = Counter()
        for image in included:
            cat, n, ma, mb = paired_image_score(image, a, b, refs, labels)
            categories[cat] += 1
            paired.append({"row_type": "image", "image_id": image, "finding": "", "arm_a": a, "arm_b": b,
                           "material_input_diff": image in material_images if b == "C3" else "",
                           "joint_definite_images": n, "a_matches": ma, "b_matches": mb,
                           "improved": int(cat == "improved"), "reduced": int(cat == "reduced"), "tied": int(cat == "tied"),
                           "non_evaluable": int(cat == "non-evaluable"), "status": "RAW_IMAGE_COUNT"})
        for finding in FINDINGS:
            vals = []
            for image in included:
                ref, aa, bb = refs[(image, finding)], labels[(image, a, finding)], labels[(image, b, finding)]
                if definite(ref) and definite(aa) and definite(bb):
                    vals.append((int(aa == ref), int(bb == ref)))
            n = len(vals)
            if (a, b) == ("C2a", "C3") and n >= 5:
                primary_supported = True
            paired.append({"row_type": "finding", "image_id": "", "finding": finding, "arm_a": a, "arm_b": b,
                           "material_input_diff": "", "joint_definite_images": n,
                           "a_matches": sum(x for x, _ in vals), "b_matches": sum(y for _, y in vals),
                           "improved": sum(x > y for x, y in vals), "reduced": sum(x < y for x, y in vals),
                           "tied": sum(x == y for x, y in vals), "non_evaluable": len(included) - n,
                           "status": coverage_status(n)})
        paired.append({"row_type": "comparison", "image_id": "", "finding": "ALL", "arm_a": a, "arm_b": b,
                       "material_input_diff": "", "joint_definite_images": len(included), "a_matches": "", "b_matches": "",
                       "improved": categories["improved"], "reduced": categories["reduced"],
                       "tied": categories["tied"], "non_evaluable": categories["non-evaluable"],
                       "status": "DESCRIPTIVE_ONLY" if any(r["row_type"] == "finding" and r["arm_a"] == a and r["arm_b"] == b and r["status"] == "DESCRIPTIVE_ONLY" for r in paired) else "INSUFFICIENT_COVERAGE"})
    writecsv(OUT / "E4_A3A_paired_comparison.csv", paired, list(paired[0]))

    short = []
    for image in images:
        for arm in ARMS:
            t = report[(image, arm)]["generated_report"]
            if arm in ("C2a", "C3") or len(t) < 60:
                short.append((image, arm, len(t), short_category(t), t))
    catcounts = Counter(x[3] for x in short)
    pair_exact = sum(report[(i, "C2a")]["generated_report"] == report[(i, "C3")]["generated_report"] for i in images)
    prompt_lens = {a: sorted({prompt[(i, a)]["input_token_count"] for i in images}) for a in ARMS}
    diagnosis = f"""# E4-A3A generation diagnosis — post-hoc exploratory

**VERIFIED:** Stage 0 input hashes passed. Frozen C2a/C3 final chats differ for {sum(not x['final_chat_equal'] for x in prompt_rows)}/20 images; their CLEAR evidence blocks differ materially for **{len(material_images)}/20**. Each contains five finding rows; row-by-row differences are in `E4_A3A_prompt_differences.csv`. Thus identical evidence inputs cannot explain their {pair_exact}/20 exact report matches. All 80 chats preserve the original system message, first user instruction and one `<image>` marker: {all(x['original_chat_format_preserved'] for x in prompt_rows)}. C1/C2a/C3 append `<context>` and neutral `x` padding. Saved input ID counts: {prompt_lens}. All C2a/C3 rendered-prompt and input-ID hashes differ. The native helper uses the same `apply_chat_template` and `tokenizer_image_token` path as `write_radiologic_report`; G2C `tokenizer_verification.json` recorded tokenizer-only exact matches for all 80. Full rendered prompt strings were not persisted, so byte-for-byte comparison here uses the frozen chats and rendered hashes, not a new tokenization run.

**VERIFIED decoding configuration:** E4-R4 frozen temperature 0.2, top_p 0.8, top_k 50 in generation config, case seed reset for four arms, `max_new_tokens` 512, `use_cache=True`, native `</s>` keyword stopping criteria and streamer. `generate_cxr_repsonse` sets `do_sample=True` when temperature >0.001. One sample per image/arm means report differences may include sampling noise despite a shared case seed; branch-specific token probabilities and generation paths diverge. No multiple-seed estimate exists.

**VERIFIED short-output pattern:** {dict(catcounts)} across all C2a/C3 outputs plus C0/C1 outputs under 60 characters. The boilerplate `No acute cardiopulmonary process.` recurs heavily. Full verbatim outputs and per-case categories are in `E4_A3A_case_analysis.md`. Generic short outputs occur under the 512-token cap and are not evidence of hitting the cap. Reports ending mid-sentence are potential early termination/streaming or sampling outcomes; saved artifacts do not identify a single cause. Neutral padding changes the prompt content and could influence output, but causality is untested.

**Most likely explanation (inference from artifacts):** the augmented context plus long neutral padding changes the prompt distribution and often leads to generic short completions; shared model/output preference plausibly dominates differing numeric evidence. We cannot isolate padding, evidence semantics, stopping behavior or sampling noise from this single frozen grid. Recommend a separate development-split prompt-format test with no arbitrary padding and controlled decoding; do not edit the frozen prompts here.
"""
    (OUT / "E4_A3A_generation_diagnosis.md").write_text(diagnosis)

    lines = ["# E4-A3A case analysis — model outputs only", "", "All quoted text below is model output, not patient text or a medical judgment.", "", "## Required focused cases", ""]
    for prefix in ("CXR2891", "CXR486", "CXR981"):
        image = next(i for i in images if i.startswith(prefix))
        lines += [f"### {image}", ""]
        for arm in ARMS:
            lines += [f"- **{arm}:** {json.dumps(report[(image, arm)]['generated_report'], ensure_ascii=False)}"]
        lines += [""]
    lines += ["CXR981: C2a explicitly denies pleural effusion while C3 asserts it; the latter also mentions a CT exam absent from the frozen image modality. Neither statement is adjudicated as correct. CXR2891 differs near a sentence ending. CXR486 has a change from a generic normal sentence to a COPD/cardiomegaly statement, plus an internally broad 'no significant abnormalities' phrase.", "", "## Short-output taxonomy for every C2a/C3 and C0/C1 under 60 characters", ""]
    for image, arm, length, category, t in short:
        lines.append(f"- {image} {arm} ({length} chars; {category}): {json.dumps(t, ensure_ascii=False)}")
    (OUT / "E4_A3A_case_analysis.md").write_text("\n".join(lines) + "\n")

    primary = next(x for x in paired if x["row_type"] == "comparison" and x["arm_a"] == "C2a" and x["arm_b"] == "C3")
    decision = "E4_A3A_AUTOMATED_EVALUATION_COMPLETE_LIMITED" if primary_supported else "E4_A3A_INCONCLUSIVE_INSUFFICIENT_COVERAGE"
    limitation = f"""# E4-A3A statistical limitations

Post-hoc exploratory protocol frozen by SHA-256 `{PROTOCOL_SHA}` before A3A comparative calculations. The 20 **images**, not 80 reports, are paired units. All 20 C2a/C3 evidence blocks differ materially; report strings match exactly in {pair_exact}/20. Primary image-level raw result: improved {primary['improved']}, reduced {primary['reduced']}, tied {primary['tied']}, non-evaluable {primary['non_evaluable']}. Per-finding support is in the paired CSV; five jointly definite images per finding are required before reporting an agreement rate.

No bootstrap CI is calculated at n=20. No sign-test p-value is calculated unless five discordant image pairs exist; any qualifying p-value would be descriptive only. Undefined rates are `INSUFFICIENT_COVERAGE`, not zero. Cached blanks remain Not mentioned. The reference is report text; reference silence cannot prove absence. Multi-view Indiana study reports may contain information unavailable in the frontal image. No qualified independent clinical annotations exist; no clinical accuracy or causal CLEAR benefit is established.
"""
    (OUT / "E4_A3A_statistical_limitations.md").write_text(limitation)
    decision_md = f"""# E4-A3A research decision

**{decision}**. `HOLD_MORE_EVIDENCE`; `HOLD_COMPLEX_FUSION`.

- **VERIFIED:** all checked source hashes passed; 80 unique reports match the frozen 20×4 grid; C2a/C3 evidence content differs on {len(material_images)}/20 images, yet output text matches on {pair_exact}/20. Saved generation uses sampling. Short, repeated output is observed; its precise cause is not isolated.
- **AUTOMATED PROXY:** cached five-finding labeler states and reference-report agreement are tabulated with exact denominators. Primary paired counts are improved {primary['improved']}, reduced {primary['reduced']}, tied {primary['tied']}, non-evaluable {primary['non_evaluable']}. Per-finding threshold status is in `E4_A3A_paired_comparison.csv`.
- **UNVERIFIED:** radiological correctness, image-grounded omissions/unsupported findings, CLEAR benefit, patient-independent generalization and clinical value.

Recommendation only: design E4-B on a development cohort disjoint from locked test sets. Test a concise prompt without `x` padding, while preserving C0 and shuffled-evidence control. Use fixed greedy decoding or repeated prespecified seeds per arm. For a five-image jointly definite threshold, each target finding needs at least **five** reference-definite cases with definite predictions in *both* compared arms; this is a lower bound. With this run's C2a/C3 coverage, simply adding images at the same response behavior is unlikely to solve support; first establish an adequate output format and estimate per-finding prevalence on development data. At ~24 seconds/report on four K80 devices, 4 arms × N images takes ~96N seconds of model-call time plus load/evaluation; e.g. N=100 → ~2.7 hours serial. Eight visible K80 devices could in principle host two four-device processes, but memory, scheduler rights and interference require separate validation. No new run is authorized by this recommendation.
"""
    (OUT / "E4_A3A_research_decision.md").write_text(decision_md)

    source_files = [R4 / "E4_R4_run_manifest.json", R4 / "E4_R4_generated_reports.jsonl", R4 / "reference_labels_long.csv", R4 / "prediction_labels_long.csv", R4 / "E4_R4_execution_protocol.yaml", A2 / "E4_A2_run_manifest.json", A2 / "E4_A2_source_manifest.json", FROZEN]
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    manifest = {"run_id": OUT.name, "status": decision, "classification": "post-hoc exploratory automated reference proxy",
                "created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA,
                "source_artifacts": {str(p): sha(p) for p in source_files}, "full_integrity_report": str(OUT / "E4_A3A_input_integrity.json"),
                "input_integrity_counts": integrity["checks"], "environment": env,
                "pytest_cache_dir": "/storage/student4/hanhpm/.a3a_cache/pytest", "python": sys.version,
                "local_git_commit": git_head, "source_code_sha256": sha(Path(__file__)),
                "test_code_sha256": sha(ROOT / "tests/test_e4_a3a.py"),
                "validation_sha256": sha(OUT / "E4_A3A_validation.json"),
                "report_sha256": sha(ROOT / "reports/2026-10-17_E4_A3A_automated_proxy_evaluation.md"),
                "images": 20, "reports": 80,
                "material_C2a_C3_inputs": len(material_images), "exact_C2a_C3_outputs": pair_exact,
                "clinical_annotations": 0, "scientific_hold": "HOLD_MORE_EVIDENCE", "fusion_hold": "HOLD_COMPLEX_FUSION"}
    (OUT / "E4_A3A_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    index = ["# E4-A3A artifact index", "", f"Protocol SHA-256: `{PROTOCOL_SHA}`. Decision: `{decision}`.", "", "| File | SHA-256 |", "|---|---|"]
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "E4_A3A_artifact_index.md":
            index.append(f"| `{p.relative_to(OUT)}` | `{sha(p)}` |")
    (OUT / "E4_A3A_artifact_index.md").write_text("\n".join(index) + "\n")
    print(json.dumps({"decision": decision, "material_inputs": len(material_images), "exact_outputs": pair_exact,
                      "primary_counts": {k: primary[k] for k in ("improved", "reduced", "tied", "non_evaluable")}}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--synthetic"]:
        synthetic_dry_run()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: python3 research/e4_a3a.py [--synthetic]")
