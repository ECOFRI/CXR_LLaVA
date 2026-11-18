"""CPU-only post-hoc analysis of frozen E4-B1 text and CheXpert labels."""

import csv
import hashlib
import json
import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from research.e4_b1_analyze import label_state
from research.e4_b1_prepare import ARMS, FIVE, GENERIC_NORMAL, SIX, oracle_line


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
OUT = ROOT / "research/runs/20261017_E4_B1a_posthoc_discordant_analysis_v1"
FORMATS = ("F1", "F2", "F3")
DEFINITE = ("Positive", "Negative")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verify_inputs():
    """Fail before creating any B1a artifact if a B1 source hash differs."""
    manifest_path = SOURCE / "E4_B1_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "E4_B1_PROMPT_ROUTE_REJECTED":
        raise RuntimeError("Unexpected E4-B1 decision")
    verified = {}
    for recorded_path, expected in manifest["input_sha256"].items():
        path = SOURCE / Path(recorded_path).name
        actual = sha(path)
        if actual != expected:
            raise RuntimeError("E4-B1 manifest hash mismatch: " + path.name)
        verified[path.name] = {"expected_sha256": expected, "actual_sha256": actual, "source": "E4_B1_run_manifest.json"}
    index_text = (SOURCE / "E4_B1_artifact_index.md").read_text(encoding="utf-8")
    block_name = "E4_B1_evidence_blocks.jsonl"
    index_lines = [line for line in index_text.splitlines() if line.startswith("| `" + block_name + "` | `")]
    if len(index_lines) != 1:
        raise RuntimeError("E4-B1 evidence-block index entry missing or ambiguous")
    expected = index_lines[0].split("`")[3]
    actual = sha(SOURCE / block_name)
    if actual != expected:
        raise RuntimeError("E4-B1 evidence-block hash mismatch")
    verified[block_name] = {"expected_sha256": expected, "actual_sha256": actual, "source": "E4_B1_artifact_index.md"}
    return manifest, sha(manifest_path), verified


def direction_state(direction):
    return {"favors present": "Positive", "favors absent": "Negative"}[direction]


def oracle_shuffled_discordant(oracle, shuffled_direction):
    return oracle in DEFINITE and shuffled_direction != oracle


def collapse(report):
    text = report.strip()
    return len(text) < 60 or text in GENERIC_NORMAL


def mention(state):
    return state in ("Positive", "Negative", "Uncertain")


def change_category(matched_state, shuffled_state, matched_direction, shuffled_direction, same_text):
    """Exclusive categories; retain partial/unknown target transitions as residual."""
    if matched_direction == shuffled_direction:
        raise ValueError("Directions must disagree")
    if matched_state == matched_direction and shuffled_state == shuffled_direction:
        return "follows_evidence"
    if matched_state == shuffled_direction and shuffled_state == matched_direction:
        return "opposes_evidence"
    if matched_state != shuffled_state:
        return "other_target_state_change"
    if same_text:
        return "unchanged_text_and_target_state"
    return "text_changed_target_state_same"


def load_checked():
    cohort_rows = read_csv(SOURCE / "E4_B1_cohort.csv")
    score_rows = read_csv(SOURCE / "E4_B1_clear_scores.csv")
    donor_rows = read_csv(SOURCE / "E4_B1_derangement.csv")
    reports = read_jsonl(SOURCE / "E4_B1_reports.jsonl")
    blocks = read_jsonl(SOURCE / "E4_B1_evidence_blocks.jsonl")
    labeled = read_csv(SOURCE / "E4_B1_labeler_output.csv")
    cohort = {row["image_id"]: row for row in cohort_rows}
    scores = {(row["image_id"], row["finding"]): direction_state(row["direction"]) for row in score_rows}
    donors = {row["image_id"]: row["donor_image_id"] for row in donor_rows}
    by_report = {(row["image_id"], row["arm"]): row for row in reports}
    by_block = {(row["image_id"], row["arm"]): row["evidence_block"] for row in blocks}
    expected_grid = {(image, arm) for image in cohort for arm in ARMS}
    if len(cohort_rows) != 24 or len(cohort) != 24:
        raise RuntimeError("Cohort ID mismatch")
    if len(cohort) != 24 or len(scores) != 120 or len(score_rows) != 120 or set(scores) != {(i, f) for i in cohort for f in FIVE}:
        raise RuntimeError("Cohort/CLEAR score grid incomplete")
    if len(donors) != 24 or set(donors) != set(cohort) or set(donors.values()) != set(cohort) or any(i == d for i, d in donors.items()):
        raise RuntimeError("Derangement invalid")
    if len(reports) != len(labeled) != len(blocks) or len(reports) != 264 or set(by_report) != expected_grid or set(by_block) != expected_grid:
        raise RuntimeError("Report/label/block grid incomplete")
    states = {}
    for report, label in zip(reports, labeled):
        if label["Reports"] != report["generated_report"]:
            raise RuntimeError("Labeler/report order mismatch")
        for finding in SIX:
            states[(report["image_id"], report["arm"], finding)] = label_state(label.get(finding))
    for image in cohort:
        for finding in SIX:
            if cohort[image][finding] not in ("Positive", "Negative", "Uncertain", "Not mentioned"):
                raise RuntimeError("Unexpected oracle state")
            for arm in ("F2-O", "F3-O"):
                if by_block[(image, arm)].count(oracle_line(finding, cohort[image][finding])) != 1:
                    raise RuntimeError("Oracle sentence/block inventory mismatch")
            if by_block[(image, "F1-O")].count(f"{finding} | {cohort[image][finding]}") != 1:
                raise RuntimeError("F1 oracle table inventory mismatch")
    return cohort, scores, donors, by_report, by_block, states


def aggregate(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return groups


def rate(numerator, denominator):
    return f"{numerator}/{denominator}" if denominator else "0/0 (N/A)"


def markdown_table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|", *["| " + " | ".join(str(x) for x in row) + " |" for row in rows]]


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("E4-B1a requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    b1_manifest, b1_manifest_sha, verified = verify_inputs()
    cohort, scores, donors, reports, blocks, states = load_checked()
    OUT.mkdir(parents=True, exist_ok=True)
    images = sorted(cohort)
    reference_definite = [(i, f) for i in images for f in SIX if cohort[i][f] in DEFINITE]
    positive_cells = [(i, f) for i in images for f in SIX if cohort[i][f] == "Positive"]
    eligible_direction_images = [i for i in images if any(scores[(i, f)] != scores[(donors[i], f)] for f in FIVE)]

    discordant = []
    clear_proxy = []
    for image in images:
        for finding in FIVE:
            oracle = cohort[image][finding]
            if oracle not in DEFINITE:
                continue
            matched = scores[(image, finding)]
            shuffled = scores[(donors[image], finding)]
            clear_proxy.append({"image_id": image, "finding": finding, "oracle_state": oracle,
                                "matched_direction": matched, "shuffled_direction": shuffled,
                                "matched_agrees_reference": int(matched == oracle),
                                "shuffled_agrees_reference": int(shuffled == oracle)})
            if not oracle_shuffled_discordant(oracle, shuffled):
                continue
            for fmt in FORMATS:
                c0, s, o = (states[(image, arm, finding)] for arm in ("C0", fmt + "-S", fmt + "-O"))
                discordant.append({"format": fmt, "image_id": image, "finding": finding,
                                   "oracle_state": oracle, "matched_direction": matched, "shuffled_direction": shuffled,
                                   "C0_state": c0, "S_state": s, "O_state": o,
                                   "C0_matches_oracle": int(c0 == oracle), "S_matches_oracle": int(s == oracle),
                                   "O_matches_oracle": int(o == oracle), "S_follows_shuffled": int(s == shuffled),
                                   "O_S_state_unchanged": int(o == s),
                                   "O_S_text_unchanged": int(reports[(image, fmt + "-O")]["generated_report"] == reports[(image, fmt + "-S")]["generated_report"]),
                                   "O_not_mentioned": int(o == "Not mentioned"), "S_not_mentioned": int(s == "Not mentioned"),
                                   "both_not_mentioned": int(o == s == "Not mentioned"),
                                   "O_uncertain": int(o == "Uncertain"), "S_uncertain": int(s == "Uncertain"),
                                   "O_extraction_failure": int(o == "Extraction failure"), "S_extraction_failure": int(s == "Extraction failure")})

    positive = []
    for image, finding in positive_cells:
        for arm in ARMS:
            state = states[(image, arm, finding)]
            positive.append({"image_id": image, "finding": finding, "arm": arm,
                             "oracle_state": "Positive", "output_state": state,
                             "output_positive": int(state == "Positive")})

    direction_rows = []
    for fmt in FORMATS:
        for image in images:
            for finding in FIVE:
                m_direction, s_direction = scores[(image, finding)], scores[(donors[image], finding)]
                if m_direction == s_direction:
                    continue
                m_state, s_state = states[(image, fmt + "-M", finding)], states[(image, fmt + "-S", finding)]
                same_text = reports[(image, fmt + "-M")]["generated_report"] == reports[(image, fmt + "-S")]["generated_report"]
                direction_rows.append({"comparison": "M_vs_S", "format": fmt, "image_id": image, "finding": finding,
                                       "matched_direction": m_direction, "shuffled_direction": s_direction,
                                       "first_output_state": m_state, "second_output_state": s_state,
                                       "target_state_changed": int(m_state != s_state), "exact_text_changed": int(not same_text),
                                       "category": change_category(m_state, s_state, m_direction, s_direction, same_text)})
    for image in images:
        for finding in SIX:
            c0, n1 = states[(image, "C0", finding)], states[(image, "N1", finding)]
            same_text = reports[(image, "C0")]["generated_report"] == reports[(image, "N1")]["generated_report"]
            direction_rows.append({"comparison": "N1_vs_C0", "format": "NEUTRAL", "image_id": image, "finding": finding,
                                   "matched_direction": "UNAVAILABLE", "shuffled_direction": "UNAVAILABLE",
                                   "first_output_state": c0, "second_output_state": n1,
                                   "target_state_changed": int(c0 != n1), "exact_text_changed": int(not same_text),
                                   "category": "state_changed_no_evidence_direction" if c0 != n1 else (
                                       "text_changed_target_state_same" if not same_text else "unchanged_text_and_target_state")})

    arm_summary = []
    for arm in ARMS:
        rows = [reports[(image, arm)] for image in images]
        chars = [len(row["generated_report"]) for row in rows]
        tokens = [int(row["output_token_count"]) for row in rows]
        mentions = sum(mention(states[(image, arm, finding)]) for image in images for finding in SIX)
        arm_summary.append({"arm": arm, "reports": 24, "collapse_n": sum(collapse(row["generated_report"]) for row in rows),
                            "collapse_denominator": 24, "mention_cells": mentions, "mention_denominator": 144,
                            "mean_mentions_per_report": mentions / 24,
                            "chars_sum": sum(chars), "chars_mean": statistics.mean(chars), "chars_median": statistics.median(chars),
                            "chars_min": min(chars), "chars_max": max(chars),
                            "tokens_sum": sum(tokens), "tokens_mean": statistics.mean(tokens), "tokens_median": statistics.median(tokens),
                            "tokens_min": min(tokens), "tokens_max": max(tokens),
                            "exact_text_changed_vs_C0_images": sum(reports[(image, arm)]["generated_report"] != reports[(image, "C0")]["generated_report"] for image in images),
                            "any_six_state_changed_vs_C0_images": sum(any(states[(image, arm, f)] != states[(image, "C0", f)] for f in SIX) for image in images)})

    oracle_inventory = []
    for image in images:
        counts = Counter(cohort[image][finding] for finding in SIX)
        oracle_inventory.append({"image_id": image, "oracle_positive_sentences": counts["Positive"],
                                 "oracle_negative_sentences": counts["Negative"], "oracle_uncertain_sentences": counts["Uncertain"],
                                 "oracle_not_mentioned_sentences": counts["Not mentioned"], "oracle_total_sentences": 6,
                                 "F1_O_mentioned_findings": sum(mention(states[(image, "F1-O", f)]) for f in SIX),
                                 "F2_O_mentioned_findings": sum(mention(states[(image, "F2-O", f)]) for f in SIX),
                                 "F3_O_mentioned_findings": sum(mention(states[(image, "F3-O", f)]) for f in SIX),
                                 "F2_O_collapsed": int(collapse(reports[(image, "F2-O")]["generated_report"])),
                                 "F2_O_output_chars": len(reports[(image, "F2-O")]["generated_report"]),
                                 "F2_O_output_tokens": int(reports[(image, "F2-O")]["output_token_count"])})

    save_csv(OUT / "E4_B1a_discordant_cells.csv", discordant, list(discordant[0]))
    save_csv(OUT / "E4_B1a_positive_cells.csv", positive, list(positive[0]))
    save_csv(OUT / "E4_B1a_direction_of_change.csv", direction_rows, list(direction_rows[0]))
    save_csv(OUT / "E4_B1a_arm_summary.csv", arm_summary, list(arm_summary[0]))
    save_csv(OUT / "E4_B1a_oracle_block_inventory.csv", oracle_inventory, list(oracle_inventory[0]))
    save_csv(OUT / "E4_B1a_clear_reference_proxy.csv", clear_proxy, list(clear_proxy[0]))

    lines = ["# E4-B1a — post-hoc discordant analysis", "", "**AUTOMATED PROXY, POST-HOC.** E4-B1 decision remains `E4_B1_PROMPT_ROUTE_REJECTED`; thresholds and E4-B1 artifacts were not changed. All denominators below are exact counts, with no p-values or confidence intervals.", "",
             "## Methods and denominators", "", "The unit is an image–finding cell. CheXpert output states remain Positive, Negative, Uncertain, Not mentioned, or Extraction failure. CLEAR direction is threshold-free `favors present`→Positive and `favors absent`→Negative. Pneumothorax has no CLEAR score and is excluded from direction-discordant sets. O–S discordance requires a Positive/Negative reference-report oracle and shuffled CLEAR direction opposite to it. The F1-O categorical table differs from the F1-M/S numeric table, limiting causal interpretation. `Unchanged` below means unchanged extracted state; exact text equality is separately counted.", "",
             f"There are **{len(reference_definite)}** Positive/Negative oracle cells across six findings, **{len(clear_proxy)}** across the five CLEAR-scored findings, **{len(positive_cells)}** oracle-Positive cells across six findings, and **{len(eligible_direction_images)}/24** images with at least one matched/shuffled CLEAR direction difference. These sets are distinct.", "",
             "## 1–2. O–S discordant cells", ""]
    def discordant_table(group_rows, with_name):
        vals = []
        for name, rows in group_rows:
            n = len(rows)
            vals.append([name, n, rate(sum(x["O_matches_oracle"] for x in rows), n),
                         rate(sum(x["S_follows_shuffled"] for x in rows), n),
                         rate(sum(x["O_S_state_unchanged"] for x in rows), n),
                         rate(sum(x["O_S_text_unchanged"] for x in rows), n),
                         rate(sum(x["O_not_mentioned"] for x in rows), n),
                         rate(sum(x["S_not_mentioned"] for x in rows), n)])
        return markdown_table([with_name, "Cells", "O follows oracle", "S follows shuffled", "O=S state", "O=S text", "O Not mentioned", "S Not mentioned"], vals)
    lines += discordant_table([(fmt, [r for r in discordant if r["format"] == fmt]) for fmt in FORMATS], "Format")
    lines += ["", "By oracle state:", ""]
    lines += discordant_table([(fmt + "/" + state, [r for r in discordant if r["format"] == fmt and r["oracle_state"] == state]) for fmt in FORMATS for state in DEFINITE], "Format/state")
    lines += ["", "By finding (only findings with CLEAR scores):", ""]
    lines += discordant_table([(fmt + "/" + f, [r for r in discordant if r["format"] == fmt and r["finding"] == f]) for fmt in FORMATS for f in FIVE], "Format/finding")
    lines += ["", "By finding and oracle state (retains sparse/zero denominators):", ""]
    lines += discordant_table([(fmt + "/" + f + "/" + state, [r for r in discordant if r["format"] == fmt and r["finding"] == f and r["oracle_state"] == state]) for fmt in FORMATS for f in FIVE for state in DEFINITE], "Format/finding/state")

    lines += ["", "## 3. M–S direction of change", "", "For each direction-discordant five-finding cell: `follows_evidence` requires M and S both to output their respective evidence directions; `opposes_evidence` requires the reverse pair. `text_changed_target_state_same` means the exact report text differs while this finding's extracted state does not; it does **not** prove the changed words are semantically unrelated. `unchanged_text_and_target_state` requires exact report equality. The extra `other_target_state_change` category preserves one-sided, Uncertain, Not mentioned and extraction-failure transitions without mislabeling them as directional following.", ""]
    categories = ("follows_evidence", "opposes_evidence", "text_changed_target_state_same", "unchanged_text_and_target_state", "other_target_state_change")
    direction_table = []
    for fmt in FORMATS:
        rows = [r for r in direction_rows if r["comparison"] == "M_vs_S" and r["format"] == fmt]
        n = len(rows)
        counts = Counter(r["category"] for r in rows)
        direction_table.append([fmt, n, *[rate(counts[c], n) for c in categories]])
    lines += markdown_table(["Format", "Direction-discordant cells", *categories], direction_table)

    lines += ["", "## 4. Arm-level collapse, mention and length", ""]
    lines += markdown_table(["Arm", "Collapse", "Mentioned cells", "Mean findings/report", "Chars mean/median", "Tokens mean/median"],
                            [[r["arm"], rate(r["collapse_n"], 24), rate(r["mention_cells"], 144), f'{r["mean_mentions_per_report"]:.2f}/6',
                              f'{r["chars_mean"]:.1f}/{r["chars_median"]:g}', f'{r["tokens_mean"]:.1f}/{r["tokens_median"]:g}'] for r in arm_summary])

    lines += ["", "## 5. No-evidence C0 baseline on identical cells", ""]
    eligible_table = []
    for fmt in FORMATS:
        n = len(reference_definite)
        eligible_table.append([fmt, n, *[rate(sum(states[(i, arm, f)] == cohort[i][f] for i, f in reference_definite), n) for arm in ("C0", fmt + "-S", fmt + "-O")]])
    lines += markdown_table(["Format", "All definite cells", "C0 matches oracle", "S matches oracle", "O matches oracle"], eligible_table)
    discordant_baseline = []
    for fmt in FORMATS:
        rows = [r for r in discordant if r["format"] == fmt]
        n = len(rows)
        discordant_baseline.append([fmt, n, *[rate(sum(r[key] for r in rows), n) for key in ("C0_matches_oracle", "S_matches_oracle", "O_matches_oracle")]])
    lines += ["", "On the O–S discordant subset:", ""]
    lines += markdown_table(["Format", "Discordant cells", "C0 matches oracle", "S matches oracle", "O matches oracle"], discordant_baseline)
    lines += ["", "On all 34 discordant cells, O does not exceed S in any format: F1 6/34 versus 6/34, F2 7/34 versus 17/34, F3 12/34 versus 15/34. F3's C0/S/O counts are 17/34, 15/34 and 12/34. Thus these proxy counts provide **no evidence that oracle text was used in the intended direction**. This does not imply the generated text is invariant to prompts. There is no post-hoc equivalence threshold; the counts are agreement with reference-report labels, not image truth."]

    lines += ["", "## 6. Oracle-Positive cells", ""]
    lines += markdown_table(["Arm", "Output Positive / oracle-Positive cells"],
                            [[arm, rate(sum(r["output_positive"] for r in positive if r["arm"] == arm), len(positive_cells))] for arm in ARMS])

    lines += ["", "## 7. Oracle block inventory", "", "F2-O and F3-O blocks were checked sentence-by-sentence against all six saved oracle states for every image. Each block has six reference-report sentences; `Not mentioned` and `Uncertain` are explicit sentences rather than Negative evidence.", ""]
    inventory_rows = []
    for collapsed in (1, 0):
        group = [r for r in oracle_inventory if r["F2_O_collapsed"] == collapsed]
        n = len(group)
        inventory_rows.append(["F2-O collapsed" if collapsed else "F2-O not collapsed", n,
                               *[rate(sum(r[k] for r in group), 6*n) for k in ("oracle_positive_sentences", "oracle_negative_sentences", "oracle_uncertain_sentences", "oracle_not_mentioned_sentences")],
                               rate(sum(r["F2_O_mentioned_findings"] for r in group), 6*n)])
    lines += markdown_table(["Group", "Images", "Oracle Positive sentences", "Negative", "Uncertain", "Not mentioned", "F2-O output mentions"], inventory_rows)
    lines += ["", "Per-image counts, F1/F2/F3-O mentions and F2-O collapse status are in `E4_B1a_oracle_block_inventory.csv`."]

    lines += ["", "## 8. Neutral-text background versus M–S sensitivity", "", "All image-level comparisons below use the **same 19 images** with at least one CLEAR direction difference. A finding-state change means any of the six extracted states changes; exact text change is counted separately. Target-cell changes use the same five CLEAR-scored findings (19×5=95 cells); N1/C0 has no evidence direction and cannot be classified as following or opposing it.", ""]
    background = []
    for name, first, second in [("N1 vs C0", "N1", "C0")] + [(fmt + " M vs S", fmt + "-M", fmt + "-S") for fmt in FORMATS]:
        n = len(eligible_direction_images)
        text_change = sum(reports[(i, first)]["generated_report"] != reports[(i, second)]["generated_report"] for i in eligible_direction_images)
        any_six = sum(any(states[(i, first, f)] != states[(i, second, f)] for f in SIX) for i in eligible_direction_images)
        target_cells = sum(states[(i, first, f)] != states[(i, second, f)] for i in eligible_direction_images for f in FIVE)
        background.append([name, rate(text_change, n), rate(any_six, n), rate(target_cells, 5*n)])
    lines += markdown_table(["Comparison", "Exact text changed/images", "Any six-state change/images", "Five-finding state changes/cells"], background)
    lines += ["", "N1 versus C0 changes at least one extracted state on 19/19 eligible images, exceeding each M–S comparison's 3/19, 11/19 and 13/19. Arm-level collapse, mention and length for C0/N1/M/S/O are shown in section 4. The neutral-text comparison is a background change rate, not a causal subtraction or a calibrated noise correction."]

    matched_agreement = sum(r["matched_agrees_reference"] for r in clear_proxy)
    shuffled_agreement = sum(r["shuffled_agrees_reference"] for r in clear_proxy)
    lines += ["", "## 9. CLEAR direction versus reference report", "", f"Among **{len(clear_proxy)}** Positive/Negative reference-report cells for the five scored findings, CLEAR-matched direction agrees with the report in **{matched_agreement}/{len(clear_proxy)}**, and shuffled direction in **{shuffled_agreement}/{len(clear_proxy)}**. This is an **AUTOMATED PROXY, POST-HOC** comparison, not correctness relative to the radiograph. The cell-level source is `E4_B1a_clear_reference_proxy.csv`.", ""]
    lines += markdown_table(["Finding", "Definite cells", "Matched agrees report", "Shuffled agrees report"],
                            [[f, len(group := [r for r in clear_proxy if r["finding"] == f]),
                              rate(sum(r["matched_agrees_reference"] for r in group), len(group)),
                              rate(sum(r["shuffled_agrees_reference"] for r in group), len(group))] for f in FIVE])
    lines += ["", "Split by reference-report state:", ""]
    lines += markdown_table(["Oracle state", "Definite cells", "Matched agrees report", "Shuffled agrees report"],
                            [[state, len(group := [r for r in clear_proxy if r["oracle_state"] == state]),
                              rate(sum(r["matched_agrees_reference"] for r in group), len(group)),
                              rate(sum(r["shuffled_agrees_reference"] for r in group), len(group))] for state in DEFINITE])

    lines += ["", "## Design limitations and two separate questions", "", "- **Original criterion 2:** the ≥4/6 mention requirement is above C0's 71/24 = 2.96/6 baseline and was unmet by every E4-B1 arm. This is a strict development gate, not a model quality estimate.",
              "- **Original criterion 3:** oracle following used all 69 definite cells, including cells where O and S evidence agreed and Pneumothorax with no CLEAR score. Such cells dilute a contrast focused on conflicting evidence. This analysis isolates conflicts post hoc; it cannot retroactively change the frozen gate.",
              "- **Original criterion 4:** any extracted-state change measures sensitivity, not whether the change follows the evidence direction. The directional categories above retain that distinction.",
              "- Reference reports may describe views absent from the frontal input; rule-based labels can omit nuance. No radiologist adjudication or image-grounded truth was added.", "",
              "**(a) Does the model follow text evidence content?** VERIFIED: prompts/evidence differ, and exact output text changes are directly counted. AUTOMATED PROXY, POST-HOC: full directional following is rare on M–S discordant cells (F1 0/62, F2 2/62, F3 0/62), while N1/C0 changes extracted states on 19/19 eligible images. O does not beat S on the 34 O–S discordant cells in any format. Thus there is no evidence here of *reliable intended-direction following*, although the model's text is prompt-sensitive. UNVERIFIED: semantic reliance on evidence or its causal mechanism.", "",
              "**(b) Is CLEAR evidence correct for the image?** AUTOMATED PROXY, POST-HOC: matched/shuffled CLEAR directions can be compared with reference-report labels in the definite cells above. UNVERIFIED: radiograph-grounded correctness, unsupported findings, clinical utility or generalization. No E4-B2 proposal or run is made.", ""]
    (OUT / "E4_B1a_findings.md").write_text("\n".join(lines), encoding="utf-8")

    output_files = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "E4_B1a_run_manifest.json")
    run_manifest = {"status": "COMPLETE_POST_HOC_SECONDARY_ANALYSIS", "classification": "AUTOMATED PROXY, POST-HOC",
                    "E4_B1_decision_unchanged": b1_manifest["status"], "cpu_only": True, "cuda_visible_devices": "",
                    "source_manifest_sha256": b1_manifest_sha, "verified_inputs": verified,
                    "source_code_sha256": sha(Path(__file__)), "source_commit_at_analysis": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "cohort_images": len(images), "formats": list(FORMATS), "oracle_definite_six_cells": len(reference_definite),
                    "clear_scored_definite_cells": len(clear_proxy), "oracle_positive_six_cells": len(positive_cells),
                    "direction_eligible_images": len(eligible_direction_images),
                    "output_sha256": {path.name: sha(path) for path in output_files}}
    (OUT / "E4_B1a_run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"discordant_cells_per_format": len(discordant)//3, "direction_rows": len(direction_rows),
                      "positive_cells_per_arm": len(positive_cells), "source_hashes": len(verified)}, indent=2))


if __name__ == "__main__":
    main()
