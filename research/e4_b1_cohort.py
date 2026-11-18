"""Select a fixed enriched Indiana development cohort for E4-B1."""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "result/indiana_full_k80_20261004_031940_1252388/inference/cohort.csv"
LABELS = ROOT / "result/indiana_full_k80_20261004_inventory02/nlm_review_01/gt_labeling/raw_labels.csv"
MAP = ROOT / "result/indiana_full_k80_20261004_inventory02/nlm_review_01/gt_labeling/row_map.csv"
E4_IDS = ROOT / "research/runs/20261017_E4_R4_controlled_resume_v1/cohort_ids.csv"
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
SEED = 2026101701
FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax")


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def state(value):
    return {"1": "Positive", "1.0": "Positive", "0": "Negative", "0.0": "Negative",
            "-1": "Uncertain", "-1.0": "Uncertain", "": "Not mentioned"}[str(value).strip()]


def select(candidates, seed=SEED, size=24, target=4):
    rng = random.Random(seed)
    remaining = list(candidates)
    rng.shuffle(remaining)
    chosen = []
    support = Counter()
    while len(chosen) < size and any(support[f] < target and any(r[f] == "Positive" for r in remaining) for f in FINDINGS):
        ranked = sorted(remaining, key=lambda r: sum(r[f] == "Positive" and support[f] < target for f in FINDINGS), reverse=True)
        gain = sum(ranked[0][f] == "Positive" and support[f] < target for f in FINDINGS)
        if gain == 0:
            break
        tied = [r for r in ranked if sum(r[f] == "Positive" and support[f] < target for f in FINDINGS) == gain]
        row = rng.choice(tied)
        needed = [f for f in FINDINGS if row[f] == "Positive" and support[f] < target]
        row = dict(row, selection_reason="positive_enrichment:" + ",".join(needed))
        chosen.append(row)
        remaining = [r for r in remaining if r["image_id"] != row["image_id"]]
        support.update(f for f in FINDINGS if row[f] == "Positive")
    neutral = [r for r in remaining if all(r[f] != "Positive" for f in FINDINGS)]
    rng.shuffle(neutral)
    for row in neutral[:max(0, size-len(chosen))]:
        chosen.append(dict(row, selection_reason="no_positive_among_six"))
    if len(chosen) != size:
        raise ValueError(f"Only {len(chosen)} candidate rows after enrichment and neutral fill")
    return sorted(chosen, key=lambda r: r["image_id"]), support


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "E4_B1_cohort.csv").exists():
        raise FileExistsError("Cohort is immutable; choose a new run version")
    cohort, labels, mapping = rows(BASE), rows(LABELS), rows(MAP)
    assert len(cohort) == len(labels) == len(mapping) == 3667
    assert sha(LABELS) == "e96c71d6d488c4329c968c0096723b16c8d2da49db28da26da07d64fbf485673"
    excluded = {r["image_id"] for r in rows(E4_IDS)}
    assert len(excluded) == 20
    candidates = []
    for i, (base, lab, m) in enumerate(zip(cohort, labels, mapping)):
        assert int(m["row_index"]) == i and m["image_id"] == base["image_id"]
        assert lab["Reports"] == base["ground_truth_report"]
        assert hashlib.sha256(lab["Reports"].encode()).hexdigest() == m["text_sha256"]
        if base["image_id"] in excluded:
            continue
        r = {"image_id": base["image_id"], "report_id": base["report_id"],
             "image_path": base["image_path"], "image_sha256": base["image_sha256"],
             "reference_report_sha256": m["text_sha256"], "view_label": base["view_label"]}
        r.update({f: state(lab[f]) for f in FINDINGS})
        candidates.append(r)
    selected, support = select(candidates)
    assert len({r["image_id"] for r in selected}) == 24 and not excluded.intersection(r["image_id"] for r in selected)
    for r in selected:
        p = Path(r["image_path"])
        assert p.is_file() and sha(p) == r["image_sha256"]
    fields = ["image_id", "report_id", "image_path", "image_sha256", "reference_report_sha256", "view_label", *FINDINGS, "selection_reason"]
    with (OUT / "E4_B1_cohort.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)
    summary = {"seed": SEED, "cohort_size": 24, "candidate_count_after_exclusion": len(candidates),
               "excluded_E4_images": len(excluded), "support_selected": {f: support[f] for f in FINDINGS},
               "available_positive": {f: sum(r[f] == "Positive" for r in candidates) for f in FINDINGS},
               "shortfall": {f: max(0, 4-support[f]) for f in FINDINGS},
               "neutral_selected": sum(r["selection_reason"] == "no_positive_among_six" for r in selected),
               "cohort_sha256": sha(OUT / "E4_B1_cohort.csv"),
               "input_sha256": {str(p): sha(p) for p in (BASE, LABELS, MAP, E4_IDS)}}
    (OUT / "E4_B1_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
