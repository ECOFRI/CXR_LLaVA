"""Freeze E4-B1 evidence blocks and exact tokenizer-only prompts."""

import csv
import hashlib
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/runs/20261017_E4_B1_evidence_format_gate_v1"
PROTOCOL_SHA = "a6eb9661672b65816630866ec2ad8a9676179562202c4e0b2966cebb0b599269"
SIX = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax")
FIVE = SIX[:5]
ARMS = ("C0", "N1", "F1-M", "F1-S", "F1-O", "F2-M", "F2-S", "F2-O", "F3-M", "F3-S", "F3-O")
SYSTEM = "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."
BASE_USER = "<image>\nWrite a radiologic report on the given chest radiograph, including information about atelectasis, cardiomegaly, consolidation, pulmonary edema, pleural effusion, and pneumothorax.\n"
NEUTRAL = ("Please write a clear and complete description of the supplied image. Use ordinary prose and a consistent order. "
           "Describe what is visible without adding assumptions. Keep each sentence readable and avoid repetition. "
           "If an area is difficult to assess, say so plainly. Do not use information outside this image. "
           "Finish with a concise summary in the same style. Make the opening sentence direct and keep the remaining description orderly. "
           "Prefer ordinary wording that a reader can follow without ambiguity. If the image is unclear, describe that limitation plainly.")
F1_HEADER = "Auxiliary CLEAR paired similarity scores; not calibrated disease probabilities. Use the image; these scores do not establish diagnoses.\nfinding | positive cosine | negative cosine | paired positive score"
F1_ORACLE_HEADER = "Reference-report labels for a diagnostic prompt test; they are not image truth. Use the image if they conflict with it.\nfinding | reference-report state"
F2_HEADER = "Auxiliary evidence is uncertain and is not a diagnosis. Use the image as primary source."
F3_INSTRUCTION = "State your own assessment of atelectasis, cardiomegaly, consolidation, edema, pleural effusion, and pneumothorax explicitly; rely on the image if auxiliary evidence conflicts with it."
GENERIC_NORMAL = ("No acute cardiopulmonary process.", "No evidence of acute cardiopulmonary disease.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def derangement(ids, seed=2026101702):
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    result = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    assert set(result) == set(result.values()) and all(a != b for a, b in result.items())
    return result


def oracle_line(finding, state):
    f = finding.lower()
    return {"Positive": f"The reference report states {f} is present.",
            "Negative": f"The reference report states no {f}.",
            "Uncertain": f"The reference report is uncertain about {f}.",
            "Not mentioned": f"The reference report does not mention {f}."}[state]


def clear_line(finding, direction):
    f = finding.lower()
    return (f"Auxiliary evidence suggests {f} may be present." if direction == "favors present"
            else f"Auxiliary evidence suggests no {f}.")


def block(arm, item, matched, donor):
    if arm == "C0":
        return ""
    if arm == "N1":
        return NEUTRAL
    fmt, source = arm.split("-")
    scores = matched if source == "M" else donor
    if fmt == "F1":
        if source == "O":
            return F1_ORACLE_HEADER + "\n" + "\n".join(f"{f} | {item[f]}" for f in SIX)
        return F1_HEADER + "\n" + "\n".join(
            f"{f} | {scores[f]['positive_cosine_similarity']} | {scores[f]['negative_cosine_similarity']} | {scores[f]['paired_softmax_score']}"
            for f in FIVE)
    if source == "O":
        lines = [oracle_line(f, item[f]) for f in SIX]
    else:
        lines = [clear_line(f, scores[f]["direction"]) for f in FIVE]
        lines.append("No auxiliary estimate is available for pneumothorax; assess the image directly.")
    text = F2_HEADER + "\n" + "\n".join(lines)
    if fmt == "F3":
        text += "\n" + F3_INSTRUCTION
    return text


def chat_for(arm, content):
    user = BASE_USER if arm == "C0" else BASE_USER + "\n<context>\n" + content + "\n</context>"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def tokenizer_image_token(rendered, tokenizer):
    chunks = [tokenizer(part).input_ids for part in rendered.split("<image>")]
    offset = 0
    output = []
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        output.append(chunks[0][0])
    for j, chunk in enumerate(chunks):
        if j:
            output.append(-200)
        output.extend(chunk[offset:])
    return output


def main():
    from transformers import LlamaTokenizer

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert sha(OUT / "E4_B1_protocol.md") == PROTOCOL_SHA
    outputs = (OUT / "E4_B1_evidence_blocks.jsonl", OUT / "E4_B1_frozen_prompts.jsonl")
    if any(p.exists() for p in outputs):
        raise FileExistsError("Frozen prompts already exist")
    cohort = list(csv.DictReader((OUT / "E4_B1_cohort.csv").open(newline="", encoding="utf-8")))
    scores = list(csv.DictReader((OUT / "E4_B1_clear_scores.csv").open(newline="", encoding="utf-8")))
    assert len(cohort) == 24 and len(scores) == 120
    byimage = {item["image_id"]: item for item in cohort}
    scoremap = {(row["image_id"], row["finding"]): row for row in scores}
    assert len(scoremap) == 120 and all((i, f) in scoremap for i in byimage for f in FIVE)
    donors = derangement(byimage)
    tok = LlamaTokenizer.from_pretrained(str(ROOT / "model_cache/cxr-llava-v2-k80-fp32"), local_files_only=True, add_special_tokens=False)
    blocks, prompts, diffrows = [], [], []
    for image in sorted(byimage):
        item = byimage[image]
        mat = {f: scoremap[(image, f)] for f in FIVE}
        donor = {f: scoremap[(donors[image], f)] for f in FIVE}
        n_diff = sum(mat[f]["direction"] != donor[f]["direction"] for f in FIVE)
        diffrows.append({"image_id": image, "donor_image_id": donors[image], "differing_directions": n_diff})
        for arm in ARMS:
            content = block(arm, item, mat, donor)
            chat = chat_for(arm, content)
            rendered = tok.apply_chat_template(chat, tokenize=False)
            ids = tokenizer_image_token(rendered, tok)
            assert rendered.count("<image>") == 1 and ids.count(-200) == 1
            assert len(ids) - 1 + 1024 + 512 <= 2048
            blocks.append({"image_id": image, "arm": arm, "evidence_donor": donors[image] if arm.endswith("-S") else image if arm.endswith("-M") else None,
                           "evidence_block": content, "evidence_sha256": hashlib.sha256(content.encode()).hexdigest(),
                           "differing_directions": n_diff})
            prompts.append({"image_id": image, "arm": arm, "chat": chat, "rendered_prompt": rendered,
                            "input_ids": ids, "input_token_count": len(ids),
                            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                            "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            "source_sha256": item["image_sha256"], "evidence_sha256": blocks[-1]["evidence_sha256"]})
    assert len(blocks) == len(prompts) == 264
    for path, items in zip(outputs, (blocks, prompts)):
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "E4_B1_derangement.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "donor_image_id", "differing_directions"])
        w.writeheader()
        w.writerows(diffrows)
    summary = {"status": "PASS", "n_images": 24, "n_arms": 11, "prompts": 264,
               "eligible_direction_differences": sum(r["differing_directions"] > 0 for r in diffrows),
               "token_ranges": {a: [min(r["input_token_count"] for r in prompts if r["arm"] == a),
                                    max(r["input_token_count"] for r in prompts if r["arm"] == a)] for a in ARMS},
               "protocol_sha256": PROTOCOL_SHA, "scores_sha256": sha(OUT / "E4_B1_clear_scores.csv"),
               "blocks_sha256": sha(outputs[0]), "prompts_sha256": sha(outputs[1]),
               "derangement_sha256": sha(OUT / "E4_B1_derangement.csv")}
    (OUT / "E4_B1_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
