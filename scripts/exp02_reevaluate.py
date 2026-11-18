#!/usr/bin/env python
"""Offline EXP-02 evaluator for saved Indiana/CheXpert artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_PATHOLOGIES = [
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return float("nan")
    return (2 * tp) / denominator


def safe_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def is_success_row(row: pd.Series) -> bool:
    status = str(row.get("status", "")).strip().lower()
    text = str(row.get("generated_report", "") if not pd.isna(row.get("generated_report", "")) else "")
    return status == "success" and bool(text.strip())


def canonicalize_predictions(predictions: pd.DataFrame, subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"image_id", "ground_truth_report", "generated_report", "status"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"prediction file is missing columns: {sorted(missing)}")
    if "image_id" not in subset.columns:
        raise ValueError("subset file is missing image_id")

    work = predictions.copy()
    if work['image_id'].isna().any() or subset['image_id'].isna().any():
        raise ValueError('Missing image_id')
    work["source_row_index"] = np.arange(len(work))
    work["image_id"] = work["image_id"].astype(str)
    subset_ids = set(subset["image_id"].astype(str))
    outside = sorted(set(work["image_id"]) - subset_ids)
    if outside:
        raise ValueError(f"prediction contains image_id outside subset: {outside[:10]}")

    conflicts = []
    canonical_rows = []
    for image_id, group in work.groupby("image_id", sort=False):
        if group['ground_truth_report'].fillna('').nunique() != 1:
            raise ValueError(f'Conflicting GT across records: {image_id}')
        statuses = group.apply(is_success_row, axis=1).tolist()
        if len(group) > 1 or len(set(statuses)) > 1:
            conflicts.append(
                {
                    "image_id": image_id,
                    "raw_rows": int(len(group)),
                    "source_row_indexes": ";".join(str(int(v)) for v in group["source_row_index"]),
                    "success_flags": ";".join(str(bool(v)) for v in statuses),
                    "statuses": ";".join(str(v) for v in group["status"].fillna("")),
                }
            )
        canonical_rows.append(group.iloc[-1].copy())

    canonical = pd.DataFrame(canonical_rows, columns=work.columns).reset_index(drop=True)
    canonical["canonical_success"] = canonical.apply(is_success_row, axis=1)
    canonical["canonical_status"] = np.where(canonical["canonical_success"], "success", "failed")
    empty_success = (
        canonical["status"].astype(str).str.strip().str.lower().eq("success")
        & ~canonical["generated_report"].fillna("").astype(str).str.strip().astype(bool)
    )
    canonical["canonical_reason"] = ""
    canonical.loc[empty_success, "canonical_reason"] = "success_status_with_empty_generated_report"

    subset_order = subset.copy().reset_index(drop=True)
    subset_order["subset_row_index"] = np.arange(len(subset_order))
    subset_order["image_id"] = subset_order["image_id"].astype(str)
    if subset_order["image_id"].duplicated().any():
        duplicated = subset_order.loc[subset_order["image_id"].duplicated(), "image_id"].tolist()
        raise ValueError(f"subset contains duplicated image_id: {duplicated[:10]}")

    merged = subset_order.merge(
        canonical,
        on="image_id",
        how="left",
        suffixes=("_subset", ""),
    ).sort_values("subset_row_index")
    merged["canonical_status"] = merged["canonical_status"].fillna("not_attempted")
    merged["canonical_success"] = merged["canonical_success"].fillna(False).astype(bool)
    merged["canonical_reason"] = merged["canonical_reason"].fillna("not_attempted")
    if 'ground_truth_report_subset' in merged:
        attempted = merged['canonical_status'].ne('not_attempted')
        if not merged.loc[attempted, 'ground_truth_report'].fillna('').equals(
            merged.loc[attempted, 'ground_truth_report_subset'].fillna('')
        ):
            raise ValueError('Prediction GT does not match subset GT')

    return merged, pd.DataFrame(conflicts)


def read_headerless_reports(path: Path) -> list[str]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            if len(row) != 1:
                raise ValueError(f"{path} row {index} has {len(row)} columns, expected 1")
            rows.append(row[0])
    return rows


def align_labels(
    canonical: pd.DataFrame,
    row_map: pd.DataFrame,
    gt_input: list[str],
    gen_input: list[str],
    gt_labels: pd.DataFrame,
    gen_labels: pd.DataFrame,
    pathologies=None,
) -> tuple[pd.DataFrame, dict]:
    row_map = row_map.copy()
    row_map["image_id"] = row_map["image_id"].astype(str)
    expected_rows = list(range(len(row_map)))
    if row_map["chexpert_row"].tolist() != expected_rows:
        raise ValueError("row_map chexpert_row is not contiguous from 0")
    if row_map["image_id"].duplicated().any():
        raise ValueError("row_map has duplicated image_id")

    lengths = {
        "row_map": len(row_map),
        "gt_input": len(gt_input),
        "gen_input": len(gen_input),
        "gt_labels": len(gt_labels),
        "gen_labels": len(gen_labels),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"CheXpert row count mismatch: {lengths}")
    if "Reports" not in gt_labels.columns or "Reports" not in gen_labels.columns:
        raise ValueError("label CSVs must contain Reports column")
    if gt_labels["Reports"].fillna("").astype(str).tolist() != gt_input:
        raise ValueError("GT label Reports column does not match headerless input")
    if gen_labels["Reports"].fillna("").astype(str).tolist() != gen_input:
        raise ValueError("Generated label Reports column does not match headerless input")

    success = canonical.loc[canonical["canonical_success"]].copy()
    success["image_id"] = success["image_id"].astype(str)
    aligned = row_map.merge(success, on="image_id", how="left", suffixes=("_rowmap", ""))
    if aligned["canonical_success"].isna().any():
        missing = aligned.loc[aligned["canonical_success"].isna(), "image_id"].tolist()
        raise ValueError(f"row_map contains images without canonical success: {missing[:10]}")

    extra = set(success["image_id"]) - set(row_map["image_id"])
    if extra:
        raise ValueError(f"canonical success without labels: {sorted(extra)[:10]}")

    if "report_id" in aligned.columns and "report_id_rowmap" in aligned.columns:
        left = aligned["report_id"].astype(str)
        right = aligned["report_id_rowmap"].astype(str)
        if not left.equals(right):
            raise ValueError("row_map report_id does not match canonical prediction")

    gt_text = aligned["ground_truth_report"].fillna("").astype(str).tolist()
    gen_text = aligned["generated_report"].fillna("").astype(str).tolist()
    if gt_text != gt_input:
        raise ValueError("canonical GT text does not match GT CheXpert input order")
    if gen_text != gen_input:
        raise ValueError("canonical generated text does not match generated CheXpert input order")

    for pathology in (TARGET_PATHOLOGIES if pathologies is None else pathologies):
        if pathology not in gt_labels.columns or pathology not in gen_labels.columns:
            raise ValueError(f"missing pathology column: {pathology}")
        aligned[f"gt_{pathology}"] = gt_labels[pathology].to_numpy()
        aligned[f"pred_{pathology}"] = gen_labels[pathology].to_numpy()

    return aligned, lengths


def label_state(value) -> str:
    if pd.isna(value):
        return "unmentioned"
    number = safe_float(value)
    if number == 1:
        return "positive"
    if number == 0:
        return "negative"
    if number == -1:
        return "uncertain"
    return "other"


def metric_status(tp: int, fp: int, fn: int, n_valid: int) -> str:
    if n_valid == 0:
        return "undefined_no_valid_pairs"
    if (2 * tp + fp + fn) == 0:
        return "undefined_no_positive_reference_or_prediction"
    return "defined"


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return precision, recall, f1_from_counts(tp, fp, fn)


def bootstrap_f1(gt: np.ndarray, pred: np.ndarray, iterations: int, seed: int) -> dict:
    n_valid = len(gt)
    if n_valid == 0:
        return {
            "ci_low_95": float("nan"),
            "ci_high_95": float("nan"),
            "bootstrap_total": int(iterations),
            "bootstrap_defined": 0,
            "bootstrap_undefined": int(iterations),
            "ci_reason": "no_valid_pairs",
        }

    rng = np.random.default_rng(seed)
    samples = []
    undefined = 0
    for _ in range(iterations):
        indexes = rng.integers(0, n_valid, size=n_valid)
        g = gt[indexes]
        p = pred[indexes]
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        value = f1_from_counts(tp, fp, fn)
        if math.isnan(value):
            undefined += 1
        else:
            samples.append(value)

    if not samples:
        return {
            "ci_low_95": float("nan"),
            "ci_high_95": float("nan"),
            "bootstrap_total": int(iterations),
            "bootstrap_defined": 0,
            "bootstrap_undefined": int(undefined),
            "ci_reason": "all_resamples_undefined",
        }

    values = np.asarray(samples, dtype=float)
    return {
        "ci_low_95": float(np.percentile(values, 2.5, method="linear")),
        "ci_high_95": float(np.percentile(values, 97.5, method="linear")),
        "bootstrap_total": int(iterations),
        "bootstrap_defined": int(len(values)),
        "bootstrap_undefined": int(undefined),
        "ci_reason": "conditional_on_defined_resamples",
    }


def compute_metrics(aligned: pd.DataFrame, n_requested: int, n_success: int, iterations: int, seed: int, pathologies=None):
    metric_rows = []
    ci_rows = []
    coverage_rows = []
    error_rows = []
    omission_rows = []

    for pathology in (TARGET_PATHOLOGIES if pathologies is None else pathologies):
        gt_col = f"gt_{pathology}"
        pred_col = f"pred_{pathology}"
        states = pd.DataFrame(
            {
                "image_id": aligned["image_id"],
                "pathology": pathology,
                "gt_state": aligned[gt_col].map(label_state),
                "pred_state": aligned[pred_col].map(label_state),
                "ground_truth_report": aligned["ground_truth_report"],
                "generated_report": aligned["generated_report"],
            }
        )
        coverage_rows.append(states)
        omission_rows.append(
            states.loc[(states["gt_state"] == "positive") & (states["pred_state"].isin(["uncertain", "unmentioned"]))]
        )

        gt = aligned[gt_col]
        pred = aligned[pred_col]
        valid = gt.isin([0, 1]) & pred.isin([0, 1])
        gt_valid = gt.loc[valid].astype(int).to_numpy()
        pred_valid = pred.loc[valid].astype(int).to_numpy()

        tp = int(((gt_valid == 1) & (pred_valid == 1)).sum())
        fp = int(((gt_valid == 0) & (pred_valid == 1)).sum())
        tn = int(((gt_valid == 0) & (pred_valid == 0)).sum())
        fn = int(((gt_valid == 1) & (pred_valid == 0)).sum())
        n_valid = int(valid.sum())
        n_gt_positive = int((gt_valid == 1).sum())
        n_gt_negative = int((gt_valid == 0).sum())
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        status = metric_status(tp, fp, fn, n_valid)
        low_support = bool(n_valid < 20 or n_gt_positive < 5)
        row = {
            "pathology": pathology,
            "n_requested": int(n_requested),
            "n_success": int(n_success),
            "n_valid": n_valid,
            "coverage": n_valid / n_success if n_success else float("nan"),
            "n_gt_positive": n_gt_positive,
            "n_gt_negative": n_gt_negative,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_status": status,
            "low_support": low_support,
        }
        metric_rows.append(row)

        ci = bootstrap_f1(gt_valid, pred_valid, iterations, seed)
        if status != "defined":
            ci["ci_low_95"] = float("nan")
            ci["ci_high_95"] = float("nan")
            ci["ci_reason"] = status
        ci_rows.append({"pathology": pathology, "f1": f1, "seed": seed, **ci})

        mismatches = aligned.loc[valid & (gt.astype(float) != pred.astype(float))].copy()
        for _, err in mismatches.iterrows():
            gt_label = int(err[gt_col])
            pred_label = int(err[pred_col])
            error_rows.append(
                {
                    "image_id": err["image_id"],
                    "report_id": err.get("report_id", err.get("report_id_rowmap", "")),
                    "pathology": pathology,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "error_type": "FN" if gt_label == 1 and pred_label == 0 else "FP",
                    "ground_truth_report": err["ground_truth_report"],
                    "generated_report": err["generated_report"],
                }
            )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(ci_rows),
        pd.concat(coverage_rows, ignore_index=True),
        pd.DataFrame(error_rows),
        pd.concat(omission_rows, ignore_index=True) if omission_rows else pd.DataFrame(),
    )


def summarize_system(canonical: pd.DataFrame, size: int) -> dict:
    n_requested = len(canonical)
    n_success = int(canonical["canonical_success"].sum())
    n_failed = int((canonical["canonical_status"] == "failed").sum())
    n_not_attempted = int((canonical["canonical_status"] == "not_attempted").sum())
    success = canonical.loc[canonical["canonical_success"]].copy()
    latencies = pd.to_numeric(success.get("latency_sec", pd.Series(dtype=float)), errors="coerce").dropna()
    vram = pd.to_numeric(success.get("peak_vram_gb", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "subset_size": int(size),
        "n_requested": int(n_requested),
        "n_success": n_success,
        "n_failed": n_failed,
        "n_not_attempted": n_not_attempted,
        "failure_rate": n_failed / n_requested if n_requested else float("nan"),
        "completion_rate": n_success / n_requested if n_requested else float("nan"),
        "latency_count": int(len(latencies)),
        "latency_missing": int(n_success - len(latencies)),
        "mean_latency_sec": float(latencies.mean()) if len(latencies) else float("nan"),
        "median_latency_sec": float(latencies.median()) if len(latencies) else float("nan"),
        "min_latency_sec": float(latencies.min()) if len(latencies) else float("nan"),
        "max_latency_sec": float(latencies.max()) if len(latencies) else float("nan"),
        "vram_count": int(len(vram)),
        "vram_missing": int(n_success - len(vram)),
        "peak_vram_gb_max": float(vram.max()) if len(vram) else float("nan"),
        "peak_vram_gb_mean": float(vram.mean()) if len(vram) else float("nan"),
    }


def compare_old_metrics(source: Path, size: int, metrics: pd.DataFrame, dataset="indiana") -> pd.DataFrame:
    old_path = source / "metrics" / f"{dataset}_f1_subset_{size}_pathology_metrics.csv"
    if not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    keep = [c for c in ["pathology", "n_valid", "tp", "fp", "tn", "fn", "f1"] if c in old.columns]
    old = old[keep].rename(columns={c: f"old_{c}" for c in keep if c != "pathology"})
    new = metrics[["pathology", "n_valid", "tp", "fp", "tn", "fn", "f1", "f1_status"]].rename(
        columns={c: f"new_{c}" for c in ["n_valid", "tp", "fp", "tn", "fn", "f1"]}
    )
    merged = old.merge(new, on="pathology", how="outer")
    merged["reason"] = ""
    changed = []
    for _, row in merged.iterrows():
        reasons = []
        old_f1 = safe_float(row.get("old_f1"))
        new_f1 = safe_float(row.get("new_f1"))
        if (math.isnan(old_f1) and not math.isnan(new_f1)) or (
            not math.isnan(old_f1) and not math.isnan(new_f1) and abs(old_f1 - new_f1) > 1e-12
        ):
            reasons.append("F1 policy fixed: zero TP with FP/FN is defined as 0")
        if not reasons and math.isnan(old_f1) and math.isnan(new_f1):
            reasons.append("unchanged undefined")
        elif not reasons:
            reasons.append("unchanged")
        changed.append("; ".join(reasons))
    merged["reason"] = changed
    return merged


def source_paths_for_size(source: Path, size: int, dataset="indiana") -> dict[str, Path]:
    return {
        "subset": source / "subsets" / f"{dataset}_f1_subset_{size}_seed42.csv",
        "predictions": source / "predictions" / f"{dataset}_f1_subset_{size}_predictions.csv",
        "row_map": source / "chexpert_inputs" / f"{dataset}_f1_subset_{size}_row_map.csv",
        "gt_input": source / "chexpert_inputs" / f"{dataset}_f1_subset_{size}_gt_reports.csv",
        "generated_input": source / "chexpert_inputs" / f"{dataset}_f1_subset_{size}_generated_reports.csv",
        "gt_labels": source / "chexpert_labels" / f"{dataset}_f1_subset_{size}_gt_chexpert_labels.csv",
        "generated_labels": source / "chexpert_labels" / f"{dataset}_f1_subset_{size}_generated_chexpert_labels.csv",
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def reevaluate(source: Path, output: Path, sizes: list[int], iterations: int, seed: int, dataset="indiana") -> None:
    source_config = source / 'run_config.json'
    if source_config.is_file():
        recorded_dataset = json.loads(source_config.read_text()).get('dataset')
        if recorded_dataset is not None and recorded_dataset != dataset:
            raise ValueError('Requested dataset does not match inference run_config.json')
    if iterations < 1 or any(size < 1 for size in sizes):
        raise ValueError('Sizes and bootstrap iterations must be positive')
    pathologies = TARGET_PATHOLOGIES if dataset == "indiana" else [p for p in TARGET_PATHOLOGIES if p != "Lung Opacity"]
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    for subdir in ["audit", "canonical_predictions", "metrics", "review", "logs"]:
        (output / subdir).mkdir(parents=True, exist_ok=False)

    run_id = output.name
    write_json(
        output / "run_config.json",
        {
            "run_id": run_id,
            "run_kind": "offline_reevaluation",
            "dataset": dataset,
            "pathologies": pathologies,
            "source": str(source),
            "output": str(output),
            "sizes": sizes,
            "bootstrap_iterations": iterations,
            "seed": seed,
            "start_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )

    all_hashes = {}
    system_rows = []
    status_rows = []

    for size in sizes:
        paths = source_paths_for_size(source, size, dataset)
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing source files for size {size}: {missing}")
        all_hashes[str(size)] = {name: sha256_file(path) for name, path in paths.items()}

        subset = pd.read_csv(paths["subset"], dtype={'image_id': str, 'report_id': str})
        predictions = pd.read_csv(paths["predictions"], dtype={'image_id': str, 'report_id': str})
        row_map = pd.read_csv(paths["row_map"], dtype={'image_id': str, 'report_id': str})
        gt_input = read_headerless_reports(paths["gt_input"])
        gen_input = read_headerless_reports(paths["generated_input"])
        gt_labels = pd.read_csv(paths["gt_labels"])
        gen_labels = pd.read_csv(paths["generated_labels"])

        canonical, conflicts = canonicalize_predictions(predictions, subset)
        aligned, lengths = align_labels(canonical, row_map, gt_input, gen_input, gt_labels, gen_labels, pathologies)
        n_requested = int(len(canonical))
        n_success = int(canonical["canonical_success"].sum())
        metrics, ci, coverage, errors, omissions = compute_metrics(aligned, n_requested, n_success, iterations, seed, pathologies)
        system = summarize_system(canonical, size)
        system_rows.append(system)

        prefix = f"{dataset}_f1_subset_{size}"
        canonical.to_csv(output / "canonical_predictions" / f"{prefix}_canonical_predictions.csv", index=False)
        conflicts.to_csv(output / "audit" / f"{prefix}_dedup_conflicts.csv", index=False)
        metrics.to_csv(output / "metrics" / f"{prefix}_pathology_metrics_v2.csv", index=False)
        write_json(output / 'metrics' / f'{prefix}_summary.json', {
            'dataset': dataset, 'cases_requested': n_requested, 'cases_successful': n_success,
            'pathologies': pathologies,
            'macro_precision': float(metrics['precision'].mean()),
            'macro_recall': float(metrics['recall'].mean()),
            'macro_f1': float(metrics['f1'].mean()),
            'defined_f1_pathologies': int(metrics['f1'].notna().sum()),
            'total_pathologies': len(pathologies),
            'policy': 'Both labels must be 0/1; -1 and missing excluded. Macro means omit undefined values.',
        })
        ci.to_csv(output / "metrics" / f"{prefix}_bootstrap_ci_v2.csv", index=False)
        coverage.to_csv(output / "audit" / f"{prefix}_label_coverage_long.csv", index=False)
        errors.to_csv(output / "review" / f"{prefix}_label_errors_v2.csv", index=False)
        omissions.to_csv(output / "review" / f"{prefix}_omission_queue.csv", index=False)
        compare_old_metrics(source, size, metrics, dataset).to_csv(
            output / "metrics" / f"{prefix}_old_new_metric_diff.csv", index=False
        )
        write_json(
            output / "audit" / f"{prefix}_alignment_checks.json",
            {
                "size": size,
                "source_rows": len(predictions),
                "subset_rows": len(subset),
                "unique_prediction_image_ids": int(predictions["image_id"].nunique()),
                "canonical_rows": len(canonical),
                "canonical_success": n_success,
                "canonical_failed": int((canonical["canonical_status"] == "failed").sum()),
                "canonical_not_attempted": int((canonical["canonical_status"] == "not_attempted").sum()),
                "label_lengths": lengths,
                "alignment": "pass",
            },
        )
        status_rows.append({"phase": "B", "size": size, "status": "PASS", **system})

    pd.DataFrame(system_rows).to_csv(output / "metrics" / "system_metrics_v2.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output / "logs" / "phase_status.csv", index=False)
    write_json(output / "source_hashes.json", all_hashes)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    config["end_utc"] = utc_now()
    config["final_status"] = "PASS_B_OFFLINE"
    write_json(output / "run_config.json", config)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["indiana", "mimic"], default="indiana")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", required=True, type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make-run-id", action="store_true", help="print a UTC UUID run id and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.make_run_id:
        print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        return 0
    reevaluate(args.source, args.output, args.sizes, args.bootstrap_iterations, args.seed, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
