from __future__ import annotations

import argparse
import csv
from pathlib import Path

from e4_adapter import AdapterError, distinct_records, normalize_label, parse_findings, read_csv, write_csv


def convert_wide(wide_path, mapping_path, findings_path, kind, key_column="report_key"):
    if kind not in ("reference", "prediction"):
        raise AdapterError(f"Invalid kind: {kind}")
    findings = parse_findings(findings_path)
    keys = (key_column, "image_id", "arm_id") if kind == "prediction" else (key_column, "image_id")
    mapping_rows = read_csv(mapping_path, keys)
    mapping = distinct_records(mapping_rows, (key_column,), "labeler source mapping")
    wide_rows = read_csv(wide_path, (key_column, *findings))
    wide = distinct_records(wide_rows, (key_column,), "labeler output")
    if set(mapping) != set(wide):
        raise AdapterError(
            f"Labeler output/mapping mismatch: missing={len(set(mapping) - set(wide))} "
            f"extra={len(set(wide) - set(mapping))}"
        )

    result = []
    for source_key in sorted(mapping):
        row = wide[source_key]
        mapped = mapping[source_key]
        for finding in findings:
            label = normalize_label(row[finding])
            if kind == "reference":
                result.append({
                    "image_id": mapped["image_id"].strip(),
                    "finding": finding,
                    "reference_label": label,
                })
            else:
                result.append({
                    "image_id": mapped["image_id"].strip(),
                    "arm_id": mapped["arm_id"].strip(),
                    "finding": finding,
                    "prediction_label": label,
                })
    identity_columns = ("image_id", "arm_id", "finding") if kind == "prediction" else ("image_id", "finding")
    distinct_records(result, identity_columns, "converted labeler output")
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert pinned labeler wide outputs using an explicit ID mapping")
    parser.add_argument("--wide-csv", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--findings", required=True)
    parser.add_argument("--kind", choices=("reference", "prediction"), required=True)
    parser.add_argument("--key-column", default="report_key")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        rows = convert_wide(args.wide_csv, args.mapping_csv, args.findings, args.kind, args.key_column)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, rows)
        print(f"PASS_LABELER_BRIDGE: {len(rows)} records -> {output}")
    except (AdapterError, OSError) as error:
        parser.exit(2, f"BLOCKED: {error}\n")


if __name__ == "__main__":
    main()
