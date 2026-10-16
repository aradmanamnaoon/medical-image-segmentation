"""Validate a local case/patient split manifest without sharing patient metadata.

Manifest CSV headers: case_id, patient_id, split.
Accepted split names: train, val, test.

No original dataset, patient records, hashes, row-level identifiers, or model
weights are downloaded, uploaded, or logged by this tool.

IMPORTANT: passing this audit checks only grouping consistency in the supplied
manifest. It does not prove patient IDs are correctly mapped, case originality,
dataset rights, clinical generalization, or absence of near-duplicate leakage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


class SplitAuditError(ValueError):
    """Split metadata failed a declared structural integrity rule."""


REQUIRED = {"case_id", "patient_id", "split"}
ALLOWED_SPLITS = {"train", "val", "test"}


def audit_manifest(path: Path) -> dict[str, object]:
    """Check supplied case/patient/split metadata, returning counts only."""
    seen_cases: set[str] = set()
    patients: dict[str, str] = {}
    split_cases: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = {
        "train": set(), "val": set(), "test": set()
    }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise SplitAuditError(
                "Manifest must contain case_id, patient_id and split columns"
            )

        for line, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            split = (row.get("split") or "").strip().lower()
            if not case_id or not patient_id:
                raise SplitAuditError(f"Missing case or patient identifier on row {line}")
            if split not in ALLOWED_SPLITS:
                raise SplitAuditError(f"Invalid split on row {line}")
            if case_id in seen_cases:
                raise SplitAuditError(f"Duplicate case identifier on row {line}")

            assigned_split = patients.get(patient_id)
            if assigned_split is not None and assigned_split != split:
                raise SplitAuditError(
                    f"Patient/group overlap between splits detected on row {line}"
                )

            seen_cases.add(case_id)
            patients[patient_id] = split
            split_cases[split] += 1
            split_patients[split].add(patient_id)

    if not seen_cases:
        raise SplitAuditError("Manifest is empty")
    if not split_cases["train"] or not split_cases["val"]:
        raise SplitAuditError("At least one train and one validation case are required")

    return {
        "status": "structural_group_integrity_passed",
        "case_count": len(seen_cases),
        "group_count": len(patients),
        "cases_by_split": {key: split_cases[key] for key in ("train", "val", "test")},
        "groups_by_split": {
            key: len(split_patients[key]) for key in ("train", "val", "test")
        },
        "scope": "Supplied metadata only; does not verify source patient identities or numerical metrics",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_manifest(args.manifest)
    except (SplitAuditError, OSError, csv.Error) as error:
        # Avoid logging any patient/case identifier or sensitive file contents.
        # Error messages emitted above contain only row numbers.
        parser.exit(2, f"Manifest integrity check failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
