"""Synthetic-only tests for the metadata split audit (no MONAI or real data)."""
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from split_audit import audit_manifest, SplitAuditError


class SplitAuditTests(unittest.TestCase):
    def run_case(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text("case_id,patient_id,split\n" + "\n".join(rows) + "\n", encoding="utf-8")
            return audit_manifest(path)

    def test_disjoint_patient_groups(self):
        result = self.run_case([
            "case01,patientA,train",
            "case02,patientA,train",
            "case03,patientB,val",
            "case04,patientC,test",
        ])
        self.assertEqual(result["case_count"], 4)
        self.assertEqual(result["group_count"], 3)
        self.assertEqual(result["cases_by_split"]["train"], 2)
        self.assertEqual(result["status"], "structural_group_integrity_passed")

    def test_same_patient_across_splits_fails(self):
        with self.assertRaisesRegex(SplitAuditError, "overlap"):
            self.run_case(["case01,p1,train", "case02,p1,val"])

    def test_duplicate_case_fails(self):
        with self.assertRaisesRegex(SplitAuditError, "Duplicate"):
            self.run_case(["case01,p1,train", "case01,p2,val"])

    def test_missing_patient_id_fails(self):
        with self.assertRaisesRegex(SplitAuditError, "Missing"):
            self.run_case(["case01,,train", "case02,p2,val"])

    def test_unexpected_split_fails(self):
        with self.assertRaisesRegex(SplitAuditError, "Invalid"):
            self.run_case(["case01,p1,training", "case02,p2,val"])

    def test_requires_validation_and_training(self):
        with self.assertRaisesRegex(SplitAuditError, "train and one validation"):
            self.run_case(["case01,p1,train"])

    def test_empty_manifest_fails(self):
        with self.assertRaisesRegex(SplitAuditError, "empty"):
            self.run_case([])


if __name__ == "__main__":
    unittest.main()
