"""CPU-only tests for run discovery; no model imports."""

import json
import tempfile
import unittest
from pathlib import Path

from research.project import catalog, main


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_catalog_nested_stages_and_bad_json(self):
        run = self.root / "research/runs/run1"
        (run / "E5").mkdir(parents=True)
        (run / "bundle_manifest.json").write_text(json.dumps({"status": "COMPLETE", "decision": "HOLD"}))
        (run / "E5/run_manifest.json").write_text(json.dumps({"status": "BLOCKED"}))
        (run / "environment_manifest.json").write_text("{}")
        empty = run.parent / "empty"
        empty.mkdir()
        bad = run.parent / "bad"
        bad.mkdir()
        (bad / "run_manifest.json").write_text("bad json")
        rows = catalog(self.root)
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["status"] for r in rows], ["INVALID_MANIFEST", "NO_RUN_MANIFEST", "COMPLETE", "BLOCKED"])
        self.assertEqual(rows[2]["decision"], "HOLD")
        self.assertEqual(rows[3]["component"], "E5")

    def test_status_output_refuses_overwrite(self):
        (self.root / "research/runs").mkdir(parents=True)
        output = self.root / "existing.md"
        output.write_text("original")
        with self.assertRaises(FileExistsError):
            main(["--root", str(self.root), "status", "--output", str(output)])
        self.assertEqual(output.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
