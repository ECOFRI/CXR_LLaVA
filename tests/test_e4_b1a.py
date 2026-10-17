"""Small synthetic checks for the CPU-only E4-B1a post-hoc definitions."""

import unittest

from research.e4_b1_analyze import label_state
from research.e4_b1a_posthoc import (
    change_category,
    collapse,
    direction_state,
    mention,
    oracle_shuffled_discordant,
    verify_inputs,
)


class E4B1aTests(unittest.TestCase):
    def test_source_hashes_match_before_analysis(self):
        manifest, digest, checked = verify_inputs()
        self.assertEqual(manifest["status"], "E4_B1_PROMPT_ROUTE_REJECTED")
        self.assertEqual(len(digest), 64)
        self.assertIn("E4_B1_reports.jsonl", checked)
        self.assertIn("E4_B1_evidence_blocks.jsonl", checked)
        self.assertTrue(all(x["expected_sha256"] == x["actual_sha256"] for x in checked.values()))

    def test_oracle_discordance_requires_definite_reference(self):
        self.assertTrue(oracle_shuffled_discordant("Positive", "Negative"))
        self.assertTrue(oracle_shuffled_discordant("Negative", "Positive"))
        self.assertFalse(oracle_shuffled_discordant("Positive", "Positive"))
        self.assertFalse(oracle_shuffled_discordant("Uncertain", "Positive"))
        self.assertFalse(oracle_shuffled_discordant("Not mentioned", "Negative"))

    def test_direction_categories_do_not_merge_partial_changes(self):
        self.assertEqual(change_category("Positive", "Negative", "Positive", "Negative", False), "follows_evidence")
        self.assertEqual(change_category("Negative", "Positive", "Positive", "Negative", False), "opposes_evidence")
        self.assertEqual(change_category("Not mentioned", "Positive", "Positive", "Negative", False), "other_target_state_change")
        self.assertEqual(change_category("Uncertain", "Uncertain", "Positive", "Negative", False), "text_changed_target_state_same")
        self.assertEqual(change_category("Not mentioned", "Not mentioned", "Positive", "Negative", True), "unchanged_text_and_target_state")
        with self.assertRaises(ValueError):
            change_category("Positive", "Negative", "Positive", "Positive", False)

    def test_missing_and_uncertain_are_not_negative(self):
        self.assertEqual(label_state(""), "Not mentioned")
        self.assertEqual(label_state("-1.0"), "Uncertain")
        self.assertEqual(label_state(None), "Extraction failure")
        self.assertEqual(direction_state("favors absent"), "Negative")
        self.assertFalse(mention("Not mentioned"))
        self.assertFalse(mention("Extraction failure"))
        self.assertTrue(mention("Uncertain"))

    def test_collapse_uses_original_definition(self):
        self.assertTrue(collapse("No acute cardiopulmonary process. "))
        self.assertTrue(collapse("short"))
        self.assertFalse(collapse("A longer synthetic report that is well above sixty characters and discusses the image."))


if __name__ == "__main__":
    unittest.main()
