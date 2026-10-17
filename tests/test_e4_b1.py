"""Synthetic-only E4-B1 protocol invariants."""

import unittest

from research.e4_b1_cohort import FINDINGS, select
from research.e4_b1_prepare import ARMS, FIVE, SIX, block, chat_for, derangement, oracle_line
from research.e4_b1_gpu import output_stop_reason
from research.e4_b1_analyze import gate, label_state


class E4B1Tests(unittest.TestCase):
    def test_derangement(self):
        ids = [f"S{i:02d}" for i in range(24)]
        a = derangement(ids)
        self.assertEqual(a, derangement(ids))
        self.assertEqual(set(a), set(ids))
        self.assertEqual(set(a.values()), set(ids))
        self.assertTrue(all(k != v for k, v in a.items()))

    def test_oracle_mapping(self):
        values = [oracle_line("Edema", x) for x in ("Positive", "Negative", "Uncertain", "Not mentioned")]
        self.assertEqual(len(set(values)), 4)
        self.assertIn("uncertain", values[2])
        self.assertIn("does not mention", values[3])

    def test_templates_keep_original_chat(self):
        item = {f: "Not mentioned" for f in SIX}
        scores = {f: {"positive_cosine_similarity": "0.2", "negative_cosine_similarity": "0.1",
                      "paired_softmax_score": "0.51", "direction": "favors present"} for f in FIVE}
        for arm in ARMS:
            content = block(arm, item, scores, scores)
            chat = chat_for(arm, content)
            self.assertEqual(chat[-1]["content"].count("<image>"), 1)
            if arm == "C0":
                self.assertNotIn("<context>", chat[-1]["content"])
            else:
                self.assertIn("<context>", chat[-1]["content"])
        self.assertIn("finding | positive cosine", block("F1-M", item, scores, scores))
        self.assertIn("reference-report state", block("F1-O", item, scores, scores))
        self.assertIn("State your own assessment", block("F3-M", item, scores, scores))
        self.assertNotIn("cardiomegaly", block("N1", item, scores, scores).lower())

    def test_cohort_enrichment_synthetic(self):
        candidates = []
        for i in range(30):
            row = {"image_id": f"S{i:02d}"}
            row.update({f: "Positive" if i < 4 else "Negative" for f in FINDINGS})
            candidates.append(row)
        chosen, support = select(candidates)
        self.assertEqual(len(chosen), 24)
        self.assertTrue(all(support[f] >= 4 for f in FINDINGS))
        self.assertEqual(sum(x["selection_reason"] == "no_positive_among_six" for x in chosen), 20)

    def test_stop_reason(self):
        self.assertEqual(output_stop_reason([2], 2, False, 512), "EOS")
        self.assertEqual(output_stop_reason([10], 2, True, 512), "STOP_KEYWORD")
        self.assertEqual(output_stop_reason([10] * 512, 2, False, 512), "MAX_TOKENS")
        self.assertEqual(output_stop_reason([10], 2, False, 512), "UNDETERMINED")

    def test_label_states_and_denominator(self):
        self.assertEqual([label_state(x) for x in ("1.0", "0.0", "-1.0", "", None)],
                         ["Positive", "Negative", "Uncertain", "Not mentioned", "Extraction failure"])
        cohort = {f"S{i:02d}": {f: "Uncertain" if i < 4 else "Negative" for f in SIX} for i in range(24)}
        records = [{"image_id": i, "arm": a, "generated_report": "A sufficiently long synthetic sentence for this dry run."}
                   for i in cohort for a in ARMS]
        states = {(i, a, f): "Negative" for i in cohort for a in ARMS for f in SIX}
        _, fmt, _ = gate(records, states, cohort, {i: 1 for i in cohort})
        self.assertEqual([r["oracle_eligible_cells"] for r in fmt], [120, 120, 120])
