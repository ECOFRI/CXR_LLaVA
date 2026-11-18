"""Synthetic CPU tests of sampler reproducibility and crash-tail recovery."""
import io
import unittest
from research.rrg_f1.cohort import seeded_order, restore_stream


class ResumeTests(unittest.TestCase):
    def test_seeded_eligible_sampling_is_same_after_resume(self):
        candidates = list(range(100))
        eligible = {i for i in candidates if i % 3 != 0}
        order = seeded_order(candidates, 42)
        wanted = [i for i in order if i in eligible][:30]
        first = [i for i in order[:17] if i in eligible]
        resumed = (first + [i for i in order[17:] if i in eligible])[:30]
        self.assertEqual(wanted, resumed)
        self.assertEqual(len(set(wanted)), 30)
        self.assertNotEqual(order, seeded_order(candidates, 43))

    def test_uncheckpointed_tail_is_removed_before_append(self):
        stream = io.BytesIO(b'committed\nuncommitted\n')
        restore_stream(stream, len(b'committed\n'))
        stream.write(b'resumed\n')
        self.assertEqual(stream.getvalue(), b'committed\nresumed\n')

    def test_missing_committed_bytes_stop(self):
        with self.assertRaises(ValueError):
            restore_stream(io.BytesIO(b'x'), 2)


if __name__ == '__main__':
    unittest.main()
