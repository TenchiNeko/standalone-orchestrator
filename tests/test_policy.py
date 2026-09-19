import tempfile
import unittest
from pathlib import Path

from orchestrator_v2.jev import JevAdapter
from orchestrator_v2.state import StateStore


class PolicyTests(unittest.TestCase):
    def test_jev_off_is_deterministic_and_never_authorizes(self):
        result = JevAdapter("off").classify_failure("tests failed")
        self.assertIsNone(result.answer)
        self.assertEqual(result.mode, "off")

    def test_unknown_mutation_is_recorded_not_repeated(self):
        with tempfile.TemporaryDirectory() as d:
            store = StateStore(Path(d) / "state.sqlite3")
            store.mutation_intent("t", "write_file", "k1")
            store.mutation_ack("t", "k1", "unknown")
            events = store.events("t")
            self.assertEqual(events[-1]["kind"], "mutation_result")
            self.assertIn('"unknown"', events[-1]["payload"])


if __name__ == "__main__": unittest.main()
