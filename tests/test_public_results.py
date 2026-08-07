import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_public_results import build_report


class PublicResultsTest(unittest.TestCase):
    def test_public_results_pass(self):
        report = build_report()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["aggregate_rows"], 180)
        self.assertEqual(report["query_rows"], 18000)
        self.assertEqual(report["unique_query_units"], 1335)
        self.assertEqual(len(report["datasets"]), 4)
        self.assertEqual(len(report["methods"]), 3)


if __name__ == "__main__":
    unittest.main()
