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
        self.assertEqual(report["query_seed_units"], 6000)
        self.assertEqual(report["candidate_distribution_rows"], 12)
        self.assertEqual(report["corrected_throughput_rows"], 180)
        self.assertEqual(report["latency_boundary_rows"], 180)
        self.assertEqual(report["ratio_rows"], 60)
        self.assertEqual(report["unique_query_units"], 1335)
        self.assertAlmostEqual(report["nfcorpus_cutoff20_mean_actual_candidate_count"], 15.308, places=12)
        self.assertEqual(report["nfcorpus_cutoff20_zero_candidate_count"], 39)
        self.assertAlmostEqual(report["int8_fp32_end_to_end_ratio_mean"], 0.787322079555077, places=12)
        self.assertEqual(report["main_latency_ratio_field"], "int8_end_to_end_latency_ratio_mean")
        self.assertEqual(report["aggregate_file"], "formal_aggregate_metrics_v2.csv")
        self.assertAlmostEqual(report["legacy_reranker_only_ratio_mean"], 0.7773979139667812, places=12)
        self.assertEqual(len(report["datasets"]), 4)
        self.assertEqual(len(report["methods"]), 3)


if __name__ == "__main__":
    unittest.main()
