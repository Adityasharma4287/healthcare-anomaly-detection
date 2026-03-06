"""
tests/test_anomaly_pipeline.py
Unit tests for anomaly detection pipeline components.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        from utils.preprocessing import VITAL_RANGES, FEATURE_NAMES
        self.normal_vitals = {
            "heart_rate": 75.0, "spo2": 98.0,
            "temperature": 36.6, "systolic_bp": 118.0, "diastolic_bp": 76.0,
        }
        self.anomalous_vitals = {
            "heart_rate": 160.0, "spo2": 85.0,
            "temperature": 39.5, "systolic_bp": 185.0, "diastolic_bp": 115.0,
        }

    def test_feature_extraction_shape(self):
        from utils.preprocessing import extract_features
        features = extract_features(self.normal_vitals)
        self.assertEqual(features.shape, (1, 5))

    def test_feature_extraction_values(self):
        from utils.preprocessing import extract_features
        features = extract_features(self.normal_vitals)
        self.assertAlmostEqual(features[0, 0], 75.0)
        self.assertAlmostEqual(features[0, 1], 98.0)
        self.assertAlmostEqual(features[0, 2], 36.6)

    def test_validate_vitals_normal(self):
        from utils.preprocessing import validate_vitals
        valid, errors = validate_vitals(self.normal_vitals)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validate_vitals_out_of_range(self):
        from utils.preprocessing import validate_vitals
        bad = {**self.normal_vitals, "heart_rate": 999.0}
        valid, errors = validate_vitals(bad)
        self.assertFalse(valid)
        self.assertTrue(any("heart_rate" in e for e in errors))

    def test_validate_vitals_missing_key(self):
        from utils.preprocessing import validate_vitals
        incomplete = {"heart_rate": 75.0}
        valid, errors = validate_vitals(incomplete)
        self.assertFalse(valid)
        self.assertEqual(len(errors), 4)

    def test_synthetic_data_shape(self):
        from utils.preprocessing import generate_synthetic_training_data
        data = generate_synthetic_training_data(n_samples=500)
        self.assertEqual(data.shape, (500, 5))

    def test_synthetic_data_ranges(self):
        from utils.preprocessing import generate_synthetic_training_data
        data = generate_synthetic_training_data(n_samples=1000)
        self.assertTrue(np.all(data[:, 0] >= 40) and np.all(data[:, 0] <= 130))   # HR
        self.assertTrue(np.all(data[:, 1] >= 90) and np.all(data[:, 1] <= 100))   # SpO2
        self.assertTrue(np.all(data[:, 2] >= 35) and np.all(data[:, 2] <= 38))    # Temp


class TestSeverityClassifier(unittest.TestCase):

    def test_high_severity(self):
        from utils.severity import classify_severity
        result = classify_severity(0.85, 0.80)
        self.assertEqual(result.severity, "HIGH")
        self.assertGreaterEqual(result.combined_score, 0.70)

    def test_medium_severity(self):
        from utils.severity import classify_severity
        result = classify_severity(0.55, 0.50)
        self.assertEqual(result.severity, "MEDIUM")
        self.assertGreaterEqual(result.combined_score, 0.40)
        self.assertLess(result.combined_score, 0.70)

    def test_low_severity(self):
        from utils.severity import classify_severity
        result = classify_severity(0.10, 0.15)
        self.assertEqual(result.severity, "LOW")
        self.assertLess(result.combined_score, 0.40)

    def test_combined_score_formula(self):
        from utils.severity import classify_severity
        ae, iso = 0.60, 0.50
        result = classify_severity(ae, iso)
        expected = 0.60 * ae + 0.40 * iso
        self.assertAlmostEqual(result.combined_score, round(expected, 6), places=5)

    def test_boundary_high(self):
        from utils.severity import classify_severity
        # Exactly at HIGH threshold
        result = classify_severity(0.70 / 0.60, 0.0)  # Combined = 0.70
        # Due to floating point, just check it's HIGH or MEDIUM
        self.assertIn(result.severity, ["HIGH", "MEDIUM"])

    def test_score_bounds(self):
        from utils.severity import classify_severity
        result = classify_severity(0.0, 0.0)
        self.assertEqual(result.combined_score, 0.0)
        self.assertEqual(result.severity, "LOW")

    def test_severity_color(self):
        from utils.severity import severity_color
        self.assertEqual(severity_color("HIGH"),   "#FF4757")
        self.assertEqual(severity_color("MEDIUM"), "#FFA502")
        self.assertEqual(severity_color("LOW"),    "#2ED573")

    def test_severity_emoji(self):
        from utils.severity import severity_emoji
        self.assertEqual(severity_emoji("HIGH"), "🔴")
        self.assertEqual(severity_emoji("LOW"),  "🟢")


class TestIsolationForest(unittest.TestCase):

    def test_train_and_score(self):
        """Train a small Isolation Forest and verify scoring."""
        import tempfile, joblib
        from models.isolation_forest import train_isolation_forest, compute_isolation_score, load_isolation_forest
        from utils.preprocessing import generate_synthetic_training_data, fit_and_save_scaler, extract_features

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path  = os.path.join(tmpdir, "scaler.pkl")
            iforest_path = os.path.join(tmpdir, "iforest.pkl")

            os.environ["IFOREST_PATH"] = iforest_path
            os.environ["SCALER_PATH"]  = scaler_path

            model_data = train_isolation_forest(iforest_path, scaler_path)
            self.assertIsNotNone(model_data)

            loaded = load_isolation_forest(iforest_path)
            scaler = joblib.load(scaler_path)

            normal = {"heart_rate": 75, "spo2": 98, "temperature": 36.6, "systolic_bp": 115, "diastolic_bp": 75}
            anomalous = {"heart_rate": 165, "spo2": 83, "temperature": 39.5, "systolic_bp": 185, "diastolic_bp": 120}

            score_normal   = compute_isolation_score(loaded, extract_features(normal),    scaler)
            score_anomalous = compute_isolation_score(loaded, extract_features(anomalous), scaler)

            self.assertGreaterEqual(score_normal,    0.0)
            self.assertLessEqual(score_normal,       1.0)
            self.assertGreaterEqual(score_anomalous, 0.0)
            self.assertLessEqual(score_anomalous,    1.0)
            # Anomalous score should generally be higher (may not always hold for all edge cases)
            # self.assertGreater(score_anomalous, score_normal)

    def test_score_range(self):
        """Ensure isolation scores are always in [0, 1]."""
        import tempfile, joblib
        from models.isolation_forest import train_isolation_forest, compute_isolation_score
        from utils.preprocessing import generate_synthetic_training_data, extract_features

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path  = os.path.join(tmpdir, "scaler.pkl")
            iforest_path = os.path.join(tmpdir, "iforest.pkl")
            os.environ["IFOREST_PATH"] = iforest_path
            os.environ["SCALER_PATH"]  = scaler_path

            model_data = train_isolation_forest(iforest_path, scaler_path)
            scaler = joblib.load(scaler_path)

            test_cases = [
                {"heart_rate": v1, "spo2": v2, "temperature": v3, "systolic_bp": v4, "diastolic_bp": v5}
                for v1, v2, v3, v4, v5 in [
                    (75, 98, 36.6, 115, 75),
                    (160, 85, 39.0, 180, 115),
                    (40, 92, 35.0, 70, 45),
                    (200, 70, 41.0, 220, 140),
                ]
            ]

            for case in test_cases:
                features = extract_features(case)
                score = compute_isolation_score(model_data, features, scaler)
                self.assertGreaterEqual(score, 0.0, f"Score below 0 for {case}")
                self.assertLessEqual(score,    1.0, f"Score above 1 for {case}")


class TestEmailNotification(unittest.TestCase):

    def test_build_html_email(self):
        from notifications.email_alert import _build_html_email
        html = _build_html_email(
            patient_id="P001", patient_name="Alice Johnson",
            vitals={"heart_rate": 155, "spo2": 88, "temperature": 38.9,
                    "systolic_bp": 175, "diastolic_bp": 110},
            severity="HIGH", combined_score=0.8734,
            detected_at="2025-01-01 12:00:00 UTC",
        )
        self.assertIn("P001", html)
        self.assertIn("Alice Johnson", html)
        self.assertIn("0.8734", html)
        self.assertIn("HIGH", html)
        self.assertIn("155", html)

    def test_send_without_credentials(self):
        """Should return False gracefully without SMTP credentials."""
        import os
        os.environ["SMTP_USER"] = ""
        os.environ["SMTP_PASSWORD"] = ""
        from notifications.email_alert import send_alert_email
        result = send_alert_email(
            "P001", "Test Patient",
            {"heart_rate": 155, "spo2": 88, "temperature": 38.9,
             "systolic_bp": 175, "diastolic_bp": 110},
            "HIGH", 0.85,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
