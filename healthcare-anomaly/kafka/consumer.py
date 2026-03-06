"""
kafka/consumer.py
Kafka consumer that processes incoming patient vital signs, runs
anomaly detection models (Autoencoder + Isolation Forest), classifies
severity, stores results in PostgreSQL, and triggers email alerts.
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(levelname)s %(message)s"
)

KAFKA_BROKER   = os.getenv("KAFKA_BROKER",   "localhost:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC",    "patient-vitals")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "anomaly-detector-group")
MODEL_PATH     = os.getenv("MODEL_PATH",     "./models/autoencoder_model.h5")
IFOREST_PATH   = os.getenv("IFOREST_PATH",   "./models/isolation_forest.pkl")
SCALER_PATH    = os.getenv("SCALER_PATH",    "./models/scaler.pkl")


class AnomalyPipeline:
    """Loads models and runs the full anomaly detection pipeline."""

    def __init__(self):
        self.autoencoder   = None
        self.iforest_data  = None
        self.scaler        = None
        self._load_models()

    def _load_models(self):
        """Load all ML models and scaler."""
        from utils.preprocessing import load_scaler

        # Load scaler
        try:
            self.scaler = load_scaler(SCALER_PATH)
            logger.info("Scaler loaded successfully.")
        except FileNotFoundError:
            logger.warning("Scaler not found. Using mock scoring.")

        # Load autoencoder
        try:
            from models.autoencoder import load_autoencoder
            self.autoencoder = load_autoencoder(MODEL_PATH)
            logger.info("Autoencoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Autoencoder not loaded ({e}). Using mock scoring.")

        # Load Isolation Forest
        try:
            from models.isolation_forest import load_isolation_forest
            self.iforest_data = load_isolation_forest(IFOREST_PATH)
            logger.info("Isolation Forest loaded successfully.")
        except Exception as e:
            logger.warning(f"Isolation Forest not loaded ({e}). Using mock scoring.")

    def process(self, vitals: dict) -> dict:
        """
        Run the full anomaly detection pipeline on a vitals record.
        Returns enriched result with scores and severity.
        """
        import numpy as np
        from utils.preprocessing import extract_features
        from utils.severity import classify_severity

        features = extract_features(vitals)

        # Compute scores
        if self.autoencoder and self.scaler:
            from models.autoencoder import compute_autoencoder_score
            ae_score = compute_autoencoder_score(self.autoencoder, self.scaler, features)
        else:
            import random
            ae_score = round(random.betavariate(1.5, 8), 6)

        if self.iforest_data and self.scaler:
            from models.isolation_forest import compute_isolation_score
            if_score = compute_isolation_score(self.iforest_data, features, self.scaler)
        else:
            import random
            if_score = round(random.betavariate(1.5, 8), 6)

        result = classify_severity(ae_score, if_score)

        return {
            **vitals,
            "autoencoder_score": ae_score,
            "isolation_score":   if_score,
            "combined_score":    result.combined_score,
            "severity":          result.severity,
            "description":       result.description,
            "action_required":   result.action_required,
            "processed_at":      datetime.utcnow().isoformat(),
        }


class VitalsConsumer:
    """Kafka consumer that feeds vitals through the anomaly detection pipeline."""

    def __init__(self):
        self.pipeline = AnomalyPipeline()
        self.consumer = self._create_consumer()
        self.processed_count = 0
        self.alert_count = 0

    def _create_consumer(self):
        if not KAFKA_AVAILABLE:
            logger.warning("kafka-python not installed. Running in mock mode.")
            return None
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                consumer_timeout_ms=1000,
            )
            logger.info(f"Kafka consumer connected — topic: {KAFKA_TOPIC}, group: {KAFKA_GROUP_ID}")
            return consumer
        except NoBrokersAvailable:
            logger.error(f"Cannot connect to Kafka at {KAFKA_BROKER}. Running in mock mode.")
            return None

    def _store_result(self, vitals: dict, result: dict):
        """Persist vitals and anomaly scores to PostgreSQL."""
        try:
            from database.db_connection import get_db, insert_patient_vitals, insert_anomaly_score, insert_alert_log

            with get_db() as db:
                vital_id = insert_patient_vitals(db, {
                    "patient_id":   result["patient_id"],
                    "patient_name": result.get("patient_name", "Unknown"),
                    "heart_rate":   result["heart_rate"],
                    "spo2":         result["spo2"],
                    "temperature":  result["temperature"],
                    "systolic_bp":  result["systolic_bp"],
                    "diastolic_bp": result["diastolic_bp"],
                })

                anomaly_id = insert_anomaly_score(db, {
                    "vital_id":          vital_id,
                    "patient_id":        result["patient_id"],
                    "autoencoder_score": result["autoencoder_score"],
                    "isolation_score":   result["isolation_score"],
                    "combined_score":    result["combined_score"],
                    "severity":          result["severity"],
                })

                if result["severity"] == "HIGH":
                    email_sent = self._send_alert(result)
                    insert_alert_log(db, {
                        "anomaly_id":     anomaly_id,
                        "patient_id":     result["patient_id"],
                        "patient_name":   result.get("patient_name", "Unknown"),
                        "severity":       result["severity"],
                        "combined_score": result["combined_score"],
                        "message":        result["description"],
                        "email_sent":     email_sent,
                        "email_recipient": os.getenv("ALERT_RECIPIENT", ""),
                    })
                    self.alert_count += 1

        except Exception as e:
            logger.error(f"Failed to store result: {e}")

    def _send_alert(self, result: dict) -> bool:
        """Send email alert for HIGH severity anomalies."""
        try:
            from notifications.email_alert import send_alert_email
            return send_alert_email(
                patient_id=result["patient_id"],
                patient_name=result.get("patient_name", "Unknown"),
                vitals={
                    "heart_rate":   result["heart_rate"],
                    "spo2":         result["spo2"],
                    "temperature":  result["temperature"],
                    "systolic_bp":  result["systolic_bp"],
                    "diastolic_bp": result["diastolic_bp"],
                },
                severity=result["severity"],
                combined_score=result["combined_score"],
            )
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    def _log_result(self, result: dict):
        """Log processed result to console."""
        severity = result["severity"]
        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
        logger.info(
            f"{emoji} [{severity}] Patient {result['patient_id']} | "
            f"Score={result['combined_score']:.4f} | "
            f"HR={result['heart_rate']} SpO2={result['spo2']} "
            f"Temp={result['temperature']} BP={result['systolic_bp']}/{result['diastolic_bp']}"
        )

    def run(self):
        """Main consumer loop."""
        logger.info("Starting anomaly detection consumer...")

        if self.consumer is None:
            logger.info("Running in mock mode (no Kafka). Generating synthetic events...")
            self._run_mock_mode()
            return

        try:
            while True:
                for message in self.consumer:
                    try:
                        vitals = message.value
                        result = self.pipeline.process(vitals)
                        self._log_result(result)
                        self._store_result(vitals, result)
                        self.processed_count += 1

                        if self.processed_count % 100 == 0:
                            logger.info(
                                f"Stats: processed={self.processed_count}, "
                                f"alerts={self.alert_count}"
                            )
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user.")
        finally:
            self.consumer.close()
            logger.info(
                f"Consumer closed. Total processed: {self.processed_count}, "
                f"Alerts sent: {self.alert_count}"
            )

    def _run_mock_mode(self):
        """Generate and process synthetic vitals in mock mode (no Kafka)."""
        import random
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kafka.producer import PATIENTS, generate_vitals

        logger.info("Mock mode: press Ctrl+C to stop.")
        try:
            while True:
                patient = random.choice(PATIENTS)
                vitals = generate_vitals(patient)
                result = self.pipeline.process(vitals)
                self._log_result(result)
                self._store_result(vitals, result)
                self.processed_count += 1
                time.sleep(1.5)
        except KeyboardInterrupt:
            logger.info(f"Mock mode stopped. Processed: {self.processed_count}")


if __name__ == "__main__":
    consumer = VitalsConsumer()
    consumer.run()
