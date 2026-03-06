"""
kafka/producer.py
Kafka producer that simulates real-time patient vital signs streaming.
Generates realistic vital sign data for multiple patients and publishes
to the 'patient-vitals' Kafka topic.
"""

import os
import sys
import json
import time
import random
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [PRODUCER] %(message)s")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.getenv("KAFKA_TOPIC",  "patient-vitals")

# ─── Simulated Patient Profiles ───────────────────────────────────────────────
PATIENTS = [
    {"id": "P001", "name": "Alice Johnson",   "age": 45, "condition": "normal"},
    {"id": "P002", "name": "Bob Martinez",    "age": 62, "condition": "hypertension"},
    {"id": "P003", "name": "Carol Williams",  "age": 38, "condition": "normal"},
    {"id": "P004", "name": "David Chen",      "age": 71, "condition": "cardiac"},
    {"id": "P005", "name": "Emma Thompson",   "age": 55, "condition": "respiratory"},
    {"id": "P006", "name": "Frank Nguyen",    "age": 48, "condition": "normal"},
    {"id": "P007", "name": "Grace Lee",       "age": 29, "condition": "normal"},
    {"id": "P008", "name": "Henry Davis",     "age": 66, "condition": "diabetic"},
]

# Anomaly injection probability per patient condition
ANOMALY_RATES = {
    "normal":      0.03,
    "hypertension": 0.12,
    "cardiac":     0.15,
    "respiratory": 0.12,
    "diabetic":    0.08,
}


def generate_vitals(patient: dict, force_anomaly: bool = False) -> dict:
    """Generate realistic vital signs for a patient."""
    anomaly_rate = ANOMALY_RATES.get(patient["condition"], 0.05)
    is_anomaly = force_anomaly or random.random() < anomaly_rate

    if is_anomaly:
        # Inject anomalous values with condition-specific patterns
        condition = patient["condition"]
        if condition == "cardiac":
            heart_rate  = random.uniform(130, 175)
            spo2        = random.uniform(88, 94)
            systolic    = random.uniform(155, 200)
            diastolic   = random.uniform(95, 130)
            temperature = random.uniform(36.0, 37.8)
        elif condition == "hypertension":
            heart_rate  = random.uniform(85, 115)
            spo2        = random.uniform(94, 98)
            systolic    = random.uniform(160, 210)
            diastolic   = random.uniform(100, 135)
            temperature = random.uniform(36.2, 37.5)
        elif condition == "respiratory":
            heart_rate  = random.uniform(95, 125)
            spo2        = random.uniform(84, 93)
            systolic    = random.uniform(100, 135)
            diastolic   = random.uniform(65, 88)
            temperature = random.uniform(37.5, 39.5)
        else:
            # Generic anomaly
            heart_rate  = random.choice([random.uniform(35, 50), random.uniform(140, 185)])
            spo2        = random.uniform(82, 93)
            temperature = random.choice([random.uniform(34.0, 35.5), random.uniform(38.5, 41.0)])
            systolic    = random.choice([random.uniform(70, 85), random.uniform(155, 205)])
            diastolic   = random.choice([random.uniform(45, 58), random.uniform(100, 130)])
    else:
        # Normal physiological values with small random variation
        heart_rate  = random.gauss(75,  8)  + (5  if patient["condition"] == "cardiac"      else 0)
        spo2        = random.gauss(98,  0.6)
        temperature = random.gauss(36.6, 0.25)
        systolic    = random.gauss(115, 7)  + (15 if patient["condition"] == "hypertension" else 0)
        diastolic   = random.gauss(75,  5)  + (10 if patient["condition"] == "hypertension" else 0)

    return {
        "event_id":    str(uuid.uuid4()),
        "patient_id":  patient["id"],
        "patient_name": patient["name"],
        "heart_rate":   round(max(30, min(220, heart_rate)), 1),
        "spo2":         round(max(70, min(100, spo2)), 1),
        "temperature":  round(max(34, min(42, temperature)), 2),
        "systolic_bp":  round(max(60, min(220, systolic)), 1),
        "diastolic_bp": round(max(40, min(140, diastolic)), 1),
        "timestamp":    datetime.utcnow().isoformat(),
        "is_anomaly":   is_anomaly,
    }


def create_producer() -> "KafkaProducer | None":
    """Create and return a KafkaProducer instance."""
    if not KAFKA_AVAILABLE:
        logger.warning("kafka-python not installed. Running in mock mode.")
        return None
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            linger_ms=5,
        )
        logger.info(f"Kafka producer connected to {KAFKA_BROKER}")
        return producer
    except NoBrokersAvailable:
        logger.error(f"Cannot connect to Kafka at {KAFKA_BROKER}. Running in mock mode.")
        return None


def run_producer(interval_seconds: float = 2.0, burst_mode: bool = False):
    """
    Continuously produce patient vital sign events to Kafka.

    Args:
        interval_seconds: Time between producing events for each patient.
        burst_mode: If True, send many events quickly for testing.
    """
    producer = create_producer()
    logger.info(f"Starting vitals producer — topic: {KAFKA_TOPIC}, interval: {interval_seconds}s")
    logger.info(f"Monitoring {len(PATIENTS)} patients")

    event_count = 0
    try:
        while True:
            for patient in PATIENTS:
                vitals = generate_vitals(patient)
                message_key = patient["id"]

                if producer:
                    future = producer.send(
                        KAFKA_TOPIC,
                        key=message_key,
                        value=vitals,
                    )
                    try:
                        metadata = future.get(timeout=5)
                        logger.debug(
                            f"Sent: {patient['id']} | HR={vitals['heart_rate']} "
                            f"SpO2={vitals['spo2']} | partition={metadata.partition}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to send message: {e}")
                else:
                    # Mock mode: just log the vitals
                    logger.info(f"[MOCK] Patient {patient['id']}: {vitals}")

                event_count += 1

            if producer:
                producer.flush()

            logger.info(f"Batch sent — total events: {event_count}")

            if burst_mode:
                time.sleep(0.1)
            else:
                time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Producer stopped by user.")
    finally:
        if producer:
            producer.close()
            logger.info("Kafka producer closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Patient Vitals Kafka Producer")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between producing events (default: 2.0)")
    parser.add_argument("--burst", action="store_true",
                        help="Send events rapidly for load testing")
    args = parser.parse_args()
    run_producer(interval_seconds=args.interval, burst_mode=args.burst)
