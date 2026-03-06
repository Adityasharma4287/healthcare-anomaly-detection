#!/usr/bin/env python3
"""
run.py
Convenience startup script for the Healthcare Anomaly Detection System.
Launches Flask API server (and optionally Kafka consumer in a thread).
"""

import os
import sys
import threading
import logging
import argparse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("run")


def start_consumer():
    """Start Kafka consumer in background thread."""
    logger.info("Starting Kafka consumer thread...")
    try:
        from kafka.consumer import VitalsConsumer
        consumer = VitalsConsumer()
        consumer.run()
    except Exception as e:
        logger.error(f"Consumer thread error: {e}")


def start_producer():
    """Start Kafka producer."""
    logger.info("Starting Kafka producer...")
    from kafka.producer import run_producer
    run_producer()


def start_api(with_consumer: bool = False):
    """Start Flask API server."""
    if with_consumer:
        t = threading.Thread(target=start_consumer, daemon=True)
        t.start()
        logger.info("Kafka consumer thread started.")

    from api.app import create_app
    app = create_app()
    host  = os.getenv("FLASK_HOST", "0.0.0.0")
    port  = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    print("\n" + "═" * 60)
    print("  🫀  MediSense AI — Healthcare Anomaly Detection System")
    print("═" * 60)
    print(f"  Dashboard: http://localhost:{port}/")
    print(f"  API docs:  http://localhost:{port}/api/health")
    print(f"  Mode:      {'Development' if debug else 'Production'}")
    print("═" * 60 + "\n")

    app.run(host=host, port=port, debug=debug, use_reloader=False)


def train_models():
    """Train both ML models."""
    print("Training Isolation Forest...")
    from models.isolation_forest import train_isolation_forest
    train_isolation_forest()

    print("Training Autoencoder...")
    from models.autoencoder import train_autoencoder
    train_autoencoder()

    print("✓ All models trained successfully.")


def run_tests():
    """Run unit test suite."""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Healthcare Anomaly Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  api          Start Flask API server (default)
  api --full   Start API + Kafka consumer together
  producer     Start Kafka producer (vitals streamer)
  consumer     Start Kafka consumer (anomaly detector)
  train        Train ML models (Autoencoder + Isolation Forest)
  test         Run unit tests
        """
    )
    parser.add_argument("command", nargs="?", default="api",
                        choices=["api", "producer", "consumer", "train", "test"])
    parser.add_argument("--full", action="store_true",
                        help="Start API with embedded consumer (for API command)")
    args = parser.parse_args()

    if args.command == "api":
        start_api(with_consumer=args.full)
    elif args.command == "producer":
        start_producer()
    elif args.command == "consumer":
        start_consumer()
    elif args.command == "train":
        train_models()
    elif args.command == "test":
        sys.exit(run_tests())
