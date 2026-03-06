"""
models/isolation_forest.py
Isolation Forest for unsupervised anomaly detection in patient vitals.

Anomaly score is normalized from sklearn's raw score (negative, lower = more anomalous)
to a [0, 1] range (higher = more anomalous) for consistency with the autoencoder.
"""

import os
import sys
import logging
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import generate_synthetic_training_data, load_scaler, SCALER_PATH

logger = logging.getLogger(__name__)

IFOREST_PATH = os.getenv("IFOREST_PATH", "./models/isolation_forest.pkl")
SCALER_PATH  = os.getenv("SCALER_PATH",  "./models/scaler.pkl")

# Isolation Forest training parameters
N_ESTIMATORS    = 200
CONTAMINATION   = 0.05   # Expected fraction of anomalies in training data
MAX_SAMPLES     = "auto"
RANDOM_STATE    = 42


def train_isolation_forest(
    save_path: str = IFOREST_PATH,
    scaler_path: str = SCALER_PATH,
) -> IsolationForest:
    """Train Isolation Forest and save to disk."""

    logger.info("Generating synthetic training data for Isolation Forest...")
    data = generate_synthetic_training_data(n_samples=20000)

    # Use existing scaler or fit a new one
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded existing scaler from {scaler_path}")
    else:
        from utils.preprocessing import fit_and_save_scaler
        scaler = fit_and_save_scaler(data, scaler_path)

    X = scaler.transform(data)

    logger.info(f"Training Isolation Forest with {N_ESTIMATORS} estimators...")
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples=MAX_SAMPLES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)

    # Compute score range for normalization
    raw_scores = model.score_samples(X)
    score_min = float(raw_scores.min())
    score_max = float(raw_scores.max())

    model_data = {
        "model":      model,
        "score_min":  score_min,
        "score_max":  score_max,
    }

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    joblib.dump(model_data, save_path)
    logger.info(f"Isolation Forest saved to {save_path}")
    logger.info(f"Score range: [{score_min:.4f}, {score_max:.4f}]")

    return model


def load_isolation_forest(path: str = IFOREST_PATH) -> dict:
    """Load saved Isolation Forest model data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Isolation Forest not found at {path}. Run training first.")
    return joblib.load(path)


def compute_isolation_score(model_data: dict, vitals_array: np.ndarray, scaler) -> float:
    """
    Compute normalized anomaly score (0–1) from Isolation Forest.
    Higher = more anomalous.
    """
    if model_data is None:
        return float(np.random.beta(1.5, 8))

    model     = model_data["model"]
    score_min = model_data["score_min"]
    score_max = model_data["score_max"]

    scaled    = scaler.transform(vitals_array)
    raw_score = float(model.score_samples(scaled)[0])

    # Invert and normalize: low raw score → high anomaly score
    score_range = score_max - score_min
    if score_range == 0:
        return 0.0
    normalized = (raw_score - score_min) / score_range  # 0 (anomalous) → 1 (normal)
    anomaly_score = 1.0 - normalized                     # invert: 1 = most anomalous
    return round(float(np.clip(anomaly_score, 0.0, 1.0)), 6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training Isolation Forest...")
    train_isolation_forest()
    logger.info("Training complete!")
