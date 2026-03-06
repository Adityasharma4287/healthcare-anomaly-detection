"""
utils/preprocessing.py
Data normalization and feature extraction for vital signs.
"""

import numpy as np
import joblib
import os
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["heart_rate", "spo2", "temperature", "systolic_bp", "diastolic_bp"]

# Normal reference ranges for each vital sign
VITAL_RANGES = {
    "heart_rate":    {"min": 40,  "max": 200, "normal_low": 60,  "normal_high": 100},
    "spo2":          {"min": 70,  "max": 100, "normal_low": 95,  "normal_high": 100},
    "temperature":   {"min": 34,  "max": 42,  "normal_low": 36.1,"normal_high": 37.2},
    "systolic_bp":   {"min": 60,  "max": 220, "normal_low": 90,  "normal_high": 120},
    "diastolic_bp":  {"min": 40,  "max": 140, "normal_low": 60,  "normal_high": 80},
}


def extract_features(vitals: dict) -> np.ndarray:
    """Extract feature vector from a vitals dictionary."""
    return np.array([[
        vitals["heart_rate"],
        vitals["spo2"],
        vitals["temperature"],
        vitals["systolic_bp"],
        vitals["diastolic_bp"],
    ]], dtype=np.float32)


def normalize_features(features: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Normalize features using a fitted scaler."""
    return scaler.transform(features)


def validate_vitals(vitals: dict) -> tuple[bool, list]:
    """
    Validate that vital sign values are within physiologically possible ranges.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    for key, ranges in VITAL_RANGES.items():
        value = vitals.get(key)
        if value is None:
            errors.append(f"Missing vital: {key}")
            continue
        if not (ranges["min"] <= value <= ranges["max"]):
            errors.append(
                f"{key}={value} out of physiological range "
                f"[{ranges['min']}, {ranges['max']}]"
            )
    return len(errors) == 0, errors


def generate_synthetic_training_data(n_samples: int = 15000) -> np.ndarray:
    """
    Generate synthetic normal vital sign data for model training.
    Samples are drawn from realistic normal distributions.
    """
    np.random.seed(42)
    heart_rate   = np.random.normal(75, 8,    n_samples).clip(55, 105)
    spo2         = np.random.normal(98, 0.8,  n_samples).clip(94, 100)
    temperature  = np.random.normal(36.6, 0.3, n_samples).clip(35.8, 37.5)
    systolic_bp  = np.random.normal(115, 8,   n_samples).clip(90, 135)
    diastolic_bp = np.random.normal(75, 6,    n_samples).clip(55, 90)
    return np.column_stack([heart_rate, spo2, temperature, systolic_bp, diastolic_bp])


def load_scaler(path: str) -> MinMaxScaler:
    """Load a saved MinMaxScaler from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler not found at {path}. Run training first.")
    return joblib.load(path)


def fit_and_save_scaler(data: np.ndarray, path: str) -> MinMaxScaler:
    """Fit a MinMaxScaler on training data and save it."""
    scaler = MinMaxScaler()
    scaler.fit(data)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")
    return scaler
