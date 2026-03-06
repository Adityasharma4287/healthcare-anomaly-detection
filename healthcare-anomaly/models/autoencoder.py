"""
models/autoencoder.py
Autoencoder Neural Network for healthcare anomaly detection.

Architecture:
  Encoder: 5 → 16 → 8 → 4 (latent space)
  Decoder: 4 → 8 → 16 → 5 (reconstruction)

Anomaly score = mean squared reconstruction error (normalized 0–1).
"""

import os
import sys
import logging
import numpy as np
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Autoencoder will use mock scoring.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import (
    generate_synthetic_training_data,
    fit_and_save_scaler,
    load_scaler,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

MODEL_PATH  = os.getenv("MODEL_PATH",  "./models/autoencoder_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")


def build_autoencoder(input_dim: int = 5) -> "keras.Model":
    """Build and compile the autoencoder model."""
    inputs = keras.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(4, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(8, activation="relu")(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    model = keras.Model(inputs, decoded, name="healthcare_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_autoencoder(save_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Train the autoencoder on synthetic normal vital sign data."""
    if not TF_AVAILABLE:
        logger.error("TensorFlow required for training.")
        return None

    logger.info("Generating synthetic training data...")
    data = generate_synthetic_training_data(n_samples=20000)
    scaler = fit_and_save_scaler(data, scaler_path)
    X = scaler.transform(data)

    # 80/20 train-val split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]

    model = build_autoencoder(input_dim=X.shape[1])
    model.summary(print_fn=logger.info)

    cb = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=256,
        validation_data=(X_val, X_val),
        callbacks=cb,
        verbose=1,
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    model.save(save_path)
    logger.info(f"Autoencoder saved to {save_path}")

    # Compute threshold on training data
    recon = model.predict(X_train, verbose=0)
    mse = np.mean(np.square(X_train - recon), axis=1)
    threshold = float(np.percentile(mse, 95))
    logger.info(f"95th percentile MSE threshold: {threshold:.6f}")
    joblib.dump({"threshold": threshold}, save_path.replace(".h5", "_threshold.pkl"))

    return model, history


def load_autoencoder(path: str = MODEL_PATH):
    """Load a saved autoencoder from disk."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Autoencoder model not found at {path}.")
    return keras.models.load_model(path)


def compute_autoencoder_score(model, scaler, vitals_array: np.ndarray) -> float:
    """
    Compute normalized anomaly score (0–1) via reconstruction error.
    Higher = more anomalous.
    """
    if model is None:
        # Fallback mock scoring when TF unavailable
        return float(np.random.beta(1.5, 8))

    scaled = scaler.transform(vitals_array)
    recon = model.predict(scaled, verbose=0)
    mse = float(np.mean(np.square(scaled - recon)))

    # Load threshold for normalization
    threshold_path = MODEL_PATH.replace(".h5", "_threshold.pkl")
    if os.path.exists(threshold_path):
        threshold_data = joblib.load(threshold_path)
        threshold = threshold_data.get("threshold", 0.05)
    else:
        threshold = 0.05

    # Normalize: score = mse / (2 * threshold), clipped to [0, 1]
    score = min(mse / (2 * threshold), 1.0)
    return round(score, 6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training Autoencoder Neural Network...")
    train_autoencoder()
    logger.info("Training complete!")
