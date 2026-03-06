"""
utils/severity.py
Classify anomaly severity based on combined anomaly scores.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

THRESHOLD_HIGH   = float(os.getenv("ANOMALY_THRESHOLD_HIGH",   "0.70"))
THRESHOLD_MEDIUM = float(os.getenv("ANOMALY_THRESHOLD_MEDIUM", "0.40"))


@dataclass
class SeverityResult:
    severity: str        # LOW | MEDIUM | HIGH
    combined_score: float
    autoencoder_score: float
    isolation_score: float
    description: str
    action_required: str


def classify_severity(autoencoder_score: float, isolation_score: float) -> SeverityResult:
    """
    Combine autoencoder reconstruction error and isolation forest score
    into a single anomaly score and classify severity.

    - Autoencoder score:  0.0 (normal) → 1.0 (highly anomalous)
    - Isolation score:    0.0 (normal) → 1.0 (highly anomalous)
    - Combined score:     weighted average (60% autoencoder, 40% isolation)
    """
    combined = round(0.60 * autoencoder_score + 0.40 * isolation_score, 6)

    if combined >= THRESHOLD_HIGH:
        return SeverityResult(
            severity="HIGH",
            combined_score=combined,
            autoencoder_score=autoencoder_score,
            isolation_score=isolation_score,
            description="Critical anomaly detected. Patient vitals significantly deviate from normal.",
            action_required="Immediate clinical intervention required. Email notification sent.",
        )
    elif combined >= THRESHOLD_MEDIUM:
        return SeverityResult(
            severity="MEDIUM",
            combined_score=combined,
            autoencoder_score=autoencoder_score,
            isolation_score=isolation_score,
            description="Moderate anomaly detected. Vitals show concerning deviation.",
            action_required="Notify care team and increase monitoring frequency.",
        )
    else:
        return SeverityResult(
            severity="LOW",
            combined_score=combined,
            autoencoder_score=autoencoder_score,
            isolation_score=isolation_score,
            description="Minor deviation detected. Within acceptable monitoring range.",
            action_required="Log and continue routine monitoring.",
        )


def severity_color(severity: str) -> str:
    """Return a hex color for dashboard display."""
    return {"HIGH": "#FF4757", "MEDIUM": "#FFA502", "LOW": "#2ED573"}.get(severity, "#888")


def severity_emoji(severity: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "⚪")
