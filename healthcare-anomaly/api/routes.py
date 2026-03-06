"""
api/routes.py
Flask REST API endpoints for the healthcare anomaly detection system.
"""

import os
import json
import logging
import random
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")

# ─── Mock data generator (used when DB is unavailable) ────────────────────────

MOCK_PATIENTS = [
    {"id": "P001", "name": "Alice Johnson"},
    {"id": "P002", "name": "Bob Martinez"},
    {"id": "P003", "name": "Carol Williams"},
    {"id": "P004", "name": "David Chen"},
    {"id": "P005", "name": "Emma Thompson"},
    {"id": "P006", "name": "Frank Nguyen"},
    {"id": "P007", "name": "Grace Lee"},
    {"id": "P008", "name": "Henry Davis"},
]

def _mock_vital():
    """Generate a realistic mock vital signs record."""
    p = random.choice(MOCK_PATIENTS)
    severity = random.choices(["LOW", "MEDIUM", "HIGH"], weights=[75, 18, 7])[0]
    score = {
        "LOW":    round(random.uniform(0.05, 0.39), 4),
        "MEDIUM": round(random.uniform(0.40, 0.69), 4),
        "HIGH":   round(random.uniform(0.70, 0.98), 4),
    }[severity]
    return {
        "id":               str(random.randint(100000, 999999)),
        "patient_id":       p["id"],
        "patient_name":     p["name"],
        "heart_rate":       round(random.gauss(78, 18), 1),
        "spo2":             round(random.gauss(97.5, 2), 1),
        "temperature":      round(random.gauss(36.7, 0.5), 2),
        "systolic_bp":      round(random.gauss(118, 15), 1),
        "diastolic_bp":     round(random.gauss(76, 10), 1),
        "autoencoder_score": round(score * random.uniform(0.8, 1.1), 4),
        "isolation_score":  round(score * random.uniform(0.7, 1.2), 4),
        "combined_score":   score,
        "severity":         severity,
        "detected_at":      (datetime.utcnow() - timedelta(seconds=random.randint(0, 3600))).isoformat(),
    }


def _try_db(fn, fallback):
    """Try executing a database operation; return fallback on failure."""
    try:
        from database.db_connection import get_db
        with get_db() as db:
            return fn(db)
    except Exception as e:
        logger.warning(f"DB unavailable ({e}), using mock data.")
        return fallback()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@api_bp.route("/health")
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Healthcare Anomaly Detection API",
        "version": "1.0.0",
    })


@api_bp.route("/stats")
def stats():
    """Dashboard summary statistics."""
    def from_db(db):
        from database.db_connection import get_dashboard_stats
        return get_dashboard_stats(db)

    def mock():
        return {
            "active_patients":  random.randint(4, 8),
            "anomalies_24h":    random.randint(12, 60),
            "critical_alerts":  random.randint(1, 8),
            "avg_anomaly_score": round(random.uniform(0.15, 0.45), 4),
        }

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "data": data})


@api_bp.route("/anomalies/recent")
def recent_anomalies():
    """Return the most recent anomaly detections."""
    limit = min(int(request.args.get("limit", 50)), 200)

    def from_db(db):
        from database.db_connection import get_recent_anomalies
        rows = get_recent_anomalies(db, limit=limit)
        return [{**r, "detected_at": str(r["detected_at"])} for r in rows]

    def mock():
        return [_mock_vital() for _ in range(min(limit, 20))]

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "count": len(data), "data": data})


@api_bp.route("/anomalies/live")
def live_anomalies():
    """Return only HIGH and MEDIUM severity anomalies from the last hour."""
    def from_db(db):
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT * FROM recent_anomalies
            WHERE severity IN ('HIGH', 'MEDIUM')
              AND detected_at >= NOW() - INTERVAL '1 hour'
            LIMIT 30
        """))
        rows = result.fetchall()
        return [{**dict(r._mapping), "detected_at": str(dict(r._mapping)["detected_at"])} for r in rows]

    def mock():
        return [_mock_vital() for _ in range(8) if random.random() > 0.3]

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "count": len(data), "data": data})


@api_bp.route("/patients")
def list_patients():
    """Return list of all monitored patients."""
    def from_db(db):
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT DISTINCT patient_id, patient_name,
                   MAX(recorded_at) AS last_seen
            FROM patient_vitals
            GROUP BY patient_id, patient_name
            ORDER BY last_seen DESC
        """))
        return [dict(r._mapping) for r in result.fetchall()]

    def mock():
        return [{"patient_id": p["id"], "patient_name": p["name"],
                 "last_seen": datetime.utcnow().isoformat()} for p in MOCK_PATIENTS]

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "count": len(data), "data": data})


@api_bp.route("/patients/<patient_id>/vitals")
def patient_vitals(patient_id: str):
    """Return vital signs history for a specific patient."""
    limit = min(int(request.args.get("limit", 100)), 500)

    def from_db(db):
        from database.db_connection import get_patient_vitals_history
        rows = get_patient_vitals_history(db, patient_id, limit)
        return [{**r, "recorded_at": str(r["recorded_at"])} for r in rows]

    def mock():
        base = datetime.utcnow()
        return [{
            "patient_id":   patient_id,
            "patient_name": "Mock Patient",
            "heart_rate":   round(random.gauss(76, 10), 1),
            "spo2":         round(random.gauss(97.8, 1.5), 1),
            "temperature":  round(random.gauss(36.6, 0.3), 2),
            "systolic_bp":  round(random.gauss(118, 12), 1),
            "diastolic_bp": round(random.gauss(75, 8), 1),
            "recorded_at":  (base - timedelta(seconds=i * 30)).isoformat(),
        } for i in range(min(limit, 50))]

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "patient_id": patient_id, "count": len(data), "data": data})


@api_bp.route("/alerts")
def alerts():
    """Return recent alert log entries."""
    limit = min(int(request.args.get("limit", 50)), 200)

    def from_db(db):
        from database.db_connection import get_alert_stats
        from sqlalchemy import text
        rows = db.execute(text("""
            SELECT * FROM alert_log ORDER BY notified_at DESC LIMIT :limit
        """), {"limit": limit}).fetchall()
        return [{**dict(r._mapping), "notified_at": str(dict(r._mapping)["notified_at"])} for r in rows]

    def mock():
        base = datetime.utcnow()
        alerts = []
        for i in range(min(limit, 10)):
            p = random.choice(MOCK_PATIENTS)
            alerts.append({
                "id": str(i),
                "patient_id": p["id"],
                "patient_name": p["name"],
                "severity": random.choices(["HIGH", "MEDIUM"], weights=[60, 40])[0],
                "combined_score": round(random.uniform(0.55, 0.97), 4),
                "email_sent": random.choice([True, False]),
                "notified_at": (base - timedelta(minutes=i * 15)).isoformat(),
            })
        return alerts

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "count": len(data), "data": data})


@api_bp.route("/alerts/stats")
def alert_stats():
    """Return 24-hour alert statistics by severity."""
    def from_db(db):
        from database.db_connection import get_alert_stats
        return get_alert_stats(db)

    def mock():
        return {
            "high_count":   random.randint(1, 10),
            "medium_count": random.randint(5, 25),
            "low_count":    random.randint(20, 80),
            "total_count":  random.randint(30, 110),
        }

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "data": data})


@api_bp.route("/vitals/stream")
def vitals_stream():
    """Return a batch of the most recent vital readings (for real-time chart updates)."""
    def mock():
        base = datetime.utcnow()
        return [{
            "patient_id":   p["id"],
            "patient_name": p["name"],
            "heart_rate":   round(random.gauss(76, 12), 1),
            "spo2":         round(random.gauss(97.5, 1.8), 1),
            "temperature":  round(random.gauss(36.6, 0.4), 2),
            "systolic_bp":  round(random.gauss(118, 14), 1),
            "diastolic_bp": round(random.gauss(76, 9), 1),
            "recorded_at":  base.isoformat(),
        } for p in MOCK_PATIENTS]

    def from_db(db):
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT DISTINCT ON (patient_id) *
            FROM patient_vitals
            ORDER BY patient_id, recorded_at DESC
        """))
        return [{**dict(r._mapping), "recorded_at": str(dict(r._mapping)["recorded_at"])} for r in result.fetchall()]

    data = _try_db(from_db, mock)
    return jsonify({"success": True, "data": data})
