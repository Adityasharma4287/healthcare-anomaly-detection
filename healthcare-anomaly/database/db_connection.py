"""
database/db_connection.py
PostgreSQL connection manager with connection pooling.
"""

import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

load_dotenv()
logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER', 'healthcare_admin')}:"
    f"{os.getenv('DB_PASSWORD', 'password')}@"
    f"{os.getenv('DB_HOST', 'localhost')}:"
    f"{os.getenv('DB_PORT', '5432')}/"
    f"{os.getenv('DB_NAME', 'healthcare_monitoring')}"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db():
    """Provide a transactional database session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def test_connection():
    """Test database connectivity."""
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        logger.info("Database connection successful.")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def insert_patient_vitals(db, data: dict) -> str:
    """Insert a patient vitals record and return its UUID."""
    result = db.execute(text("""
        INSERT INTO patient_vitals
            (patient_id, patient_name, heart_rate, spo2, temperature, systolic_bp, diastolic_bp)
        VALUES
            (:patient_id, :patient_name, :heart_rate, :spo2, :temperature, :systolic_bp, :diastolic_bp)
        RETURNING id
    """), data)
    return str(result.fetchone()[0])


def insert_anomaly_score(db, data: dict) -> str:
    """Insert an anomaly score record and return its UUID."""
    result = db.execute(text("""
        INSERT INTO anomaly_scores
            (vital_id, patient_id, autoencoder_score, isolation_score, combined_score, severity)
        VALUES
            (:vital_id, :patient_id, :autoencoder_score, :isolation_score, :combined_score, :severity)
        RETURNING id
    """), data)
    return str(result.fetchone()[0])


def insert_alert_log(db, data: dict) -> str:
    """Insert an alert log entry and return its UUID."""
    result = db.execute(text("""
        INSERT INTO alert_log
            (anomaly_id, patient_id, patient_name, severity, combined_score, message, email_sent, email_recipient)
        VALUES
            (:anomaly_id, :patient_id, :patient_name, :severity, :combined_score,
             :message, :email_sent, :email_recipient)
        RETURNING id
    """), data)
    return str(result.fetchone()[0])


def get_recent_anomalies(db, limit: int = 50) -> list:
    result = db.execute(text("""
        SELECT * FROM recent_anomalies LIMIT :limit
    """), {"limit": limit})
    return [dict(row._mapping) for row in result.fetchall()]


def get_patient_vitals_history(db, patient_id: str, limit: int = 100) -> list:
    result = db.execute(text("""
        SELECT * FROM patient_vitals
        WHERE patient_id = :patient_id
        ORDER BY recorded_at DESC
        LIMIT :limit
    """), {"patient_id": patient_id, "limit": limit})
    return [dict(row._mapping) for row in result.fetchall()]


def get_alert_stats(db) -> dict:
    result = db.execute(text("""
        SELECT
            COUNT(*) FILTER (WHERE severity = 'HIGH')   AS high_count,
            COUNT(*) FILTER (WHERE severity = 'MEDIUM') AS medium_count,
            COUNT(*) FILTER (WHERE severity = 'LOW')    AS low_count,
            COUNT(*)                                     AS total_count
        FROM alert_log
        WHERE notified_at >= NOW() - INTERVAL '24 hours'
    """))
    row = result.fetchone()
    return dict(row._mapping) if row else {}


def get_dashboard_stats(db) -> dict:
    result = db.execute(text("""
        SELECT
            (SELECT COUNT(DISTINCT patient_id) FROM patient_vitals
             WHERE recorded_at >= NOW() - INTERVAL '1 hour') AS active_patients,
            (SELECT COUNT(*) FROM anomaly_scores
             WHERE detected_at >= NOW() - INTERVAL '24 hours') AS anomalies_24h,
            (SELECT COUNT(*) FROM alert_log
             WHERE severity = 'HIGH'
             AND notified_at >= NOW() - INTERVAL '24 hours') AS critical_alerts,
            (SELECT ROUND(AVG(combined_score)::numeric, 4)
             FROM anomaly_scores
             WHERE detected_at >= NOW() - INTERVAL '1 hour') AS avg_anomaly_score
    """))
    row = result.fetchone()
    return dict(row._mapping) if row else {}
