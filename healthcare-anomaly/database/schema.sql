-- ─────────────────────────────────────────────────────────────────────────────
-- AI-Driven Healthcare Anomaly Detection System
-- PostgreSQL Schema
-- ─────────────────────────────────────────────────────────────────────────────

-- Create database (run as postgres superuser)
-- CREATE DATABASE healthcare_monitoring;
-- CREATE USER healthcare_admin WITH ENCRYPTED PASSWORD 'your_secure_password';
-- GRANT ALL PRIVILEGES ON DATABASE healthcare_monitoring TO healthcare_admin;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ─── Table: patient_vitals ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patient_vitals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      VARCHAR(50)  NOT NULL,
    patient_name    VARCHAR(100) NOT NULL DEFAULT 'Unknown',
    heart_rate      FLOAT        NOT NULL,
    spo2            FLOAT        NOT NULL,
    temperature     FLOAT        NOT NULL,
    systolic_bp     FLOAT        NOT NULL,
    diastolic_bp    FLOAT        NOT NULL,
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    source          VARCHAR(50)  NOT NULL DEFAULT 'kafka_stream'
);

CREATE INDEX IF NOT EXISTS idx_patient_vitals_patient_id   ON patient_vitals(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_vitals_recorded_at  ON patient_vitals(recorded_at DESC);

-- ─── Table: anomaly_scores ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_scores (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vital_id              UUID REFERENCES patient_vitals(id) ON DELETE CASCADE,
    patient_id            VARCHAR(50)  NOT NULL,
    autoencoder_score     FLOAT        NOT NULL,
    isolation_score       FLOAT        NOT NULL,
    combined_score        FLOAT        NOT NULL,
    severity              VARCHAR(10)  NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH')),
    detected_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_patient_id  ON anomaly_scores(patient_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_severity    ON anomaly_scores(severity);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_detected_at ON anomaly_scores(detected_at DESC);

-- ─── Table: alert_log ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alert_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anomaly_id      UUID REFERENCES anomaly_scores(id) ON DELETE SET NULL,
    patient_id      VARCHAR(50)  NOT NULL,
    patient_name    VARCHAR(100) NOT NULL DEFAULT 'Unknown',
    severity        VARCHAR(10)  NOT NULL,
    combined_score  FLOAT        NOT NULL,
    message         TEXT,
    email_sent      BOOLEAN      NOT NULL DEFAULT FALSE,
    email_recipient VARCHAR(255),
    notified_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alert_log_patient_id  ON alert_log(patient_id);
CREATE INDEX IF NOT EXISTS idx_alert_log_notified_at ON alert_log(notified_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_log_severity    ON alert_log(severity);

-- ─── View: recent_anomalies ───────────────────────────────────────────────────
CREATE OR REPLACE VIEW recent_anomalies AS
SELECT
    a.id,
    a.patient_id,
    v.patient_name,
    v.heart_rate,
    v.spo2,
    v.temperature,
    v.systolic_bp,
    v.diastolic_bp,
    a.autoencoder_score,
    a.isolation_score,
    a.combined_score,
    a.severity,
    a.detected_at
FROM anomaly_scores a
JOIN patient_vitals v ON a.vital_id = v.id
ORDER BY a.detected_at DESC;

GRANT ALL ON ALL TABLES IN SCHEMA public TO healthcare_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO healthcare_admin;
GRANT SELECT ON recent_anomalies TO healthcare_admin;
