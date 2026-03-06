"""
notifications/email_alert.py
SMTP email notification service for HIGH-severity anomaly alerts.
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SMTP_SERVER    = os.getenv("SMTP_SERVER",    "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT",  "587"))
SMTP_USER      = os.getenv("SMTP_USER",      "")
SMTP_PASSWORD  = os.getenv("SMTP_PASSWORD",  "")
ALERT_RECIPIENT = os.getenv("ALERT_RECIPIENT", "")


def _build_html_email(patient_id: str, patient_name: str, vitals: dict,
                      severity: str, combined_score: float, detected_at: str) -> str:
    """Build a styled HTML email body for the alert."""
    color = {"HIGH": "#FF4757", "MEDIUM": "#FFA502", "LOW": "#2ED573"}.get(severity, "#888")
    return f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="font-family: 'Helvetica Neue', Arial, sans-serif; background: #f4f6f9; margin:0; padding:20px;">
      <div style="max-width:600px; margin:0 auto; background:#fff; border-radius:12px;
                  box-shadow:0 4px 20px rgba(0,0,0,0.1); overflow:hidden;">

        <!-- Header -->
        <div style="background:{color}; padding:28px 32px;">
          <h1 style="color:#fff; margin:0; font-size:22px; font-weight:700; letter-spacing:-0.3px;">
            ⚠️ {severity} SEVERITY HEALTH ANOMALY DETECTED
          </h1>
          <p style="color:rgba(255,255,255,0.85); margin:8px 0 0; font-size:14px;">
            AI-Driven Healthcare Anomaly Detection System
          </p>
        </div>

        <!-- Patient Info -->
        <div style="padding:24px 32px; border-bottom:1px solid #eee;">
          <h2 style="color:#1a1a2e; margin:0 0 16px; font-size:16px;">Patient Information</h2>
          <table style="width:100%; border-collapse:collapse;">
            <tr>
              <td style="padding:6px 0; color:#666; font-size:14px; width:160px;">Patient ID</td>
              <td style="padding:6px 0; color:#1a1a2e; font-size:14px; font-weight:600;">{patient_id}</td>
            </tr>
            <tr>
              <td style="padding:6px 0; color:#666; font-size:14px;">Patient Name</td>
              <td style="padding:6px 0; color:#1a1a2e; font-size:14px; font-weight:600;">{patient_name}</td>
            </tr>
            <tr>
              <td style="padding:6px 0; color:#666; font-size:14px;">Detection Time</td>
              <td style="padding:6px 0; color:#1a1a2e; font-size:14px;">{detected_at}</td>
            </tr>
            <tr>
              <td style="padding:6px 0; color:#666; font-size:14px;">Anomaly Score</td>
              <td style="padding:6px 0; font-size:14px;">
                <span style="background:{color}; color:#fff; padding:2px 10px;
                             border-radius:20px; font-weight:700;">{combined_score:.4f}</span>
              </td>
            </tr>
          </table>
        </div>

        <!-- Vital Signs -->
        <div style="padding:24px 32px; border-bottom:1px solid #eee;">
          <h2 style="color:#1a1a2e; margin:0 0 16px; font-size:16px;">Recorded Vital Signs</h2>
          <table style="width:100%; border-collapse:collapse; background:#f8f9fa; border-radius:8px; overflow:hidden;">
            <thead>
              <tr style="background:#e9ecef;">
                <th style="padding:10px 16px; text-align:left; font-size:13px; color:#495057;">Vital Sign</th>
                <th style="padding:10px 16px; text-align:right; font-size:13px; color:#495057;">Value</th>
                <th style="padding:10px 16px; text-align:right; font-size:13px; color:#495057;">Normal Range</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-top:1px solid #dee2e6;">
                <td style="padding:10px 16px; font-size:14px; color:#333;">Heart Rate</td>
                <td style="padding:10px 16px; font-size:14px; font-weight:600; color:#333; text-align:right;">{vitals.get('heart_rate', 'N/A')} BPM</td>
                <td style="padding:10px 16px; font-size:13px; color:#888; text-align:right;">60–100 BPM</td>
              </tr>
              <tr style="border-top:1px solid #dee2e6; background:#fff;">
                <td style="padding:10px 16px; font-size:14px; color:#333;">SpO₂</td>
                <td style="padding:10px 16px; font-size:14px; font-weight:600; color:#333; text-align:right;">{vitals.get('spo2', 'N/A')}%</td>
                <td style="padding:10px 16px; font-size:13px; color:#888; text-align:right;">95–100%</td>
              </tr>
              <tr style="border-top:1px solid #dee2e6;">
                <td style="padding:10px 16px; font-size:14px; color:#333;">Temperature</td>
                <td style="padding:10px 16px; font-size:14px; font-weight:600; color:#333; text-align:right;">{vitals.get('temperature', 'N/A')}°C</td>
                <td style="padding:10px 16px; font-size:13px; color:#888; text-align:right;">36.1–37.2°C</td>
              </tr>
              <tr style="border-top:1px solid #dee2e6; background:#fff;">
                <td style="padding:10px 16px; font-size:14px; color:#333;">Blood Pressure</td>
                <td style="padding:10px 16px; font-size:14px; font-weight:600; color:#333; text-align:right;">{vitals.get('systolic_bp', 'N/A')}/{vitals.get('diastolic_bp', 'N/A')} mmHg</td>
                <td style="padding:10px 16px; font-size:13px; color:#888; text-align:right;">90–120/60–80 mmHg</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Action Required -->
        <div style="padding:24px 32px; background:#fff8f8; border-left:4px solid {color}; margin:0 32px 24px; border-radius:4px;">
          <h3 style="color:{color}; margin:0 0 8px; font-size:14px; text-transform:uppercase; letter-spacing:1px;">
            Action Required
          </h3>
          <p style="color:#333; font-size:14px; margin:0; line-height:1.6;">
            Immediate clinical review is recommended. Please check on the patient at the
            earliest opportunity and assess the vital sign readings in clinical context.
          </p>
        </div>

        <!-- Footer -->
        <div style="padding:20px 32px; background:#f8f9fa; border-top:1px solid #eee;">
          <p style="color:#999; font-size:12px; margin:0; text-align:center;">
            This alert was automatically generated by the AI Healthcare Anomaly Detection System.<br>
            Please do not reply to this email. Contact your system administrator for support.
          </p>
        </div>
      </div>
    </body>
    </html>
    """


def send_alert_email(
    patient_id: str,
    patient_name: str,
    vitals: dict,
    severity: str,
    combined_score: float,
    recipient: str = None,
) -> bool:
    """
    Send a HIGH-severity anomaly alert email via SMTP.
    Returns True on success, False on failure.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured. Email not sent.")
        return False

    recipient = recipient or ALERT_RECIPIENT
    if not recipient:
        logger.warning("No email recipient configured.")
        return False

    detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = (
        f"[{severity} ALERT] Patient {patient_id} — "
        f"Anomaly Score {combined_score:.4f} | {detected_at}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Healthcare Monitor <{SMTP_USER}>"
    msg["To"]      = recipient

    plain_text = (
        f"SEVERITY: {severity}\n"
        f"Patient ID: {patient_id}\n"
        f"Patient Name: {patient_name}\n"
        f"Anomaly Score: {combined_score:.4f}\n"
        f"Detected At: {detected_at}\n\n"
        f"Vitals:\n"
        f"  Heart Rate: {vitals.get('heart_rate')} BPM\n"
        f"  SpO2: {vitals.get('spo2')}%\n"
        f"  Temperature: {vitals.get('temperature')}°C\n"
        f"  BP: {vitals.get('systolic_bp')}/{vitals.get('diastolic_bp')} mmHg\n"
    )

    html_content = _build_html_email(
        patient_id, patient_name, vitals, severity, combined_score, detected_at
    )

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipient, msg.as_string())
        logger.info(f"Alert email sent to {recipient} for patient {patient_id}")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending alert: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return False
