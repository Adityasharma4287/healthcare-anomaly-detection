# 🫀 MediSense AI — Healthcare Anomaly Detection System

> Real-time patient vital signs monitoring powered by Autoencoder Neural Networks, Isolation Forest, Apache Kafka, and Flask.

---

## Architecture

```
Patient Devices / Simulators
        │
        ▼
  Kafka Producer  ──►  Apache Kafka (patient-vitals topic)
                               │
                               ▼
                      Kafka Consumer
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
           Autoencoder NN        Isolation Forest
          (reconstruction        (outlier scoring)
               error)
                    └──────────┬──────────┘
                               ▼
                    Severity Classifier
                    LOW / MEDIUM / HIGH
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
          PostgreSQL       Flask API      Email Alert
           Database        (REST)        (SMTP/HIGH)
               │               │
               └───────────────┘
                       │
                       ▼
                  Dashboard
              (Real-time charts)
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10, Java 17, Apache Kafka 3.4, PostgreSQL 15
python3 --version    # 3.10.x
java -version        # openjdk 17
psql --version       # PostgreSQL 15
```

### 2. Environment Setup

```bash
git clone <repo-url> && cd healthcare-anomaly-detection

python3.10 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your credentials
```

### 3. Start Kafka & ZooKeeper

```bash
# In Kafka directory:
bin/zookeeper-server-start.sh config/zookeeper.properties &
sleep 5
bin/kafka-server-start.sh config/server.properties &

# Create topic
bin/kafka-topics.sh --create \
  --topic patient-vitals \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

### 4. Initialize Database

```bash
psql -U postgres -c "CREATE DATABASE healthcare_monitoring;"
psql -U postgres -c "CREATE USER healthcare_admin WITH PASSWORD 'your_password';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE healthcare_monitoring TO healthcare_admin;"
psql -U healthcare_admin -d healthcare_monitoring -f database/schema.sql
```

### 5. Train ML Models

```bash
python run.py train
```

### 6. Run the System

```bash
# Terminal 1 — Flask API + Dashboard
python run.py api

# Terminal 2 — Kafka Producer (patient vitals simulator)
python run.py producer

# Terminal 3 — Kafka Consumer (anomaly detector)
python run.py consumer
```

**Or start API + Consumer together:**
```bash
python run.py api --full
```

### 7. Open Dashboard

```
http://localhost:5000/
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health check |
| `/api/stats` | GET | Dashboard KPI statistics |
| `/api/anomalies/recent` | GET | Recent anomaly detections |
| `/api/anomalies/live` | GET | HIGH/MEDIUM anomalies (last hour) |
| `/api/patients` | GET | All monitored patients |
| `/api/patients/{id}/vitals` | GET | Patient vital history |
| `/api/alerts` | GET | Alert log entries |
| `/api/alerts/stats` | GET | 24h alert statistics |
| `/api/vitals/stream` | GET | Latest vitals per patient |

---

## Project Structure

```
healthcare-anomaly/
├── kafka/
│   ├── producer.py       # Simulates patient vitals streaming
│   └── consumer.py       # Processes vitals with ML pipeline
├── models/
│   ├── autoencoder.py    # Autoencoder NN (TensorFlow/Keras)
│   └── isolation_forest.py # Isolation Forest (Scikit-Learn)
├── database/
│   ├── schema.sql        # PostgreSQL schema
│   └── db_connection.py  # Connection manager
├── api/
│   ├── app.py            # Flask application factory
│   └── routes.py         # REST API endpoints
├── dashboard/
│   └── templates/
│       └── index.html    # Interactive monitoring dashboard
├── notifications/
│   └── email_alert.py    # SMTP alert service
├── utils/
│   ├── preprocessing.py  # Feature extraction & normalization
│   └── severity.py       # LOW/MEDIUM/HIGH classification
├── tests/
│   └── test_anomaly_pipeline.py
├── run.py                # Unified startup script
├── requirements.txt
└── .env.example
```

---

## Severity Levels

| Severity | Score Range | Action |
|---|---|---|
| 🟢 LOW | 0.00 – 0.39 | Log and monitor |
| 🟡 MEDIUM | 0.40 – 0.69 | Notify care team |
| 🔴 HIGH | 0.70 – 1.00 | Immediate action + email alert |

---

## Running Tests

```bash
python run.py test
# or
python -m pytest tests/ -v
```
