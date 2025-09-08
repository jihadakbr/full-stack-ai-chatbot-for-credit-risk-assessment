# Full-Stack AI Chatbot for Credit Risk Assessment

## 📌 Project Overview
This project implements an **end-to-end AI chatbot system** for **Credit Risk Assessment**.  
It includes two chatbots:
- **Borrower Chatbot (WhatsApp via Twilio)**: allows borrowers to apply and receive predictions with explainability reports.
- **Officer Chatbot (Telegram)**: allows loan officers to query stats, run SQL, and receive reports.

The system integrates **Machine Learning (PySpark, SHAP, MLflow)**, **RAG (LangChain + FAISS)**, **Kafka streaming** and **Airflow orchestration** for both real-time and scheduled batch processing, alongside **Docker CI/CD** and **Superset dashboards**.

![Project Workflow Diagram](https://raw.githubusercontent.com/jihadakbr/full-stack-ai-chatbot-for-credit-risk-assessment/refs/heads/main/img/reduced/project-workflow.png)

---

## 📺 Demo Video

You can watch a demonstration of the full chatbot system in action here:  
[Google Drive Link](https://drive.google.com/file/d/1kg0kX8CYU9vJSZJu7dZVOIZ3SVZjkzgH/view?usp=sharing)

![Loan Officer](https://raw.githubusercontent.com/jihadakbr/full-stack-ai-chatbot-for-credit-risk-assessment/refs/heads/main/img/reduced/loan-officer-bot.png)

---

## ✨ Features

- **Borrower Chatbot (WhatsApp via Twilio)**  
  - New applicants can interact with a WhatsApp bot (`twilio_bot.py`) to ask about loan-related questions (RAG with LangChain + FAISS), register for a loan, and simulate credit checks.  
  - Data is sent to a REST API (`api.py`) and published to a Kafka topic. Applicants receive results via PDF reports with SHAP-based explanations.  

- **Officer Chatbot (Telegram)**  
  - Loan officers can explore loan data (`telegram_bot.py`), ask loan-related questions (RAG with LangChain + FAISS), and query PostgreSQL for analytics using `/sql` commands.  
  - Officers receive borrower registration alerts directly in the Telegram bot.  
  - A separate bot (`telegram_live_pred_bot.py`) provides officers with live prediction results.  

- **Batch Processing (Airflow)**  
  - Kafka → HDFS (raw dataset) → Python transformation (clean dataset).  
  - Airflow DAGs orchestrate ETL jobs and batch predictions.  
  - Results are stored in both PostgreSQL and HDFS.  

- **Real-Time Prediction (Kafka)**  
  - Kafka → FlexiLoan Live Prediction bot (`telegram_live_pred_bot.py`).

- **Containerized with Docker & Docker Compose**  
  - Kafka, Airflow, Spark, bots, API all containerized.  
  - HDFS runs as a **pseudo-cluster on local host**.  

- **CI/CD with GitHub Actions**  
  - ✅ CI: Linting (ruff, isort, black), security checks (bandit).  
  - 📦 Build & push Docker images to GitHub Container Registry (GHCR).  
  - 🚀 Deploy workflow runs on **self-hosted runner** (your local PC), pulling the latest images and restarting containers.

---

## 🗂️ Project Structure

```
full-stack-ai-chatbot-for-credit-risk-assessment/
├── .github/workflows/          # GitHub Actions workflows
│   ├── build-and-push.yml      # Build & publish images
│   ├── ci.yml                  # Static checks
│   └── deploy.yml              # Self-hosted deploy
├── airflow/                    # Airflow DAGs & configs
│   ├── dags/                   # ETL, ML, and prediction DAGs
│   └── requirements.txt
├── borrower_chatbot/           # Borrower WhatsApp bot (Twilio)
├── dataset/                    # Sample applicant datasets
├── img/                        # Images for this project
├── info/                       # Text policies, glossary, faq
├── mlruns/                     # MLflow experiment tracking data
├── model_deployment/           # ML model artifacts + schema
├── notebook/                   # Data preprocessing and model training
├── officer_chatbot/            # Officer Telegram bots
├── rag_vectorstore/            # FAISS + pickle index
├── .dockerignore
├── .gitignore
├── Dockerfile                  # Application image (chatbots, API, Spark deps)
├── Dockerfile.airflow          # Airflow image with Spark deps
├── LICENSE
├── api_predict_test.py         # Test script for the API
├── docker-compose.deploy.yml   # Deployment stack for CI/CD
├── docker-compose.yml          # Local development stack
├── kafka_consumer.py           # Test script for real-time predictions
├── kafka_producer.py           # Script for creating new applicants
├── ngrok.yml
├── README.md
├── requirements.txt
└── start.sh                    # Entrypoint for chatbots and ngrok
```

---

## ⚙️ Setup (Local Development)

### 1. Prerequisites
- Docker & Docker Compose  
- Java 11 (for PySpark)  
- Python 3.10  
- PostgreSQL (external, not inside Docker)  
- HDFS running locally in pseudo-cluster mode  

### 2. Environment Variables
Create a `.env` file with:

```env
# Database
DB_USER=...
DB_PASSWORD=...
DB_HOST=<server IP or hostname>
DB_PORT=5432
DB_DATABASE=airflow_meta

# Airflow
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin
AIRFLOW_ADMIN_FIRSTNAME=Air
AIRFLOW_ADMIN_LASTNAME=Flow
AIRFLOW_ADMIN_EMAIL=admin@example.com
AIRFLOW__CORE__FERNET_KEY=...
AIRFLOW__WEBSERVER__SECRET_KEY=...
CRON_PRED_SCHEDULE=*/5 * * * *

# Twilio
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=whatsapp:+123456789
MY_WHATSAPP_NUMBER=whatsapp:+987654321

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_LIVE_BOT_TOKEN=...

# Ngrok
NGROK_AUTHTOKEN=...

# Other
APP_TZ=Asia/Jakarta
HDFS_NAMENODE=host.docker.internal:9000
```

### 3. Start the stack
```bash
docker compose -f docker-compose.deploy.yml --env-file .env up -d
```

Stop it:
```bash
docker compose -f docker-compose.deploy.yml --env-file .env down
```

---

## 📱 Usage

### Borrower Chatbot
1. Start REST API (`api.py`) → exposed at port 5000.  
2. Ngrok tunnel created in `start.sh`.  
3. Twilio bot (`twilio_bot.py`) forwards WhatsApp messages to API.  

### Officer Chatbot
1. `telegram_bot.py` → explore credit data, NL2SQL queries.  
2. `telegram_live_pred_bot.py` → consume Kafka + run real-time predictions.  

### Airflow
- Open [http://localhost:8080](http://localhost:8080) (user / password from `.env`).  
- DAGs:  
  - `credit_risk_pipeline` → Kafka → HDFS → Python clean → Postgres.  
  - Scheduled via cron (default: every 5 min, configurable with `CRON_PRED_SCHEDULE`).  

---

## 🔄 CI/CD Workflows

### 1. CI (Static Checks)
- Runs on every push/PR.  
- Tools: `ruff`, `isort`, `black`, `bandit`.  
- Prevents bad code from merging.

### 2. Build & Push
- Builds two images:
  - `ghcr.io/<your_user>/<repo>/app:latest`
  - `ghcr.io/<your_user>/<repo>/airflow:latest`
- Pushes them to GHCR.

### 3. Deploy
- Runs on your **self-hosted runner** (your local PC).  
- Steps:
  1. Logs into GHCR.  
  2. Generates `.env` from GitHub Secrets.  
  3. Pulls latest images.  
  4. Restarts containers with `docker-compose.deploy.yml`.

To check logs:
```bash
docker compose -f docker-compose.deploy.yml logs -f
```

---

## 🐛 Troubleshooting

- **Airflow Permission Error (logs)**  
  ```bash
  chmod -R 777 airflow/logs
  ```
- **Java PySpark error** → check the HDFS pseudo-cluster connection.
- **Kafka not ready** → increase `healthcheck` retries in compose.  
- **Ngrok URL** → check logs:
  ```bash
  docker logs loan_chatbot | grep ngrok
  ```
- **Self-hosted runner stuck** → ensure `./run.sh` is active and Docker Desktop is running.  

---

## 📖 References
- [Apache Airflow](https://airflow.apache.org/)  
- [Apache Kafka](https://kafka.apache.org/)  
- [Apache Spark](https://spark.apache.org/)  
- [Twilio WhatsApp](https://www.twilio.com/whatsapp)  
- [Telegram Bot API](https://core.telegram.org/bots/api)  
- [GitHub Actions](https://docs.github.com/en/actions)  

---

## 👨‍💻 Author
**Jihad Akbar**  
- Email: [jihadakbr@gmail.com](mailto:jihadakbr@gmail.com)
- LinkedIn: [linkedin.com/in/jihadakbr](https://www.linkedin.com/in/jihadakbr)
- Portfolio: [jihadakbr.github.io](https://jihadakbr.github.io/)
