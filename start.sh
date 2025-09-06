#!/bin/bash
set -euo pipefail

# Initialize PIDS as an array
PIDS=()

# graceful shutdown
cleanup() {
  echo 'Shutting down...'
  # kill background pids if we recorded them
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  # fallbacks
  pkill -f "ngrok" || true
  pkill -f "flask" || true
  pkill -f "officer_chatbot.telegram_bot" || true
  pkill -f "telegram_live_pred_bot.py" || true
}
trap cleanup SIGINT SIGTERM

cd /app

# basics
mkdir -p /app/generated_pdfs

# load .env if present
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

# defaults for in-compose networking (overridden by .env if set)
export KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}"
export KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-kafka:9092}"
export DB_HOST="${DB_HOST:-postgres}"
export DB_PORT="${DB_PORT:-5432}"

# make airflow/dags modules importable from anywhere
export PYTHONPATH="/app:/app/airflow/dags:${PYTHONPATH:-}"

# ngrok auth
if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
  ngrok config add-authtoken "$NGROK_AUTHTOKEN" || true
fi

# helpers
wait_for_http() {
  local url="$1"
  local tries="${2:-60}"
  local delay="${3:-2}"
  echo "Waiting for HTTP ${url} ..."
  for i in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "Ready: ${url}"
      return 0
    fi
    sleep "$delay"
  done
  echo "ERROR: timeout waiting for ${url}" >&2
  return 1
}

wait_for_tcp() {
  local host="$1" port="$2" tries="${3:-90}" delay="${4:-2}"
  echo "Waiting for TCP ${host}:${port} ..."
  python - "$host" "$port" "$tries" "$delay" <<'PY'
import socket, sys, time
host, port, tries, delay = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
for i in range(tries):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("Ready:", host, port)
            sys.exit(0)
    except Exception:
        time.sleep(delay)
print("ERROR: timeout waiting for", host, port, file=sys.stderr)
sys.exit(1)
PY
}

# borrower API (port 5000)
python -m flask --app /app/airflow/dags/api.py run --host=0.0.0.0 --port=5000 &
PIDS+=("$!")

# ngrok (dashboard on 4040)
ngrok start --all --config /app/ngrok.yml >/dev/null 2>&1 &
PIDS+=("$!")
sleep 5

# wait for API to be healthy
wait_for_http "http://localhost:5000/health"

# fetch ngrok public URL for port 5000
NGROK1_URL="$(curl -fsS http://localhost:4040/api/tunnels | jq -r '.tunnels[] | select(.config.addr=="http://localhost:5000") | .public_url')"
echo "=========================================================="
echo "ngrok1 URL (for /predict): ${NGROK1_URL:-<not found>}"

# borrower WhatsApp bot (port 5001)
export NGROK1_URL="${NGROK1_URL:-}"
python -m flask --app borrower_chatbot/twilio_bot.py run --host=0.0.0.0 --port=5001 &
PIDS+=("$!")

# officer bots
# - wait for Kafka before starting the live prediction bot

# start the officer Telegram (RAG + /sql) bot (no Kafka dependency)
python -m officer_chatbot.telegram_bot &
PIDS+=("$!")

# wait for Kafka broker then start live prediction bot
_k_host="${KAFKA_BOOTSTRAP_SERVERS%%,*}"
_k_port="${_k_host##*:}"
_k_host="${_k_host%%:*}"
# fallbacks if the user set only host or only port weirdly
_k_host="${_k_host:-kafka}"
_k_port="${_k_port:-9092}"

wait_for_tcp "$_k_host" "$_k_port" 120 2
python /app/officer_chatbot/telegram_live_pred_bot.py &
PIDS+=("$!")

# show tunnels
sleep 3
echo "All active ngrok tunnels:"
curl -fsS http://localhost:4040/api/tunnels | jq -r '.tunnels[] | .public_url'
echo "=========================================================="

# keep foreground
wait
