import os
import sys
import json
import logging
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from kafka import KafkaConsumer

# Import Airflow DAG modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'airflow', 'dags'))
sys.path.insert(0, project_root)
print(f"project_root={project_root}")

from airflow.dags.predictor import run_prediction # noqa: E402
from airflow.dags.data_agg_clean import data_agg_clean_full # noqa: E402


# Settings (env vars)
KAFKA_BOOTSTRAP       = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC           = os.getenv("KAFKA_TOPIC", "borrowers")
KAFKA_AUTO_OFFSET     = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")  # "latest" for fresh-only

TELEGRAM_TOKEN        = os.getenv("TELEGRAM_LIVE_BOT_TOKEN")
CHAT_ID               = os.getenv("TELEGRAM_CHAT_ID")

TZ                    = ZoneInfo(os.getenv("APP_TZ", "Asia/Jakarta"))

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)


# Telegram helper
def _send_telegram_message(text_html: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("TELEGRAM_LIVE_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": False,
    }
    r = requests.post(url, json=payload, timeout=30)
    if not r.ok:
        logging.error("sendMessage failed: %s %s", r.status_code, r.text)


# Formatting helpers
def _fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def _first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _html_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _fmt_thousands(x) -> str:
    try:
        f = float(x)
        if f.is_integer():
            return f"{int(f):,}"
        return f"{f:,.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def _extract_pdefault(pred: dict | None) -> float | None:
    """
    The predictor returns:
      - prediction: "Default" or "Non-Default"
      - confidence_pct: e.g. "78.17%" (prob of predicted class)
    Convert to P(default) in [0,1].
    """
    if not isinstance(pred, dict):
        return None
    label = (pred.get("prediction") or "").strip().lower()
    conf_pct = pred.get("confidence_pct")

    conf_val = None
    if isinstance(conf_pct, str):
        s = conf_pct.strip().replace("%", "")
        try:
            conf_val = float(s) / 100.0
        except Exception:
            conf_val = None

    if conf_val is not None and label:
        return conf_val if label == "default" else (1.0 - conf_val)
    return None

def _predicted_class_conf(pred: dict | None, p_default: float | None) -> float | None:
    """
    Return the probability of the predicted class (0..1).
    Prefer predictor's confidence_pct; else derive from p_default.
    """
    if isinstance(pred, dict) and isinstance(pred.get("confidence_pct"), str):
        s = pred["confidence_pct"].strip().replace("%", "")
        try:
            return float(s) / 100.0
        except Exception:
            pass
    if isinstance(pred, dict) and p_default is not None:
        label = (pred.get("prediction") or "").strip().lower()
        return p_default if label == "default" else (1.0 - p_default)
    return None

def _prediction_line(pred: dict | None, p_default: float | None) -> str:
    """
    Build: 'Prediction: Default (70.36%) 🟢/🔴'
    Uses predictor's label + confidence_pct; if missing, derives from p_default.
    """
    label = (pred.get("prediction") if isinstance(pred, dict) else None) or "—"

    # Prefer predictor's own confidence text if present
    if isinstance(pred, dict) and pred.get("confidence_pct"):
        conf_text = pred["confidence_pct"]
    else:
        conf_pred = _predicted_class_conf(pred, p_default)
        conf_text = _fmt_pct(conf_pred) if conf_pred is not None else "—"

    # Add icon based on label
    icon = ""
    if label.lower() == "default":
        icon = " 🔴"
    elif label.lower() == "non-default":
        icon = " 🟢"

    return f"<b>Prediction:</b> {label} ({conf_text}){icon}"


def _top10_factors(expl) -> list[str]:
    """
    Return up to 10 aligned lines with:
        +0.613 (↑ risk)
        -0.070 (↓ risk)
    """
    rows = []
    if isinstance(expl, dict):
        items = sorted(expl.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        # compute name width for alignment (cap to 28)
        maxw = min(28, max((len(str(k)) for k, _ in items), default=0))
        for feat, val in items:
            name = str(feat)[:maxw]
            try:
                fval = float(val)
                arrow = "↑ risk" if fval > 0 else ("↓ risk" if fval < 0 else "neutral")
                rows.append(f"{name:<{maxw}}  {fval:+.3f} ({arrow})")
            except Exception:
                rows.append(f"{name:<{maxw}}  {val}")
    elif isinstance(expl, list):
        for x in expl[:10]:
            rows.append(str(x))
    return rows


def _build_message_html(borrower_raw: dict, borrower_agg: dict, pred: dict | None) -> str:
    sk = str(_first_nonempty(
        borrower_raw.get("SK_ID_CURR"),
        borrower_raw.get("sk_id_curr"),
        borrower_agg.get("SK_ID_CURR") if isinstance(borrower_agg, dict) else None,
        (pred or {}).get("SK_ID_CURR"),
        (pred or {}).get("sk_id_curr"),
    ) or "UNKNOWN")

    # For predicted-class confidence
    p_default = _extract_pdefault(pred)

    amt_credit = _first_nonempty(
        borrower_raw.get("AMT_CREDIT"),
        borrower_agg.get("AMT_CREDIT") if isinstance(borrower_agg, dict) else None,
    )
    amt_annuity = _first_nonempty(
        borrower_raw.get("AMT_ANNUITY"),
        borrower_agg.get("AMT_ANNUITY") if isinstance(borrower_agg, dict) else None,
    )
    income = _first_nonempty(
        borrower_raw.get("AMT_INCOME_TOTAL"),
        borrower_agg.get("AMT_INCOME_TOTAL") if isinstance(borrower_agg, dict) else None,
    )

    tstamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        "📡 Real-Time Prediction",
        f"🕒 Time: {_html_escape(tstamp)}",
        f"\n<b>SK_ID_CURR:</b> <code>{_html_escape(sk)}</code>",
        _prediction_line(pred, p_default),
        ]

    if amt_credit is not None:
        lines.append(f"\n<b>AMT_CREDIT:</b> {_html_escape(_fmt_thousands(amt_credit))}")
    if amt_annuity is not None:
        lines.append(f"<b>AMT_ANNUITY:</b> {_html_escape(_fmt_thousands(amt_annuity))}")
    if income is not None:
        lines.append(f"<b>AMT_INCOME_TOTAL:</b> {_html_escape(_fmt_thousands(income))}\n")

    # Top 10 factors (aligned with up/down risk arrows)
    factors = _top10_factors((pred or {}).get("explanation"))
    if factors:
        lines.append("<b>Top factors:</b>")
        lines.append('<pre><code class="language-text">{}</code></pre>'.format("\n".join(factors)))
    
    return "\n".join(lines)


# Main
def main():
    logging.info("Starting Live Prediction Bot (Kafka → Telegram)")
    logging.info("topic=%s bootstrap=%s offset_reset=%s", KAFKA_TOPIC, KAFKA_BOOTSTRAP, KAFKA_AUTO_OFFSET)

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset=KAFKA_AUTO_OFFSET,  # default "latest" for fresh-only
        enable_auto_commit=True,
        group_id="borrower_live_bot_group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    logging.info("Listening on topic '%s' @ %s", KAFKA_TOPIC, KAFKA_BOOTSTRAP)

    for msg in consumer:
        try:
            borrower_raw = msg.value
            borrower_agg = data_agg_clean_full(borrower_raw)

            # No PDF
            pred = run_prediction(borrower_agg, make_pdf=False)

            text_html = _build_message_html(borrower_raw, borrower_agg, pred)
            _send_telegram_message(text_html)

        except Exception as e:
            logging.exception("Failed to process message: %s", e)

if __name__ == "__main__":
    main()

