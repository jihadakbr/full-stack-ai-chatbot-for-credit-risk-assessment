from __future__ import annotations
from pathlib import Path
import json
import os
import time
from datetime import datetime, timezone

# Keep light imports only at parse-time
try:
    import pandas as pd  # light enough okay to keep
except Exception:
    pd = None

# .env loading (airflow already passes envs, safe to skip if unavailable)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Local modules
from data_agg_clean import data_agg_clean_full
from db import get_db_new_table, get_db_url

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_path = PROJECT_ROOT / ".env"
if load_dotenv is not None and env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)

# Prefer KAFKA_BOOTSTRAP_SERVERS, fallback to KAFKA_BOOTSTRAP, default to in-network
KAFKA_BOOTSTRAP = (
    os.getenv("KAFKA_BOOTSTRAP_SERVERS") or os.getenv("KAFKA_BOOTSTRAP") or "kafka:9092"
)
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "borrowers")

AIRFLOW_HOME = Path(os.getenv("AIRFLOW_HOME", str(Path.home() / "airflow-local")))
DATA_ROOT = AIRFLOW_HOME / "datalake"
RAW_DIR = DATA_ROOT / "raw"
CLEAN_DIR = DATA_ROOT / "clean"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

raw_schedule = os.getenv("CRON_PRED_SCHEDULE")
SCHEDULE = None if raw_schedule in (None, "None") else raw_schedule

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

dag = DAG(
    dag_id="credit_risk_pipeline",
    default_args=default_args,
    # schedule_interval="*/5 * * * *",  # every 5 minutes
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    description="Local: Kafka -> raw jsonl -> clean parquet -> predict",
    tags=["local", "kafka", "etl"],
)


def consume_kafka_to_raw(**context):
    # move heavy imports here so DAG can parse even if packages install later
    from kafka import KafkaConsumer
    from pyspark.sql import SparkSession

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"borrowers_{ts}.jsonl"

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="airflow_credit_pipeline",
        consumer_timeout_ms=8000,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_interval_ms=600000,
    )

    records = []
    count = 0
    end_time = time.time() + 30
    with out_path.open("w", encoding="utf-8") as f:
        while time.time() < end_time:
            try:
                msg = next(consumer)
                rec = msg.value
                records.append(rec)
                json.dump(rec, f)
                f.write("\n")
                count += 1
            except StopIteration:
                time.sleep(1)

    if not records:
        context["ti"].xcom_push(key="raw_path", value=str(out_path))
        print(f"[consume] no events; wrote empty file {out_path}")
        return str(out_path)

    hdfs_base = "hdfs://localhost:9000/user/jihadakbr/credit_risk/raw/new_applicants_airflow_kafka"
    hdfs_target = f"{hdfs_base}/ingest_ts={ts}"

    nn = os.getenv("HDFS_NAMENODE", "localhost:9000")

    spark = (
        SparkSession.builder.master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.executor.extraJavaOptions", "-Djava.net.preferIPv4Stack=true")
        .config("spark.hadoop.fs.defaultFS", f"hdfs://{nn}")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .config("spark.hadoop.dfs.replication", "1")
        .getOrCreate()
    )

    try:
        if records:
            sdf = spark.createDataFrame(records)
            hdfs_base = f"hdfs://{nn}/user/jihadakbr/credit_risk/raw/new_applicants_airflow_kafka"
            hdfs_target = f"{hdfs_base}/ingest_ts={ts}"
            sdf.write.mode("overwrite").parquet(hdfs_target)
            print(f"[consume] wrote {count} records to HDFS Parquet at {hdfs_target}")
        else:
            print(f"[consume] no events; wrote empty file {out_path}")
    finally:
        spark.stop()

    context["ti"].xcom_push(key="raw_path", value=str(out_path))
    return str(out_path)


def clean_and_predict(**context):
    # move heavy imports here
    import pandas as pd
    from predictor import run_prediction
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.dialects.postgresql import JSONB

    raw_path = Path(
        context["ti"].xcom_pull(key="raw_path", task_ids="consume_kafka_to_raw")
    )
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        print("[clean_and_predict] No raw file or empty file; nothing to process.")
        return None

    df = pd.read_json(raw_path, lines=True)

    try:
        df_agg = pd.DataFrame(
            [data_agg_clean_full(row) for row in df.to_dict(orient="records")]
        )
    except Exception as e:
        print(f"[clean_and_predict] Error applying data_agg_clean_full: {e}")
        return None

    # Predict
    payloads = []
    for idx, row in df_agg.iterrows():
        try:
            payloads.append(run_prediction(row.to_dict(), make_pdf=False))
        except Exception as e:
            print(f"[clean_and_predict] Prediction error for row {idx}: {e}")
            payloads.append(None)

    # 1/0 target expected by the SQL schema
    df_agg["TARGET"] = [
        1 if (p and p.get("prediction") == "Default") else 0 if p else None
        for p in payloads
    ]

    # Save cleaned data to Parquet
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = CLEAN_DIR / f"borrowers_clean_{ts}.parquet"
    df_agg.to_parquet(out_path, index=False)
    print(f"[clean_and_predict] Saved cleaned data with predictions to {out_path}")

    # Load to Postgres
    db_url = get_db_url()
    new_table = get_db_new_table()
    schema = "public"

    # Build dtype map
    dtype_map = {}
    if "PRED_EXPLANATION" in df_agg.columns:
        dtype_map["PRED_EXPLANATION"] = JSONB

    engine = create_engine(db_url)
    insp = inspect(engine)

    try:
        if insp.has_table(new_table, schema=schema):
            pg_cols_info = insp.get_columns(new_table, schema=schema)
            pg_columns = [c["name"] for c in pg_cols_info]

            df_columns = df_agg.columns.tolist()
            missing_in_pg = [c for c in df_columns if c not in pg_columns]
            extra_in_pg = [c for c in pg_columns if c not in df_columns]
            print(
                "[clean_and_predict] Columns in DataFrame but NOT in PostgreSQL:",
                missing_in_pg,
            )
            print(
                "[clean_and_predict] Columns in PostgreSQL but NOT in DataFrame:",
                extra_in_pg,
            )

            write_cols = [c for c in df_columns if c in pg_columns]
            if write_cols:
                df_to_write = df_agg[write_cols].copy()
                df_to_write.to_sql(
                    name=new_table,
                    con=engine,
                    schema=schema,
                    if_exists="append",
                    index=False,
                    method="multi",
                    dtype={k: v for k, v in dtype_map.items() if k in write_cols}
                    or None,
                )
                print("[clean_and_predict] Data appended into PostgreSQL.")
            else:
                print("[clean_and_predict] No overlapping columns; nothing to write.")
        else:
            print(
                f"[clean_and_predict] Destination table {schema}.{new_table} does not exist; creating it."
            )
            df_to_write = df_agg.copy()
            df_to_write.to_sql(
                name=new_table,
                con=engine,
                schema=schema,
                if_exists="fail",
                index=False,
                method="multi",
                dtype=dtype_map or None,
            )
            print(
                "[clean_and_predict] Table created and data inserted into PostgreSQL."
            )
    except Exception as e:
        print(f"[clean_and_predict] Error while writing to PostgreSQL: {e}")
        raise

    context["ti"].xcom_push(key="clean_path", value=str(out_path))
    print(f"[clean_and_predict] Wrote {len(df_agg)} rows to {out_path}")
    return str(out_path)


with dag:
    t1 = PythonOperator(
        task_id="consume_kafka_to_raw", python_callable=consume_kafka_to_raw
    )
    t2 = PythonOperator(task_id="clean_and_predict", python_callable=clean_and_predict)
    t1 >> t2
