import sys
import os
import json
from kafka import KafkaConsumer

# Add the dags folder to the Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "airflow", "dags")
)
sys.path.insert(0, project_root)
print(f"project_root={project_root}")

from airflow.dags.predictor import run_prediction  # noqa: E402
from airflow.dags.data_agg_clean import data_agg_clean_full  # noqa: E402

consumer = KafkaConsumer(
    "borrowers",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="borrower_bot_group",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
)

print("Listening for borrower data on 'borrowers'...")
for msg in consumer:
    borrower_data = msg.value  # Raw Kafka message

    # Call data_agg_clean_full from utils
    borrower_data_agg = data_agg_clean_full(borrower_data)
    # print(f"================ Aggregated borrower data: {borrower_data_agg}")

    # Pass the aggregated features to prediction
    # result = run_prediction(borrower_data_agg, make_pdf=False)  # set True to get PDFs

    # print("Kafka prediction:", result)

    run_prediction(borrower_data_agg, make_pdf=False)  # set True to get PDFs
