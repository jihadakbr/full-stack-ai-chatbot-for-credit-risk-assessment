import json
import time
import numpy as np
import pandas as pd
from kafka import KafkaProducer

producer = KafkaProducer(
    # bootstrap_servers='localhost:9092', # Dev Mode 
    bootstrap_servers='localhost:29092', # Docker Mode
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_borrower_data():
    # Load and combine datasets
    df1 = pd.read_csv("dataset/new_applicant_default_raw.csv")
    df2 = pd.read_csv("dataset/new_applicant_non_default_raw.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Set TARGET to NaN
    if "TARGET" in df.columns:
        df["TARGET"] = np.nan

    # Replace SK_ID_CURR with sequence starting at 900000
    if "SK_ID_CURR" in df.columns:
        df["SK_ID_CURR"] = range(900000, 900000 + len(df))

    # Pick one random row from the combined dataset
    borrower = df.sample(1).iloc[0].to_dict()

    return borrower

# One message only
# if __name__ == "__main__":
#     data = generate_borrower_data()
#     print(" ")
#     print("=" * 117)
#     print(f"[ {time.strftime('%Y-%m-%d %H:%M:%S')} ] Running real-time borrower data simulation:")
#     print("=" * 117)
#     print(data)
#     producer.send('borrowers', data)
#     producer.flush()
#     time.sleep(10)  # wait 10 seconds before sending next message

if __name__ == "__main__":
    while True:
        data = generate_borrower_data()
        print(" ")
        print("=" * 117)
        print(f"[ {time.strftime('%Y-%m-%d %H:%M:%S')} ] Running real-time borrower data simulation:")
        print("=" * 117)
        print(data)
        producer.send('borrowers', data)
        producer.flush()
        time.sleep(10)  # wait 10 seconds before sending next message
