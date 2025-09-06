import json
import requests

url = "http://localhost:5000/predict"
payload = {
        "SK_ID_CURR": 100002,
        "EXT_SOURCE_1": 0.15,
        "GOODS_CREDIT_RATIO": 2.5,
        "EXT_SOURCE_3": 0.25,
        "EXT_SOURCE_2": 0.30,
        "ORGANIZATION_TYPE_ENCODED": 8.0,
        "DAYS_BIRTH": -20000,  # ~55 years old
        "DAYS_EMPLOYED": -300,  # <1 year employed
        "NAME_EDUCATION_TYPE": 0,  # 0=Secondary
        "CODE_GENDER": 0,  # 0=Female
        "AMT_ANNUITY": 45000.0,
        "AMT_CREDIT": 500000.0,
        "FLAG_OWN_CAR": 0,  # 0=No
        "DAYS_ID_PUBLISH": -3000,
        "ATI_RATIO": 0.15,
        "OWN_CAR_AGE": 0.0,
        "LIVINGAREA_MEDI": 45.2,
        "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
        "FLAG_DOCUMENT_3": 0,
        "NAME_FAMILY_STATUS_Married": 0,
        "NAME_INCOME_TYPE_Working": 1
}
resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
print("HTTP:", resp.status_code)

try:
    data = resp.json()
except Exception:
    print(resp.text)
    raise

if resp.status_code != 200:
    print("Server error payload:", json.dumps(data, indent=2))
    raise SystemExit(1)

print("Prediction:", data.get("prediction", "N/A"))
print("\nMessage:\n", data.get("message", "N/A"))
explanation = data.get("explanation", {})
for k, v in sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"{k}: {v:.3f}")