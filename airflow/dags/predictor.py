from credit_model import (
    explain_prediction,
    get_input_schema,
    get_model,
    get_pipeline_model,
    get_spark,
    get_top_20_features,
)
from data_agg_pred import prepare_features_21
from pdf_credit_report import generate_pdf
from pyspark.sql import Row


def _align_cast_by_schema(features: dict, schema):
    import pandas as pd
    from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

    def _to_py(v):
        if v is None:
            return None
        try:
            return None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
        except Exception:
            return v

    def _coerce(v, st):
        v = _to_py(v)
        try:
            if isinstance(st, (DoubleType, FloatType)):
                # No mean/median: hard-fill missing/invalid with 0.0
                return float(v) if v is not None else 0.0
            if isinstance(st, (LongType, IntegerType)):
                # Integers can’t hold NaN; hard-fill with 0
                return int(v) if v is not None else 0
            return v
        except Exception:
            # On bad strings etc., fall back to constant
            return 0.0 if isinstance(st, (DoubleType, FloatType)) else 0

    return {f.name: _coerce(features.get(f.name), f.dataType) for f in schema.fields}


def run_prediction(user_input: dict, *, make_pdf: bool = True) -> dict:
    spark = get_spark()
    schema = get_input_schema()
    pipeline = get_pipeline_model()
    model = get_model()
    top_20 = get_top_20_features()

    feats = prepare_features_21(user_input)
    row_payload = _align_cast_by_schema(feats, schema)
    if row_payload.get("SK_ID_CURR") is None:
        raise ValueError("SK_ID_CURR is required and cannot be null.")

    print("===== Encoding Check =====")
    print(row_payload)

    sdf = spark.createDataFrame([Row(**row_payload)], schema=schema)

    result_df = pipeline.transform(sdf)

    prob_of_label = None
    # class_probs = None

    if "probability" in result_df.columns:
        row = result_df.select("prediction", "probability").collect()[0]
        pred = float(row["prediction"])
        prob_vec = row["probability"]
        try:
            prob_list = prob_vec.toArray().tolist()
        except AttributeError:
            prob_list = list(prob_vec)

        probs = {
            "Non-Default": float(prob_list[0]) if len(prob_list) > 0 else None,
            "Default": float(prob_list[1]) if len(prob_list) > 1 else None,
        }
        label = "Default" if pred == 1.0 else "Non-Default"
        prob_of_label = probs.get(label)
        # class_probs = probs
    else:
        pred = float(result_df.select("prediction").collect()[0][0])
        label = "Default" if pred == 1.0 else "Non-Default"

    explanation = explain_prediction(feats, model, top_20)
    explanation = dict(
        sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    # Build a percentage string like "85.62%"
    confidence_pct_str = (
        f"{prob_of_label * 100:.2f}%" if prob_of_label is not None else None
    )

    prob_text = f" ({confidence_pct_str})" if confidence_pct_str else ""
    message = (
        f"📊 *Credit Risk Prediction*: {label}{prob_text}\n\n"
        f"🔍 See the *PDF* for full details"
    )

    payload = {
        "prediction": label,
        "confidence_pct": confidence_pct_str,
        "explanation": explanation,
        "message": message,
    }

    print(payload)
    if make_pdf:
        pdf_filename = generate_pdf(label, prob_text, explanation)
        payload["pdf_path"] = f"/pdfs/{pdf_filename}"

    return payload


# borrower_input = {
#     "SK_ID_CURR": 900093,
#     "EXT_SOURCE_1": 0.15,
#     "GOODS_CREDIT_RATIO": 2.5,
#     "EXT_SOURCE_3": 0.25,
#     "EXT_SOURCE_2": 0.30,
#     "ORGANIZATION_TYPE_ENCODED": 8.0,
#     "DAYS_BIRTH": -20000,
#     "DAYS_EMPLOYED": -300,
#     "NAME_EDUCATION_TYPE": 0,
#     "CODE_GENDER": 0,
#     "AMT_ANNUITY": 45000.0,
#     "AMT_CREDIT": 500000.0,
#     "FLAG_OWN_CAR": 0,
#     "DAYS_ID_PUBLISH": -3000,
#     "ATI_RATIO": 0.15,
#     "OWN_CAR_AGE": 0.0,
#     "LIVINGAREA_MEDI": 45.2,
#     "DEF_30_CNT_SOCIAL_CIRCLE": 2.0,
#     "FLAG_DOCUMENT_3": 0,
#     "NAME_FAMILY_STATUS_Married": 0,
#     "NAME_INCOME_TYPE_Working": 1
# }

# run_prediction(borrower_input)
