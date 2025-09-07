from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORG_MEANS_PATH = PROJECT_ROOT / "dataset" / "org_means.json"

try:
    with open(ORG_MEANS_PATH, "r", encoding="utf-8") as f:
        ORG_MEANS = json.load(f)
except Exception:
    ORG_MEANS = {}

MOST_FREQ_GENDER = "F"

EDUCATION_ORDER = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

FEATURES_21 = [
    "SK_ID_CURR",
    "EXT_SOURCE_1",
    "GOODS_CREDIT_RATIO",
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "ORGANIZATION_TYPE_ENCODED",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "CODE_GENDER",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "FLAG_OWN_CAR",
    "DAYS_ID_PUBLISH",
    "ATI_RATIO",
    "OWN_CAR_AGE",
    "LIVINGAREA_MEDI",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_3",
    "NAME_FAMILY_STATUS_Married",
    "NAME_INCOME_TYPE_Working",
]

ALIASES_21 = {
    "sk_id_curr": "SK_ID_CURR",
    "ext_source_1": "EXT_SOURCE_1",
    "goods_credit_ratio": "GOODS_CREDIT_RATIO",
    "ext_source_3": "EXT_SOURCE_3",
    "ext_source_2": "EXT_SOURCE_2",
    "organization_type_encoded": "ORGANIZATION_TYPE_ENCODED",
    "days_birth": "DAYS_BIRTH",
    "days_employed": "DAYS_EMPLOYED",
    "name_education_type": "NAME_EDUCATION_TYPE",
    "code_gender": "CODE_GENDER",
    "amt_annuity": "AMT_ANNUITY",
    "amt_credit": "AMT_CREDIT",
    "flag_own_car": "FLAG_OWN_CAR",
    "days_id_publish": "DAYS_ID_PUBLISH",
    "ati_ratio": "ATI_RATIO",
    "own_car_age": "OWN_CAR_AGE",
    "livingarea_medi": "LIVINGAREA_MEDI",
    "def_30_cnt_social_circle": "DEF_30_CNT_SOCIAL_CIRCLE",
    "flag_document_3": "FLAG_DOCUMENT_3",
    "name_family_status_married": "NAME_FAMILY_STATUS_Married",
    "name_income_type_working": "NAME_INCOME_TYPE_Working",
}

numeric_like = [
    "EXT_SOURCE_1",
    "GOODS_CREDIT_RATIO",
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "ORGANIZATION_TYPE_ENCODED",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "CODE_GENDER",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "FLAG_OWN_CAR",
    "DAYS_ID_PUBLISH",
    "ATI_RATIO",
    "OWN_CAR_AGE",
    "LIVINGAREA_MEDI",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "FLAG_DOCUMENT_3",
    "NAME_FAMILY_STATUS_Married",
    "NAME_INCOME_TYPE_Working",
]


def _normalize_for_model_21(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        key = ALIASES_21.get(k.lower(), None)
        if key is None:
            # heuristic: UPPER + replace spaces/hyphens with underscore
            key = k.upper().replace(" ", "_").replace("-", "_")
        out[key] = v
    return out


def prepare_features_21(raw: dict) -> dict:
    # Accept snake_case / lowercase / canonical names
    raw = _normalize_for_model_21(raw)

    # Fast path: if the caller already supplies the final 21 features, just coerce types and return
    if all(k in raw for k in FEATURES_21):
        df = pd.DataFrame([raw])

        # Coerce known numeric-like fields
        for col in numeric_like:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure SK_ID_CURR is an int or None
        df["SK_ID_CURR"] = pd.to_numeric(df.get("SK_ID_CURR"), errors="coerce").astype(
            "Int64"
        )
        out = {f: df.iloc[0].get(f, np.nan) for f in FEATURES_21}
        sk = out["SK_ID_CURR"]
        out["SK_ID_CURR"] = int(sk) if pd.notna(sk) else None

        return out

    df = pd.DataFrame([raw])
    # Raw + Agg Columns Required to Create Prediction Data
    base_cols = [
        "SK_ID_CURR",
        "EXT_SOURCE_1",
        "AMT_GOODS_PRICE",
        "AMT_CREDIT",
        "EXT_SOURCE_3",
        "EXT_SOURCE_2",
        "ORGANIZATION_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "NAME_EDUCATION_TYPE",
        "CODE_GENDER",
        "AMT_ANNUITY",
        "FLAG_OWN_CAR",
        "DAYS_ID_PUBLISH",
        "AMT_INCOME_TOTAL",
        "OWN_CAR_AGE",
        "LIVINGAREA_MEDI",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "FLAG_DOCUMENT_3",
        "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Agg of GOODS_CREDIT_RATIO
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]

    # Encoding of ORGANIZATION_TYPE
    df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", "Unknown")

    ## Derive ORG encoding only if missing/NaN
    if ("ORGANIZATION_TYPE_ENCODED" not in df.columns) or pd.isna(
        df.loc[0, "ORGANIZATION_TYPE_ENCODED"]
    ):
        df["ORGANIZATION_TYPE_ENCODED"] = df["ORGANIZATION_TYPE"].map(ORG_MEANS)

    df["ORGANIZATION_TYPE_ENCODED"] = df["ORGANIZATION_TYPE_ENCODED"].astype(float)

    # Encoding of NAME_EDUCATION_TYPE
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDER)

    # Encoding of CODE_GENDER
    df["CODE_GENDER"] = (
        df["CODE_GENDER"].fillna(MOST_FREQ_GENDER).replace("XNA", MOST_FREQ_GENDER)
    )
    df["CODE_GENDER"] = df["CODE_GENDER"].apply(
        lambda x: 1 if str(x).upper() == "F" else 0
    )

    # Encoding of FLAG_OWN_CAR
    def _own_car_to_num(v):
        s = str(v).strip().upper()
        if s == "Y":
            return 1
        if s == "N":
            return 0
        try:
            return int(v)
        except Exception:
            return np.nan

    df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].apply(_own_car_to_num)

    # Agg of ATI_RATIO
    df["ATI_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # OWN_CAR_AGE
    df["OWN_CAR_AGE"] = pd.to_numeric(df["OWN_CAR_AGE"], errors="coerce").clip(upper=60)

    # Encoding of NAME_FAMILY_STATUS, NAME_INCOME_TYPE
    df = pd.get_dummies(
        df,
        columns=["NAME_FAMILY_STATUS", "NAME_INCOME_TYPE"],
        dtype=int,
        drop_first=True,
    )
    for must in ["NAME_FAMILY_STATUS_Married", "NAME_INCOME_TYPE_Working"]:
        if must not in df.columns:
            df[must] = 0

    # Coerce known numeric-like fields
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure SK_ID_CURR is an int or None
    df["SK_ID_CURR"] = pd.to_numeric(df["SK_ID_CURR"], errors="coerce").astype("Int64")
    out = {f: df.iloc[0].get(f, np.nan) for f in FEATURES_21}
    sk = out["SK_ID_CURR"]
    out["SK_ID_CURR"] = int(sk) if pd.notna(sk) else None

    return out
