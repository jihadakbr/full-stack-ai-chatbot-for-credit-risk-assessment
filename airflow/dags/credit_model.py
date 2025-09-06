from pathlib import Path as _Path
import os as _os
import numpy as _np
import json as _json
import pandas as _pd
from dotenv import load_dotenv


PROJECT_ROOT = _Path(__file__).resolve().parents[2]  # Goes up 3 levels
env_path = PROJECT_ROOT / ".env"
# print(f"env_path: {env_path}")

load_dotenv(dotenv_path=env_path, override=True)


_SPARK = _INPUT_SCHEMA = _PIPELINE_MODEL = _MODEL = _TOP_20_FEATURES = None


EXPERIMENT_ID = _os.getenv("MLFLOW_EXPERIMENT_ID")
RUN_ID = _os.getenv("MLFLOW_RUN_ID")


def _project_root() -> _Path:
    return _Path(__file__).resolve().parents[2] # Goes up 3 levels


# Initialize Spark
def get_spark():
    global _SPARK
    if _SPARK is None:
        from pyspark.sql import SparkSession
        _SPARK = SparkSession.builder.appName("BorrowerChatbot").getOrCreate()
    return _SPARK


def get_input_schema():
    global _INPUT_SCHEMA
    if _INPUT_SCHEMA is None:
        from pyspark.sql.types import StructType, StructField, IntegerType
        schema_path = _project_root() / "model_deployment" / "input_schema.json"
        print(f"schema_path: {schema_path}")
        with open(schema_path, "r") as f:
            schema_json = _json.load(f)
        base = StructType.fromJson(schema_json)
        _INPUT_SCHEMA = StructType([StructField("SK_ID_CURR", IntegerType(), False)] + base.fields)
    return _INPUT_SCHEMA


# get_input_schema()


def _ensure_tracking_uri():
    import mlflow
    # Comment this for docker (Docker mode)
    # _os.environ.pop("MLFLOW_TRACKING_URI", None) # Dev mode
    # _os.environ.pop("MLFLOW_TRACKING_DIR", None) # Dev mode

    # if .env doesn't exist or ENV MLFLOW_TRACKING_DIR=/app/mlruns, fallback to mlruns
    tracking_dir = _os.environ.get("MLFLOW_TRACKING_DIR") or str(_project_root() / "mlruns")
    print(f"tracking_dir: {tracking_dir}")
    mlflow.set_tracking_uri(f"file://{_os.path.abspath(tracking_dir)}")
    return tracking_dir, mlflow


# _ensure_tracking_uri()


def get_pipeline_model():
    global _PIPELINE_MODEL, _MODEL, _TOP_20_FEATURES
    if _PIPELINE_MODEL is None:
        get_spark()  # <-- ensure SparkSession/Context exists BEFORE mlflow.spark.load_model
        tracking_dir, mlflow = _ensure_tracking_uri()
        with mlflow.start_run(run_id=RUN_ID):
            run = mlflow.get_run(RUN_ID)
            _TOP_20_FEATURES = eval(run.data.params["top_features"])
        model_uri = f"file://{_os.path.abspath(tracking_dir)}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/production_model"
        _PIPELINE_MODEL = mlflow.spark.load_model(model_uri)
        _MODEL = _PIPELINE_MODEL.stages[-1]
    return _PIPELINE_MODEL


# get_pipeline_model()


def get_model():
    if _PIPELINE_MODEL is None:
        get_pipeline_model()
    return _MODEL


def get_top_20_features():
    if _TOP_20_FEATURES is None:
        get_pipeline_model()
    return _TOP_20_FEATURES


def _kernel_predict_matrix(data):
    from pyspark.sql import Row
    spark = get_spark()
    schema = get_input_schema()
    pipeline = get_pipeline_model()
    top_20 = get_top_20_features()
    sdf = spark.createDataFrame(_pd.DataFrame(data, columns=top_20), schema=schema)
    preds = pipeline.transform(sdf)
    return _np.array([row['probability'][1] for row in preds.select('probability').collect()])


def explain_prediction(input_dict, model, feature_names):
    import shap, pandas as pd
    df = pd.DataFrame([input_dict])[feature_names]
    if "RandomForest" in str(type(model)) or "GBTC" in str(type(model)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df).values[0]
        print("===== Tree-Based Model =====")
    else:
        explainer = shap.KernelExplainer(_kernel_predict_matrix, df)
        shap_values = explainer(df).values[0][:, 1]
        print("===== Non Tree-Based Model =====")
    out = {col: float(shap_values[i]) for i, col in enumerate(df.columns)}
    return dict(sorted(out.items(), key=lambda x: abs(x[1]), reverse=True))

