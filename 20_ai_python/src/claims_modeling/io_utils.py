import os
import json
import pandas as pd
import yaml

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".xlsx":
        return pd.read_excel(path)
    raise ValueError("unsupported file extension")

def write_table(df, path):
    ensure_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
        return
    if ext == ".csv":
        df.to_csv(path, index=False)
        return
    if ext == ".xlsx":
        df.to_excel(path, index=False)
        return
    raise ValueError("unsupported file extension")

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def validate_required_columns(df, schema_json_path):
    schema = load_json(schema_json_path)
    cols = [c["name"] for c in schema["columns"]]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("missing columns: " + ",".join(missing))
    return True
