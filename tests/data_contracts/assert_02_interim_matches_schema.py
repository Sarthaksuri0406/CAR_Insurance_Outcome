import json
import pandas as pd

def assert_schema(df, schema_path):
    with open(schema_path,"r") as f:
        schema = json.load(f)
    cols = [c["name"] for c in schema["columns"]]
    assert set(cols).issubset(set(df.columns))
