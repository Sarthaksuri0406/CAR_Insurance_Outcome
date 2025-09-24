import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def infer_feature_types(df, target, drop=None):
    drop = drop or []
    cols = [c for c in df.columns if c not in drop + [target]]
    cat = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("string") or str(df[c].dtype) == "category"]
    num = [c for c in cols if c not in cat]
    return cat, num

def make_preprocessor(cat_cols, num_cols, with_scaling=True):
    num_imputer = SimpleImputer(strategy="median")
    steps_num = [("imputer", num_imputer)]
    if with_scaling:
        steps_num.append(("scaler", StandardScaler(with_mean=False)))
    num_proc = Pipeline(steps=steps_num)
    cat_proc = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    pre = ColumnTransformer(transformers=[("num", num_proc, num_cols),("cat", cat_proc, cat_cols)], remainder="drop", sparse_threshold=0)
    return pre
