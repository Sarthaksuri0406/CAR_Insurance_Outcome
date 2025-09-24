import pandas as pd
import numpy as np
from claims_modeling.features import infer_feature_types, make_preprocessor

def test_infer_and_preprocess():
    df = pd.DataFrame({
        "claim_id":[1,2,3,4],
        "driver_age":[30,45,52,28],
        "gender":["M","F","M","F"],
        "annual_premium":[400.0,520.0,610.0,390.0],
        "has_claim":[0,1,0,1]
    })
    cat, num = infer_feature_types(df, "has_claim", drop=["claim_id"])
    pre = make_preprocessor(cat, num, with_scaling=True)
    X = df[[c for c in df.columns if c not in ["has_claim","claim_id"]]]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == 4
