import os
import pandas as pd
from claims_modeling.features import infer_feature_types, make_preprocessor
from claims_modeling.model import build_pipeline, save_model, load_model

def test_train_and_save(tmp_path):
    df = pd.DataFrame({
        "driver_age":[30,45,52,28,31,47,62,26],
        "gender":["M","F","M","F","M","F","M","F"],
        "annual_premium":[400.0,520.0,610.0,390.0,410.0,530.0,615.0,395.0],
        "has_claim":[0,1,0,1,0,1,0,1]
    })
    cat, num = infer_feature_types(df, "has_claim")
    pre = make_preprocessor(cat, num, with_scaling=True)
    pipe = build_pipeline(pre, {"solver":"lbfgs","penalty":"l2","C":1.0,"max_iter":1000})
    pipe.fit(df.drop(columns=["has_claim"]), df["has_claim"])
    model_path = tmp_path / "m.joblib"
    save_model(pipe, str(model_path))
    m2 = load_model(str(model_path))
    p = m2.predict_proba(df.drop(columns=["has_claim"]))[:,1]
    assert len(p) == len(df)
