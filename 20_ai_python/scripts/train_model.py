import os
import json
import argparse
import pandas as pd
from datetime import datetime
from claims_modeling.io_utils import read_table, write_table, load_yaml, validate_required_columns
from claims_modeling.features import infer_feature_types, make_preprocessor
from claims_modeling.data_prep import split_xy, train_valid_split
from claims_modeling.model import build_pipeline, save_model
from claims_modeling.calibration import calibrate_prefit
from claims_modeling.evaluate import metrics_basic, metrics_at_threshold, ks_statistic, lift_table, calibration_bins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/02_interim/claims_clean_sample.csv")
    ap.add_argument("--target", type=str, default="has_claim")
    ap.add_argument("--schema", type=str, default="config/schema/raw_claims_schema.json")
    ap.add_argument("--config", type=str, default="config/model/logistic_regression.yml")
    ap.add_argument("--sampling", type=str, default="config/model/sampling.yml")
    ap.add_argument("--outdir", type=str, default="20_ai_python/models")
    args = ap.parse_args()

    df = read_table(args.input)
    validate_required_columns(df, args.schema)
    cfg = load_yaml(args.config)
    samp = load_yaml(args.sampling)
    drop = ["claim_id"]
    X, y = split_xy(df, args.target, drop=drop)
    cat, num = infer_feature_types(df, args.target, drop=drop)
    pre = make_preprocessor(cat, num, with_scaling=True)
    pipe = build_pipeline(pre, cfg["model"])
    X_tr, X_va, y_tr, y_va = train_valid_split(X, y, test_size=samp["split"]["test_size"], random_state=samp["split"]["random_state"], stratify=bool(samp["split"]["stratify"]))
    pipe.fit(X_tr, y_tr)
    model = pipe
    proba_va = model.predict_proba(X_va)[:,1]
    mb = metrics_basic(y_va, proba_va)
    ks = ks_statistic(y_va.values, proba_va)
    lb = lift_table(y_va.values, proba_va, bins=10)
    cal_bins, ece = calibration_bins(y_va.values, proba_va, bins=10)
    if cfg.get("calibration",{}).get("enabled",False):
        model = calibrate_prefit(model, X_va, y_va, method=cfg["calibration"].get("method","isotonic"))
        proba_va = model.predict_proba(X_va)[:,1]
        mb = metrics_basic(y_va, proba_va)
        ks = ks_statistic(y_va.values, proba_va)
        lb = lift_table(y_va.values, proba_va, bins=10)
        cal_bins, ece = calibration_bins(y_va.values, proba_va, bins=10)
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "logistic_regression.joblib")
    save_model(model, model_path)
    meta = {}
    meta["trained_at"] = datetime.utcnow().isoformat()
    meta["model_path"] = model_path
    meta["features"] = {"categorical":cat,"numerical":num}
    meta["metrics"] = {"validation":mb,"ks":ks,"ece":ece}
    with open(os.path.join(args.outdir,"metadata.json"),"w") as f:
        json.dump(meta, f, indent=2)
    lb.to_csv(os.path.join("20_ai_python/experiments","lift_validation.csv"), index=False)
    cal_bins.to_csv(os.path.join("20_ai_python/experiments","calibration_validation.csv"), index=False)
    runs_csv = os.path.join("20_ai_python/experiments","runs.csv")
    header = not os.path.exists(runs_csv)
    with open(runs_csv,"a") as f:
        if header:
            f.write("timestamp,model,roc_auc,pr_auc,ks,ece,model_path\n")
        f.write(f"{datetime.utcnow().isoformat()},logistic_regression,{mb['roc_auc']},{mb['pr_auc']},{ks},{ece},{model_path}\n")

if __name__ == "__main__":
    main()
