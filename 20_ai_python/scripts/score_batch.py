import os
import glob
import argparse
import pandas as pd
from datetime import datetime
from claims_modeling.io_utils import read_table, write_table, load_yaml
from claims_modeling.model import load_model

def latest_model(path="20_ai_python/models", pattern="*.joblib"):
    files = glob.glob(os.path.join(path, pattern))
    if not files:
        raise FileNotFoundError("no model found")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/02_interim/claims_clean_sample.csv")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--thresholds", type=str, default="config/model/thresholds.yml")
    ap.add_argument("--outdir", type=str, default="data/05_predictions")
    args = ap.parse_args()

    df = read_table(args.input)
    mpath = args.model or latest_model()
    model = load_model(mpath)
    features = [c for c in df.columns if c not in ["claim_id","has_claim"]]
    proba = model.predict_proba(df[features])[:,1]
    out = df.copy()
    out["p_claim"] = proba
    th = load_yaml(args.thresholds)
    t = float(th["thresholds"]["operating_point_default"])
    out["decision_at_threshold"] = (out["p_claim"] >= t).astype(int)
    bins = th["thresholds"]["risk_buckets"]
    out["risk_bucket"] = pd.cut(out["p_claim"], bins=bins, labels=False, include_lowest=True)
    out["score_dt"] = pd.to_datetime(datetime.utcnow())
    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    path = os.path.join(args.outdir, f"preds_{ts}.csv")
    write_table(out, path)
    print(path)

if __name__ == "__main__":
    main()
