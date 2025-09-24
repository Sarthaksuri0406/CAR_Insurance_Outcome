import os
import glob
import argparse
import pandas as pd
from claims_modeling.io_utils import read_table
from claims_modeling.export_for_bi import export_star_schema

def latest_preds(path="data/05_predictions", pattern="preds_*.csv"):
    files = glob.glob(os.path.join(path, pattern))
    if not files:
        raise FileNotFoundError("no predictions found")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="data/06_bi_exports")
    ap.add_argument("--sample_excel", type=str, default="data/07_excel_exports/sample_10k_scored.xlsx")
    args = ap.parse_args()
    preds_path = args.preds or latest_preds()
    df = read_table(preds_path)
    export_star_schema(df, args.outdir)
    sample = df.head(10000)
    os.makedirs(os.path.dirname(args.sample_excel), exist_ok=True)
    sample.to_excel(args.sample_excel, index=False)
    print(args.outdir)

if __name__ == "__main__":
    main()
