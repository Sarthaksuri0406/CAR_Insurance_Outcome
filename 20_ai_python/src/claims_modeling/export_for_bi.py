import os
import pandas as pd
from datetime import datetime

def make_dim_time(dates):
    df = pd.DataFrame({"date":pd.to_datetime(pd.Series(list(dates)).astype(str))})
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df.drop_duplicates().sort_values("date")

def export_star_schema(scored_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fact = scored_df.copy()
    fact["score_dt"] = pd.to_datetime(fact["score_dt"])
    fact["accident_date"] = pd.to_datetime(fact["accident_date"])
    dim_policy = fact.groupby("policy_id").size().reset_index(name="records")
    dim_time = make_dim_time(pd.concat([fact["accident_date"], fact["score_dt"].dt.date.astype(str)]).unique())
    fact_out = fact[["claim_id","policy_id","accident_date","has_claim","p_claim","decision_at_threshold","risk_bucket","score_dt"]].copy()
    fact_out.to_csv(os.path.join(out_dir,"facts_claims_scored.csv"), index=False)
    dim_policy.to_csv(os.path.join(out_dir,"dim_policy.csv"), index=False)
    dim_time.to_csv(os.path.join(out_dir,"dim_time.csv"), index=False)
