import pandas as pd
from claims_modeling.export_for_bi import export_star_schema
import os

def test_bi_exports(tmp_path):
    df = pd.DataFrame({
        "claim_id":[1,2,3],
        "policy_id":[10,10,11],
        "accident_date":["2025-01-01","2025-01-02","2025-01-03"],
        "has_claim":[0,1,0],
        "p_claim":[0.1,0.8,0.3],
        "decision_at_threshold":[0,1,0],
        "risk_bucket":[0,4,1],
        "score_dt":["2025-09-24T00:00:00","2025-09-24T00:00:00","2025-09-24T00:00:00"]
    })
    out = tmp_path / "out"
    export_star_schema(df, str(out))
    assert (out / "facts_claims_scored.csv").exists()
    assert (out / "dim_policy.csv").exists()
    assert (out / "dim_time.csv").exists()
