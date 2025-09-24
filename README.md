# 🚗📈 Modeling Car Insurance Claim Outcomes

**Tech stack:** Power BI • Excel • Python (AI/ML) • Alteryx  
**Goal:** Train and deploy a calibrated logistic regression model that predicts the probability a car insurance claim occurs, publish score outputs for BI, and enable Excel QA and governance.

---

## ✨ Highlights
- End‑to‑end pipeline: **Alteryx → Python (AI) → Excel QA → Power BI**
- Clean project contracts with versioned **schemas** and **BI star‑schema** exports
- Reproducible training and scoring via **scripts** and **config**
- Calibrated probabilities, **risk buckets**, and ready‑made **DAX** measures
- Synthetic sample data to run locally immediately

---

## 🗺️ Architecture & Flow

```mermaid
flowchart LR
    A[00_raw sources] --> B[Alteryx workflows]
    B --> C[02_interim cleaned]
    C --> D[Python: train + validate]
    D --> E[Model artifact]
    C --> F[Python: score batch]
    E --> F
    F --> G[05_predictions]
    G --> H[06_bi_exports (fact + dims)]
    H --> I[Power BI dataset/report]
    G --> J[07_excel_exports sample]
    I --> K[Dashboards]
    J --> L[Excel QA & sign‑offs]
```

---

## 🧱 Folder Map (top levels)
```
Modeling-Car-Insurance-Claim-Outcomes/
├─ config/
│  ├─ schema/
│  └─ model/
├─ data/
│  ├─ 00_raw/
│  ├─ 02_interim/
│  ├─ 05_predictions/
│  ├─ 06_bi_exports/
│  └─ 07_excel_exports/
├─ 10_alteryx/
│  └─ workflows/
├─ 20_ai_python/
│  ├─ src/claims_modeling/
│  └─ scripts/
├─ 30_powerbi/
│  ├─ dax/
│  └─ powerquery/
├─ 40_excel/
│  ├─ templates/
│  └─ qa_checks/
├─ orchestration/
└─ tests/

```

---

## 🚀 Quickstart

### 1) Environment
```bash
# Option A: pip
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate claims_outcomes_env
```

### 2) Seed data
A 1,000‑row synthetic dataset is included at `data/02_interim/claims_clean_sample.csv`.

### 3) Train → Score → Export
```bash
# Train logistic regression (with optional isotonic calibration)
python 20_ai_python/scripts/train_model.py --input data/02_interim/claims_clean_sample.csv

# Score a batch using the latest model
python 20_ai_python/scripts/score_batch.py --input data/02_interim/claims_clean_sample.csv

# Export Power BI star schema + Excel sample
python 20_ai_python/scripts/export_bi_tables.py
```

Or run everything at once:
```bash
bash orchestration/run_all.sh
# Windows:
orchestration\run_all.bat
```

---

## 🧠 Modeling

- Algorithm: `sklearn.linear_model.LogisticRegression` with `class_weight=balanced`
- Preprocessing: numeric median imputation + scaling; categorical most‑frequent + One‑Hot
- Calibration: optional isotonic (`config/model/logistic_regression.yml` → `calibration.enabled`)
- Threshold & risk buckets: `config/model/thresholds.yml`  
  - `operating_point_default`: decision threshold  
  - `risk_buckets`: edges for bucketizing predicted probabilities

**Outputs**
- `p_claim`: predicted probability
- `decision_at_threshold`: 0/1 decision at configured threshold
- `risk_bucket`: ordinal bucket index
- `score_dt`: UTC timestamp when the score was generated

---

## 📦 Data Contracts

### Alteryx → Python (`data/02_interim/claims_clean_*.parquet|csv`)
| Column | Type | Notes |
|---|---|---|
| claim_id | int64 | Primary key |
| policy_id | int64 | Foreign key |
| accident_date | datetime | ISO 8601 |
| driver_age | float | |
| vehicle_age | float | |
| annual_premium | float | |
| credit_score | float | |
| gender | string | |
| marital_status | string | |
| region | string | |
| vehicle_segment | string | |
| previous_claims | int64 | |
| has_claim | int64 | Target |

### Python → Power BI (`data/06_bi_exports/`)
- `facts_claims_scored.csv`: `claim_id, policy_id, accident_date, has_claim, p_claim, decision_at_threshold, risk_bucket, score_dt`
- `dim_policy.csv`: simple policy grain
- `dim_time.csv`: date dimensions for accident and scoring timestamps

---

## 📊 Power BI Setup

1. Open `30_powerbi/datasets/ClaimsOutcomes.pbids` to auto‑bind to `data/06_bi_exports/`
2. Load `facts_claims_scored.csv`, `dim_policy.csv`, `dim_time.csv`
3. Import DAX: `30_powerbi/dax/measures.dax`, `30_powerbi/dax/calc_columns.dax`
4. Refresh, validate measures like:
   - `Claim Count`
   - `Observed Claim Rate`
   - `Average Predicted Probability`
   - `Lift Top 10 Percent`

Tip: Lock data model relationships on `policy_id` and date keys as needed.

---

## 📗 Excel QA

- Open the generated sample at `data/07_excel_exports/sample_10k_scored.xlsx`
- Templates in `40_excel/templates/`:
  - `data_dictionary_template.xlsx`
  - `business_rules_template.xlsx`
- QA workbooks in `40_excel/qa_checks/` for schema checks, sampling bias, and threshold scenarios

---

## 🧩 Configuration Guide

- `config/model/logistic_regression.yml` model and calibration
- `config/model/sampling.yml` train/validation split
- `config/model/thresholds.yml` decision threshold + risk buckets
- `config/schema/*.json` column contracts
- `config/powerbi/parameters.json` BI paths
- `config/powerbi/pbids/*.pbids` dataset connection bootstrap

---

## 🔍 Evaluation Artifacts

- Validation metrics and charts:
  - `20_ai_python/experiments/runs.csv`
  - `20_ai_python/experiments/lift_validation.csv`
  - `20_ai_python/experiments/calibration_validation.csv`

Key metrics tracked:
- ROC AUC, PR AUC, Log Loss, Brier Score
- KS statistic, decile lift, calibration error (ECE)

---

## 🛠️ Alteryx Stage

Open workflows in `10_alteryx/workflows/` to run profiling, cleaning, enrichment, and publication to `data/02_interim/`. Bind inputs from `data/00_raw/` and ensure outputs match `config/schema/raw_claims_schema.json`.

---

## 🔐 Security & PII

- Do not commit PII to the repo
- Keep raw source drops outside of version control
- Mark PII in `config/schema` and enforce via data‑contract tests

---

## 🧪 Tests

Lightweight checks under `20_ai_python/tests/` and `tests/`:
- Feature pipeline shape and transform
- Model train/save/load round‑trip
- BI export presence and schema
- Drift (PSI) and calibration ECE utilities

Run with:
```bash
pytest -q
```

---

## ♻️ Repro Tips

- Pin environments via `requirements.txt` or `environment.yml`
- Keep artifacts in `20_ai_python/models/`; metadata in `metadata.json`
- Use `runs.csv` to compare seeds and settings across experiments

---

## ❓ FAQ

**Q:** Can I change the threshold per business segment?  
**A:** Yes. Duplicate `thresholds.yml` per segment or add segment rules in `score_batch.py` before export.

**Q:** Where do I add new features?  
**A:** Create them in Alteryx or extend `features.py` and retrain.

**Q:** How do I add a new BI dimension?  
**A:** Extend `export_for_bi.py` to emit the dimension and relate it in Power BI.

---

## 📅 Changelog
- 2025-09-24: Added this README

---

## 📄 License
Specify your license here.
