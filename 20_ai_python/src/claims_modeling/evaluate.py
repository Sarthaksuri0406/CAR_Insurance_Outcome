import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_fscore_support, confusion_matrix, brier_score_loss

def metrics_basic(y_true, proba):
    m = {}
    m["roc_auc"] = float(roc_auc_score(y_true, proba))
    m["pr_auc"] = float(average_precision_score(y_true, proba))
    m["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-6, 1-1e-6)))
    m["brier"] = float(brier_score_loss(y_true, proba))
    return m

def metrics_at_threshold(y_true, proba, threshold=0.5):
    y_pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = float((tp+tn)/max(1,(tp+tn+fp+fn)))
    return {"precision":float(p),"recall":float(r),"f1":float(f1),"accuracy":acc,"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

def ks_statistic(y_true, proba):
    pos = np.sort(proba[y_true==1])
    neg = np.sort(proba[y_true==0])
    grid = np.linspace(0,1,101)
    cdf_pos = np.searchsorted(pos, grid, side="right")/max(1,len(pos))
    cdf_neg = np.searchsorted(neg, grid, side="right")/max(1,len(neg))
    ks = float(np.max(np.abs(cdf_pos-cdf_neg)))
    return ks

def lift_table(y_true, proba, bins=10):
    df = pd.DataFrame({"y":y_true,"p":proba})
    df["decile"] = pd.qcut(df["p"], q=bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(n=("y","size"), events=("y","sum"))
    g["non_events"] = g["n"] - g["events"]
    g["event_rate"] = g["events"]/g["n"]
    overall = df["y"].mean()
    g["lift"] = g["event_rate"]/overall
    g = g.sort_index(ascending=False).reset_index()
    g["cum_events"] = g["events"].cumsum()
    g["cum_n"] = g["n"].cumsum()
    g["cum_event_rate"] = g["cum_events"]/g["cum_n"]
    g["cum_lift"] = g["cum_event_rate"]/overall
    return g

def calibration_bins(y_true, proba, bins=10):
    df = pd.DataFrame({"y":y_true,"p":proba})
    df["bin"] = pd.cut(df["p"], bins=bins, labels=False, include_lowest=True)
    agg = df.groupby("bin").agg(n=("y","size"), mean_pred=("p","mean"), mean_obs=("y","mean")).reset_index()
    ece = float(np.average(np.abs(agg["mean_pred"]-agg["mean_obs"]), weights=agg["n"]))
    return agg, ece
