import pandas as pd
from sklearn.model_selection import train_test_split

def split_xy(df, target, drop=None):
    drop = drop or []
    cols = [c for c in df.columns if c not in drop + [target]]
    X = df[cols].copy()
    y = df[target].copy()
    return X, y

def train_valid_split(X, y, test_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    return X_tr, X_va, y_tr, y_va
