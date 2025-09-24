from sklearn.calibration import CalibratedClassifierCV

def calibrate_prefit(model, X_valid, y_valid, method="isotonic"):
    cal = CalibratedClassifierCV(base_estimator=model, method=method, cv="prefit")
    cal.fit(X_valid, y_valid)
    return cal
