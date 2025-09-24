import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline(preprocessor, params):
    lr = LogisticRegression(
        solver=params.get("solver","lbfgs"),
        penalty=params.get("penalty","l2"),
        C=float(params.get("C",1.0)),
        class_weight=params.get("class_weight",None),
        max_iter=int(params.get("max_iter",1000)),
        random_state=int(params.get("random_state",42))
    )
    pipe = Pipeline(steps=[("pre", preprocessor),("clf", lr)])
    return pipe

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
