import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import joblib


CSV_PATH = "data.csv"  
TARGET   = "Spotify Popularity"  
RANDOM_STATE = 42

def load_csv_robust(path):
    encodings = ["latin-1", "cp1252", "utf-8-sig", "utf-16"]
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

assert Path(CSV_PATH).exists(), f"No existe el archivo: {CSV_PATH}"
df = load_csv_robust(CSV_PATH)

subset_keys = [c for c in ["Track", "Artist", "Release Date"] if c in df.columns]
if subset_keys:
    df = df.drop_duplicates(subset=subset_keys)

if TARGET not in df.columns:
    raise ValueError(f"El target '{TARGET}' no está en el dataset. Columnas: {list(df.columns)}")

df = df[df[TARGET].notna()]

def pre_base_ops(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "Release Date" in X.columns:
        fecha = pd.to_datetime(X["Release Date"], errors="coerce")
        X["release_year"]  = fecha.dt.year
        X["release_month"] = fecha.dt.month
        X["release_day"]   = fecha.dt.day
        X = X.drop(columns=["Release Date"])

    for col in X.columns:
        if X[col].dtype == "object":
            s = X[col].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
            conv = pd.to_numeric(s, errors="coerce")
            ratio = 1.0 - conv.isna().mean()
            if ratio >= 0.7:  
                X[col] = conv

    return X

pre_base = FunctionTransformer(pre_base_ops, validate=False)

cat_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10)

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", cat_encoder),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, selector(dtype_include=np.number)),
        ("cat", cat_pipe, selector(dtype_include=object)),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

y = df[TARGET]
X = df.drop(columns=[TARGET])

is_classification = (
    (y.dtype.kind in "iu" and y.nunique() <= 10) or
    (set(pd.Series(y).dropna().unique()) <= {0, 1})
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y if is_classification and y.nunique() > 1 else None
)

if is_classification:
    score_fn = f_classif
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
else:
    score_fn = f_regression
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

sel = SelectKBest(score_func=score_fn, k="all")  

pipe = Pipeline(steps=[
    ("pre_base", pre_base),
    ("pre", preprocessor),
    ("sel", sel),
    ("model", model)
])

if is_classification:
    scoring = "accuracy"
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }
else:
    scoring = "r2"
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring=scoring,
    verbose=1
)

grid.fit(X_train, y_train)

print(">> Mejores parámetros:", grid.best_params_)

best_pipe = grid.best_estimator_

y_pred_train = best_pipe.predict(X_train)
y_pred_test  = best_pipe.predict(X_test)

if is_classification:
    acc_tr = accuracy_score(y_train, y_pred_train)
    acc_te = accuracy_score(y_test, y_pred_test)
    f1_tr  = f1_score(y_train, y_pred_train, average="weighted")
    f1_te  = f1_score(y_test, y_pred_test,  average="weighted")
    print(f"Train accuracy: {acc_tr:.4f} | F1: {f1_tr:.4f}")
    print(f"Test  accuracy: {acc_te:.4f} | F1: {f1_te:.4f}")
else:
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test)
    mae_tr = mean_absolute_error(y_train, y_pred_train)
    mae_te = mean_absolute_error(y_test, y_pred_test)
    print(f"Train R2: {r2_tr:.4f} | MAE: {mae_tr:.4f}")
    print(f"Test  R2: {r2_te:.4f} | MAE: {mae_te:.4f}")


