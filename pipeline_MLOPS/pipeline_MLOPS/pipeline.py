import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import re
import unicodedata
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import joblib


RANDOM_STATE  = 42
CV_SPLITS     = 3          
GRID_NJOBS    = 2          
PRE_DISPATCH  = 2          


def _normalize_col(s: str) -> str:
    """Normaliza nombres de columnas: espacios raros → espacio, colapsa, trim, lowercase."""
    s = str(s)
    s = s.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ").replace("\u2007", " ")
    s = "".join((" " if unicodedata.category(c) == "Zs" else c) for c in s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def resolve_target_column(df: pd.DataFrame, target_arg: str) -> str:
    """Resuelve el nombre real del target aunque traiga espacios/nbspace/mayúsculas diferentes."""
    norm_map = {_normalize_col(c): c for c in df.columns}
    tnorm = _normalize_col(target_arg)
    if tnorm in norm_map:
        return norm_map[tnorm]
    candidates = [orig for n, orig in norm_map.items() if tnorm in n or n in tnorm]
    hint = f" ¿Quisiste decir uno de: {candidates}?" if candidates else ""
    raise ValueError(f"No se pudo resolver el target '{target_arg}'. Columnas: {list(df.columns)}.{hint}")

def load_csv_robust(path):
    """Intenta varias codificaciones y separadores (engine='python' autodetecta sep)."""
    encodings = ["latin-1", "cp1252", "utf-8-sig", "utf-16"]
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def pre_base_ops(X: pd.DataFrame) -> pd.DataFrame:
    """FE básica: expandir fecha, convertir strings numéricos y reducir alta cardinalidad."""
    X = X.copy()

    # Expandir fecha
    if "Release Date" in X.columns:
        fecha = pd.to_datetime(X["Release Date"], errors="coerce")
        X["release_year"]  = fecha.dt.year
        X["release_month"] = fecha.dt.month
        X["release_day"]   = fecha.dt.day
        X = X.drop(columns=["Release Date"])

    # Convertir strings con números (≥70% convertibles) a numéricos
    for col in list(X.columns):
        if X[col].dtype == "object":
            s = X[col].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
            conv = pd.to_numeric(s, errors="coerce")
            if (1.0 - conv.isna().mean()) >= 0.70:
                X[col] = conv

    for col in ["Track", "Artist"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    return X

pre_base = FunctionTransformer(pre_base_ops, validate=False)


def build_and_train(csv_path: str, target: str, out_path: str = "best_model.pkl"):
    csv_path = Path(csv_path)
    assert csv_path.exists(), f"No existe el archivo: {csv_path}"
    df = load_csv_robust(csv_path)

    subset_keys = [c for c in ["Track", "Artist", "Release Date"] if c in df.columns]
    if subset_keys:
        df = df.drop_duplicates(subset=subset_keys)

    target_col = resolve_target_column(df, target)
    print(f">> Usando columna target: {repr(target_col)}")

    df = df[df[target_col].notna()]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    is_classification = (
        (y.dtype.kind in "iu" and y.nunique() <= 10) or
        (set(pd.Series(y).dropna().unique()) <= {0, 1})
    )

    cat_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10))  
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

    if is_classification:
        score_fn = f_classif
        base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
        base_gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
        stack = StackingClassifier(
            estimators=[("rf", base_rf), ("gb", base_gb)],
            final_estimator=LogisticRegression(max_iter=500),
            n_jobs=1
        )
        models = {"rf": base_rf, "gb": base_gb, "stack": stack}
        scoring = "accuracy"
    else:
        score_fn = f_regression
        base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
        base_gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
        stack = StackingRegressor(
            estimators=[("rf", base_rf), ("gb", base_gb)],
            final_estimator=GradientBoostingRegressor(),
            n_jobs=1
        )
        models = {"rf": base_rf, "gb": base_gb, "stack": stack}
        scoring = "r2"

    sel = SelectKBest(score_func=score_fn, k="all")

    pipe = Pipeline(steps=[
        ("pre_base", pre_base),
        ("pre", preprocessor),
        ("sel", sel),
        ("model", list(models.values())[0])  
    ])

    cv = (StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
          if is_classification else
          KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE))

    param_grid = [
        {
            "model": [models["rf"]],
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        {
            "model": [models["gb"]],
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
        {
            "model": [models["stack"]],
        }
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        n_jobs=GRID_NJOBS,
        pre_dispatch=PRE_DISPATCH,
        scoring=scoring,
        verbose=2,
        error_score="raise"
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if is_classification and y.nunique() > 1 else None
    )

    grid.fit(X_tr, y_tr)
    best_pipe = grid.best_estimator_
    print(">> Mejores parámetros:", grid.best_params_)

    y_pred_tr = best_pipe.predict(X_tr)
    y_pred_te = best_pipe.predict(X_te)

    if is_classification:
        acc_tr = accuracy_score(y_tr, y_pred_tr)
        acc_te = accuracy_score(y_te, y_pred_te)
        f1_tr  = f1_score(y_tr, y_pred_tr, average="weighted")
        f1_te  = f1_score(y_te, y_pred_te, average="weighted")
        print(f"Train accuracy: {acc_tr:.4f} | F1: {f1_tr:.4f}")
        print(f"Test  accuracy: {acc_te:.4f} | F1: {f1_te:.4f}")
    else:
        r2_tr = r2_score(y_tr, y_pred_tr)
        r2_te = r2_score(y_te, y_pred_te)
        mae_tr = mean_absolute_error(y_tr, y_pred_tr)
        mae_te = mean_absolute_error(y_te, y_pred_te)
        print(f"Train R2: {r2_tr:.4f} | MAE: {mae_tr:.4f}")
        print(f"Test  R2: {r2_te:.4f} | MAE: {mae_te:.4f}")

    joblib.dump(best_pipe, out_path)
    print(f">> Modelo guardado en {out_path}")


def cli():
    p = argparse.ArgumentParser(prog="pipeline-mlops-train", description="Entrena y guarda el mejor pipeline.")
    p.add_argument("csv", help="Ruta al CSV (ej. data.csv)")
    p.add_argument("-o", "--output", default="pipeline_best.joblib", help="Ruta de salida del modelo (joblib/pkl)")
    p.add_argument("--target", required=True, help="Nombre de la columna target (se resuelve aunque tenga espacios raros)")
    args = p.parse_args()
    build_and_train(args.csv, args.target, args.output)

if __name__ == "__main__":
    cli()
