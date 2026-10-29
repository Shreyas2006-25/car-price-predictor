"""Train a used-car price model on the CarDekho dataset.

Usage:  python train.py [--data data/cardekho.csv]

Outputs (in ./model):
    price_model.joblib   - full sklearn pipeline (preprocessing + regressor)
    meta.json            - metrics, dropdown options, ranges used by the GUI
"""
import argparse, json, datetime, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

ROOT = Path(__file__).parent
CAT = ["Brand", "Model", "Fuel", "Seller_Type", "Transmission", "Owner"]
NUM = ["Age", "KM_Driven", "Log_KM"]
FEATURES = CAT + NUM


def load(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().dropna()
    ref_year = int(df["Year"].max()) + 1           # age measured relative to newest listing
    df = df[(df["KM_Driven"] < 500_000) & (df["Selling_Price"] > 20_000)]  # drop clear outliers
    return df.reset_index(drop=True), ref_year


def add_features(df, ref_year):
    X = df.copy()
    X["Age"] = ref_year - X["Year"]
    X["Log_KM"] = np.log1p(X["KM_Driven"])
    return X[FEATURES]


def make_pipe(regressor):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    # model log(price): prices are right-skewed, errors become relative
    return Pipeline([("pre", pre),
                     ("reg", TransformedTargetRegressor(regressor, func=np.log1p, inverse_func=np.expm1))])


def make_hgb_pipe():
    """Gradient boosting with native categoricals; the ~1000 car models are target-encoded."""
    low = ["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"]
    pre = ColumnTransformer([
        ("lo", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), low),
        ("te", TargetEncoder(target_type="continuous", smooth=5.0), ["Model"]),
        ("num", "passthrough", NUM)])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20, max_depth=6,
                                        l2_regularization=2.0, categorical_features=[True] * 5 + [False] * (1 + len(NUM)),
                                        random_state=42)
    return Pipeline([("pre", pre), ("reg", TransformedTargetRegressor(reg, func=np.log1p, inverse_func=np.expm1))])


def main(data_path):
    df, ref_year = load(data_path)
    X, y = add_features(df, ref_year), df["Selling_Price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Rows after cleaning: {len(df)}  (train {len(Xtr)}, test {len(Xte)})")

    candidates = {
        "Ridge": lambda: make_pipe(Ridge(alpha=0.3)),
        "Random Forest": lambda: make_pipe(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42)),
        "Gradient Boosting": lambda: make_pipe(GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                                                         max_depth=4, subsample=0.8, random_state=42)),
        "Hist GB (target-encoded)": make_hgb_pipe,
    }
    results, best_name, best_cv = {}, None, -1e9
    for name, build in candidates.items():
        pipe = build()
        cv = cross_val_score(pipe, Xtr, ytr, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_percentage_error", n_jobs=1)
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        results[name] = dict(cv_mape=float(-cv.mean()), test_r2=float(r2_score(yte, p)),
                             test_mae=float(mean_absolute_error(yte, p)),
                             test_mape=float(mean_absolute_percentage_error(yte, p)))
        print(f"{name:18s} CV MAPE={-cv.mean():.1%}  test R2={results[name]['test_r2']:.3f}  "
              f"MAE=Rs {results[name]['test_mae']:,.0f}  MAPE={results[name]['test_mape']:.1%}")
        if cv.mean() > best_cv:
            best_name, best_cv = name, cv.mean()

    print(f"\nBest model: {best_name}")
    best = candidates[best_name]().fit(Xtr, ytr)
    pte = best.predict(Xte)
    resid = np.abs(pte - yte) / yte
    # empirical error band (80% of test predictions fall within this relative error)
    band = float(np.quantile(resid, 0.8))

    # final model: refit on all data for deployment
    final = candidates[best_name]().fit(X, y)

    models_by_brand = {b: sorted(g["Model"].unique().tolist()) for b, g in df.groupby("Brand")}
    meta = dict(
        trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
        best_model=best_name, ref_year=ref_year, rows=len(df),
        metrics=results, test_sample=dict(
            actual=yte.tolist()[:400], predicted=[float(v) for v in pte[:400]]),
        error_band_80=band,
        options=dict(
            brands=sorted(df["Brand"].unique().tolist()), models_by_brand=models_by_brand,
            fuel=sorted(df["Fuel"].unique().tolist()),
            seller=sorted(df["Seller_Type"].unique().tolist()),
            transmission=sorted(df["Transmission"].unique().tolist()),
            owner=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
            year_min=int(df["Year"].min()), year_max=int(df["Year"].max()),
            km_max=int(df["KM_Driven"].max())),
        price_stats=dict(median=float(y.median()), p5=float(y.quantile(.05)), p95=float(y.quantile(.95))),
    )
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "model" / "price_model.joblib", compress=3)
    (ROOT / "model" / "meta.json").write_text(json.dumps(meta))
    print("Saved model/price_model.joblib and model/meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "cardekho.csv"))
    main(ap.parse_args().data)
