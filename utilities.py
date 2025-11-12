import yfinance as yf
from fredapi import Fred

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, MultipleLocator

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, ConfusionMatrixDisplay
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.inspection import permutation_importance  # (optional: RF only)

from pathlib import Path
from dotenv import load_dotenv
import os



# Exploratory Data Analysis of Input Variables for the Financial Turbulence Model 
# Other pd.DatarFrame data is suitable too
def exploratory_data_analysis_pd_dataframe(data: pd.DataFrame):
    print("The shape:")
    print(data.shape)
    print("\nThe number of na data:")
    print(data.isna().sum())
    print("\nThe Info of the input dataframe:")
    print(data.info())
    print("\nThe numberic profile of the input data:")
    print(data.describe())
    print("\nThe skewness of the input dataframe:")
    print(data.skew())
    print("\nThe kurtosis of the input dataframe:")
    print(data.kurtosis())

# Plot the max drawdown for sp500 in the stage 1 before the k means clustering 
def plot_maxdrawdown_series(
    s,
    title: str = 'US Equity Drawdown',
    ymin: float = -0.45,
    ymax: float = 0.0,
    ystep: float = 0.05,
    year_freq: int = 1,                 # 1 = every year, 2 = every 2 years, etc.
    # date_fmt: str = '1/1/%Y',           # fallback to '01/01/%Y' on Windows if needed
    figsize=(14, 5),
    rotate_labels: int = 90
):
    """
    Plot a drawdown series with yearly x-axis ticks and percent y-axis.

    Parameters
    ----------
    s : pd.Series
        Time series (index must be datetime-like; values in decimal, e.g., -0.25 for -25%).
    title : str
        Plot title.
    ymin, ymax : float
        Y-axis limits.
    ystep : float
        Major tick step on Y axis (in decimal, e.g., 0.05 = 5%).
    year_freq : int
        Year tick spacing (1 = yearly, 2 = every two years, etc.).
    date_fmt : str
        Date label format for major ticks (e.g., '1/1/%Y' or '01/01/%Y').
    figsize : tuple
        Figure size.
    rotate_labels : int
        X tick label rotation in degrees.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    # --- ensure datetime index, sorted, and tz-naive ---
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_convert(None)
    except Exception:
        pass
    s = s.sort_index()

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=2)

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Y-axis as percent
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis='y', alpha=0.35)

    # X-axis: yearly ticks on Jan 1
    ax.set_xlim(s.index.min(), s.index.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_freq))
    # try:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    # except ValueError:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('01/01/%Y'))
    ax.tick_params(axis='x', labelrotation=rotate_labels)

    # Clean frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax





#################### Stacking model in stage 01####################

# -----------------------------
# Utilities used by all models
# -----------------------------

def _ensure_xy(df: pd.DataFrame, target_col: str):
    df = df.sort_index()
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    # basic NA handling like RF style
    X = X.ffill().bfill()
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], y.loc[mask]

def _chron_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

def _format_report(report_dict: dict) -> str:
    """Turn classification_report(output_dict=True) into a neat, sklearn-like text block."""
    # Keep common keys if present
    lines = []
    headers = ["precision", "recall", "f1-score", "support"]
    # Per-class
    classes = [k for k in report_dict.keys() if k.isdigit()]
    for cls in sorted(classes, key=lambda z: int(z)):
        row = report_dict[cls]
        lines.append(f"{cls:>7}  {row['precision']:.3f}  {row['recall']:.3f}  {row['f1-score']:.3f}  {int(row['support']):>6}")
    # Accuracy (if present)
    if 'accuracy' in report_dict:
        lines.append(f"{'accuracy':>7}  {'':>5}  {'':>5}  {report_dict['accuracy']:.3f}  {int(sum(report_dict[c]['support'] for c in classes)):>6}")
    # Macro/weighted
    for agg in ['macro avg', 'weighted avg']:
        if agg in report_dict:
            row = report_dict[agg]
            lines.append(f"{agg:>7}  {row['precision']:.3f}  {row['recall']:.3f}  {row['f1-score']:.3f}  {int(row['support']):>6}")
    # Header
    head = f"{'':>7}  {'precision'}  {'recall'}  {'f1-score'}  {'support'}"
    return "Classification report (test):\n" + head + "\n" + "\n".join(lines)

def _plot_confusion_matrix(cm, y_test, title):
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def _compute_plot_roc(y_test, proba, classes, title="ROC Curve (Test)"):
    """
    Binary: plot standard ROC and return AUC.
    Multiclass: plot macro-averaged ROC (OvR) and return macro AUC (ovr).
    """
    classes = np.array(classes)
    if len(classes) == 2:
        # find column for positive class = max label
        pos_idx = np.where(classes == classes.max())[0][0]
        fpr, tpr, _ = roc_curve(y_test, proba[:, pos_idx])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0,1],[0,1],'k--', lw=1)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(title); plt.legend(loc='lower right')
        plt.tight_layout(); plt.show()
        return roc_auc
    else:
        # multiclass OvR macro-average AUC
        Y = pd.get_dummies(y_test, columns=classes).reindex(columns=classes, fill_value=0).values
        # align proba columns to classes
        roc_auc = roc_auc_score(Y, proba[:, np.searchsorted(classes, classes)], multi_class='ovr', average='macro')
        # (Optional) just plot a diagonal + title with macro AUC for simplicity
        plt.figure(figsize=(6,4))
        plt.plot([0,1],[0,1],'k--', lw=1)
        plt.title(f"{title} (macro OvR AUC={roc_auc:.3f})")
        plt.tight_layout(); plt.show()
        return roc_auc

def _print_header(best_params, best_cv_score, report_text, head_df):
    print(f"Best params: {best_params}")
    if best_cv_score is not None:
        print(f"Best CV score (F1): {best_cv_score:.12f}")
    print(report_text)
    print("\nHead of prediction table:")
    print(head_df)

    # print("\nWhat is this output?\n"
    #       "The classification report summarizes precision/recall/F1 and support per class on the test set.\n"
    #       "The confusion matrix (figure) shows how predicted labels compare with actual labels.\n"
    #       "The ROC curve (figure) visualizes the true-positive/false-positive trade-off; its AUC condenses performance into one number.\n"
    #       "The prediction table* lists, by timestamp, the actual regime, the model’s predicted regime, and the model-estimated probability(s).")

def _build_pred_table(index, y_actual, y_pred, proba, classes):
    out = pd.DataFrame(index=index.copy())
    out["regime_actual"] = y_actual.values
    out["regime_pred"]   = y_pred
    classes = np.array(classes)
    if len(classes) == 2:
        # probability of class 1 (assume max label is positive)
        pos_idx = np.where(classes == classes.max())[0][0]
        out["regime_prob1"] = proba[:, pos_idx]
    else:
        # one column per class
        for i, c in enumerate(classes):
            out[f"prob_{c}"] = proba[:, i]
    return out

# =========================================================
# 1) RANDOM-FOREST (supports binary or multiclass regimes)
# =========================================================
def train_rf_regime_classifier(
    df: pd.DataFrame,
    target_col: str = "regime",
    test_size: float = 0.2,
    n_splits: int = 5,
    scoring: str = "f1_macro",
    tune: bool = True,
    param_grid: dict | None = None,
    random_state: int = 42,
    plot: bool = True
) -> dict:
    X, y = _ensure_xy(df, target_col)
    X_train, X_test, y_train, y_test = _chron_split(X, y, test_size)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, weights)}

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight
        ))
    ])

    if param_grid is None:
        param_grid = {
            'rf__n_estimators': [150, 300, 600],
            'rf__max_depth'   : [4, 8, 12, 16, 24],
            'rf__min_samples_leaf': [1, 2, 3, 4, 5],
            'rf__max_features': ['sqrt', 'log2'],
            'rf__bootstrap'   : [True]
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    if tune:
        search = GridSearchCV(pipe, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
    else:
        model = pipe.fit(X_train, y_train)
        best_params = {}
        best_cv_score = None

    y_pred  = model.predict(X_test)
    rf      = model.named_steps['rf']
    proba   = model.predict_proba(X_test)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, output_dict=True, digits=4)
    report_text = _format_report(report)

    # Feature importances (Gini)
    gini_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    roc_auc = None
    if plot:
        _plot_confusion_matrix(cm, y_test, "Confusion Matrix (RF, Test)")
        roc_auc = _compute_plot_roc(y_test, proba, rf.classes_, "ROC Curve (Test, RF)")

    # Build standardized prediction table
    proba_reordered = proba[:, np.searchsorted(rf.classes_, np.unique(y_test))]
    pred_table = _build_pred_table(X_test.index, y_test, y_pred, proba_reordered, np.unique(y_test))

    # Print header & sample
    _print_header(best_params, best_cv_score, report_text, pred_table.head())

    return {
        "model": model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "report_dict": report,
        "report_text": report_text,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_test": y_test,
        "y_proba": proba_reordered,
        "pred_table": pred_table,
        "gini_importances": gini_importances,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test_series": y_test,
        "classes": rf.classes_
    }

# =========================================================
# 2) GAUSSIAN NAIVE BAYES (binary by default; multiclass OK)
# =========================================================
def train_gnb_regime_classifier(
    df: pd.DataFrame,
    target_col: str = "regime",
    test_size: float = 0.2,
    n_splits: int = 5,
    scoring: str = "f1",
    tune: bool = True,
    param_grid: dict | None = None,
    plot: bool = True
) -> dict:
    X, y = _ensure_xy(df, target_col)
    X_train, X_test, y_train, y_test = _chron_split(X, y, test_size)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('gnb', GaussianNB())
    ])

    if param_grid is None:
        param_grid = {
            'gnb__var_smoothing': np.logspace(-12, -6, 7),
            'gnb__priors': [None]  # keep API; explicit [0.5,0.5] optional
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    if tune:
        search = GridSearchCV(pipe, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
    else:
        model = pipe.fit(X_train, y_train)
        best_params = {}
        best_cv_score = None

    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    report_text = _format_report(report)

    roc_auc = None
    if plot:
        _plot_confusion_matrix(cm, y_test, "Confusion Matrix (GNB, Test)")
        roc_auc = _compute_plot_roc(y_test, proba, model.named_steps['gnb'].classes_, "ROC Curve (Test, GNB)")

    # Standardized prediction table
    gnb = model.named_steps['gnb']
    proba_reordered = proba[:, np.searchsorted(gnb.classes_, np.unique(y_test))]
    pred_table = _build_pred_table(X_test.index, y_test, y_pred, proba_reordered, np.unique(y_test))

    _print_header(best_params, best_cv_score, report_text, pred_table.head())

    return {
        "model": model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "report_dict": report,
        "report_text": report_text,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_test": y_test,
        "y_proba": proba_reordered,
        "pred_table": pred_table,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test_series": y_test,
        "classes": gnb.classes_
    }

# =========================================================
# 3) SVC (scaled; kernel/soft-margin tuning; binary/multiclass)
# =========================================================
def train_svc_regime_classifier(
    df: pd.DataFrame,
    target_col: str = "regime",
    test_size: float = 0.2,
    n_splits: int = 5,
    scoring: str = "f1",
    tune: bool = True,
    param_grid: dict | None = None,
    plot: bool = True,
    random_state: int = 42
) -> dict:
    X, y = _ensure_xy(df, target_col)
    X_train, X_test, y_train, y_test = _chron_split(X, y, test_size)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('svc', SVC(probability=True, class_weight='balanced', random_state=random_state))
    ])

    if param_grid is None:
        param_grid = [
            {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 5, 10]},
            {'svc__kernel': ['rbf'],    'svc__C': [0.1, 1, 5, 10], 'svc__gamma': ['scale', 'auto', 0.1, 0.01, 0.001]},
            {'svc__kernel': ['poly'],   'svc__C': [0.1, 1, 5],     'svc__gamma': ['scale', 'auto', 0.1, 0.01],
             'svc__degree': [2, 3, 4], 'svc__coef0': [0.0, 0.5, 1.0]},
            {'svc__kernel': ['sigmoid'],'svc__C': [0.1, 1, 5, 10], 'svc__gamma': ['scale', 'auto', 0.1, 0.01],
             'svc__coef0': [0.0, 0.5, 1.0]}
        ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    if tune:
        search = GridSearchCV(pipe, param_grid, cv=tscv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
        best_cv_score = search.best_score_
    else:
        model = pipe.fit(X_train, y_train)
        best_params = {}
        best_cv_score = None

    y_pred = model.predict(X_test)
    proba  = model.predict_proba(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    report_text = _format_report(report)

    roc_auc = None
    if plot:
        _plot_confusion_matrix(cm, y_test, "Confusion Matrix (SVC, Test)")
        roc_auc = _compute_plot_roc(y_test, proba, model.named_steps['svc'].classes_, "ROC Curve (Test, SVC)")

    svc = model.named_steps['svc']
    proba_reordered = proba[:, np.searchsorted(svc.classes_, np.unique(y_test))]
    pred_table = _build_pred_table(X_test.index, y_test, y_pred, proba_reordered, np.unique(y_test))

    _print_header(best_params, best_cv_score, report_text, pred_table.head())

    return {
        "model": model,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "report_dict": report,
        "report_text": report_text,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_test": y_test,
        "y_proba": proba_reordered,
        "pred_table": pred_table,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test_series": y_test,
        "classes": svc.classes_
    }

# =========================================================
# 4) TIME-SERIES-SAFE STACKING (RF/SVC/GNB -> LR meta)
# =========================================================
def _ts_default_grids():
    return {
        "rf":  {"rf__n_estimators":[200,400,800], "rf__max_depth":[None,8,12,16,24],
                "rf__min_samples_leaf":[1,2,4], "rf__max_features":["sqrt","log2",0.5], "rf__bootstrap":[True]},
        "svc": {"svc__kernel":["rbf","linear"], "svc__C":[0.1,1,3,10], "svc__gamma":["scale",0.1,0.01,0.001]},
        "gnb": {"gnb__var_smoothing": np.logspace(-12,-7,6)},
        "meta":{"C":[0.1,1,3,10]},
    }

def _tune_with_tscv(pipe, grid, X, y, n_splits=5, scoring="f1_macro"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(estimator=pipe, param_grid=grid, cv=tscv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_

def train_ts_safe_stacking(
    df: pd.DataFrame,
    target_col: str = "regime",
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: str = "f1_macro",
    tune_base: bool = True,
    tune_meta: bool = True,
    param_grids: dict | None = None,
    plot: bool = True
) -> dict:
    # 1) data
    X, y = _ensure_xy(df, target_col)
    X_tr, X_te, y_tr, y_te = _chron_split(X, y, test_size)

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {c: w for c, w in zip(classes, weights)}
    is_binary = len(classes) == 2

    # 2) base models
    rf_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1, class_weight=class_weight))
    ])
    svc_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state))
    ])
    gnb_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gnb", GaussianNB(var_smoothing=1e-9))
    ])
    base_models = [("rf", rf_pipe), ("svc", svc_pipe), ("gnb", gnb_pipe)]

    # 3) tune base
    defaults = _ts_default_grids()
    grids = defaults if param_grids is None else {**defaults, **param_grids}
    tuned_info = {}
    tuned_models = []
    for name, pipe in base_models:
        if tune_base:
            best_est, best_params, best_cv_score = _tune_with_tscv(pipe, grids[name], X_tr, y_tr, n_splits=n_splits, scoring=scoring)
            tuned_info[name] = {"best_params": best_params, "best_cv_score": best_cv_score}
            tuned_models.append((name, best_est))
        else:
            tuned_models.append((name, pipe))
    base_models = tuned_models

    # 4) OOF generation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_list, oof_index = [], []
    for (tr_idx, val_idx) in tscv.split(X_tr, y_tr):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
        fold_feats = []
        for name, mdl in base_models:
            mdl.fit(X_tr_f, y_tr_f)
            proba = mdl.predict_proba(X_va_f)
            # align class order
            idx = np.searchsorted(mdl.classes_, classes)
            proba = proba[:, idx]
            if is_binary:
                # keep positive class column only (max label)
                pos_idx = np.where(classes == classes.max())[0][0]
                proba = proba[:, [pos_idx]]
            fold_feats.append(proba)
        fold_meta = np.hstack(fold_feats)
        oof_list.append(fold_meta)
        oof_index.append(val_idx)

    n_meta_cols = sum([(1 if is_binary else len(classes)) for _ in base_models])
    oof_meta = np.zeros((len(X_tr), n_meta_cols), dtype=float)
    oof_mask = np.zeros(len(X_tr), dtype=bool)
    for meta_block, val_idx in zip(oof_list, oof_index):
        oof_meta[val_idx, :] = meta_block
        oof_mask[val_idx] = True

    X_meta_train = oof_meta[oof_mask]
    y_meta_train = y_tr.iloc[oof_mask]

    # 5) meta learner
    meta = LogisticRegression(multi_class="auto", class_weight="balanced", max_iter=500, random_state=random_state, solver="lbfgs", C=1.0)
    meta_tune_info = {}
    if tune_meta:
        meta_cv = KFold(n_splits=min(5, len(np.unique(y_meta_train))), shuffle=False)
        gs_meta = GridSearchCV(meta, {"C": grids["meta"]["C"]}, cv=meta_cv, scoring=scoring, n_jobs=-1, refit=True, verbose=0)
        gs_meta.fit(X_meta_train, y_meta_train)
        meta = gs_meta.best_estimator_
        meta_tune_info = {"best_params": gs_meta.best_params_, "best_cv_score": gs_meta.best_score_}
    else:
        meta.fit(X_meta_train, y_meta_train)

    # 6) refit bases on full train & build test meta features
    test_feats, fitted_bases = [], []
    for name, mdl in base_models:
        mdl.fit(X_tr, y_tr)
        fitted_bases.append((name, mdl))
        proba_te = mdl.predict_proba(X_te)
        idx = np.searchsorted(mdl.classes_, classes)
        proba_te = proba_te[:, idx]
        if is_binary:
            pos_idx = np.where(classes == classes.max())[0][0]
            proba_te = proba_te[:, [pos_idx]]
        test_feats.append(proba_te)

    X_meta_test = np.hstack(test_feats)
    y_pred = meta.predict(X_meta_test)

    # 7) metrics & printing
    report = classification_report(y_te, y_pred, output_dict=True, digits=4)
    report_text = _format_report(report)
    cm = confusion_matrix(y_te, y_pred)

    # meta probabilities aligned to classes
    proba_meta = meta.predict_proba(X_meta_test)
    idx = np.searchsorted(meta.classes_, classes)
    proba_meta = proba_meta[:, idx]

    roc_auc = None
    if plot:
        _plot_confusion_matrix(cm, y_te, "Confusion Matrix (Stacking, Test)")
        roc_auc = _compute_plot_roc(y_te, proba_meta, classes, "ROC Curve (Test, Stacking)")

    # standardized prediction table
    pred_table = _build_pred_table(X_te.index, y_te, y_pred, proba_meta, np.unique(y_te))

    # summarize “best” info at top-level for convenience
    # choose meta best if tuned, else None
    best_params = {"meta": meta_tune_info.get("best_params")} if meta_tune_info else {}
    best_cv_score = meta_tune_info.get("best_cv_score", None)

    _print_header(best_params, best_cv_score, report_text, pred_table.head())

    return {
        "base_models": fitted_bases,
        "meta_model": meta,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "report_dict": report,
        "report_text": report_text,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_test": y_te,
        "y_proba": proba_meta,
        "pred_table": pred_table,
        "classes": classes,
        "tuned_info": tuned_info,
        "meta_tune_info": meta_tune_info,
        "oof_mask_ratio": oof_mask.mean()
    }
