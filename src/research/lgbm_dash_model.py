"""LightGBM Signal Dashboard — per-symbol fwd-return + P(positive) models.

For each (symbol, horizon) we train two models on the daily feature panel:
  - Regressor   → predicts fwd log-return
  - Classifier  → predicts P(fwd > 0)

Walk-forward validation gives honest OOS metrics (IC, AUC, hit-rate). Final
production model is fit on full history for live inference.

Cache layout: models/lgbm_dash/{SYM}.pkl (single pickle per symbol holds all
horizons + meta). One train pass for a symbol = ~30-90s on a typical laptop.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, r2_score, brier_score_loss, log_loss

from src.research.feature_panel import (
    load_panel, FEATURES, FWD_HORIZONS, available_symbols,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "lgbm_dash"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---- LightGBM hyperparameters ----
# Lowered complexity vs v1 (num_leaves 63->15, min_child 80->200, depth cap)
# because daily bars with multi-month overlapping forward targets give very few
# *effective* independent samples. A deep tree memorizes -> jumpy predictions.
# Huber loss on the regressor curbs magnitude overshoot (the negative-R2 problem).
REG_PARAMS = dict(
    n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
    min_child_samples=200, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=0.7,
    objective="huber", alpha=0.9, verbose=-1, n_jobs=-1,
)
CLF_PARAMS = dict(
    n_estimators=300, learning_rate=0.02, num_leaves=15, max_depth=4,
    min_child_samples=200, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=0.7,
    objective="binary", verbose=-1, n_jobs=-1,
)

# Number of seed-bagged models averaged for the FINAL (live-inference) model.
# Bagging reduces prediction variance -> smoother day-to-day signal.
N_BAG = 7

# Convert a forward horizon in trading-day bars to a calendar-day embargo gap
# inserted between train-end and test-start in walk-forward, so training targets
# never overlap the test window (purged CV, Lopez de Prado).
def _embargo_days(h_bars: int) -> int:
    return int(round(h_bars * 365.0 / 252.0)) + 2


def _fit_bag_reg(X, y, n: int = N_BAG) -> list:
    """Fit N seed-bagged regressors. Averaging their output reduces variance."""
    models = []
    for s in range(n):
        p = {**REG_PARAMS, "random_state": s, "bagging_seed": s,
             "feature_fraction_seed": s + 100}
        models.append(lgb.LGBMRegressor(**p).fit(X, y))
    return models


def _fit_bag_clf(X, y, n: int = N_BAG) -> list:
    models = []
    for s in range(n):
        p = {**CLF_PARAMS, "random_state": s, "bagging_seed": s,
             "feature_fraction_seed": s + 100}
        models.append(lgb.LGBMClassifier(**p).fit(X, y))
    return models


def _bag_predict_reg(models: list, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def _bag_predict_proba(models: list, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def _avg_importance(models: list, feats: list) -> dict:
    if not models:
        return {}
    arr = np.mean([m.feature_importances_ for m in models], axis=0)
    return dict(zip(feats, arr.tolist()))


@dataclass
class HorizonModel:
    horizon: str
    bars: int
    reg_models: list = field(default_factory=list)   # list[LGBMRegressor] (bagged)
    clf_models: list = field(default_factory=list)   # list[LGBMClassifier] (bagged)
    # Regressor metrics
    ic_oos: float = float("nan")           # Spearman(reg_pred, fwd_ret)
    r2_oos: float = float("nan")           # r2_score(fwd_ret, reg_pred)
    hit_oos: float = float("nan")          # sign(reg_pred)==sign(fwd_ret)
    # Classifier metrics
    auc_oos: float = float("nan")          # ROC AUC of P(+)
    brier_oos: float = float("nan")        # mean((P(+) - y)^2), lower better
    ic_clf_oos: float = float("nan")       # Spearman(P(+), fwd_ret)
    hit_clf_oos: float = float("nan")      # ((P(+)>0.5)==(fwd>0)).mean()
    log_loss_oos: float = float("nan")     # cross-entropy, lower better
    pseudo_r2_oos: float = float("nan")    # McFadden: 1 - LL_model/LL_baseline
    n_oos: int = 0
    n_train: int = 0
    feature_importance_reg: dict[str, float] = field(default_factory=dict)
    feature_importance_clf: dict[str, float] = field(default_factory=dict)


@dataclass
class SymbolBundle:
    symbol: str
    trained_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    features: list[str] = field(default_factory=list)
    horizons: dict[str, HorizonModel] = field(default_factory=dict)
    panel_rows: int = 0
    panel_start: str = ""
    panel_end: str = ""


def _walk_forward_oos(
    df: pd.DataFrame, target_col: str, classifier: bool,
    embargo_days: int,
    warmup_days: int = 180 * 6, step_days: int = 60,  # ≈6y warmup, 60d step
    expanding: bool = True,
) -> tuple[pd.Series, dict]:
    """Purged walk-forward predictions on daily bars.

    A gap of `embargo_days` is inserted between train-end and test-start so that
    training samples (whose forward target reaches `embargo_days` ahead) never
    overlap the test window. Without this, the last horizon's worth of training
    rows leak future info into the test fold and inflate OOS metrics.

    `expanding=True` trains on ALL history up to the embargo boundary (averages
    over multiple macro cycles -> avoids the stale-regime inversion a short
    rolling window suffers at long horizons). `warmup_days` is the minimum
    history before the first test fold.
    """
    df = df.sort_values("date").reset_index(drop=True)
    times = pd.to_datetime(df["timestamp"], utc=True)
    start_day = times.min().floor("D")
    end_day = times.max().floor("D")
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    embargo = pd.Timedelta(days=embargo_days)

    cur = start_day + pd.Timedelta(days=warmup_days)
    folds = 0
    while cur < end_day:
        # Train ends embargo days BEFORE the test window opens (purge overlap).
        lo = start_day if expanding else cur - pd.Timedelta(days=warmup_days)
        train_mask = (times >= lo) & (times < cur - embargo)
        test_end = cur + pd.Timedelta(days=step_days)
        test_mask = (times >= cur) & (times < test_end)
        train = df.loc[train_mask].dropna(subset=FEATURES + [target_col])
        test = df.loc[test_mask].dropna(subset=FEATURES)
        if len(train) < 200 or len(test) == 0:
            cur = test_end; continue
        if classifier:
            y = (train[target_col] > 0).astype(int)
            if y.nunique() < 2:
                cur = test_end; continue
            m = lgb.LGBMClassifier(**CLF_PARAMS)
            m.fit(train[FEATURES], y)
            preds.loc[test.index] = m.predict_proba(test[FEATURES])[:, 1]
        else:
            m = lgb.LGBMRegressor(**REG_PARAMS)
            m.fit(train[FEATURES], train[target_col])
            preds.loc[test.index] = m.predict(test[FEATURES])
        folds += 1
        cur = test_end

    metrics = {"folds": folds, "n_oos": int(preds.notna().sum())}
    return preds, metrics


def train_symbol(
    symbol: str,
    warmup_days: int = 180 * 6,  # min history before first OOS fold (expanding)
    step_days: int = 60,
) -> SymbolBundle:
    df = load_panel([symbol])
    if df.empty:
        raise ValueError(f"no panel data for {symbol}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    bundle = SymbolBundle(
        symbol=symbol, features=list(FEATURES),
        panel_rows=len(df),
        panel_start=str(df["date"].min()),
        panel_end=str(df["date"].max()),
    )

    for h_label, h_bars in FWD_HORIZONS.items():
        target = f"fwd_{h_label}"
        if target not in df.columns:
            continue
        embargo_days = _embargo_days(h_bars)
        # Walk-forward OOS (purged, expanding window)
        preds_reg, _ = _walk_forward_oos(df, target, classifier=False,
                                         embargo_days=embargo_days,
                                         warmup_days=warmup_days, step_days=step_days)
        preds_clf, _ = _walk_forward_oos(df, target, classifier=True,
                                         embargo_days=embargo_days,
                                         warmup_days=warmup_days, step_days=step_days)
        oos = df.assign(pred_reg=preds_reg, pred_clf=preds_clf).dropna(
            subset=["pred_reg", "pred_clf", target])
        if len(oos) < 50:
            logger.warning("skip %s %s: OOS sparse n=%d", symbol, h_label, len(oos))
            continue
        ic = float(spearmanr(oos["pred_reg"], oos[target]).statistic)
        try:
            r2 = float(r2_score(oos[target], oos["pred_reg"]))
        except Exception:
            r2 = float("nan")
        y_true = (oos[target] > 0).astype(int)
        try:
            auc = float(roc_auc_score(y_true, oos["pred_clf"]))
        except Exception:
            auc = float("nan")
        try:
            brier = float(brier_score_loss(y_true, oos["pred_clf"]))
        except Exception:
            brier = float("nan")
        hit = float(((oos["pred_reg"] > 0).astype(int) == y_true).mean())

        # Classifier-specific extras
        try:
            ic_clf = float(spearmanr(oos["pred_clf"], oos[target]).statistic)
        except Exception:
            ic_clf = float("nan")
        try:
            hit_clf = float(((oos["pred_clf"] > 0.5).astype(int) == y_true).mean())
        except Exception:
            hit_clf = float("nan")
        try:
            ll = float(log_loss(y_true, oos["pred_clf"].clip(1e-6, 1 - 1e-6)))
        except Exception:
            ll = float("nan")
        try:
            base = float(y_true.mean())
            ll_base = float(log_loss(y_true, np.full_like(y_true, base, dtype=float)))
            pseudo_r2 = float(1 - (ll / ll_base)) if ll_base > 0 else float("nan")
        except Exception:
            pseudo_r2 = float("nan")

        # Final models on FULL history — seed-bagged for low-variance live signal
        final = df.dropna(subset=FEATURES + [target])
        reg_models = _fit_bag_reg(final[FEATURES], final[target])
        y_full = (final[target] > 0).astype(int)
        clf_models = (_fit_bag_clf(final[FEATURES], y_full)
                      if y_full.nunique() == 2 else [])

        fi_reg = _avg_importance(reg_models, FEATURES)
        fi_clf = _avg_importance(clf_models, FEATURES)

        bundle.horizons[h_label] = HorizonModel(
            horizon=h_label, bars=h_bars,
            reg_models=reg_models, clf_models=clf_models,
            ic_oos=ic, r2_oos=r2, hit_oos=hit,
            auc_oos=auc, brier_oos=brier,
            ic_clf_oos=ic_clf, hit_clf_oos=hit_clf,
            log_loss_oos=ll, pseudo_r2_oos=pseudo_r2,
            n_oos=int(len(oos)), n_train=int(len(final)),
            feature_importance_reg=fi_reg,
            feature_importance_clf=fi_clf,
        )
        logger.info("%s %s  IC=%+.4f  R²=%+.4f  AUC=%.3f  hit=%.3f  brier=%.4f  n_oos=%d  n_train=%d",
                    symbol, h_label, ic, r2, auc, hit, brier, len(oos), len(final))
    save_bundle(bundle)
    return bundle


def _bundle_to_dict(b: SymbolBundle) -> dict:
    return {
        "symbol": b.symbol,
        "trained_at": b.trained_at,
        "features": list(b.features),
        "panel_rows": b.panel_rows,
        "panel_start": b.panel_start,
        "panel_end": b.panel_end,
        "horizons": {
            h: {
                "horizon": m.horizon, "bars": m.bars,
                "reg_models": m.reg_models, "clf_models": m.clf_models,
                "ic_oos": m.ic_oos, "r2_oos": m.r2_oos, "hit_oos": m.hit_oos,
                "auc_oos": m.auc_oos, "brier_oos": m.brier_oos,
                "ic_clf_oos": m.ic_clf_oos, "hit_clf_oos": m.hit_clf_oos,
                "log_loss_oos": m.log_loss_oos, "pseudo_r2_oos": m.pseudo_r2_oos,
                "n_oos": m.n_oos, "n_train": m.n_train,
                "feature_importance_reg": dict(m.feature_importance_reg),
                "feature_importance_clf": dict(m.feature_importance_clf),
            }
            for h, m in b.horizons.items()
        },
    }


def _dict_to_bundle(d: dict) -> SymbolBundle:
    bundle = SymbolBundle(
        symbol=d["symbol"], trained_at=d.get("trained_at", pd.Timestamp.utcnow()),
        features=list(d.get("features", [])),
        panel_rows=int(d.get("panel_rows", 0)),
        panel_start=str(d.get("panel_start", "")),
        panel_end=str(d.get("panel_end", "")),
    )
    for h, mh in d.get("horizons", {}).items():
        # Backward-compat: old pickles stored single reg_model/clf_model.
        reg_models = mh.get("reg_models")
        if reg_models is None:
            reg_models = [mh["reg_model"]] if mh.get("reg_model") is not None else []
        clf_models = mh.get("clf_models")
        if clf_models is None:
            clf_models = [mh["clf_model"]] if mh.get("clf_model") is not None else []
        bundle.horizons[h] = HorizonModel(
            horizon=mh["horizon"], bars=mh["bars"],
            reg_models=reg_models, clf_models=clf_models,
            ic_oos=mh.get("ic_oos", float("nan")),
            r2_oos=mh.get("r2_oos", float("nan")),
            hit_oos=mh.get("hit_oos", float("nan")),
            auc_oos=mh.get("auc_oos", float("nan")),
            brier_oos=mh.get("brier_oos", float("nan")),
            ic_clf_oos=mh.get("ic_clf_oos", float("nan")),
            hit_clf_oos=mh.get("hit_clf_oos", float("nan")),
            log_loss_oos=mh.get("log_loss_oos", float("nan")),
            pseudo_r2_oos=mh.get("pseudo_r2_oos", float("nan")),
            n_oos=int(mh.get("n_oos", 0)),
            n_train=int(mh.get("n_train", 0)),
            feature_importance_reg=dict(mh.get("feature_importance_reg", {})),
            feature_importance_clf=dict(mh.get("feature_importance_clf", {})),
        )
    return bundle


def save_bundle(bundle: SymbolBundle) -> Path:
    out = MODELS_DIR / f"{bundle.symbol}.pkl"
    with open(out, "wb") as f:
        pickle.dump(_bundle_to_dict(bundle), f)
    return out


def load_bundle(symbol: str) -> SymbolBundle | None:
    p = MODELS_DIR / f"{symbol}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return _dict_to_bundle(obj)
    # Legacy dataclass pickle (from old runs) — drop & retrain
    return None


def is_trained(symbol: str) -> bool:
    return (MODELS_DIR / f"{symbol}.pkl").exists()


def list_trained() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.pkl"))


def predict_current(bundle: SymbolBundle, current_row: pd.Series) -> pd.DataFrame:
    """Return per-horizon (exp_log_return, exp_pct, p_positive)."""
    rows = []
    feats = bundle.features
    X = pd.DataFrame([{f: float(current_row[f]) if not pd.isna(current_row.get(f))
                       else np.nan for f in feats}])
    for h, m in bundle.horizons.items():
        if not m.reg_models: continue
        log_ret = float(_bag_predict_reg(m.reg_models, X)[0])
        p_pos = float(_bag_predict_proba(m.clf_models, X)[0]) if m.clf_models else float("nan")
        rows.append({
            "horizon": h,
            "bars": m.bars,
            "exp_log_ret": log_ret,
            "exp_pct": (np.exp(log_ret) - 1) * 100,
            "p_positive": p_pos,
            "ic_oos": m.ic_oos,
            "r2_oos": m.r2_oos,
            "auc_oos": m.auc_oos,
            "hit_oos": m.hit_oos,
            "brier_oos": m.brier_oos,
            "ic_clf_oos": m.ic_clf_oos,
            "hit_clf_oos": m.hit_clf_oos,
            "log_loss_oos": m.log_loss_oos,
            "pseudo_r2_oos": m.pseudo_r2_oos,
            "n_oos": m.n_oos,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["SPY"])
    ap.add_argument("--warmup-days", type=int, default=180 * 6)
    ap.add_argument("--step-days", type=int, default=60)
    args = ap.parse_args()
    syms = args.symbols if args.symbols else ["SPY"]
    for s in syms:
        try:
            b = train_symbol(s, warmup_days=args.warmup_days, step_days=args.step_days)
            print(f"\n=== {s} trained, {len(b.horizons)} horizons, panel {b.panel_start}..{b.panel_end} ===")
            for h, m in b.horizons.items():
                print(f"  {h}: IC={m.ic_oos:+.4f}  AUC={m.auc_oos:.3f}  hit={m.hit_oos:.3f}  n_oos={m.n_oos}")
        except Exception as e:
            logger.exception("train failed for %s: %s", s, e)
