"""Volatility-regime detection via a 5-state Gaussian Hidden Markov Model.

Features (all daily, from data/raw/daily/*.parquet, sourced from yfinance):
  - vix          : current CBOE VIX level
  - rv{N}        : realized volatility of the underlying (annualized std of
                   log returns over N trading days) — "historical standard deviation"
  - term_spread  : VIX3M - VIX  (3-month minus spot; <0 = backwardation/stress)
  - macd_runsum  : MACD histogram accumulated within the current sign-leg
                   (resets when the histogram flips sign), divided by that day's
                   close — a price-normalized momentum-of-trend signal

A GaussianHMM (hmmlearn) is fit on the standardized features. Its latent states
are then *ordered by mean VIX* and mapped onto five named regimes:
    Extremely Low, Low, Medium, High, Extremely High.

If hmmlearn is unavailable at runtime, we fall back to a sklearn GaussianMixture
(same features, same ordering) so the tab still works — noted in the returned meta.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAILY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily"

REGIME_NAMES = ["Extremely Low", "Low", "Medium", "High", "Extremely High"]
REGIME_COLORS = {
    "Extremely Low":  "#2166ac",   # deep blue = calm
    "Low":            "#67a9cf",
    "Medium":         "#f7f7f7",
    "High":           "#ef8a62",
    "Extremely High": "#b2182b",   # deep red = panic
}
FEATURES = ["vix", "rv", "term_spread", "macd_runsum"]


@dataclass
class RegimeResult:
    df: pd.DataFrame                    # date-indexed: features + regime, regime_idx, prob_*
    method: str                        # "HMM" or "GaussianMixture (fallback)"
    transmat: np.ndarray | None        # 5x5 ordered transition matrix (HMM only)
    regime_means: pd.DataFrame         # per-regime mean of each feature
    underlying: str
    params: dict = field(default_factory=dict)

    @property
    def current(self) -> str:
        return str(self.df["regime"].iloc[-1])

    @property
    def current_probs(self) -> dict:
        row = self.df.iloc[-1]
        return {n: float(row.get(f"prob_{i}", np.nan))
                for i, n in enumerate(REGIME_NAMES)}


def _load_close(sym: str) -> pd.Series | None:
    p = DAILY_DIR / f"{sym}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)[["timestamp", "close"]].copy()
    d["date"] = (pd.to_datetime(d["timestamp"], utc=True)
                 .dt.tz_localize(None).dt.normalize())
    s = d.groupby("date")["close"].last().sort_index()
    return s


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int = 9) -> pd.Series:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def build_features(underlying: str = "SPY", rv_window: int = 20,
                   macd_fast: int = 10, macd_slow: int = 30) -> pd.DataFrame:
    vix = _load_close("VIX")
    if vix is None:
        raise FileNotFoundError("VIX.parquet not found in data/raw/daily")
    vix3m = _load_close("VIX3M")
    px = _load_close(underlying)
    if px is None:
        raise FileNotFoundError(f"{underlying}.parquet not found in data/raw/daily")

    df = pd.DataFrame(index=vix.index)
    df["vix"] = vix
    # realized (historical) volatility, annualized %
    logret = np.log(px / px.shift(1))
    rv = logret.rolling(rv_window).std() * np.sqrt(252) * 100.0
    df["rv"] = rv.reindex(df.index)
    # term structure: 3M minus spot
    if vix3m is not None:
        df["term_spread"] = (vix3m - vix).reindex(df.index)
    else:
        df["term_spread"] = np.nan
    # MACD histogram accumulated within the current sign-leg: the running total
    # resets to zero each time the histogram flips sign, so a big positive value
    # means a long sustained up-leg and a big negative a deep down-leg. Divided
    # by the day's close to normalize across price levels.
    hist = _macd_hist(px, macd_fast, macd_slow)
    sgn = np.sign(hist)
    leg = (sgn != sgn.shift()).cumsum()            # new id at each sign flip
    leg_cum = hist.groupby(leg).cumsum()
    df["macd_runsum"] = (leg_cum / px).reindex(df.index)

    df = df.dropna(subset=FEATURES)
    return df


def _empirical_transmat(states: np.ndarray, n: int = 5) -> np.ndarray:
    """Maximum-likelihood Markov transition matrix from an observed state path:
    P[i,j] = count(i->j) / count(i). Lets the GaussianMixture fallback (no
    hmmlearn) still provide a transition matrix so forecasts work."""
    T = np.zeros((n, n), dtype=float)
    s = np.asarray(states, dtype=int)
    for a, b in zip(s[:-1], s[1:]):
        T[a, b] += 1.0
    rs = T.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return T / rs


def fit_regimes(underlying: str = "SPY", rv_window: int = 20,
                macd_fast: int = 10, macd_slow: int = 30,
                seed: int = 42) -> RegimeResult:
    df = build_features(underlying, rv_window, macd_fast, macd_slow)
    X = df[FEATURES].to_numpy(dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    method = "HMM"
    transmat = None
    probs = None
    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=5, covariance_type="full",
                            n_iter=300, random_state=seed, tol=1e-3)
        model.fit(Xs)
        raw_states = model.predict(Xs)
        raw_probs = model.predict_proba(Xs)
        transmat_raw = model.transmat_
    except Exception as e:
        logger.warning("hmmlearn unavailable/failed (%s); using GaussianMixture", e)
        from sklearn.mixture import GaussianMixture
        method = "GaussianMixture (fallback)"
        gm = GaussianMixture(n_components=5, covariance_type="full",
                             random_state=seed, n_init=5)
        gm.fit(Xs)
        raw_states = gm.predict(Xs)
        raw_probs = gm.predict_proba(Xs)
        transmat_raw = None

    # Order latent states by mean (raw) VIX -> 0=calmest .. 4=most extreme
    tmp = pd.DataFrame({"state": raw_states, "vix": df["vix"].to_numpy()})
    order = (tmp.groupby("state")["vix"].mean().sort_values().index.tolist())
    remap = {old: new for new, old in enumerate(order)}          # old->rank
    ranks = np.array([remap[s] for s in raw_states])

    out = df.copy()
    out["regime_idx"] = ranks
    out["regime"] = [REGIME_NAMES[r] for r in ranks]
    # reorder posterior probability columns to regime order
    prob_ordered = raw_probs[:, order]
    for i in range(5):
        out[f"prob_{i}"] = prob_ordered[:, i]

    if transmat_raw is not None:
        transmat = transmat_raw[np.ix_(order, order)]
    else:
        # GaussianMixture fallback: estimate the Markov transition matrix from
        # the ordered regime-state path so the forecast + transition matrix work
        # without hmmlearn.
        transmat = _empirical_transmat(ranks, 5)
        method += " + empirical transitions"

    regime_means = (out.groupby("regime")[FEATURES].mean()
                    .reindex(REGIME_NAMES))

    return RegimeResult(
        df=out, method=method, transmat=transmat, regime_means=regime_means,
        underlying=underlying,
        params=dict(rv_window=rv_window, macd_fast=macd_fast,
                    macd_slow=macd_slow, seed=seed),
    )


# Forecast horizons in trading days (1w .. 1y)
FORECAST_HORIZONS = {
    "1w": 5, "2w": 10, "3w": 15, "1m": 21, "2m": 42,
    "3m": 63, "6m": 126, "9m": 189, "1y": 252,
}


def forecast(res: "RegimeResult") -> pd.DataFrame | None:
    """Markov forecast of the regime distribution at each horizon.

    Starts from today's posterior state distribution and propagates it through
    the HMM transition matrix: p(h) = pi0 @ P**h. Adds the most-likely regime
    and the probability-weighted expected VIX per horizon.

    Returns None when no transition matrix is available (GaussianMixture fallback),
    since a Markov forecast needs transition dynamics.

    NOTE: as the horizon grows, P**h converges to the chain's stationary
    distribution, so long-horizon forecasts (6m+) approach the unconditional
    base-rate frequencies rather than a conditioned call.
    """
    if res.transmat is None:
        return None
    P = np.asarray(res.transmat, dtype=float)
    P = P / P.sum(axis=1, keepdims=True)            # guard: rows sum to 1
    pi0 = np.array([res.current_probs[n] for n in REGIME_NAMES], dtype=float)
    if not np.isfinite(pi0).all() or pi0.sum() == 0:
        pi0 = np.zeros(5); pi0[res.df["regime_idx"].iloc[-1]] = 1.0
    pi0 = pi0 / pi0.sum()

    vix_by_regime = res.regime_means["vix"].reindex(REGIME_NAMES).to_numpy(dtype=float)
    rows = {}
    exp_vix = {}
    for label, h in FORECAST_HORIZONS.items():
        dist = pi0 @ np.linalg.matrix_power(P, h)
        rows[label] = dist
        exp_vix[label] = float(np.nansum(dist * vix_by_regime))

    fdf = pd.DataFrame(rows, index=REGIME_NAMES).T           # horizons x regimes
    fdf["most_likely"] = fdf[REGIME_NAMES].idxmax(axis=1)
    fdf["p(most_likely)"] = fdf[REGIME_NAMES].max(axis=1)
    fdf["exp_VIX"] = pd.Series(exp_vix)
    return fdf
