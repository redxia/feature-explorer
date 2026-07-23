"""Daily feature panel + forward returns, for the feature_explorer UI.

Loads daily bars from data/raw/daily/{SYMBOL}.parquet, computes a wide set of
daily technical features, and joins forward log-returns at common research
horizons (5/10/15/21/42/63/126/189/252 trading days = 1w/2w/3w/1m/2m/3m/6m/9m/12m).

Returns one DataFrame indexed by (symbol, date), columns = features + fwd_*.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DAILY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily"

# Forward-return horizons in trading days
FWD_HORIZONS = {
    "1w":  5,
    "2w":  10,
    "3w":  15,
    "1m":  21,
    "2m":  42,
    "3m":  63,
    "6m":  126,
    "9m":  189,
    "12m": 252,
}


# Feature catalogue (display name -> column key)
FEATURES = [
    "rsi_14",
    "rsi_5",
    "days_os_20",  # # of days RSI<30 in last 20 trading days (capitulation count)
    "atr_pct_14",
    "dist_sma20_pct",
    "dist_sma50_pct",
    "dist_sma200_pct",
    "sma50_sma200_ratio",
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",
    "log_ret_60",
    "log_ret_252",
    "realized_vol_20",
    "realized_vol_60",
    "vol_ratio_20",
    "obv_60d_chg",
    "bb_width_20",
    "bb_pct_b_20",
    "macd_hist",
    "drawdown_from_252h",
    "days_since_5pct_dd",  # trading days since last 5% drawdown — complacency clock
    "month_of_year",
    "dow",
    # ---- VIX-derived (cross-source: CBOE Volatility Index) ----
    "vix",
    "vix_change_5d",
    "vix_pct_rank_252",
    "vix_minus_rv20",
    # ---- VIX3M term-structure (3-month VIX) ----
    "vix3m_over_vix",     # >1 = contango (calm), <1 = backwardation (fear)
    # ---- Macro/credit/rates regime ----
    "vvix",                # vol of VIX, "fear of fear"
    "hyg_lqd_log",         # log(HYG/LQD), credit risk appetite
    "term_spread_10y_3m",  # 10Y - 3M Treasury yield spread (yield curve slope)
]


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr / close


_VIX_CACHE: pd.DataFrame | None = None


def _load_vix() -> pd.DataFrame | None:
    """Lazy-load VIX + VIX3M daily close. Returns df keyed by NY date.
    Columns: vix, vix_change_5d, vix_pct_rank_252, vix3m, vix3m_over_vix, vix3m_minus_vix.
    """
    global _VIX_CACHE
    if _VIX_CACHE is not None:
        return _VIX_CACHE
    p = DAILY_DIR / "VIX.parquet"
    if not p.exists():
        return None
    v = pd.read_parquet(p)[["timestamp", "close"]].rename(columns={"close": "vix"})
    v["timestamp"] = pd.to_datetime(v["timestamp"], utc=True)
    v["date"] = v["timestamp"].dt.tz_convert("UTC").dt.date
    v = v.sort_values("date").reset_index(drop=True)
    v["vix_change_5d"] = v["vix"].diff(5)
    v["vix_pct_rank_252"] = v["vix"].rolling(252).rank(pct=True)
    out = v[["date", "vix", "vix_change_5d", "vix_pct_rank_252"]].copy()

    # VIX3M optional — merge if cached
    p3 = DAILY_DIR / "VIX3M.parquet"
    if p3.exists():
        v3 = pd.read_parquet(p3)[["timestamp", "close"]].rename(columns={"close": "vix3m"})
        v3["date"] = pd.to_datetime(v3["timestamp"], utc=True).dt.tz_convert(
            "UTC").dt.date
        v3 = v3[["date", "vix3m"]]
        out = out.merge(v3, on="date", how="left")
        out["vix3m_over_vix"] = out["vix3m"] / out["vix"]
        out["vix3m_minus_vix"] = out["vix3m"] - out["vix"]
    else:
        out["vix3m"] = np.nan
        out["vix3m_over_vix"] = np.nan
        out["vix3m_minus_vix"] = np.nan

    # VVIX (vol of VIX) — optional
    p_vvix = DAILY_DIR / "VVIX.parquet"
    if p_vvix.exists():
        vv = pd.read_parquet(p_vvix)[["timestamp", "close"]].rename(columns={"close": "vvix"})
        vv["date"] = pd.to_datetime(vv["timestamp"], utc=True).dt.tz_convert(
            "UTC").dt.date
        out = out.merge(vv[["date", "vvix"]], on="date", how="left")
    else:
        out["vvix"] = np.nan

    # HYG/LQD credit spread proxy
    p_hyg = DAILY_DIR / "HYG.parquet"
    p_lqd = DAILY_DIR / "LQD.parquet"
    if p_hyg.exists() and p_lqd.exists():
        hyg = pd.read_parquet(p_hyg)[["timestamp", "close"]].rename(columns={"close": "hyg"})
        lqd = pd.read_parquet(p_lqd)[["timestamp", "close"]].rename(columns={"close": "lqd"})
        for d in (hyg, lqd):
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "UTC").dt.date
        out = out.merge(hyg[["date", "hyg"]], on="date", how="left")
        out = out.merge(lqd[["date", "lqd"]], on="date", how="left")
        out["hyg_lqd_log"] = np.log(out["hyg"]) - np.log(out["lqd"])
    else:
        out["hyg_lqd_log"] = np.nan

    # 10Y-3M Treasury term spread
    p_tnx = DAILY_DIR / "TNX.parquet"
    p_irx = DAILY_DIR / "IRX.parquet"
    if p_tnx.exists() and p_irx.exists():
        tnx = pd.read_parquet(p_tnx)[["timestamp", "close"]].rename(columns={"close": "tnx"})
        irx = pd.read_parquet(p_irx)[["timestamp", "close"]].rename(columns={"close": "irx"})
        for d in (tnx, irx):
            d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(
                "UTC").dt.date
        out = out.merge(tnx[["date", "tnx"]], on="date", how="left")
        out = out.merge(irx[["date", "irx"]], on="date", how="left")
        out["term_spread_10y_3m"] = out["tnx"] - out["irx"]
    else:
        out["term_spread_10y_3m"] = np.nan

    _VIX_CACHE = out
    return _VIX_CACHE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.date

    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["rsi_14"] = _rsi(c, 14)
    df["rsi_5"] = _rsi(c, 5)
    df["days_os_20"] = (df["rsi_14"] < 30).astype(int).rolling(20).sum()
    df["atr_pct_14"] = _atr_pct(h, l, c, 14)

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    df["dist_sma20_pct"] = (c - sma20) / sma20
    df["dist_sma50_pct"] = (c - sma50) / sma50
    df["dist_sma200_pct"] = (c - sma200) / sma200
    df["sma50_sma200_ratio"] = sma50 / sma200 - 1.0

    log_ret = np.log(c / c.shift(1))
    df["log_ret_1"] = log_ret
    for k in (5, 20, 60, 252):
        df[f"log_ret_{k}"] = np.log(c / c.shift(k))

    df["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["realized_vol_60"] = log_ret.rolling(60).std() * np.sqrt(252)

    vol_ma20 = v.rolling(20).mean()
    df["vol_ratio_20"] = v / vol_ma20
    # OBV (On-Balance Volume) 60d change, scaled by 60d avg volume.
    # OBV cumulative sum of signed volume; rolling 60d delta normalised.
    daily_ret = c.pct_change()
    obv_signed = np.sign(daily_ret).fillna(0) * v
    obv_cum = obv_signed.cumsum()
    vol_ma60 = v.rolling(60).mean().replace(0, np.nan)
    df["obv_60d_chg"] = obv_cum.diff(60) / vol_ma60

    sd20 = c.rolling(20).std()
    bb_up = sma20 + 2 * sd20
    bb_dn = sma20 - 2 * sd20
    df["bb_width_20"] = (bb_up - bb_dn) / sma20
    df["bb_pct_b_20"] = (c - bb_dn) / (bb_up - bb_dn)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal

    rolling_max = c.rolling(252).max()
    df["drawdown_from_252h"] = c / rolling_max - 1.0

    # Days since last 5% drawdown — complacency / late-cycle clock
    expanding_max = c.expanding().max()
    in_dd = ((c / expanding_max - 1.0) <= -0.05).astype(int)
    df["days_since_5pct_dd"] = (1 - in_dd).groupby(in_dd.cumsum()).cumcount()

    ts_et = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df["month_of_year"] = ts_et.dt.month
    df["dow"] = ts_et.dt.dayofweek

    # Merge VIX-derived features (cross-source). Symbol's own date keys to VIX date.
    vix_df = _load_vix()
    if vix_df is not None:
        df = df.merge(vix_df, on="date", how="left")
        # vix_minus_rv20 = VIX (% annualized implied vol) − realized_vol_20 (decimal annualized)
        # Convert realized_vol_20 (decimal) to % to align scale with VIX (already in %).
        df["vix_minus_rv20"] = df["vix"] - df["realized_vol_20"] * 100
    else:
        df["vix"] = np.nan
        df["vix_change_5d"] = np.nan
        df["vix_pct_rank_252"] = np.nan
        df["vix_minus_rv20"] = np.nan
        df["vix3m_over_vix"] = np.nan
        df["vix3m_minus_vix"] = np.nan
        df["vvix"] = np.nan
        df["hyg_lqd_log"] = np.nan
        df["term_spread_10y_3m"] = np.nan

    # Forward returns
    for label, h in FWD_HORIZONS.items():
        df[f"fwd_{label}"] = np.log(c.shift(-h) / c)
    return df


def load_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    """Load all symbols, compute features, return long-format DataFrame."""
    if not DAILY_DIR.exists():
        raise FileNotFoundError(f"Run fetch_alpaca_daily first: {DAILY_DIR}")
    if symbols is None:
        symbols = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    frames = []
    for sym in symbols:
        p = DAILY_DIR / f"{sym}.parquet"
        if not p.exists():
            logger.warning("missing %s", p); continue
        df = pd.read_parquet(p)
        df["symbol"] = sym
        df = add_features(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def available_symbols() -> list[str]:
    if not DAILY_DIR.exists():
        return []
    excluded = {"VIX", "VIX3M", "VVIX", "MOVE", "TNX", "IRX",
                "HYG", "LQD", "TLT"}  # feature sources, not trade targets
    return sorted(p.stem for p in DAILY_DIR.glob("*.parquet")
                  if p.stem not in excluded)


def feature_columns() -> list[str]:
    return FEATURES.copy()


def horizon_labels() -> list[str]:
    return list(FWD_HORIZONS.keys())
