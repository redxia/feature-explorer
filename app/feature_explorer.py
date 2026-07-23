"""Streamlit UI: explore daily features vs forward returns.

Run:
    streamlit run app/feature_explorer.py

Tabs:
  1. Single-feature scatter — feature on x, fwd return on y, one color per
     horizon (1w, 2w, 3w, 1m, 2m, 3m, 6m, 9m, 12m), with OLS line per horizon.
     Shows current feature value as vertical marker.
  2. Correlation matrix — pick multiple features, heatmap of corr(feature, fwd_h)
     for every horizon.
  3. Tearsheet — quantile-bucket the picked feature, show mean fwd return per
     bucket per horizon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.research.feature_panel import (  # noqa: E402
    load_panel, available_symbols, feature_columns, horizon_labels, FWD_HORIZONS,
)

st.set_page_config(page_title="Feature Explorer", layout="wide")


# ---------- caching ----------

@st.cache_data(show_spinner="Loading daily panel...", persist="disk")
def cached_panel(symbols: tuple[str, ...]) -> pd.DataFrame:
    return load_panel(list(symbols))


@st.cache_resource
def cached_per_symbol(sym: str) -> pd.DataFrame:
    """Per-symbol cache so adding a symbol to multiselect only rebuilds that one."""
    return load_panel([sym])


# ---------- sidebar ----------

st.sidebar.header("Data")
syms = available_symbols()
if not syms:
    st.error("No daily bars yet. Run `python -m src.data.fetch_alpaca_daily --years 6`.")
    st.stop()

default_sym = "QQQ" if "QQQ" in syms else ("SPY" if "SPY" in syms else syms[0])
selected_syms = st.sidebar.multiselect(
    "Symbols (samples pooled across symbols)",
    options=syms,
    default=[default_sym],
)
if not selected_syms:
    st.warning("Pick at least one symbol")
    st.stop()

# Per-symbol cache; concat is fast vs reloading whole panel
panel = pd.concat([cached_per_symbol(s) for s in selected_syms], ignore_index=True)

st.sidebar.header("Filters")
min_date = pd.to_datetime(panel["date"].min())
max_date = pd.to_datetime(panel["date"].max())
default_start = max(min_date.date(), pd.Timestamp("2008-01-01").date())
date_range = st.sidebar.date_input(
    "Date range (default starts 2008-01-01)",
    value=(default_start, max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = date_range
    panel = panel[(pd.to_datetime(panel["date"]) >= pd.Timestamp(d0))
                  & (pd.to_datetime(panel["date"]) <= pd.Timestamp(d1))]

horizons_pick = st.sidebar.multiselect(
    "Horizons to plot",
    options=horizon_labels(),
    default=["1w", "2w", "3w", "1m", "2m", "3m"],
)

# ---- Data refresh ----
st.sidebar.divider()
st.sidebar.subheader("Update data")
refresh_scope = st.sidebar.radio(
    "Scope",
    ["Full universe (~135 syms)", "Selected symbols only"],
    index=0,
)
auto_retrain = st.sidebar.checkbox(
    "Auto-retrain LightGBM models after refresh",
    value=True,
    help="If checked, every symbol with an existing trained pickle in "
         "models/lgbm_dash/ will be retrained on the fresh panel. ~30-90s per symbol.",
)
_trained_now = []
try:
    from src.research import lgbm_dash_model as _lgbm_list
    _trained_now = _lgbm_list.list_trained()
except Exception:
    pass
skip_retrain = st.sidebar.multiselect(
    "Skip these on auto-retrain (faster)",
    options=_trained_now,
    default=["SPY"] if "SPY" in _trained_now else [],
    help="Symbols listed here keep their existing model on refresh. "
         "Skipping SPY ~halves retrain time when only SPY+QQQ are trained.",
)
refresh_years = st.sidebar.number_input(
    "Years of history", min_value=5, max_value=30, value=20, step=1
)
if st.sidebar.button("🔄 Refresh data from yfinance", use_container_width=True):
    from src.data.fetch_yf_daily import fetch_and_cache as yf_fetch
    import yfinance as yf
    from pathlib import Path

    if refresh_scope.startswith("Full"):
        from src.data.fetch_alpaca_stock_bars import load_universe_symbols
        targets = load_universe_symbols()
    else:
        targets = list(selected_syms)

    # Always refresh VIX + macro feature sources so the latest bar isn't NaN-merged
    macro_sources = {
        "VIX": "^VIX", "VIX3M": "^VIX3M", "VVIX": "^VVIX", "MOVE": "^MOVE",
        "TNX": "^TNX", "IRX": "^IRX",
        "TLT": "TLT", "HYG": "HYG", "LQD": "LQD",
    }
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "daily"
    progress = st.sidebar.progress(0.0, text=f"Fetching macro sources + {len(targets)} symbol(s)...")
    total = len(targets) + len(macro_sources)
    done = 0
    written = 0

    for sym, yfsym in macro_sources.items():
        try:
            d = yf.download(yfsym, period="max", interval="1d",
                            auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
            if not d.empty:
                d = d.reset_index().rename(columns={
                    "Date": "timestamp", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"})
                d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
                d["symbol"] = sym
                d["vwap_alpaca"] = np.nan
                d["trade_count"] = np.nan
                cols = ["timestamp", "symbol", "open", "high", "low", "close",
                        "volume", "vwap_alpaca", "trade_count"]
                d[[c for c in cols if c in d.columns]].to_parquet(
                    out_dir / f"{sym}.parquet", index=False)
                written += 1
        except Exception as e:
            st.sidebar.warning(f"{sym}: {e}")
        done += 1
        progress.progress(done / total, text=f"Macro {sym} ({done}/{total})")

    for sym in targets:
        try:
            yf_fetch([sym], years=int(refresh_years))
            written += 1
        except Exception as e:
            st.sidebar.warning(f"{sym}: {e}")
        done += 1
        progress.progress(done / total, text=f"Symbol {sym} ({done}/{total})")
    progress.empty()
    st.sidebar.success(f"Refreshed {written}/{total} sources. Reloading...")

    # Invalidate Streamlit caches
    st.cache_data.clear()
    st.cache_resource.clear()
    # Bust the in-process VIX cache (module-level global)
    try:
        from src.research import feature_panel as _fp
        _fp._VIX_CACHE = None
    except Exception:
        pass

    # Auto-retrain any LightGBM model with existing pickle
    if auto_retrain:
        try:
            from src.research import lgbm_dash_model as _lgbm
            trained_syms = [s for s in _lgbm.list_trained() if s not in set(skip_retrain)]
            skipped = [s for s in _lgbm.list_trained() if s in set(skip_retrain)]
            if trained_syms:
                rt = st.sidebar.progress(
                    0.0, text=f"Retraining {len(trained_syms)} LightGBM models...")
                for i, s in enumerate(trained_syms):
                    try:
                        _lgbm.train_symbol(s)
                    except Exception as e:
                        st.sidebar.warning(f"retrain {s} failed: {e}")
                    rt.progress((i + 1) / len(trained_syms),
                                text=f"Retrained {s} ({i+1}/{len(trained_syms)})")
                rt.empty()
                msg = f"Retrained {len(trained_syms)}: {', '.join(trained_syms)}"
                if skipped:
                    msg += f" | skipped {', '.join(skipped)}"
                st.sidebar.success(msg)
        except Exception as e:
            st.sidebar.warning(f"Auto-retrain skipped: {e}")

    # Drop user's date-range so the widget re-defaults to new max date
    for k in list(st.session_state.keys()):
        if "date" in k.lower() or "range" in k.lower():
            del st.session_state[k]
    st.rerun()

st.sidebar.caption(
    f"Last loaded: {pd.to_datetime(panel['date'].max()).date()}"
)

# ---------- permanent refresh + retrain (runs on GitHub Actions) ----------
# Streamlit Cloud's disk is ephemeral, so to make a refresh PERMANENT we trigger
# the repo's daily-update workflow. It fetches fresh data + retrains on Python
# 3.11 and commits to the repo, which auto-redeploys this app. Needs a GitHub
# token in app Secrets (GH_TOKEN). Same workflow also runs on a daily schedule.
st.sidebar.markdown("---")
st.sidebar.subheader("Permanent refresh + retrain")
st.sidebar.caption(
    "Runs the workflow on GitHub (Python 3.11): fetch fresh data, retrain, and "
    "commit so it survives reboots. ~15-25 min, then the app auto-redeploys."
)
if st.sidebar.button("♻️ Refresh + retrain permanently (GitHub)",
                     use_container_width=True):
    import os
    import requests
    _tok = st.secrets.get("GH_TOKEN", os.environ.get("GH_TOKEN", ""))
    _repo = st.secrets.get("GH_REPO", os.environ.get("GH_REPO",
                                                     "redxia/feature-explorer"))
    _wf = st.secrets.get("GH_WORKFLOW", os.environ.get("GH_WORKFLOW",
                                                       "daily-update.yml"))
    _branch = st.secrets.get("GH_BRANCH", os.environ.get("GH_BRANCH", "main"))
    if not _tok:
        st.sidebar.error(
            "Add GH_TOKEN (a GitHub token with Actions read/write) to this "
            "app's Secrets to enable permanent rebuilds.")
    else:
        try:
            _r = requests.post(
                f"https://api.github.com/repos/{_repo}/actions/workflows/"
                f"{_wf}/dispatches",
                headers={
                    "Authorization": f"Bearer {_tok}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"ref": _branch},
                timeout=20,
            )
            if _r.status_code == 204:
                st.sidebar.success(
                    "Rebuild started on GitHub. It takes ~15-25 min; this app "
                    "auto-redeploys with fresh data + model when it finishes. "
                    "Watch progress on the repo's Actions tab.")
            else:
                st.sidebar.error(
                    f"Dispatch failed ({_r.status_code}): {_r.text[:200]}")
        except Exception as _e:
            st.sidebar.error(f"Could not reach GitHub: {_e}")

# ---------- tabs ----------

(tab_lgbm, tab_dash, tab_scatter, tab_scenario, tab_corr, tab_buckets,
 tab_regime) = st.tabs([
    "🤖 LightGBM Signal Dashboard", "Signal Dashboard (empirical)",
    "Scatter vs fwd returns", "What-if scenario",
    "Correlation matrix", "Quantile buckets", "🌡️ Vol Regimes (HMM)"
])


# ===== Tab 0: Signal Dashboard =====
with tab_dash:
    st.subheader("Trade signal dashboard")

    with st.expander("📖 Feature glossary — what does each feature mean?"):
        st.markdown("""
*All features computed from daily OHLCV. "fwd return" = log return over the next N trading days.*

| feature | what it measures | typical range | sign convention |
|---|---|---|---|
| **`rsi_14`** | Relative Strength Index, 14-day. Momentum oscillator: gains vs losses over 14 bars | 0-100 | <30 oversold (buy), >70 overbought (sell) |
| **`rsi_5`** | Same as RSI-14 but shorter (5-day). Faster, noisier | 0-100 | same |
| **`days_os_20`** | Count of days RSI_14 was <30 in last 20 trading days. Capitulation density | 0-20 | high count = repeated oversold readings → strong forward upside (especially in bear markets) |
| **`atr_pct_14`** | Average True Range / price. Daily volatility expressed as % of price | 0.005-0.05 | high = volatile regime, predicts higher fwd returns historically |
| **`dist_sma20_pct`** | Distance from 20-day simple moving average, as % | -0.15 to +0.15 | negative = below 20dma (oversold short-term) |
| **`dist_sma50_pct`** | Distance from 50-day SMA | -0.30 to +0.30 | negative = below 50dma (medium-term weakness) |
| **`dist_sma200_pct`** | Distance from 200-day SMA. Bull/bear regime line | -0.50 to +0.50 | negative = bear market territory |
| **`sma50_sma200_ratio`** | (SMA50 / SMA200) − 1. Golden/death cross indicator | -0.20 to +0.20 | positive = uptrend (golden cross), negative = downtrend |
| **`log_ret_1`** | Yesterday's log return. 1-day momentum | -0.05 to +0.05 | mean-reverts strongly intraday, weak signal |
| **`log_ret_5`** | 5-day log return. Weekly momentum | -0.10 to +0.10 | usually mean-reverts |
| **`log_ret_20`** | 20-day log return. Monthly momentum | -0.20 to +0.20 | mean-reverts at 1-3m horizon |
| **`log_ret_60`** | 60-day log return. Quarterly momentum | -0.40 to +0.40 | inversely predicts next 1-3m |
| **`log_ret_252`** | 1-year log return | -0.60 to +0.60 | strong long-term reversal: bad year predicts good year (and vice versa) |
| **`realized_vol_20`** | Stdev of daily log returns × √252, 20-day window. Annualized vol | 0.05-0.80 | high vol = positive fwd return signal historically |
| **`realized_vol_60`** | Same, 60-day window. Smoother | 0.05-0.80 | same |
| **`vol_ratio_20`** | Today's volume / 20-day avg volume | 0.3-5.0 | spikes during news, mostly noise as signal |
| **`obv_60d_chg`** | OBV (On-Balance Volume) 60-day change ÷ 60d avg volume. Cumulative signed-volume momentum | -10 to +10 | NEGATIVE ρ vs fwd returns: heavy crowd-buying (high OBV) precedes drawdowns (mean reversion); heavy distribution precedes rallies |
| **`bb_width_20`** | Bollinger band width = (upper-lower) / SMA20. Normalized vol | 0.02-0.30 | wide = expansion regime, predicts positive fwd returns |
| **`bb_pct_b_20`** | Where price sits in Bollinger bands. (close - lower) / (upper - lower) | 0-1 | 0 = at lower band (oversold), 1 = at upper (overbought) |
| **`macd_hist`** | MACD histogram = MACD line − signal line | -5 to +5 | positive = upward momentum, but weak predictive edge |
| **`drawdown_from_252h`** | (close / 1y rolling max) − 1. How far below 1-year peak | -0.50 to 0.0 | deep drawdown = "buy fear" signal, predicts positive fwd returns |
| **`days_since_5pct_dd`** | Trading days since last 5% drawdown from running max. Complacency clock | 0 to 500+ | NEGATIVE ρ vs fwd returns: long stretches without a 5% pullback = late-cycle complacency = pullback risk. Sell/hedge tilt |
| **`month_of_year`** | Calendar month (1-12). Seasonality control | 1-12 | small positive in Apr/Nov, near zero overall |
| **`dow`** | Day of week (0=Mon, 4=Fri) | 0-4 | basically zero edge, included for completeness |
| **`vix`** | CBOE VIX index level. Implied 30-day SPX vol from options | 9-80 | high VIX = fear regime, often precedes mean-reversion rallies |
| **`vix_change_5d`** | VIX move over last 5 trading days | -25 to +25 | spike = panic; large drop = fear unwind, momentum continuation |
| **`vix_pct_rank_252`** | Where current VIX sits in 252-day distribution | 0-1 | 0.95 = top 5% (extreme fear); 0.05 = complacency |
| **`vix_minus_rv20`** | VIX − realized_vol_20 (% scale). "Vol risk premium" | -10 to +20 | positive = market pricing more vol than realized (normal); negative = realized > implied (vol shock just hit) |
| **`vix3m_over_vix`** | VIX3M / VIX (3-month over spot). VIX term structure ratio | 0.7 to 1.4 | **>1 = contango (calm regime)**, **<1 = backwardation (panic)**. Negative ρ vs fwd returns: backwardation predicts upside |
| **`vvix`** | VVIX index level — implied vol of VIX itself ("vol of vol") | 60-200 | high VVIX = uncertainty about future fear. Positive ρ vs fwd returns: high VVIX historically precedes upside |
| **`hyg_lqd_log`** | log(HYG) − log(LQD). High-yield bonds vs investment-grade bonds | -0.5 to +0.5 | rising = credit risk-on (junk outperforming). Strong NEGATIVE ρ: exuberant credit precedes equity drawdowns |
| **`term_spread_10y_3m`** | 10Y Treasury yield − 3M T-bill yield (yield curve slope) | -3 to +5 | <0 = inverted curve (recession signal). Negative ρ vs fwd returns: steep curve early-cycle, flat/inverted late-cycle |

**Key insight:** *vol-based features* (atr_pct, realized_vol, bb_width) and *drawdown* are the strongest predictors. *Day/month* are noise. *RSI/SMA distances* are mid-strength mean-reversion signals.
""")

    with st.expander("What is this dashboard? (click to expand)", expanded=False):
        st.markdown("""
**Goal:** tell you whether to be long, short, or flat on a stock/ETF, and over
what horizon, based on where its features sit *right now* relative to the past
15 years.

**How it works in 3 steps:**
1. **Read current state.** Take latest daily bar of the target symbol. Pull every
   feature value (RSI, distance from moving averages, volatility, drawdown, etc.).
2. **Look up history.** For each feature, find every past day where this feature
   was within ±0.5 standard deviations of the current value. Average what
   actually happened in the next 1 week, 2 weeks, ..., 12 months.
3. **Combine signals.** Weight each feature's forecast by its historical
   correlation strength (|Spearman ρ|). Strong-edge features count more. Sum
   into one expected % return per horizon.

**Reading the numbers:**
- **`pct` column** = where current value sits in 15y history. 90% means today
  is higher than 90% of past days. 10% means lower than 90% of past days.
- **`ρ` (rho)** = Spearman rank correlation between this feature and forward
  returns. Range −1 to +1. |ρ| > 0.15 = real edge. 0 = no signal.
- **`exp_1m_%`** = average % return seen historically when this feature was at
  current level, holding 1 month.
- **Composite forecast bar** = weighted average across all 21 features.

**This is empirical, not predictive AI.** Reads like: "in the past 15y, when SPY
looked like this, here's what happened next on average." It cannot account for
news, regime change, or tail events.
""")

    target = st.selectbox(
        "Target symbol", options=selected_syms, index=0, key="dash_target"
    )
    target_panel = panel[panel["symbol"] == target].sort_values("date").reset_index(drop=True)
    if target_panel.empty:
        st.warning("No data for target"); st.stop()
    cur = target_panel.iloc[-1]

    # Feature signal direction priors from full sample (signed Spearman)
    feats_all = feature_columns()
    feat_stats = []
    for f in feats_all:
        if pd.isna(cur.get(f)):
            continue
        # Historical percentile of current value
        hist = target_panel[f].dropna()
        if len(hist) < 100:
            continue
        cur_val = float(cur[f])
        pct = float((hist < cur_val).mean())  # 0..1

        per_horizon = {}
        for h in horizon_labels():
            col = f"fwd_{h}"
            sub = target_panel.dropna(subset=[f, col])
            if len(sub) < 100:
                per_horizon[h] = {"sp": np.nan, "mean_fwd": np.nan, "n": 0,
                                  "pct_green": np.nan, "pct_red": np.nan,
                                  "median_fwd": np.nan}
                continue
            sp = float(spearmanr(sub[f], sub[col]).statistic)
            # Empirical fwd return when feature within ±0.5σ of current value
            tol = sub[f].std() * 0.5
            window = sub[(sub[f] > cur_val - tol) & (sub[f] < cur_val + tol)][col]
            if len(window):
                mean_fwd = float(window.mean())
                median_fwd = float(window.median())
                pct_green = float((window > 0).mean())
                pct_red = float((window < 0).mean())
            else:
                mean_fwd = median_fwd = pct_green = pct_red = np.nan
            per_horizon[h] = {"sp": sp, "mean_fwd": mean_fwd, "n": len(window),
                              "pct_green": pct_green, "pct_red": pct_red,
                              "median_fwd": median_fwd}
        feat_stats.append({
            "feature": f, "cur_val": cur_val, "pct": pct,
            "per_horizon": per_horizon,
        })

    if not feat_stats:
        st.warning("Insufficient data"); st.stop()

    # Composite per-horizon forecast: weight each feature's empirical_mean by |spearman|
    horizons = horizon_labels()
    composite = {h: {"wfwd": 0.0, "wgreen": 0.0, "wsum": 0.0} for h in horizons}
    for fs in feat_stats:
        for h, ph in fs["per_horizon"].items():
            sp, mean_fwd, n, green = ph["sp"], ph["mean_fwd"], ph["n"], ph["pct_green"]
            if np.isnan(sp) or np.isnan(mean_fwd) or n < 30:
                continue
            w = abs(sp)
            composite[h]["wfwd"] += w * mean_fwd
            composite[h]["wgreen"] += w * (green if not np.isnan(green) else 0.5)
            composite[h]["wsum"] += w

    composite_df = pd.DataFrame([
        {
            "horizon": h,
            "expected_log_return": composite[h]["wfwd"] / composite[h]["wsum"]
                if composite[h]["wsum"] > 0 else np.nan,
            "win_rate": composite[h]["wgreen"] / composite[h]["wsum"]
                if composite[h]["wsum"] > 0 else np.nan,
        }
        for h in horizons
    ])
    composite_df["expected_pct"] = (np.exp(composite_df["expected_log_return"]) - 1) * 100
    composite_df["win_rate_pct"] = composite_df["win_rate"] * 100

    # ---- panel TOP: trade rec (was Panel C — moved up) ----
    st.markdown("## 🎯 Trade Recommendation")
    st.markdown("""
*Headline call per horizon. Bullish/Bearish chip per timeframe so you can see
exactly where the edge sits.*
""")

    # All horizons (user-requested set)
    rec_horizons = ["1w", "2w", "3w", "1m", "2m", "3m", "6m", "9m", "12m"]
    horiz_idx = composite_df.set_index("horizon")
    rec_rows = []
    for h in rec_horizons:
        if h not in horiz_idx.index:
            continue
        v = float(horiz_idx.loc[h, "expected_pct"])
        wr = float(horiz_idx.loc[h, "win_rate_pct"])
        if np.isnan(v):
            verdict = "n/a"; emoji = "⚪"
        elif v > 1.0:
            verdict = "BULLISH"; emoji = "🟢"
        elif v > 0.3:
            verdict = "lean bullish"; emoji = "🟩"
        elif v < -1.0:
            verdict = "BEARISH"; emoji = "🔴"
        elif v < -0.3:
            verdict = "lean bearish"; emoji = "🟥"
        else:
            verdict = "neutral"; emoji = "⚪"
        rec_rows.append({
            "horizon": h,
            "exp_%": round(v, 2),
            "win_rate_%": round(wr, 0) if not np.isnan(wr) else None,
            "verdict": f"{emoji} {verdict}",
        })
    rec_df = pd.DataFrame(rec_rows)

    # Display as columns of metrics (one per horizon)
    cols = st.columns(len(rec_horizons))
    for i, h in enumerate(rec_horizons):
        if h not in horiz_idx.index: continue
        v = float(horiz_idx.loc[h, "expected_pct"])
        wr = float(horiz_idx.loc[h, "win_rate_pct"])
        if np.isnan(v):
            cols[i].metric(h, "—"); continue
        if v > 0.3:
            chip = "🟢 BULL" if v > 1.0 else "🟩 lean bull"
        elif v < -0.3:
            chip = "🔴 BEAR" if v < -1.0 else "🟥 lean bear"
        else:
            chip = "⚪ neutral"
        wr_txt = f"{wr:.0f}% win" if not np.isnan(wr) else ""
        cols[i].metric(h, f"{v:+.2f}%", delta=f"{chip}  {wr_txt}")

    st.dataframe(rec_df.set_index("horizon").T, use_container_width=True)

    # Aggregates
    short_avg = composite_df.set_index("horizon").loc[
        ["1w","2w","3w","1m","2m","3m"], "expected_pct"
    ].mean()
    long_avg = composite_df.set_index("horizon").loc[
        ["6m","9m","12m"], "expected_pct"
    ].mean()
    conf = min(1.0, max(0.0, (abs(short_avg) + abs(long_avg)) / 4.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Short-term avg (1w-3m)", f"{short_avg:.2f}%",
              delta="bullish" if short_avg > 0.5 else ("bearish" if short_avg < -0.5 else "neutral"))
    c2.metric("Long-term avg (6m-12m)", f"{long_avg:.2f}%",
              delta="bullish" if long_avg > 0.5 else ("bearish" if long_avg < -0.5 else "neutral"))
    c3.metric("Conviction", f"{conf*100:.0f}%")

    st.caption("**Decision thresholds:** "
               "exp>+1% = BULLISH | +0.3 to +1% = lean bullish | "
               "−0.3 to +0.3% = neutral | −1 to −0.3% = lean bearish | exp<−1% = BEARISH. "
               "Conviction <25% = stand aside.")

    # Top 15 drivers — all horizons (1w-6m), ranked by avg |ρ| × |expected| at 1m
    st.markdown("##### Top 15 driver features — expected % and win-rate at every horizon")
    st.caption("Ranked by 1m contribution. Green cells positive expected, gray cells neutral. "
               "Win-rate = historical % of windows that closed positive when feature near current value.")
    rec_horiz = ["1w", "2w", "3w", "1m", "2m", "3m", "6m", "9m", "12m"]

    drivers = []
    for fs in feat_stats:
        ph_1m = fs["per_horizon"]["1m"]
        if np.isnan(ph_1m["sp"]) or np.isnan(ph_1m["mean_fwd"]): continue
        contrib_pct = (np.exp(ph_1m["mean_fwd"]) - 1) * 100 * abs(ph_1m["sp"])
        drivers.append((fs, contrib_pct))
    drivers.sort(key=lambda x: abs(x[1]), reverse=True)

    rows = []
    for fs, contrib in drivers[:15]:
        row = {
            "feature": fs["feature"],
            "cur_val": round(fs["cur_val"], 4),
            "hist_pct": f"{fs['pct']*100:.0f}%",
        }
        for h in rec_horiz:
            ph = fs["per_horizon"][h]
            row[f"{h}_exp%"] = (round((np.exp(ph["mean_fwd"])-1)*100, 2)
                                if not np.isnan(ph["mean_fwd"]) else None)
            row[f"{h}_win%"] = (round(ph["pct_green"]*100, 0)
                                if not np.isnan(ph["pct_green"]) else None)
        row["1m_ρ"] = round(fs["per_horizon"]["1m"]["sp"], 3)
        rows.append(row)
    drivers_df = pd.DataFrame(rows)
    # Conditional color on exp% columns
    exp_cols = [c for c in drivers_df.columns if c.endswith("_exp%")]
    win_cols = [c for c in drivers_df.columns if c.endswith("_win%")]
    sty = drivers_df.style.background_gradient(
        subset=exp_cols, cmap="RdYlGn", vmin=-3, vmax=3
    ).background_gradient(
        subset=win_cols, cmap="Greens", vmin=40, vmax=70
    )
    st.dataframe(sty, use_container_width=True)

    st.divider()

    # ---- panel A: current feature posture ----
    st.markdown(f"### Panel A — {target} current feature posture")
    st.markdown("""
*One row per feature. **Sorted by `edge_rank_|ρ|` — most informative features for this symbol are at the top.***

| column | meaning |
|---|---|
| `edge_rank_|ρ|` | average absolute Spearman ρ across ALL horizons. Higher = stronger predictive edge for this symbol. Sort key. |
| `cur_val` | feature's value on today's bar |
| `hist_pct` | percentile of cur_val in 15y history (50% = median, 90% = unusually high) |
| `edge_short_avg_ρ` | average Spearman ρ across 1w-3m horizons (signed). Positive = high feature → high return. Negative = high feature → low return |
| `exp_1m_%` / `exp_3m_%` / `exp_6m_%` | average historical % return when this feature was near today's value, held 1m/3m/6m |

*Top rows = features that move the needle. Bottom rows = noise (ignore those).*
""")
    posture_rows = []
    for fs in feat_stats:
        f = fs["feature"]
        # Direction call from sign of avg-spearman across short horizons
        short_sps = [fs["per_horizon"][h]["sp"] for h in ("1w","2w","3w","1m","2m","3m")
                     if not np.isnan(fs["per_horizon"][h]["sp"])]
        avg_sp = float(np.mean(short_sps)) if short_sps else 0.0
        # Bullish if (positive feature & positive sp) or (negative feature & negative sp)
        # Better: directional contribution = sign(sp) * (current pct - 0.5) * 2
        # But we want: "given current value, what is sign of expected fwd return"
        # = sign(sp) only if percentile != 50; magnitude = |sp| * (pct - 0.5)
        # Use empirical_mean[1m] as cleaner read
        # Edge rank score = mean |ρ| across all horizons (how informative this feature is)
        all_sps = [fs["per_horizon"][h]["sp"] for h in horizon_labels()
                   if not np.isnan(fs["per_horizon"][h]["sp"])]
        edge_score = float(np.mean(np.abs(all_sps))) if all_sps else 0.0

        def _exp(v): return round((np.exp(v)-1)*100, 2) if not np.isnan(v) else None
        def _pct_round(v): return round(v*100, 0) if not np.isnan(v) else None

        row = {
            "feature": f,
            "edge_rank_|ρ|": round(edge_score, 3),
            "cur_val": round(fs["cur_val"], 4),
            "hist_pct": f"{fs['pct']*100:.0f}%",
            "edge_short_avg_ρ": round(avg_sp, 3),
        }
        # All requested horizons: 1w, 2w, 3w, 1m, 2m, 3m, 6m
        for h in ["1w", "2w", "3w", "1m", "2m", "3m", "6m"]:
            ph = fs["per_horizon"][h]
            row[f"{h}_exp%"] = _exp(ph["mean_fwd"])
            row[f"{h}_win%"] = _pct_round(ph["pct_green"])
        posture_rows.append(row)
    # Sort by feature edge strength (top = most informative for this symbol)
    posture_df = pd.DataFrame(posture_rows).sort_values(
        "edge_rank_|ρ|", ascending=False, na_position="last"
    )
    p_exp = [c for c in posture_df.columns if c.endswith("_exp%")]
    p_win = [c for c in posture_df.columns if c.endswith("_win%")]
    sty = posture_df.style.background_gradient(
        subset=p_exp, cmap="RdYlGn", vmin=-3, vmax=3
    ).background_gradient(
        subset=p_win, cmap="Greens", vmin=40, vmax=70
    )
    st.dataframe(sty, use_container_width=True, hide_index=True)

    # ---- panel B: composite forecast bars ----
    st.markdown("### Panel B — combined forecast across all features, per horizon")
    st.markdown("""
*All 21 features collapsed into one number per horizon. Strong-edge features
count more (|ρ| weighting).*

- **Green bar = expected positive return** at that horizon → bullish lean
- **Red bar = expected negative return** → bearish lean
- **Bar magnitude** = how big the average historical move was

*Read across horizons:*
- All green = unambiguous long signal
- Short-term red, long-term green = wait/buy dips
- All red = avoid long, consider hedge or short
- Bars near zero = no signal
""")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=composite_df["horizon"],
        y=composite_df["expected_pct"],
        text=composite_df["expected_pct"].round(2).astype(str) + "%",
        textposition="outside",
        marker_color=[
            "#26A69A" if v >= 0 else "#EF5350"
            for v in composite_df["expected_pct"].fillna(0)
        ],
    ))
    fig.update_layout(
        height=400,
        yaxis_title="Expected log-return (%)",
        xaxis_title="Horizon",
        title=f"{target}: composite expected return (current state)",
        showlegend=False,
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Glossary"):
        st.markdown("""
- **Spearman ρ (rho):** rank correlation. Measures if "high feature → high
  return" or "high feature → low return" without assuming linearity. Range
  −1 to +1. |ρ| > 0.15 = meaningful edge.
- **Percentile (`hist_pct`):** position in historical distribution. 80% means
  the current value is higher than 80% of past observations.
- **±0.5σ window:** when looking up "what happened historically when the
  feature was near today's value", we widen the bin to ±0.5 standard
  deviations to get enough samples.
- **Composite weighted by |ρ|:** features with stronger historical edge get
  more vote. Noise features (low |ρ|) get near-zero weight.
- **Log return vs %:** internally we use log returns (additive across time).
  Display converts to % via exp(log_ret) − 1.

**Limitations:**
- Past ≠ future. Regime shifts (rate cycles, secular bull/bear) not modeled.
- Feature correlations are themselves correlated — composite may double-count
  similar signals (e.g. atr_pct and realized_vol).
- Tail events (March 2020, 2008) skew long horizons heavily.
- Single-symbol — does not consider sector / cross-asset confirmation.
""")


# ===== Tab: 🤖 LightGBM Signal Dashboard =====
with tab_lgbm:
    from src.research import lgbm_dash_model as lgbm_mod

    st.subheader("🤖 LightGBM Signal Dashboard")
    st.caption("Replaces ±0.5σ kernel with LightGBM regressor (fwd return) "
               "+ classifier (P(positive)). Walk-forward validated, "
               "feature interactions captured, correlated features handled natively.")

    with st.expander("Why is this more accurate than the empirical dashboard?", expanded=False):
        st.markdown("""
**Empirical (old) dashboard:**
- Bins history into ±0.5σ window around current value, averages.
- Treats every feature independently → **double-counts correlated features**
  (atr_pct + realized_vol + bb_width all measure same thing).
- 1D conditioning: cannot see joint effects like "RSI<25 AND vol>70%-tile".
- No regularization, no validation, no honest OOS estimate.

**LightGBM (this) dashboard:**
- Two models per (symbol, horizon):
  - **Regressor** → predicts fwd log-return (Huber loss, robust to fat tails)
  - **Classifier** → predicts P(fwd > 0)
- Inputs = all features simultaneously (vol stack, RSI, MA distances, VIX, credit, rates).
- **Tree-based gradient boosting:** captures interactions and non-linearity natively.
- **Low-complexity + seed-bagged (7 models averaged):** small trees
  (num_leaves=15, depth≤4) plus bagging → smoother, less-noisy day-to-day signal.
- **Purged + embargoed walk-forward:** a gap equal to the forecast horizon is
  inserted between train-end and test-start, so a training row's forward target
  can never overlap the test window. **This is the big fix** — without it, the
  last horizon's worth of training rows leak the future and inflate IC/AUC.
- **Expanding train window:** trains on all history up to the embargo boundary
  (averages over multiple macro cycles, avoids stale-regime inversion).

> ⚠️ **Reality check (post-purge).** Removing the overlap leak dropped 12-month
> IC from a deceptive ~0.5 to roughly 0. **That earlier number was almost
> entirely leakage.** Honest reading: single-symbol *directional* edge is weak
> at all horizons; high `hit%` at long horizons is mostly the market's upward
> drift, not skill. Use this as a regime gauge, not a precise forecaster — and
> watch the Conviction metric, which now correctly collapses toward 0 when IC≈0.
""")

    with st.expander("📐 What do IC, R², AUC, hit, Brier mean?", expanded=False):
        st.markdown("""
All five are **out-of-sample** — computed from walk-forward predictions where
the model never saw the future when making each prediction.

### Quick reference — what each value tier means

#### **IC (Spearman rank correlation)** — regressor — range −1 to +1
Does the model rank winners ahead of losers across all OOS days?

| value | meaning | trade signal? |
|---|---|---|
| > +0.20 | very strong edge (rare on daily data) | yes, high size |
| +0.10 to +0.20 | strong edge | yes |
| +0.05 to +0.10 | real edge | small size |
| +0.02 to +0.05 | weak/marginal | filter only |
| −0.02 to +0.02 | noise | ignore |
| < −0.02 | model is anti-predictive | flip sign or retrain |

Quant industry rule: a daily strategy with **average IC > 0.05** across all bets is investable.

#### **R² (sklearn coefficient of determination)** — regressor — range −∞ to +1
How much variance in fwd return the model explains vs predicting the historical mean.

| value | meaning | trade signal? |
|---|---|---|
| > +0.10 | model explains 10%+ of variance — exceptional for daily fin data | yes |
| +0.02 to +0.10 | model adds value over mean baseline | yes |
| 0.0 | tied with predicting the constant mean | depends on IC/AUC |
| −0.10 to 0.0 | model slightly worse than mean on MSE — typical for tree models with small calibration error | rely on IC/AUC instead |
| −0.30 to −0.10 | predictions overshooting in magnitude. Direction can still be right (check IC/hit) | rely on IC/AUC; consider shrinking predictions |
| < −0.30 (e.g. **−0.55**) | severe magnitude error. Sign may still be right (e.g. 84% hit + IC 0.23 + R² −0.55 like SPY 12m) — model knows direction but blows up the size estimate. Use sign for direction, ignore the magnitude | use as direction signal only |
| < −1.0 | model badly broken / overfit — retrain or drop | no |

**Key:** negative R² ≠ useless model. If IC > +0.10 and AUC > +0.55, the model has real ranking edge even when R² is negative. Negative R² just means the *magnitude* prediction is biased — typically because tree models predict more variance than reality has at long horizons.

#### **AUC (ROC area under curve)** — classifier — range 0.5 to 1.0
Probability that the classifier ranks a random "winner" day higher than a random "loser" day.

| value | meaning | trade signal? |
|---|---|---|
| > 0.70 | very strong (rare on daily fin data) | yes, high size |
| 0.60 to 0.70 | solid edge | yes |
| 0.55 to 0.60 | weak but real edge | small size / confirm |
| 0.52 to 0.55 | marginal | filter only |
| 0.48 to 0.52 | random | ignore |
| < 0.48 | inverted edge | flip sign or retrain |

#### **hit %** — regressor — range 0% to 100%
% of OOS days where regressor's sign matched realized sign.

| value | meaning | trade signal? |
|---|---|---|
| > 75% | very strong directional accuracy | yes |
| 65-75% | strong | yes |
| 55-65% | edge above base rate | small size |
| ~50-55% | base rate (drift) — signal not adding direction | ignore unless IC/AUC strong |
| < 45% | inverted | flip sign or retrain |

**Caveat:** at long horizons SPY drifts up → base rate hit% naturally 70-80% just from "always predict +". Compare to base rate, not to 50%.

#### **Brier score** — classifier — range 0 to 1, lower = better
Mean squared error of probability forecasts. Captures calibration AND sharpness.

| value | meaning | trade signal? |
|---|---|---|
| < 0.18 | excellent — probabilities well-calibrated and confident | yes |
| 0.18 to 0.22 | good — better than coin flip | yes |
| 0.22 to 0.25 | marginal — barely beats coin (Brier of always-50% = 0.25) | small size |
| > 0.25 | worse than guessing 50% — model overconfident | retrain |

#### **n_OOS** — sample size
Number of OOS bars with valid predictions. Need ≥ 500 for robust metric estimates. Below 100 the metrics are noisy.

---

### Classifier-only metrics (P(+) head)

#### **CLF_IC** — Spearman(P(+), fwd_ret) — range −1 to +1
Does the classifier's *probability magnitude* correlate with realized return magnitude? Higher P(+) = higher fwd return on average?

| value | meaning |
|---|---|
| > +0.15 | strong: probabilistic ranking matches return ranking |
| +0.05 to +0.15 | real edge |
| 0 to +0.05 | weak — classifier ranks direction OK but probability magnitude not informative |
| < 0 | bad |

#### **CLF_hit_%** — directional accuracy from classifier — range 0-100%
% of OOS bars where (P(+) > 0.5) matched (fwd > 0). Sister of REG_hit_% but using the classifier's threshold instead of regressor's sign.

| value | meaning |
|---|---|
| > 70% | strong (compare to base rate ~60%) |
| 55-70% | real edge |
| 50-55% | barely better than base rate |
| < 50% | classifier inverted |

If `REG_hit > CLF_hit` significantly: regressor better at direction. If `CLF_hit > REG_hit`: probability head better calibrated to direction.

#### **CLF_LogLoss** — cross-entropy of probability forecast — range 0 to ∞, lower = better
Strictly proper scoring rule. Penalizes overconfident wrong predictions much more than Brier.

| value | meaning |
|---|---|
| < 0.60 | excellent — sharp + well-calibrated probabilities |
| 0.60 to 0.65 | good |
| 0.65 to 0.69 | marginal (random binary classifier ≈ 0.693 = ln(2)) |
| > 0.70 | overconfident, worse than predicting 50/50 |

#### **CLF_pseudoR²** — McFadden pseudo-R² — range −∞ to ~0.4 (rare to exceed)
Classifier-side analog of regression R². Defined as `1 − LogLoss(model) / LogLoss(base_rate)`.

| value | meaning |
|---|---|
| > 0.05 | strong probability calibration vs base rate |
| 0.01 to 0.05 | model slightly beats base rate |
| 0 | tied with always predicting class frequency |
| < 0 | worse than base rate (rare unless miscalibrated) |

**Note:** McFadden pseudo-R² ranges much lower than OLS R² in scale. Anything above 0.05 is impressive on noisy financial data.

**How to read together:**
- High IC + high R² + high AUC → reliable signal across both regression and classification framings
- High hit% but low IC → directionally right but magnitude-blind (pick winners but tiny ones)
- IC>0 but R² <0 → predictions correctly ranked but biased (mean estimate poor) — usually fixable by recalibration
- Long-horizon outperforms short-horizon almost everywhere → horizon predictability scales with signal-to-noise

---

### **Conviction** — quality-weighted, agreement-aware confidence

A single 0-100% score combining three orthogonal factors. Designed so a "loud
forecast from a noisy model" doesn't fool you.

**Formula:**
```
quality_h     = max(IC_h, 0) × (1 + 4 × max(AUC_h − 0.5, 0))
agree_h       = 1 if sign(regressor) == sign(classifier_prob > 50%) else 0
eff_weight_h  = quality_h × agree_h

weighted_dir   = Σ(exp_pct_h × eff_weight_h) / Σ(eff_weight_h)
direction_mag  = min(|weighted_dir| / 2%, 1.0)        # 2% forecast → max
quality_mag    = min(avg_quality / 0.10, 1.0)         # IC*AUClift≈0.10 → max
agreement_frac = (count of horizons where reg & clf agree) / total_horizons

Conviction = direction_mag × quality_mag × agreement_frac × 100%
```

**Three factors explained in plain English:**

| factor | what it asks | range | high when... |
|---|---|---|---|
| **direction_mag** | Is the forecast big enough to act on? | 0-100% | weighted forecast ≥ ±2% |
| **quality_mag** | Does the model actually predict well at these horizons? | 0-100% | OOS IC × AUC-lift averages ≥ 0.10 (real edge) |
| **agreement_frac** | Do regressor and classifier agree on direction? | 0-100% | both heads point the same way at all horizons |

**Conviction = product of all three.** Any one factor near 0 kills conviction
(multiplicative penalty). This is intentional — a giant forecast from a
worthless 1w model with regressor-classifier disagreement should NOT score high.

**Reading the breakdown line below the metric:**
- `weighted forecast` — quality-weighted average of all horizon exp_pct values
- `size factor` — 100 × direction_mag
- `avg model quality` — mean of `quality_h` over agreeing horizons
- `quality factor` — 100 × quality_mag
- `horizon agreement` — how many horizons (out of 9) have reg & clf same direction

**Decision rules of thumb:**
- **Conviction > 50%** → trade. Strong forecast, real edge, models agree.
- **25-50%** → small-size or wait for confirmation. Edge exists but not screaming.
- **< 25%** → stand aside. Either forecast small, or model unreliable, or heads disagree.
- **Direction label (BULL/BEAR/neutral)** comes from sign of `weighted_dir`, not magnitude.

**Conviction-level playbook — how to actually use it:**

| conviction | action | reasoning |
|---|---|---|
| **>50%** | Full size trade | Big call + real edge + heads agree — rare regime |
| **35-50%** | Half-size trade | Edge present but smaller magnitude — scale risk to signal |
| **20-35%** | Filter / confirm only | Direction bias not entry trigger. Wait for technical confirmation (e.g. RSI extremes, breakout level) |
| **<20%** | Stand aside | Model says "I don't know" — respect it |

**Three things to do when conviction is low:**

**1. Trade only the high-edge horizons.** Scroll to OOS validation table. Find rows where `REG_IC > 0.10` AND `CLF_AUC > 0.55` — that's where model has real edge. Ignore short horizons (1w-3w) where IC ≈ 0.05. For SPY today: 6m/9m/12m carry signal; 1w-1m is noise.

**2. Position-size proportionally.** Don't binary-think. Allocation:
```
position_size = base_risk × conviction × directional_multiplier
e.g. base $10k × 30% conviction → $3k position
```

**3. Run What-if to find your trigger.** Drag scenario sliders to see what conditions trigger high conviction (e.g. drawdown_from_252h ≤ −0.20, vix_pct_rank > 0.9, hyg_lqd_log < −0.4). Build watchlist of trigger thresholds. Trade when reality reaches them.

**Use as regime tilt, not daily trigger:**
- Long-term bullish + short-term neutral → **buy dips, don't time entries**
- All horizons bearish + high conviction → **reduce equity exposure, rotate to TLT/cash**
- Mixed signals → **no big bets, focus on uncorrelated strategies**
- High conviction is rare — when it appears, **size up**. Most days will be 20-40%; that's normal.
""")

    target_l = st.selectbox("Target symbol", options=selected_syms, index=0,
                            key="lgbm_target")

    bundle = lgbm_mod.load_bundle(target_l)
    trained_list = lgbm_mod.list_trained()

    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 4])
    with btn_col1:
        if st.button(f"🔧 Train / retrain {target_l}", use_container_width=True,
                     key="lgbm_train_btn"):
            with st.spinner(f"Training LightGBM models for {target_l} "
                            f"(walk-forward + final, ~30-90s)..."):
                try:
                    bundle = lgbm_mod.train_symbol(target_l)
                    st.success(f"Trained {len(bundle.horizons)} horizons "
                               f"on panel {bundle.panel_start} → {bundle.panel_end}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
    with btn_col2:
        st.caption(f"**Trained symbols:** {', '.join(trained_list) if trained_list else '(none)'}")
    with btn_col3:
        if bundle is not None:
            st.caption(f"**{target_l}** trained {bundle.trained_at:%Y-%m-%d %H:%M} UTC | "
                       f"panel {bundle.panel_start} → {bundle.panel_end} | "
                       f"{bundle.panel_rows} rows")

    if bundle is None:
        st.info(f"No trained model for **{target_l}** yet. "
                f"Click 🔧 Train above. First run takes 30-90s.")
        st.stop()

    # Get current row for target
    sym_panel_l = panel[panel["symbol"] == target_l].sort_values("date").reset_index(drop=True)
    if sym_panel_l.empty:
        st.warning(f"No panel data for {target_l}"); st.stop()
    cur_row_l = sym_panel_l.iloc[-1]

    # Predict
    pred_df = lgbm_mod.predict_current(bundle, cur_row_l)
    if pred_df.empty:
        st.warning("Model has no horizons populated"); st.stop()

    # ---- Headline trade rec ----
    st.markdown("## 🎯 Trade Recommendation (LightGBM)")
    rec_horiz = ["1w", "2w", "3w", "1m", "2m", "3m", "6m", "9m", "12m"]
    rec_show = pred_df[pred_df["horizon"].isin(rec_horiz)].copy()
    rec_show["horizon"] = pd.Categorical(rec_show["horizon"], categories=rec_horiz, ordered=True)
    rec_show = rec_show.sort_values("horizon")
    cols = st.columns(len(rec_show))
    for i, (_, r) in enumerate(rec_show.iterrows()):
        v = float(r["exp_pct"]); pp = float(r["p_positive"]) * 100
        if v > 1.0: chip = "🟢 BULL"
        elif v > 0.3: chip = "🟩 lean bull"
        elif v < -1.0: chip = "🔴 BEAR"
        elif v < -0.3: chip = "🟥 lean bear"
        else: chip = "⚪ neutral"
        cols[i].metric(r["horizon"], f"{v:+.2f}%",
                       delta=f"{chip}  P(+)={pp:.0f}%")

    # ---- Composite aggregates ----
    short_mask = pred_df["horizon"].isin(["1w","2w","3w","1m","2m","3m"])
    long_mask = pred_df["horizon"].isin(["6m","9m","12m"])
    short_avg = float(pred_df.loc[short_mask, "exp_pct"].mean())
    long_avg = float(pred_df.loc[long_mask, "exp_pct"].mean()) \
        if long_mask.any() else float("nan")
    p_short = float(pred_df.loc[short_mask, "p_positive"].mean()) * 100
    p_long = float(pred_df.loc[long_mask, "p_positive"].mean()) * 100 \
        if long_mask.any() else float("nan")

    # ---- Quality-weighted, agreement-aware Conviction ----
    # 1. Per-horizon quality from OOS IC + AUC (cap negatives at 0)
    ic_w = pred_df["ic_oos"].clip(lower=0).fillna(0)               # 0..~0.25
    auc_lift = (pred_df["auc_oos"] - 0.5).clip(lower=0).fillna(0)  # 0..~0.15
    quality = ic_w * (1 + 4 * auc_lift)                            # combined

    # 2. Agreement between regressor and classifier
    sign_reg = np.sign(pred_df["exp_pct"].fillna(0))
    sign_clf = np.sign(pred_df["p_positive"].fillna(0.5) - 0.5)
    agree = ((sign_reg * sign_clf) >= 0).astype(float)             # 1 if agree, 0 if conflict
    eff_w = quality * agree                                        # zero-weight conflicting horizons

    # 3. Quality-weighted directional forecast (% units)
    total_w = float(eff_w.sum())
    if total_w > 0:
        weighted_dir = float((pred_df["exp_pct"] * eff_w).sum() / total_w)
        avg_quality = float(eff_w.sum() / max(agree.sum(), 1))
        direction_mag = min(abs(weighted_dir) / 2.0, 1.0)          # 2% move → 100%
        quality_mag = min(avg_quality / 0.10, 1.0)                 # IC*AUClift~0.10 → 100%
        agreement_frac = float(agree.sum() / len(agree))           # 0..1
        conv = direction_mag * quality_mag * agreement_frac
    else:
        weighted_dir = 0.0
        direction_mag = quality_mag = agreement_frac = 0.0
        conv = 0.0

    direction_label = ("BULL 🟢" if weighted_dir > 0.3
                        else "BEAR 🔴" if weighted_dir < -0.3
                        else "neutral ⚪")

    c1, c2, c3 = st.columns(3)
    c1.metric("Short avg (1w-3m)", f"{short_avg:+.2f}%",
              delta=f"P(+)={p_short:.0f}%")
    c2.metric("Long avg (6m-12m)",
              f"{long_avg:+.2f}%" if not np.isnan(long_avg) else "—",
              delta=f"P(+)={p_long:.0f}%" if not np.isnan(p_long) else "")
    c3.metric("Conviction", f"{conv*100:.0f}%", delta=direction_label)

    # Conviction breakdown caption
    st.caption(
        f"**Conviction breakdown** → "
        f"weighted forecast = `{weighted_dir:+.2f}%` (size factor {direction_mag*100:.0f}%) | "
        f"avg model quality `{avg_quality if total_w>0 else 0:.3f}` "
        f"(quality factor {quality_mag*100:.0f}%) | "
        f"horizon agreement `{int(agree.sum())}/{len(agree)}` "
        f"({agreement_frac*100:.0f}%) → "
        f"**{conv*100:.0f}% conviction**"
    )

    # ---- Forecast bar chart ----
    st.markdown("##### Per-horizon forecast bars")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pred_df["horizon"], y=pred_df["exp_pct"],
        text=[f"{v:+.2f}%<br>P(+)={p*100:.0f}%"
              for v, p in zip(pred_df["exp_pct"], pred_df["p_positive"])],
        textposition="outside",
        marker_color=["#26A69A" if v >= 0 else "#EF5350" for v in pred_df["exp_pct"]],
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(height=420, yaxis_title="Expected log-return (%)",
                      xaxis_title="Horizon", showlegend=False,
                      title=f"{target_l}: LightGBM forecast")
    st.plotly_chart(fig, use_container_width=True)

    # ---- OOS validation metrics ----
    st.markdown("##### Walk-forward OOS validation")
    st.caption("Honest out-of-sample performance from rolling 5y-train / 60d-test folds. "
               "IC = Spearman corr (regressor pred vs realized fwd return). "
               "AUC = classifier ROC-AUC. Hit = % of OOS bars where regressor "
               "sign matched realized sign.")
    val_df = pred_df[["horizon", "n_oos",
                      "ic_oos", "r2_oos", "hit_oos",
                      "auc_oos", "brier_oos",
                      "ic_clf_oos", "hit_clf_oos",
                      "log_loss_oos", "pseudo_r2_oos"]].copy()
    val_df["ic_oos"] = val_df["ic_oos"].round(4)
    val_df["r2_oos"] = val_df["r2_oos"].round(4)
    val_df["hit_oos"] = (val_df["hit_oos"] * 100).round(0)
    val_df["auc_oos"] = val_df["auc_oos"].round(3)
    val_df["brier_oos"] = val_df["brier_oos"].round(4)
    val_df["ic_clf_oos"] = val_df["ic_clf_oos"].round(4)
    val_df["hit_clf_oos"] = (val_df["hit_clf_oos"] * 100).round(0)
    val_df["log_loss_oos"] = val_df["log_loss_oos"].round(4)
    val_df["pseudo_r2_oos"] = val_df["pseudo_r2_oos"].round(4)
    val_df.columns = ["horizon", "n_OOS",
                      "REG_IC", "REG_R²", "REG_hit_%",
                      "CLF_AUC", "CLF_Brier",
                      "CLF_IC", "CLF_hit_%",
                      "CLF_LogLoss", "CLF_pseudoR²"]
    st.caption("**REG_*** = regressor metrics (predicting fwd log-return). "
               "**CLF_*** = classifier metrics (predicting P(fwd>0)). "
               "Columns side-by-side so you can see direction edge from both models.")
    st.dataframe(
        val_df.style
              .background_gradient(subset=["REG_IC"], cmap="RdYlGn", vmin=-0.1, vmax=0.25)
              .background_gradient(subset=["REG_R²"], cmap="RdYlGn", vmin=-0.05, vmax=0.1)
              .background_gradient(subset=["REG_hit_%"], cmap="Greens", vmin=45, vmax=80)
              .background_gradient(subset=["CLF_AUC"], cmap="RdYlGn", vmin=0.45, vmax=0.65)
              .background_gradient(subset=["CLF_Brier"], cmap="RdYlGn_r", vmin=0.18, vmax=0.27)
              .background_gradient(subset=["CLF_IC"], cmap="RdYlGn", vmin=-0.1, vmax=0.25)
              .background_gradient(subset=["CLF_hit_%"], cmap="Greens", vmin=45, vmax=80)
              .background_gradient(subset=["CLF_LogLoss"], cmap="RdYlGn_r", vmin=0.55, vmax=0.72)
              .background_gradient(subset=["CLF_pseudoR²"], cmap="RdYlGn", vmin=-0.02, vmax=0.05),
        use_container_width=True, hide_index=True,
    )

    # ---- Feature importance (regressor) ----
    st.markdown("##### Feature importance — regressor (gain-based)")
    st.caption("Per-horizon LGBM split-gain importance, normalized to 100% per column. "
               "Top features = those the model relied on most to predict that horizon.")
    fi_rows = []
    for h, m in bundle.horizons.items():
        total = sum(m.feature_importance_reg.values()) or 1
        for f, imp in m.feature_importance_reg.items():
            fi_rows.append({"feature": f, "horizon": h,
                            "importance": imp / total * 100})
    fi_df = pd.DataFrame(fi_rows)
    if not fi_df.empty:
        fi_pivot = fi_df.pivot(index="feature", columns="horizon", values="importance")
        # Order rows by mean importance
        fi_pivot["_avg"] = fi_pivot.mean(axis=1)
        fi_pivot = fi_pivot.sort_values("_avg", ascending=False).drop(columns="_avg")
        # Order columns canonically
        col_order = [h for h in horizon_labels() if h in fi_pivot.columns]
        fi_pivot = fi_pivot[col_order]
        st.dataframe(fi_pivot.style.background_gradient(cmap="Greens", vmin=0, vmax=15)
                     .format("{:.1f}"),
                     use_container_width=True)

    # ---- Feature importance (classifier) ----
    with st.expander("Feature importance — classifier"):
        fi_rows_c = []
        for h, m in bundle.horizons.items():
            if not m.feature_importance_clf: continue
            total = sum(m.feature_importance_clf.values()) or 1
            for f, imp in m.feature_importance_clf.items():
                fi_rows_c.append({"feature": f, "horizon": h,
                                  "importance": imp / total * 100})
        if fi_rows_c:
            fi_c = pd.DataFrame(fi_rows_c).pivot(index="feature", columns="horizon",
                                                  values="importance")
            fi_c["_avg"] = fi_c.mean(axis=1)
            fi_c = fi_c.sort_values("_avg", ascending=False).drop(columns="_avg")
            col_order = [h for h in horizon_labels() if h in fi_c.columns]
            fi_c = fi_c[col_order]
            st.dataframe(fi_c.style.background_gradient(cmap="Greens", vmin=0, vmax=15)
                         .format("{:.1f}"),
                         use_container_width=True)

    # ---- Current input row ----
    with st.expander("Current input vector (what the model sees)"):
        cur_dict = {f: float(cur_row_l[f]) if not pd.isna(cur_row_l.get(f)) else None
                    for f in bundle.features}
        st.dataframe(pd.DataFrame([cur_dict]).T.rename(columns={0: "value"}),
                     use_container_width=True)


# ===== Tab 1: Scatter =====
with tab_scatter:
    st.subheader("Feature vs forward log-return")

    mode = st.radio("Mode", ["Top 15 features (small multiples)", "Single feature (detailed)"],
                    horizontal=True, index=0)

    if mode.startswith("Top 15"):
        st.markdown("**Top 15 features by mean |Spearman ρ| across all horizons.**")
        # Rank features by mean |ρ|
        ranked = []
        for f in feature_columns():
            sps = []
            for h in horizon_labels():
                col = f"fwd_{h}"
                sub = panel.dropna(subset=[f, col])
                if len(sub) < 100: continue
                sps.append(abs(spearmanr(sub[f], sub[col]).statistic))
            if sps:
                ranked.append((f, float(np.mean(sps))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top10 = [f for f, _ in ranked[:15]]   # legacy var name kept; size now 15
        st.caption("Ranking: " + ", ".join(f"{f}({s:.2f})" for f, s in ranked[:15]))

        # Build long-format df for facet plot
        long_blocks = []
        for f in top10:
            for h in horizons_pick or horizon_labels():
                col = f"fwd_{h}"
                sub = panel.dropna(subset=[f, col])[[f, col]].copy()
                sub.columns = ["x", "fwd"]
                sub["feature"] = f
                sub["horizon"] = h
                long_blocks.append(sub)
        if long_blocks:
            big = pd.concat(long_blocks, ignore_index=True)
            # Subsample if huge
            if len(big) > 60_000:
                big = big.sample(60_000, random_state=0)
            fig = px.scatter(
                big, x="x", y="fwd", color="horizon",
                facet_col="feature", facet_col_wrap=3,
                category_orders={
                    "horizon": horizon_labels(),
                    "feature": top10,
                },
                opacity=0.35, trendline="ols", trendline_scope="trace",
                labels={"x": "feature value", "fwd": "fwd log-return"},
                height=2000,
            )
            fig.update_xaxes(matches=None, showticklabels=True)
            fig.update_yaxes(matches=None, showticklabels=True)

            # Map each feature → xaxis id by matching facet-title annotation
            # paper coordinates against each xaxis's paper domain.
            # Annotations carry "feature=NAME" text (still raw at this point).
            cur_sym_top = selected_syms[0]
            cur_panel = panel[panel["symbol"] == cur_sym_top].sort_values("date")
            cur_row_top = cur_panel.iloc[-1] if len(cur_panel) else None

            # Build full (xaxis_id, yaxis_id, x_mid, y_mid) for each subplot.
            # In plotly facet, xaxis N is paired with yaxis N (same trailing index).
            x_axes = {}  # 'x' / 'x2' -> (mid_x_paper)
            for ax_name in fig.layout:
                if not ax_name.startswith("xaxis"): continue
                ax = fig.layout[ax_name]
                dom = ax.domain
                if dom is None: continue
                ax_id = "x" + ax_name[len("xaxis"):]
                x_axes[ax_id] = (float(dom[0]) + float(dom[1])) / 2
            y_axes = {}
            for ax_name in fig.layout:
                if not ax_name.startswith("yaxis"): continue
                ay = fig.layout[ax_name]
                dom = ay.domain
                if dom is None: continue
                ay_id = "y" + ax_name[len("yaxis"):]
                y_axes[ay_id] = (float(dom[0]) + float(dom[1])) / 2

            # Build subplot list: (xaxis_id, yaxis_id, x_mid, y_mid)
            # Index suffix maps x↔y: 'x'/'y', 'x2'/'y2', ...
            subplots = []
            for x_id, x_mid in x_axes.items():
                suffix = x_id[1:]  # '' or '2','3',...
                y_id = "y" + suffix
                if y_id in y_axes:
                    subplots.append((x_id, y_id, x_mid, y_axes[y_id]))

            # Snapshot facet-title annotations
            facet_anns = []
            for ann in list(fig.layout.annotations):
                txt = (ann.text or "")
                label = txt.split("=", 1)[1].strip() if "=" in txt else txt.strip()
                if label not in top10:
                    continue
                if ann.xref != "paper" or ann.yref != "paper":
                    continue
                if ann.x is None or ann.y is None:
                    continue
                facet_anns.append((label, float(ann.x), float(ann.y)))

            # Match label → subplot by 2D Euclidean distance in paper coords
            feat_to_xref: dict[str, str] = {}
            used_subplots: set[str] = set()
            for label, ax_x, ax_y in facet_anns:
                if label in feat_to_xref: continue
                best = None; best_d = 1e9
                for x_id, y_id, xm, ym in subplots:
                    if x_id in used_subplots: continue
                    d = (xm - ax_x) ** 2 + (ym - ax_y) ** 2
                    if d < best_d:
                        best_d = d; best = (x_id, y_id)
                if best is not None:
                    feat_to_xref[label] = best[0]
                    used_subplots.add(best[0])

            # Now strip "feature=" from titles (display only)
            for ann in fig.layout.annotations:
                if ann.text and "=" in ann.text:
                    ann.text = ann.text.split("=", 1)[1]

            # Add vline at current value + dashed lines marking ±0.5σ window
            # (the window used in conditional win-rate / mean-fwd calc).
            if cur_row_top is not None:
                # Pre-compute std per feature from full panel (same source the
                # stats table uses — keeps window alignment consistent).
                feat_std = {}
                for f in top10:
                    s = panel[f].dropna()
                    feat_std[f] = float(s.std()) if len(s) > 30 else None

                for f in top10:
                    cv = cur_row_top.get(f)
                    if cv is None or pd.isna(cv): continue
                    xref = feat_to_xref.get(f)
                    if not xref: continue
                    yref = "y" + xref[1:]
                    cv = float(cv)
                    sigma = feat_std.get(f)
                    half = (sigma * 0.5) if sigma else None
                    try:
                        # Shaded ±0.5σ band (light yellow)
                        if half is not None:
                            fig.add_shape(
                                type="rect", xref=xref, yref=yref + " domain",
                                x0=cv - half, x1=cv + half, y0=0, y1=1,
                                fillcolor="rgba(255, 215, 0, 0.15)",
                                line=dict(width=0),
                                layer="below",
                            )
                        # Solid vline at current value
                        fig.add_shape(
                            type="line", xref=xref, yref=yref + " domain",
                            x0=cv, x1=cv, y0=0, y1=1,
                            line=dict(color="black", width=1.5, dash="dash"),
                        )
                        # Dotted lines at ±0.5σ edges
                        if half is not None:
                            fig.add_shape(
                                type="line", xref=xref, yref=yref + " domain",
                                x0=cv - half, x1=cv - half, y0=0, y1=1,
                                line=dict(color="rgba(170,140,0,0.85)",
                                          width=1, dash="dot"),
                            )
                            fig.add_shape(
                                type="line", xref=xref, yref=yref + " domain",
                                x0=cv + half, x1=cv + half, y0=0, y1=1,
                                line=dict(color="rgba(170,140,0,0.85)",
                                          width=1, dash="dot"),
                            )
                        # Label
                        label = (f"now={cv:.3f}  ±0.5σ=[{cv-half:.3f}, {cv+half:.3f}]"
                                 if half is not None else f"now={cv:.3f}")
                        fig.add_annotation(
                            xref=xref, yref=yref + " domain",
                            x=cv, y=1.02, text=label,
                            showarrow=False, font=dict(size=10, color="black"),
                            xanchor="left",
                        )
                    except Exception:
                        pass
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⬛ dashed line = current value | 🟨 shaded band = ±0.5σ window "
                       "used for conditional win-rate & mean-fwd calculations.")

            # ---- per-feature, per-horizon stats table (under the plots) ----
            st.markdown("##### Stats per feature × horizon")
            st.caption("n = sample size. pearson/spearman = correlation feature→fwd return. "
                       "slope/intercept = OLS line. mean_fwd_% = average % return at that horizon. "
                       "win_% = % of windows finished green.")
            # Use first selected symbol's current value for ±0.5σ conditional stats
            cur_sym_stats = selected_syms[0]
            cur_p = panel[panel["symbol"] == cur_sym_stats].sort_values("date")
            cur_row_stats = cur_p.iloc[-1] if len(cur_p) else None

            stat_rows = []
            for f in top10:
                cur_val = (float(cur_row_stats[f])
                           if cur_row_stats is not None and not pd.isna(cur_row_stats.get(f))
                           else None)
                for h in horizons_pick or horizon_labels():
                    col = f"fwd_{h}"
                    sub = panel.dropna(subset=[f, col])
                    if len(sub) < 30:
                        continue
                    x = sub[f].values
                    y = sub[col].values
                    pear = float(np.corrcoef(x, y)[0, 1])
                    sp = float(spearmanr(x, y).statistic)
                    slope, intercept = np.polyfit(x, y, 1)

                    # Conditional mean/win-rate at current feature value (±0.5σ window)
                    cond_mean_pct = np.nan; cond_win = np.nan; n_cond = 0
                    if cur_val is not None:
                        tol = sub[f].std() * 0.5
                        win_mask = (sub[f] >= cur_val - tol) & (sub[f] <= cur_val + tol)
                        cond_y = sub.loc[win_mask, col]
                        n_cond = int(len(cond_y))
                        if n_cond:
                            cond_mean_pct = float((np.exp(cond_y.mean()) - 1) * 100)
                            cond_win = float((cond_y > 0).mean()) * 100

                    stat_rows.append({
                        "feature": f, "horizon": h, "n": len(sub),
                        "pearson": round(pear, 3),
                        "spearman": round(sp, 3),
                        "slope": round(float(slope), 6),
                        "intercept": round(float(intercept), 6),
                        "n_cond": n_cond,
                        "mean_fwd_%": round(cond_mean_pct, 2) if not np.isnan(cond_mean_pct) else None,
                        "win_%": round(cond_win, 0) if not np.isnan(cond_win) else None,
                    })
            stats_df = pd.DataFrame(stat_rows)
            if not stats_df.empty:
                stats_df = stats_df.sort_values(["feature","horizon"])
                # Pivot summary: |spearman| heatmap by feature × horizon
                st.markdown("**|Spearman ρ| heatmap (feature × horizon)**")
                heat = stats_df.pivot(index="feature", columns="horizon",
                                      values="spearman").abs()
                heat = heat.reindex(index=top10, columns=horizon_labels())
                fig2 = go.Figure(data=go.Heatmap(
                    z=heat.values, x=heat.columns, y=heat.index,
                    zmin=0, zmax=0.3, colorscale="Greens",
                    text=heat.round(3).values, texttemplate="%{text}",
                ))
                fig2.update_layout(height=380)
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown(f"**Mean forward % return — conditional on {cur_sym_stats} at current value (±0.5σ window)**")
                mfp = stats_df.pivot(index="feature", columns="horizon", values="mean_fwd_%")
                mfp = mfp.reindex(index=top10, columns=horizons_pick or horizon_labels())
                st.dataframe(mfp.style.background_gradient(cmap="RdYlGn", vmin=-3, vmax=3),
                             use_container_width=True)

                st.markdown(f"**Win-rate % — conditional on {cur_sym_stats} at current value (±0.5σ window)**")
                wrp = stats_df.pivot(index="feature", columns="horizon", values="win_%")
                wrp = wrp.reindex(index=top10, columns=horizons_pick or horizon_labels())
                st.dataframe(wrp.style.background_gradient(cmap="Greens", vmin=40, vmax=70),
                             use_container_width=True)

                with st.expander("Full stats table (all columns)"):
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

if not mode.startswith("Top 15"):
  with tab_scatter:
    col1, col2 = st.columns([1, 3])
    with col1:
        feat = st.selectbox("Feature", options=feature_columns(),
                            index=feature_columns().index("atr_pct_14"))
        log_y = st.checkbox("Symlog y axis", value=False)
        sample_n = st.number_input("Subsample (0 = all)", value=0, min_value=0, step=500)

    df = panel.dropna(subset=[feat])
    long = []
    for h_lbl in horizons_pick:
        col = f"fwd_{h_lbl}"
        sub = df.dropna(subset=[col])[["symbol", "date", feat, col]].copy()
        sub["horizon"] = h_lbl
        sub.rename(columns={col: "fwd"}, inplace=True)
        long.append(sub)
    if not long:
        st.info("Pick at least one horizon"); st.stop()
    long_df = pd.concat(long, ignore_index=True)
    if sample_n > 0 and len(long_df) > sample_n:
        long_df = long_df.sample(sample_n, random_state=0)

    fig = px.scatter(
        long_df, x=feat, y="fwd", color="horizon",
        category_orders={"horizon": horizon_labels()},
        opacity=0.45,
        trendline="ols",
        trendline_scope="trace",
        hover_data=["symbol", "date"],
        labels={"fwd": "forward log-return"},
        height=600,
    )
    if log_y:
        fig.update_yaxes(type="log")
    # Mark current value (latest bar in panel for first selected sym)
    cur_sym = selected_syms[0]
    cur_row = panel[panel["symbol"] == cur_sym].sort_values("date").iloc[-1]
    cur_val = float(cur_row[feat]) if pd.notna(cur_row.get(feat)) else None
    if cur_val is not None:
        fig.add_vline(x=cur_val, line_dash="dash", line_color="gray",
                      annotation_text=f"{cur_sym} now = {cur_val:.4f}")
    st.plotly_chart(fig, use_container_width=True)

    # Per-horizon stats table
    rows = []
    for h_lbl in horizons_pick:
        col = f"fwd_{h_lbl}"
        sub = df.dropna(subset=[col])
        x = sub[feat].values
        y = sub[col].values
        if len(sub) < 30:
            rows.append({"horizon": h_lbl, "n": len(sub), "pearson": np.nan,
                         "spearman": np.nan, "slope": np.nan, "intercept": np.nan,
                         "mean_fwd": np.nan})
            continue
        pear = float(np.corrcoef(x, y)[0, 1])
        sp = float(spearmanr(x, y).statistic)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "horizon": h_lbl, "n": len(sub),
            "pearson": round(pear, 4),
            "spearman": round(sp, 4),
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 6),
            "mean_fwd": round(float(y.mean()), 5),
        })
    st.subheader("Per-horizon stats")
    stats_df = pd.DataFrame(rows).set_index("horizon")
    st.dataframe(stats_df, use_container_width=True)

    # Conditional fwd return at current value
    if cur_val is not None:
        st.subheader(f"Conditional forecast at {feat} ≈ {cur_val:.4f}")
        cond_rows = []
        for h_lbl in horizons_pick:
            col = f"fwd_{h_lbl}"
            sub = df.dropna(subset=[col])
            if len(sub) < 30: continue
            slope, intercept = np.polyfit(sub[feat].values, sub[col].values, 1)
            yhat = slope * cur_val + intercept
            # tolerance band: +/- 1 std of feat
            tol = sub[feat].std() * 0.5
            window = sub[(sub[feat] > cur_val - tol) & (sub[feat] < cur_val + tol)][col]
            cond_rows.append({
                "horizon": h_lbl,
                "ols_yhat": round(float(yhat), 5),
                "empirical_mean (±0.5σ window)": round(float(window.mean()), 5) if len(window) else np.nan,
                "n_window": len(window),
                "empirical_p_pos": round(float((window > 0).mean()), 3) if len(window) else np.nan,
            })
        st.dataframe(pd.DataFrame(cond_rows).set_index("horizon"), use_container_width=True)


# ===== Tab: What-if scenario =====
with tab_scenario:
    st.subheader("What-if scenario — drag feature values, watch forecast update")
    st.caption("Set hypothetical 'now' values for each top-10 feature. Conditional "
               "win-rate, mean fwd %, and scatter shading recompute live.")

    # Rank top10 features (cached-ish via spearmanr)
    ranked = []
    for f in feature_columns():
        sps = []
        for h in horizon_labels():
            sub = panel.dropna(subset=[f, f"fwd_{h}"])
            if len(sub) < 100: continue
            sps.append(abs(spearmanr(sub[f], sub[f"fwd_{h}"]).statistic))
        if sps:
            ranked.append((f, float(np.mean(sps))))
    ranked.sort(key=lambda x: x[1], reverse=True)
    top10_sc = [f for f, _ in ranked[:15]]   # legacy var; size now 15

    # Anchor symbol = first selected
    anchor_sym = selected_syms[0]
    sym_panel = panel[panel["symbol"] == anchor_sym].sort_values("date").reset_index(drop=True)
    if sym_panel.empty:
        st.warning(f"No data for {anchor_sym}"); st.stop()
    anchor_row = sym_panel.iloc[-1]

    # Reset button
    bcol1, bcol2 = st.columns([1, 5])
    if bcol1.button("Reset to actual now", key="scn_reset"):
        for f in top10_sc:
            st.session_state.pop(f"scn_{f}", None)
        st.rerun()
    bcol2.caption(f"Anchor symbol: **{anchor_sym}** (latest bar = "
                  f"{pd.to_datetime(anchor_row['date']).date()})")

    # Build sliders, 2 per row
    slider_vals: dict[str, float] = {}
    feat_std: dict[str, float] = {}
    for i in range(0, len(top10_sc), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(top10_sc): continue
            f = top10_sc[i + j]
            hist = panel[f].dropna()
            if len(hist) < 30: continue
            lo = float(hist.quantile(0.01))
            hi = float(hist.quantile(0.99))
            cur = float(anchor_row.get(f) or hist.median())
            cur = min(max(cur, lo), hi)
            step = max((hi - lo) / 200, 1e-6)
            slider_vals[f] = cols[j].slider(
                f"{f}", min_value=lo, max_value=hi,
                value=st.session_state.get(f"scn_{f}", cur),
                step=step, key=f"scn_{f}",
                help=f"Actual now = {anchor_row.get(f):.4f} | "
                     f"std={hist.std():.4f} | hist range [{lo:.3f}, {hi:.3f}]",
            )
            feat_std[f] = float(hist.std())

    # ---- Compute conditional forecast per feature × horizon at slider values ----
    rec_horiz = ["1w", "2w", "3w", "1m", "2m", "3m", "6m", "9m", "12m"]
    rows = []
    for f in top10_sc:
        cv = slider_vals[f]
        std = feat_std[f]
        tol = std * 0.5
        for h in rec_horiz:
            col = f"fwd_{h}"
            sub = panel.dropna(subset=[f, col])
            if len(sub) < 100: continue
            window = sub[(sub[f] >= cv - tol) & (sub[f] <= cv + tol)][col]
            if len(window) < 5: continue
            rows.append({
                "feature": f, "horizon": h,
                "n_window": len(window),
                "mean_fwd_%": round((np.exp(window.mean()) - 1) * 100, 2),
                "win_%": round((window > 0).mean() * 100, 0),
            })
    sc_df = pd.DataFrame(rows)

    if sc_df.empty:
        st.info("Move sliders inside the data range to populate stats."); st.stop()

    # Composite expected return per horizon (|ρ| weighted)
    weights = dict(ranked)
    comp_rows = []
    for h in rec_horiz:
        sub = sc_df[sc_df["horizon"] == h]
        if sub.empty: continue
        w = sub["feature"].map(weights).fillna(0).abs()
        if w.sum() <= 0: continue
        exp_pct = float((sub["mean_fwd_%"] * w).sum() / w.sum())
        win_avg = float((sub["win_%"] * w).sum() / w.sum())
        comp_rows.append({"horizon": h, "exp_%": round(exp_pct, 2),
                          "win_%": round(win_avg, 0)})
    comp_df = pd.DataFrame(comp_rows)

    # Composite metrics row
    st.markdown("##### Composite forecast (your sliders)")
    if not comp_df.empty:
        cols2 = st.columns(len(comp_df))
        for i, r in comp_df.iterrows():
            v = float(r["exp_%"])
            chip = ("🟢 BULL" if v > 1.0
                    else "🟩 lean bull" if v > 0.3
                    else "🔴 BEAR" if v < -1.0
                    else "🟥 lean bear" if v < -0.3
                    else "⚪ neutral")
            cols2[i].metric(r["horizon"], f"{v:+.2f}%",
                            delta=f"{chip}  {int(r['win_%'])}% win")

    # Heatmap-styled tables
    st.markdown("##### Mean fwd % per feature × horizon (your sliders)")
    mfp = sc_df.pivot(index="feature", columns="horizon", values="mean_fwd_%")
    mfp = mfp.reindex(index=top10_sc, columns=rec_horiz)
    st.dataframe(mfp.style.background_gradient(cmap="RdYlGn", vmin=-3, vmax=3),
                 use_container_width=True)

    st.markdown("##### Win-rate % per feature × horizon (your sliders)")
    wrp = sc_df.pivot(index="feature", columns="horizon", values="win_%")
    wrp = wrp.reindex(index=top10_sc, columns=rec_horiz)
    st.dataframe(wrp.style.background_gradient(cmap="Greens", vmin=40, vmax=70),
                 use_container_width=True)

    # ---- Scatter facets with the slider values as the vlines ----
    st.markdown("##### Scatter — vline & ±0.5σ band reflect your slider values")
    long_blocks = []
    for f in top10_sc:
        for h in horizons_pick or horizon_labels():
            col = f"fwd_{h}"
            sub = panel.dropna(subset=[f, col])[[f, col]].copy()
            sub.columns = ["x", "fwd"]
            sub["feature"] = f
            sub["horizon"] = h
            long_blocks.append(sub)
    if long_blocks:
        big = pd.concat(long_blocks, ignore_index=True)
        if len(big) > 60_000:
            big = big.sample(60_000, random_state=0)
        figS = px.scatter(
            big, x="x", y="fwd", color="horizon",
            facet_col="feature", facet_col_wrap=3,
            category_orders={
                "horizon": horizon_labels(),
                "feature": top10_sc,
            },
            opacity=0.30, trendline="ols", trendline_scope="trace",
            labels={"x": "feature value", "fwd": "fwd log-return"},
            height=2000,
        )
        figS.update_xaxes(matches=None, showticklabels=True)
        figS.update_yaxes(matches=None, showticklabels=True)

        # Reuse the same 2D paper-coord matching as scatter tab
        x_axes_s = {}
        for ax_name in figS.layout:
            if not ax_name.startswith("xaxis"): continue
            ax = figS.layout[ax_name]; dom = ax.domain
            if dom is None: continue
            x_axes_s["x" + ax_name[len("xaxis"):]] = (float(dom[0]) + float(dom[1])) / 2
        y_axes_s = {}
        for ax_name in figS.layout:
            if not ax_name.startswith("yaxis"): continue
            ay = figS.layout[ax_name]; dom = ay.domain
            if dom is None: continue
            y_axes_s["y" + ax_name[len("yaxis"):]] = (float(dom[0]) + float(dom[1])) / 2
        subplots_s = []
        for x_id, x_mid in x_axes_s.items():
            y_id = "y" + x_id[1:]
            if y_id in y_axes_s:
                subplots_s.append((x_id, y_id, x_mid, y_axes_s[y_id]))

        ann_list = []
        for ann in list(figS.layout.annotations):
            txt = (ann.text or "")
            label = txt.split("=", 1)[1].strip() if "=" in txt else txt.strip()
            if label not in top10_sc: continue
            if ann.xref != "paper" or ann.yref != "paper": continue
            if ann.x is None or ann.y is None: continue
            ann_list.append((label, float(ann.x), float(ann.y)))

        feat_to_xref_s: dict[str, str] = {}
        used_s: set[str] = set()
        for label, ax, ay in ann_list:
            if label in feat_to_xref_s: continue
            best = None; bd = 1e9
            for x_id, y_id, xm, ym in subplots_s:
                if x_id in used_s: continue
                d = (xm - ax) ** 2 + (ym - ay) ** 2
                if d < bd: bd = d; best = (x_id, y_id)
            if best:
                feat_to_xref_s[label] = best[0]
                used_s.add(best[0])

        for ann in figS.layout.annotations:
            if ann.text and "=" in ann.text:
                ann.text = ann.text.split("=", 1)[1]

        for f in top10_sc:
            cv = slider_vals.get(f)
            std = feat_std.get(f)
            xref = feat_to_xref_s.get(f)
            if cv is None or std is None or not xref: continue
            yref = "y" + xref[1:]
            half = std * 0.5
            try:
                figS.add_shape(type="rect", xref=xref, yref=yref+" domain",
                               x0=cv-half, x1=cv+half, y0=0, y1=1,
                               fillcolor="rgba(255,215,0,0.18)",
                               line=dict(width=0), layer="below")
                figS.add_shape(type="line", xref=xref, yref=yref+" domain",
                               x0=cv, x1=cv, y0=0, y1=1,
                               line=dict(color="black", width=1.5, dash="dash"))
                figS.add_annotation(xref=xref, yref=yref+" domain",
                                    x=cv, y=1.02,
                                    text=f"slider={cv:.3f}  ±0.5σ=[{cv-half:.3f},{cv+half:.3f}]",
                                    showarrow=False, xanchor="left",
                                    font=dict(size=10, color="black"))
            except Exception:
                pass
        st.plotly_chart(figS, use_container_width=True)


# ===== Tab 2: Correlation matrix =====
with tab_corr:
    st.subheader("Feature × horizon correlation matrix")
    cm_method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    feats_pick = st.multiselect(
        "Features (default = all)", options=feature_columns(),
        default=feature_columns(),
    )
    if feats_pick:
        out = pd.DataFrame(
            index=feats_pick,
            columns=[f"fwd_{h}" for h in horizons_pick or horizon_labels()],
            dtype=float,
        )
        for f in feats_pick:
            for h in horizons_pick or horizon_labels():
                col = f"fwd_{h}"
                sub = panel.dropna(subset=[f, col])
                if len(sub) < 30:
                    out.loc[f, col] = np.nan; continue
                if cm_method == "pearson":
                    out.loc[f, col] = float(np.corrcoef(sub[f], sub[col])[0, 1])
                else:
                    out.loc[f, col] = float(spearmanr(sub[f], sub[col]).statistic)
        out_num = out.astype(float)
        fig = go.Figure(data=go.Heatmap(
            z=out_num.values, x=out_num.columns, y=out_num.index,
            zmin=-0.3, zmax=0.3, colorscale="RdBu", reversescale=True,
            text=out_num.round(3).values, texttemplate="%{text}",
        ))
        fig.update_layout(height=max(300, 40 * len(feats_pick) + 200))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"n_obs varies by horizon (12m needs 252 days fwd). "
                   f"Total panel rows: {len(panel):,}")
        st.dataframe(out_num, use_container_width=True)


# ===== Tab 3: Quantile buckets =====
with tab_buckets:
    st.subheader("Mean forward return by feature quantile bucket")
    feat = st.selectbox("Feature", options=feature_columns(),
                        index=feature_columns().index("atr_pct_14"),
                        key="bucket_feat")
    n_buckets = st.slider("Buckets", min_value=3, max_value=20, value=10)
    df = panel.dropna(subset=[feat]).copy()
    try:
        df["bucket"] = pd.qcut(df[feat], q=n_buckets,
                               labels=[f"q{i+1}" for i in range(n_buckets)],
                               duplicates="drop")
    except Exception as e:
        st.error(f"Could not bucket: {e}"); st.stop()

    rows = []
    for h in horizons_pick or horizon_labels():
        col = f"fwd_{h}"
        sub = df.dropna(subset=[col])
        for b, grp in sub.groupby("bucket", observed=True):
            x = grp[col]
            if len(x) == 0: continue
            rows.append({
                "horizon": h, "bucket": b,
                "mean_fwd": float(x.mean()),
                "median_fwd": float(x.median()),
                "pct_green": float((x > 0).mean()),
                "pct_red":   float((x < 0).mean()),
                "best_pct":  float((np.exp(x.max())-1)*100),
                "worst_pct": float((np.exp(x.min())-1)*100),
                "n": int(len(x)),
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        st.info("No data"); st.stop()

    fig = px.bar(
        long_df, x="bucket", y="mean_fwd", color="horizon", barmode="group",
        category_orders={
            "bucket": [f"q{i+1}" for i in range(n_buckets)],
            "horizon": horizon_labels(),
        },
        labels={"mean_fwd": "mean fwd log-return"},
        height=500,
        hover_data=["n", "pct_green", "pct_red", "best_pct", "worst_pct"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Mean forward log-return per (bucket, horizon)")
    st.dataframe(long_df.pivot(index="bucket", columns="horizon", values="mean_fwd"),
                 use_container_width=True)

    st.markdown("##### Win-rate (% of windows that finished green) per (bucket, horizon)")
    win_df = long_df.copy()
    win_df["pct_green"] = (win_df["pct_green"] * 100).round(0)
    st.dataframe(win_df.pivot(index="bucket", columns="horizon", values="pct_green"),
                 use_container_width=True)

    st.markdown("##### Sample count per (bucket, horizon)")
    st.dataframe(long_df.pivot(index="bucket", columns="horizon", values="n"),
                 use_container_width=True)

    st.markdown("##### Best / worst single-window % return per bucket × horizon")
    st.caption("Tail behaviour. Read alongside win-rate to see if a positive mean comes from one giant rally or steady gains.")
    st.dataframe(long_df.pivot(index="bucket", columns="horizon", values="best_pct").round(1),
                 use_container_width=True)
    st.dataframe(long_df.pivot(index="bucket", columns="horizon", values="worst_pct").round(1),
                 use_container_width=True)


# ===== Tab 6: Volatility Regimes (Hidden Markov Model) =====
with tab_regime:
    from src.research import vol_regime as _vr

    st.subheader("🌡️ Volatility regimes — 5-state Hidden Markov Model")
    st.caption(
        "A Gaussian HMM segments history into five persistent volatility states "
        "from four daily features: **VIX** level, **realized (historical) "
        "volatility**, the **VIX3M − VIX** term spread, and a **running sum of "
        "the MACD histogram**. Latent states are ordered by mean VIX and labelled "
        "Extremely Low → Extremely High.")

    with st.expander("ℹ️ How to read this"):
        st.markdown("""
- **Persistent by design:** an HMM models the probability of *staying in* vs
  *switching* regimes, so labels don't flip on daily noise (see the transition matrix).
- **Term spread** (VIX3M − VIX): positive = calm contango; negative = stressed
  backwardation, typical of the Extremely-High regime.
- **MACD run-sum** on the underlying captures trend momentum accompanying the vol state.
- Regimes are ordered by average VIX, so *Extremely Low* is the calmest tape and
  *Extremely High* is crisis-like.
""")

    c1, c2, c3, c4 = st.columns(4)
    _under = c1.selectbox("Underlying (for realized vol + MACD)",
                          options=([s for s in ["SPY", "QQQ", "IWM", "DIA"] if s in syms]
                                   + [s for s in syms if s not in ("SPY", "QQQ", "IWM", "DIA")]),
                          index=0)
    _rv = c2.number_input("Realized-vol window (days)", 5, 120, 20, 1)
    _mf = c3.number_input("MACD fast EMA", 3, 30, 10, 1)
    _ms = c4.number_input("MACD slow EMA", 10, 60, 30, 1)
    _msum = st.slider("MACD histogram running-sum window (days)", 3, 40, 10)

    @st.cache_data(show_spinner="Fitting HMM volatility regimes...")
    def _cached_regimes(under, rv, mf, ms, msum):
        return _vr.fit_regimes(under, rv_window=rv, macd_fast=mf,
                               macd_slow=ms, macd_sum_window=msum)

    try:
        res = _cached_regimes(_under, int(_rv), int(_mf), int(_ms), int(_msum))
    except Exception as e:
        st.error(f"Could not fit regimes: {e}")
        st.stop()

    reg_df = res.df
    cur = res.current
    cur_color = _vr.REGIME_COLORS[cur]
    last = reg_df.iloc[-1]

    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{cur_color};"
        f"color:#111;font-weight:600;font-size:1.15rem'>Current regime "
        f"({reg_df.index.max().date()}): {cur}</div>",
        unsafe_allow_html=True)
    if res.method != "HMM":
        st.warning(f"hmmlearn unavailable — used {res.method}. Regimes are still "
                   "ordered by VIX but lack HMM transition dynamics.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("VIX", f"{last['vix']:.1f}")
    m2.metric(f"Realized vol {int(_rv)}d", f"{last['rv']:.1f}%")
    m3.metric("VIX3M − VIX", f"{last['term_spread']:+.2f}")
    m4.metric("MACD run-sum", f"{last['macd_runsum']:+.2f}")

    # Regime probabilities today
    probs = res.current_probs
    prob_df = pd.DataFrame({"regime": list(probs.keys()),
                            "probability": list(probs.values())})
    figp = px.bar(prob_df, x="regime", y="probability",
                  color="regime", color_discrete_map=_vr.REGIME_COLORS,
                  category_orders={"regime": _vr.REGIME_NAMES})
    figp.update_layout(showlegend=False, height=240,
                       margin=dict(l=10, r=10, t=30, b=10),
                       title="Today's state probabilities")
    st.plotly_chart(figp, use_container_width=True)

    # VIX timeline coloured by regime
    st.markdown("##### VIX history coloured by regime")
    plot_df = reg_df.reset_index().rename(columns={"index": "date"})
    if "date" not in plot_df.columns:
        plot_df = plot_df.rename(columns={plot_df.columns[0]: "date"})
    figt = px.scatter(plot_df, x="date", y="vix", color="regime",
                      color_discrete_map=_vr.REGIME_COLORS,
                      category_orders={"regime": _vr.REGIME_NAMES},
                      render_mode="webgl")
    figt.update_traces(marker=dict(size=3))
    figt.update_layout(height=380, legend_title="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figt, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("##### Per-regime feature averages")
        st.dataframe(res.regime_means.round(2), use_container_width=True)
        st.markdown("##### Days spent in each regime")
        vc = reg_df["regime"].value_counts().reindex(_vr.REGIME_NAMES).fillna(0).astype(int)
        share = (vc / vc.sum() * 100).round(1)
        st.dataframe(pd.DataFrame({"days": vc, "% of history": share}),
                     use_container_width=True)
    with colB:
        st.markdown("##### Transition matrix  P(next | current)")
        if res.transmat is not None:
            tm = pd.DataFrame(res.transmat, index=_vr.REGIME_NAMES,
                              columns=_vr.REGIME_NAMES)
            figm = go.Figure(go.Heatmap(
                z=tm.values, x=_vr.REGIME_NAMES, y=_vr.REGIME_NAMES,
                colorscale="Blues", zmin=0, zmax=1,
                text=np.round(tm.values, 2), texttemplate="%{text}"))
            figm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                               yaxis_title="current", xaxis_title="next")
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("Transition matrix requires the HMM (hmmlearn).")

    st.caption(
        f"Method: {res.method} · underlying {res.underlying} · "
        f"{len(reg_df):,} days ({reg_df.index.min().date()} to "
        f"{reg_df.index.max().date()}) · features standardized before fitting.")
