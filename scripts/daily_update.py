"""Daily refresh + retrain, headless. Mirrors the app's "Refresh data" button.

Run by GitHub Actions on a daily cron, then the workflow commits the updated
data/raw/daily/*.parquet and models/lgbm_dash/*.pkl back to the repo, which
triggers a Streamlit Cloud redeploy.

Usage:
    python -m scripts.daily_update            # full universe + retrain all trained
    python -m scripts.daily_update --years 20
    python -m scripts.daily_update --no-retrain
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.fetch_yf_daily import fetch_and_cache
from src.data.fetch_alpaca_stock_bars import load_universe_symbols
from src.research import lgbm_dash_model as lgbm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daily_update")

DAILY_DIR = ROOT / "data" / "raw" / "daily"

# Macro / volatility / rates sources the feature panel needs (yfinance tickers).
MACRO_SOURCES = {
    "VIX": "^VIX", "VIX3M": "^VIX3M", "VVIX": "^VVIX", "MOVE": "^MOVE",
    "TNX": "^TNX", "IRX": "^IRX",
    "TLT": "TLT", "HYG": "HYG", "LQD": "LQD",
}


def refresh_macro() -> int:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for sym, yfsym in MACRO_SOURCES.items():
        try:
            d = yf.download(yfsym, period="max", interval="1d",
                            auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
            if d is None or d.empty:
                log.warning("%s: no data", sym)
                continue
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
                DAILY_DIR / f"{sym}.parquet", index=False)
            written += 1
            log.info("macro %s -> %d rows", sym, len(d))
        except Exception as e:
            log.warning("macro %s failed: %s", sym, e)
    return written


def refresh_universe(years: int) -> int:
    try:
        targets = load_universe_symbols()
    except Exception as e:
        log.warning("universe load failed (%s); falling back to existing daily files", e)
        targets = sorted(p.stem for p in DAILY_DIR.glob("*.parquet")
                         if p.stem not in MACRO_SOURCES)
    written = 0
    for sym in targets:
        try:
            fetch_and_cache([sym], years=years)
            written += 1
        except Exception as e:
            log.warning("symbol %s failed: %s", sym, e)
    log.info("refreshed %d/%d universe symbols", written, len(targets))
    return written


def retrain() -> list[str]:
    trained = lgbm.list_trained()
    done = []
    for s in trained:
        try:
            b = lgbm.train_symbol(s)          # train_symbol() saves the pickle itself
            if b.horizons:
                done.append(s)
                log.info("retrained %s: %d horizons", s, len(b.horizons))
            else:
                log.error("retrain %s produced 0 horizons (kept old pickle? check data)", s)
        except Exception as e:
            log.exception("retrain %s failed: %s", s, e)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=20)
    ap.add_argument("--no-retrain", action="store_true")
    args = ap.parse_args()

    log.info("=== macro refresh ===")
    refresh_macro()
    log.info("=== universe refresh (%dy) ===", args.years)
    refresh_universe(args.years)

    if not args.no_retrain:
        log.info("=== retrain ===")
        done = retrain()
        if not done:
            log.error("No models retrained successfully — failing job so the "
                      "previous good pickles are NOT overwritten by empties.")
            return 1
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
