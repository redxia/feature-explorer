"""Daily refresh + retrain, headless. Mirrors the app's "Refresh data" button.

Run by GitHub Actions on a daily cron, then the workflow commits the updated
data/raw/daily/*.parquet and models/lgbm_dash/*.pkl back to the repo, which
triggers a Streamlit Cloud redeploy.

Design: DATA refresh and RETRAIN are decoupled. Fresh data is always committable
even if a retrain hiccups; and a bad retrain (throw or 0 horizons) restores the
previous good model from a backup so it can never clobber a working pickle.

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
    """Fetch macro/vol sources through the resilient yfinance path (browser
    impersonation + retries) so they update even from GitHub's runners."""
    from src.data import fetch_yf_daily as ft
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for sym, yfsym in MACRO_SOURCES.items():
        try:
            raw = ft._yf_download(yfsym, "max", auto_adjust=False, tries=3)
            df = ft._finalize(raw, sym)
            if df.empty:
                log.warning("macro %s: no data", sym)
                continue
            df.to_parquet(DAILY_DIR / f"{sym}.parquet", index=False)
            written += 1
            log.info("macro %s -> %d rows (last %s)", sym, len(df),
                     df["timestamp"].iloc[-1].date())
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
    """Retrain each model. train_symbol() overwrites the pickle itself, so we
    back up the existing good pickle first and RESTORE it if the retrain throws
    or yields 0 horizons — a bad retrain can never clobber a good model.
    Never raises: data refresh must be committable even if retraining hiccups.
    """
    import shutil
    trained = lgbm.list_trained()
    done = []
    for s in trained:
        pkl = lgbm.MODELS_DIR / f"{s}.pkl"
        bak = lgbm.MODELS_DIR / f"{s}.pkl.bak"
        try:
            if pkl.exists():
                shutil.copy2(pkl, bak)
            b = lgbm.train_symbol(s)          # train_symbol() saves the pickle itself
            if b.horizons:
                done.append(s)
                log.info("retrained %s: %d horizons", s, len(b.horizons))
                if bak.exists():
                    bak.unlink()
            else:
                log.error("retrain %s produced 0 horizons — restoring previous model", s)
                if bak.exists():
                    shutil.move(str(bak), str(pkl))
        except Exception as e:
            log.exception("retrain %s failed: %s — restoring previous model", s, e)
            if bak.exists():
                shutil.move(str(bak), str(pkl))
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
            # Do NOT fail the job: fresh DATA must still be committed even if a
            # retrain hiccups. Good models were restored from backup inside
            # retrain(), so nothing is lost — we keep the previous models.
            log.warning("No models retrained this run; keeping previous models. "
                        "Fresh data will still be committed.")

    # Report freshness so the Action log shows the latest bar date.
    try:
        import glob
        latest = max(pd.to_datetime(pd.read_parquet(p)["timestamp"]).max()
                     for p in glob.glob(str(DAILY_DIR / "*.parquet")))
        log.info("latest bar across daily files: %s", latest.date())
    except Exception:
        pass
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
