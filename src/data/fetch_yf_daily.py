"""Fetch daily bars from yfinance — long history (15y+).

Alpaca IEX feed caps ~5.7y; SIP feed paywalled. yfinance is free + has decades
of daily bars (SPY back to 1993).

Schema matches data/raw/daily/{SYM}.parquet so feature_panel.load_panel works
unchanged: timestamp(UTC), symbol, open, high, low, close, volume,
            vwap_alpaca(NaN), trade_count(NaN).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DAILY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily"


def _normalize(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    if df.empty:
        return df
    # yfinance returns multi-index columns when multiple tickers; for single
    # ticker it's flat. Handle both.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index().rename(columns={
        "Date": "timestamp",
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "Adj Close": "adj_close",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = sym
    df["vwap_alpaca"] = np.nan
    df["trade_count"] = np.nan
    cols = ["timestamp", "symbol", "open", "high", "low", "close",
            "volume", "vwap_alpaca", "trade_count"]
    return df[[c for c in cols if c in df.columns]].sort_values("timestamp").reset_index(drop=True)


def fetch_and_cache(symbols: list[str], years: int = 15,
                    out_dir: Path | None = None,
                    auto_adjust: bool = True) -> dict[str, Path]:
    out_dir = out_dir or DAILY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    period = f"{years}y" if years <= 10 else "max"
    for sym in symbols:
        # yfinance uses '-' for class shares (BRK-B, not BRK.B)
        yf_sym = sym.replace(".", "-")
        try:
            df = yf.download(yf_sym, period=period, interval="1d",
                             auto_adjust=auto_adjust, progress=False, threads=False)
        except Exception as e:
            logger.exception("%s download failed: %s", sym, e); continue
        if df is None or df.empty:
            logger.warning("%s: no data", sym); continue
        df = _normalize(df, sym)
        # Trim to requested years (yfinance 'max' returns full)
        if years and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365 + 30)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        path = out_dir / f"{sym}.parquet"
        df.to_parquet(path, index=False)
        paths[sym] = path
        logger.info("%s -> %d bars (%s..%s)", sym, len(df),
                    df["timestamp"].iloc[0].date() if len(df) else "?",
                    df["timestamp"].iloc[-1].date() if len(df) else "?")
        time.sleep(0.05)
    return paths


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--years", type=int, default=15)
    args = ap.parse_args()
    if args.symbols:
        symbols = args.symbols
    else:
        from src.data.fetch_alpaca_stock_bars import load_universe_symbols
        symbols = load_universe_symbols()
    paths = fetch_and_cache(symbols, years=args.years)
    print(f"Wrote {len(paths)}/{len(symbols)} daily parquets")
