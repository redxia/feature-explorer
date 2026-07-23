"""Fetch daily bars with CI-resilient sourcing.

Primary: yfinance with curl_cffi browser impersonation + retries.
Fallback: Stooq CSV (https://stooq.com) — a free source that reliably serves
from datacenter IPs (GitHub Actions), where Yahoo often throttles/returns empty.

Output schema matches data/raw/daily/{SYM}.parquet so feature_panel.load_panel
works unchanged: timestamp(UTC), symbol, open, high, low, close, volume,
vwap_alpaca(NaN), trade_count(NaN).
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

DAILY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily"


def _yf_session():
    """A curl_cffi session impersonating Chrome bypasses much of Yahoo's
    datacenter-IP blocking. Falls back to None (plain yfinance) if unavailable."""
    try:
        from curl_cffi import requests as cr
        return cr.Session(impersonate="chrome")
    except Exception:
        return None


_YF_SESSION = _yf_session()


def _finalize(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """Coerce any yfinance/Stooq daily frame into the canonical schema."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Flatten yfinance MultiIndex columns (single-ticker sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Only lift the index into a column when it is the datetime axis (yfinance);
    # for a plain RangeIndex (Stooq CSV already has a Date column) drop it so we
    # don't create a duplicate "timestamp".
    if isinstance(df.index, pd.DatetimeIndex) or str(df.index.name).strip().lower() in ("date", "datetime", "timestamp"):
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)
    ren = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("date", "timestamp", "datetime"):
            ren[c] = "timestamp"
        elif cl == "open":
            ren[c] = "open"
        elif cl == "high":
            ren[c] = "high"
        elif cl == "low":
            ren[c] = "low"
        elif cl in ("close", "adj close", "adj_close"):
            ren.setdefault(c, "close")
        elif cl == "volume":
            ren[c] = "volume"
    df = df.rename(columns=ren)
    df = df.loc[:, ~df.columns.duplicated()]
    if "timestamp" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = sym
    df["vwap_alpaca"] = np.nan
    df["trade_count"] = np.nan
    cols = ["timestamp", "symbol", "open", "high", "low", "close",
            "volume", "vwap_alpaca", "trade_count"]
    keep = [c for c in cols if c in df.columns]
    return (df[keep].dropna(subset=["timestamp", "close"])
            .sort_values("timestamp").reset_index(drop=True))


def _yf_download(yf_sym: str, period: str, auto_adjust: bool, tries: int = 3):
    for i in range(tries):
        try:
            kw = dict(period=period, interval="1d", auto_adjust=auto_adjust,
                      progress=False, threads=False)
            if _YF_SESSION is not None:
                df = yf.download(yf_sym, session=_YF_SESSION, **kw)
            else:
                df = yf.download(yf_sym, **kw)
            if df is not None and not df.empty:
                return df
            logger.info("yf %s empty (attempt %d/%d)", yf_sym, i + 1, tries)
        except Exception as e:
            logger.warning("yf %s attempt %d/%d failed: %s", yf_sym, i + 1, tries, e)
        time.sleep(1.5 * (i + 1))
    return None


def _stooq_download(sym: str):
    """Free daily CSV from Stooq. US tickers use a `.us` suffix and `-` for
    class shares (BRK.B -> brk-b.us)."""
    s = sym.lower().replace(".", "-")
    url = f"https://stooq.com/q/d/l/?s={s}.us&i=d"
    try:
        r = requests.get(url, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
        txt = r.text or ""
        if r.status_code != 200 or txt.startswith("<") or "Date" not in txt[:64]:
            logger.info("stooq %s: no usable data", sym)
            return None
        return pd.read_csv(io.StringIO(txt))
    except Exception as e:
        logger.warning("stooq %s failed: %s", sym, e)
        return None


def fetch_and_cache(symbols: list[str], years: int = 15,
                    out_dir: Path | None = None,
                    auto_adjust: bool = True) -> dict[str, Path]:
    out_dir = out_dir or DAILY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    period = f"{years}y" if years <= 10 else "max"
    for sym in symbols:
        yf_sym = sym.replace(".", "-")
        raw = _yf_download(yf_sym, period, auto_adjust)
        source = "yfinance"
        if raw is None or raw.empty:
            raw = _stooq_download(sym)
            source = "stooq"
        df = _finalize(raw, sym)
        if df.empty:
            logger.warning("%s: no data from yfinance or stooq", sym)
            continue
        if years and not df.empty:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365 + 30)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        path = out_dir / f"{sym}.parquet"
        df.to_parquet(path, index=False)
        paths[sym] = path
        logger.info("%s <- %s: %d bars (%s..%s)", sym, source, len(df),
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
