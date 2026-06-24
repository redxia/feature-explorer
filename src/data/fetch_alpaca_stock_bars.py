"""Fetch historical equity bars from Alpaca, RTH-only, cache to parquet.

Endpoint: /v2/stocks/bars (multi-symbol).  Auth required.
Output: data/raw/equity/{SYMBOL}_{TIMEFRAME}.parquet

Schema: timestamp(UTC), symbol, open, high, low, close, volume, trade_count,
        vwap_alpaca, volume_usd.

RTH filter applied post-fetch (timestamp tz-converted to ET, keep 09:30-16:00).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
PAGE_LIMIT = 10_000
ET = "America/New_York"


def _headers() -> dict[str, str]:
    key = os.environ.get("paper_alpaca_key", "")
    secret = os.environ.get("paper_alpaca_secret", "")
    if not (key and secret):
        raise RuntimeError("Set paper_alpaca_key / paper_alpaca_secret env vars")
    return {
        "Accept": "application/json",
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def fetch_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    timeframe: str = "5Min",
    feed: str = "iex",   # 'iex' free, 'sip' paid
    adjustment: str = "split",
) -> dict[str, pd.DataFrame]:
    """Fetch bars for multiple symbols, paginating per request."""
    out: dict[str, list[dict]] = {s: [] for s in symbols}
    page_token: str | None = None
    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "limit": PAGE_LIMIT,
            "feed": feed,
            "adjustment": adjustment,
        }
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(BARS_URL, headers=_headers(), params=params, timeout=60)
        if resp.status_code == 429:
            logger.warning("Rate-limited, sleeping 5s")
            time.sleep(5); continue
        resp.raise_for_status()
        payload = resp.json()
        bars_by_symbol = payload.get("bars", {}) or {}
        for sym, bars in bars_by_symbol.items():
            out[sym].extend(bars)
        page_token = payload.get("next_page_token")
        n_total = sum(len(v) for v in out.values())
        logger.info("page: total_bars=%d token=%s", n_total, "yes" if page_token else "no")
        if not page_token:
            break
        time.sleep(0.05)

    dfs: dict[str, pd.DataFrame] = {}
    for sym, bars in out.items():
        if not bars:
            logger.warning("no bars: %s", sym)
            continue
        df = pd.DataFrame(bars).rename(columns={
            "t": "timestamp", "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "vw": "vwap_alpaca", "n": "trade_count",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = sym
        df["volume_usd"] = df["close"] * df["volume"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        dfs[sym] = df
    return dfs


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts_et = df["timestamp"].dt.tz_convert(ET)
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    weekday = ts_et.dt.weekday  # 0=Mon..4=Fri
    mask = (minutes >= 9 * 60 + 30) & (minutes < 16 * 60) & (weekday < 5)
    return df.loc[mask].reset_index(drop=True)


def fetch_and_cache(
    symbols: list[str],
    days: int = 365,
    timeframe: str = "5Min",
    feed: str = "iex",
    out_dir: Path | None = None,
    batch_size: int = 50,
) -> dict[str, Path]:
    out_dir = out_dir or Path(__file__).resolve().parents[2] / "data" / "raw" / "equity"
    out_dir.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)

    paths: dict[str, Path] = {}
    # Alpaca multi-symbol limit ~200; batch to be safe
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        logger.info("Batch %d/%d: %s", i // batch_size + 1,
                    (len(symbols) + batch_size - 1) // batch_size, batch[:5])
        try:
            dfs = fetch_bars(batch, start, end, timeframe=timeframe, feed=feed)
        except Exception as e:
            logger.exception("batch failed: %s", e)
            continue
        for sym, df in dfs.items():
            df = filter_rth(df)
            if df.empty:
                logger.warning("RTH filter left 0 bars for %s", sym)
                continue
            path = out_dir / f"{sym}_{timeframe}.parquet"
            df.to_parquet(path, index=False)
            paths[sym] = path
            logger.info("%s -> %d bars %s", sym, len(df), path.name)
    return paths


def load_universe_symbols() -> list[str]:
    """Resolve full equity universe from config/universe.yaml."""
    import yaml
    path = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"
    cfg = yaml.safe_load(path.read_text())
    eq = cfg["universe"]["equity"]
    syms: list[str] = []
    for key in ("sp100", "broad_etfs", "sector_spdrs", "thematic_etfs", "leveraged_etfs"):
        syms.extend(eq.get(key, []))
    # Alpaca uses BRK.B not BRK/B; also map BF.B etc if present
    # Alpaca expects BRK.B (with dot); leave dots intact
    # de-dupe preserve order
    seen, uniq = set(), []
    for s in syms:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--timeframe", default="5Min")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="Override symbol list (default: full equity universe)")
    args = ap.parse_args()

    symbols = args.symbols or load_universe_symbols()
    print(f"Fetching {len(symbols)} symbols, {args.days}d, {args.timeframe}, feed={args.feed}")
    paths = fetch_and_cache(symbols, days=args.days, timeframe=args.timeframe, feed=args.feed)
    print(f"Wrote {len(paths)} parquets")
