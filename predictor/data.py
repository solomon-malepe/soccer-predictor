"""Load and normalise match data from football-data.co.uk.

Season files:   https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
Upcoming games: https://www.football-data.co.uk/fixtures.csv

Every loaded DataFrame is normalised to these columns:

    date, league, season, home, away, hg, ag          (hg/ag NaN for fixtures)
    pin_H pin_D pin_A pin_O25 pin_U25                 Pinnacle pre-match odds
    pinc_H pinc_D pinc_A pinc_O25 pinc_U25            Pinnacle closing odds
    soft_H soft_D soft_A soft_O25 soft_U25            best price across your soft books
    softbook_H ... softbook_U25                       which soft book had that price
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

OUTCOMES = ["H", "D", "A", "O25", "U25"]

# book code -> (1X2 column prefix, over/under 2.5 column prefix)
BOOK_PREFIXES = {
    "PS": ("PS", "P"),       # Pinnacle
    "B365": ("B365", "B365"),
    "BW": ("BW", "BW"),
    "WH": ("WH", "WH"),
    "IW": ("IW", "IW"),
    "VC": ("VC", "VC"),
    "1XB": ("1XB", "1XB"),
    "BF": ("BF", "BF"),
    "Max": ("Max", "Max"),
    "Avg": ("Avg", "Avg"),
}

# Betway SA and SportyBet are not in football-data's history, so the backtest
# uses mainstream soft books as a stand-in for your two accounts.
DEFAULT_SOFT_BOOKS = ["B365", "WH"]


def _odds_columns(prefix_1x2: str, prefix_ou: str, closing: bool) -> dict[str, str]:
    c = "C" if closing else ""
    return {
        "H": f"{prefix_1x2}{c}H",
        "D": f"{prefix_1x2}{c}D",
        "A": f"{prefix_1x2}{c}A",
        "O25": f"{prefix_ou}{c}>2.5",
        "U25": f"{prefix_ou}{c}<2.5",
    }


def _read_csv_text(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    df.columns = [str(col).strip().lstrip("﻿") for col in df.columns]
    return df


def _fetch(url: str) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 soccer-predictor"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def download_season(league: str, season: str, cache_dir: str | Path = "data/raw",
                    refresh: bool = False) -> Path:
    """Download one season CSV (season like '2425') into the cache, return its path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{league}_{season}.csv"
    if path.exists() and not refresh:
        return path
    text = _fetch(BASE_URL.format(season=season, league=league))
    path.write_text(text, encoding="utf-8")
    return path


def normalise(raw: pd.DataFrame, soft_books: list[str] | None = None,
              league: str | None = None, season: str | None = None) -> pd.DataFrame:
    """Turn a raw football-data frame into the normalised schema."""
    soft_books = soft_books or DEFAULT_SOFT_BOOKS
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"]).copy()

    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed", errors="coerce")
    out["league"] = raw["Div"] if "Div" in raw.columns else league
    if league is not None:
        out["league"] = out["league"].fillna(league)
    out["season"] = season
    out["home"] = raw["HomeTeam"].astype(str).str.strip()
    out["away"] = raw["AwayTeam"].astype(str).str.strip()
    out["hg"] = pd.to_numeric(raw.get("FTHG"), errors="coerce") if "FTHG" in raw else np.nan
    out["ag"] = pd.to_numeric(raw.get("FTAG"), errors="coerce") if "FTAG" in raw else np.nan

    def col(name: str) -> pd.Series:
        if name in raw.columns:
            return pd.to_numeric(raw[name], errors="coerce")
        return pd.Series(np.nan, index=raw.index)

    # Pinnacle: pre-match and closing
    for closing, tag in ((False, "pin"), (True, "pinc")):
        cols = _odds_columns(*BOOK_PREFIXES["PS"], closing=closing)
        for o in OUTCOMES:
            out[f"{tag}_{o}"] = col(cols[o])

    # Soft books: best available pre-match price (line shopping across accounts)
    for o in OUTCOMES:
        prices = pd.DataFrame({
            b: col(_odds_columns(*BOOK_PREFIXES.get(b, (b, b)), closing=False)[o])
            for b in soft_books
        })
        # Fall back to market average where none of the chosen books priced it
        fallback = col(_odds_columns(*BOOK_PREFIXES["Avg"], closing=False)[o])
        arr = prices.to_numpy(dtype=float)
        has_price = ~np.isnan(arr).all(axis=1)
        filled = np.where(np.isnan(arr), -np.inf, arr)
        idx = filled.argmax(axis=1)
        best = np.where(has_price, filled[np.arange(len(arr)), idx], np.nan)
        books = np.array(soft_books, dtype=object)[idx]
        out[f"soft_{o}"] = pd.Series(best, index=raw.index).fillna(fallback)
        out[f"softbook_{o}"] = np.where(has_price, books, "Avg")

    out = out.dropna(subset=["date"])
    return out.sort_values("date").reset_index(drop=True)


def load_seasons(league: str, seasons: list[str], cache_dir: str | Path = "data/raw",
                 soft_books: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load several seasons (downloading if needed) into one normalised frame."""
    frames = []
    for season in seasons:
        path = download_season(league, season, cache_dir=cache_dir, refresh=refresh)
        raw = _read_csv_text(path.read_text(encoding="utf-8"))
        frames.append(normalise(raw, soft_books=soft_books, league=league, season=season))
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return df


def load_fixtures(leagues: list[str] | None = None, soft_books: list[str] | None = None,
                  path: str | Path | None = None) -> pd.DataFrame:
    """Load upcoming fixtures (with odds) for the given leagues."""
    if path is not None:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = _fetch(FIXTURES_URL)
    raw = _read_csv_text(text)
    if leagues:
        raw = raw[raw["Div"].isin(leagues)]
    return normalise(raw, soft_books=soft_books)


def season_codes(start_year: int, end_year: int) -> list[str]:
    """season_codes(2019, 2024) -> ['1920', '2021', ..., '2425']"""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start_year, end_year + 1)]
