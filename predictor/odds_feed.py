"""Live Pinnacle odds from OddsPapi (https://oddspapi.io), free tier: 250 requests/month.

Only Pinnacle comes from here. OddsPapi's "betway" feed is betway.com, not Betway SA,
and its SportyBet feed is patchy, so those prices are still entered by hand
(checked on 15 Nov 2026 against the SA apps).

Request cost:
  upcoming Pinnacle odds, all 5 leagues   2 requests (3 leagues per call), cached 30 min
  team names (participants)               1 request, cached 30 days
  closing odds for one pick               1 request (+1 to find the fixture if needed)

Every API call is counted in data/oddspapi/usage.json so you can see the month's total.
The key is read from ODDS_API_KEY (environment or .env).
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .teams import map_names

BASE_URL = "https://api.oddspapi.io/v4"
SOCCER = 10
TOURNAMENTS = {"E0": 17, "SP1": 8, "I1": 23, "D1": 35, "F1": 34}
LEAGUE_OF = {v: k for k, v in TOURNAMENTS.items()}
# outcome -> (market id, outcome id); 101 = full-time 1X2, 1010 = total goals 2.5
# outcome -> (OddsPapi marketId, outcomeId), from GET /markets?sportId=10
MARKET_IDS = {"H": ("101", "101"), "D": ("101", "102"), "A": ("101", "103"),
              "O25": ("1010", "1010"), "U25": ("1010", "1011"),
              # experimental markets; Pinnacle often posts these only a day or two before kickoff
              "BTTS_Y": ("104", "104"), "BTTS_N": ("104", "105"),
              "O15": ("108", "108"), "U15": ("108", "109"),
              "O35": ("1012", "1012"), "U35": ("1012", "1013")}
COOLDOWN_S = {"historical-odds": 5.0, "fixtures": 2.0}   # others: 1s
FREE_MONTHLY = 250

Fetch = Callable[[str], tuple[int, object]]


class OddsPapiError(RuntimeError):
    pass


def _http_get(url: str) -> tuple[int, object]:
    req = urllib.request.Request(url, headers={"User-Agent": "soccer-predictor"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {}


def load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("ODDS_API_KEY") or None


class OddsPapi:
    def __init__(self, api_key: str | None = None, cache_dir: str | Path = "data/oddspapi",
                 fetch: Fetch | None = None, sleep: Callable[[float], None] = time.sleep):
        self.api_key = api_key or load_api_key()
        if not self.api_key:
            raise OddsPapiError("No ODDS_API_KEY found. Put ODDS_API_KEY=... in soccer-predictor/.env")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetch = fetch or _http_get
        self._sleep = sleep
        self._last_call: dict[str, float] = {}
        self.calls = 0

    # ---- plumbing -------------------------------------------------------------------
    def _count(self) -> int:
        path = self.cache_dir / "usage.json"
        try:
            usage = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            usage = {}
        month = dt.date.today().strftime("%Y-%m")
        usage[month] = usage.get(month, 0) + 1
        path.write_text(json.dumps(usage, indent=1))
        self.calls += 1
        return usage[month]

    def month_usage(self) -> int:
        try:
            return json.loads((self.cache_dir / "usage.json").read_text()).get(
                dt.date.today().strftime("%Y-%m"), 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    def _get(self, endpoint: str, params: dict, cache: str | None = None,
             max_age_s: float | None = None):
        if cache and max_age_s:
            path = self.cache_dir / f"{cache}.json"
            if path.exists() and time.time() - path.stat().st_mtime < max_age_s:
                return json.loads(path.read_text(encoding="utf-8"))
        url = f"{BASE_URL}/{endpoint}?" + urllib.parse.urlencode({**params, "apiKey": self.api_key})
        for attempt in range(3):
            wait = COOLDOWN_S.get(endpoint, 1.0) - (time.time() - self._last_call.get(endpoint, 0))
            if wait > 0:
                self._sleep(wait)
            status, body = self._fetch(url)
            self._last_call[endpoint] = time.time()
            self._count()
            if status == 429 and attempt < 2:
                retry_ms = (body.get("error", {}) if isinstance(body, dict) else {}).get("retryMs", 2000)
                self._sleep(float(retry_ms) / 1000 + 0.2)
                continue
            break
        if status == 404 and isinstance(body, dict) and body.get("error", {}).get("code", "").endswith("NOT_FOUND"):
            return []
        if status != 200:
            msg = body.get("error", {}).get("message") if isinstance(body, dict) else body
            raise OddsPapiError(f"OddsPapi {endpoint} failed (HTTP {status}): {msg}")
        if cache:
            (self.cache_dir / f"{cache}.json").write_text(json.dumps(body), encoding="utf-8")
        return body

    # ---- endpoints ------------------------------------------------------------------
    def participants(self) -> dict[str, str]:
        return self._get("participants", {"sportId": SOCCER}, cache="participants",
                         max_age_s=30 * 86400)

    def upcoming_odds(self, leagues, bookmaker: str = "pinnacle", max_age_min: float = 30,
                      keep_snapshots: bool = True) -> list[dict]:
        """Every upcoming fixture in these leagues with this bookmaker's current odds."""
        snapshot_dir = self.cache_dir / "snapshots" if keep_snapshots else None
        ids = [TOURNAMENTS[lg] for lg in leagues]
        out = []
        for i in range(0, len(ids), 3):
            chunk = ids[i:i + 3]
            key = f"odds_{bookmaker}_{'-'.join(map(str, chunk))}"
            fresh = not (self.cache_dir / f"{key}.json").exists() or \
                time.time() - (self.cache_dir / f"{key}.json").stat().st_mtime >= max_age_min * 60
            body = self._get("odds-by-tournaments",
                             {"tournamentIds": ",".join(map(str, chunk)), "bookmaker": bookmaker},
                             cache=key, max_age_s=max_age_min * 60)
            if fresh and snapshot_dir:   # keep what the odds were when picks were made
                snap = Path(snapshot_dir)
                snap.mkdir(parents=True, exist_ok=True)
                stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                with gzip.open(snap / f"{key}_{stamp}.json.gz", "wt", encoding="utf-8") as f:
                    json.dump(body, f)
            out.extend(body)
        return out

    def finished_fixtures(self, league: str, date_from, date_to) -> list[dict]:
        return self._get("fixtures", {"tournamentId": TOURNAMENTS[league], "statusId": 2,
                                      "from": str(pd.Timestamp(date_from).date()),
                                      "to": str(pd.Timestamp(date_to).date())})

    def history(self, fixture_id: str, bookmakers=("pinnacle",)) -> dict:
        return self._get("historical-odds", {"fixtureId": fixture_id,
                                             "bookmakers": ",".join(bookmakers)})


# ---- parsing (pure functions, unit-tested) ----------------------------------------------
def extract_prices(markets: dict) -> dict[str, float]:
    """Current 1X2 and O/U 2.5 prices from a bookmaker's `markets` block."""
    out = {}
    for o, (mid, oid) in MARKET_IDS.items():
        p = markets.get(mid, {}).get("outcomes", {}).get(oid, {}).get("players", {}).get("0", {})
        price = p.get("price")
        out[o] = float(price) if price and p.get("active", True) else np.nan
    return out


def closing_from_history(hist: dict, kickoff, bookmaker: str = "pinnacle") -> dict[str, float]:
    """Last price before kickoff for each outcome (later snapshots are in-play)."""
    ko = pd.Timestamp(kickoff)
    ko = ko.tz_localize("UTC") if ko.tzinfo is None else ko
    markets = hist.get("bookmakers", {}).get(bookmaker, {}).get("markets", {})
    out = {}
    for o, (mid, oid) in MARKET_IDS.items():
        snaps = markets.get(mid, {}).get("outcomes", {}).get(oid, {}).get("players", {}).get("0", [])
        pre = [s for s in snaps if s.get("price") and pd.Timestamp(s["createdAt"]) < ko]
        out[o] = float(max(pre, key=lambda s: pd.Timestamp(s["createdAt"]))["price"]) if pre else np.nan
    return out


def fixtures_frame(raw: list[dict], names: dict[str, str], fd_names: dict[str, list[str]],
                   bookmaker: str = "pinnacle") -> tuple[pd.DataFrame, list[str]]:
    """Turn odds-by-tournaments output into predict.py's fixture schema
    (date, league, home, away, pin_*, pinc_*, soft_*, softbook_*) plus fixture_id/kickoff.
    Returns (frame, unmatched team names)."""
    rows, unmatched = [], []
    by_league: dict[str, list[dict]] = {}
    raw = list({fx["fixtureId"]: fx for fx in raw}.values())
    for fx in raw:
        lg = LEAGUE_OF.get(fx.get("tournamentId"))
        if lg and fx.get("statusId", 0) == 0:
            by_league.setdefault(lg, []).append(fx)
    for lg, fixtures in by_league.items():
        api_names = {names.get(str(fx[k]), str(fx[k])) for fx in fixtures
                     for k in ("participant1Id", "participant2Id")}
        mapping, miss = map_names(api_names, fd_names.get(lg, []))
        unmatched += [f"{lg}: {m}" for m in miss]
        for fx in fixtures:
            h = mapping.get(names.get(str(fx["participant1Id"])))
            a = mapping.get(names.get(str(fx["participant2Id"])))
            if not (h and a):
                continue
            ko = pd.Timestamp(fx["startTime"])
            prices = extract_prices(fx.get("bookmakerOdds", {}).get(bookmaker, {}).get("markets", {}))
            rec = {"date": ko.tz_convert(None).normalize(), "kickoff": ko, "league": lg,
                   "season": None, "home": h, "away": a, "hg": np.nan, "ag": np.nan,
                   "fixture_id": fx["fixtureId"]}
            for o in MARKET_IDS:   # every market we read, including the experimental ones
                rec[f"pin_{o}"] = prices[o]
                rec[f"pinc_{o}"] = np.nan
                rec[f"soft_{o}"] = np.nan
                rec[f"softbook_{o}"] = ""
            rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["kickoff", "league", "home"]).reset_index(drop=True)
    return df, sorted(set(unmatched))


def live_fixtures(client: OddsPapi, leagues, fd_names: dict[str, list[str]],
                  days: float = 7, now=None) -> tuple[pd.DataFrame, list[str]]:
    """Upcoming fixtures (next `days` days) with Pinnacle's current prices."""
    raw = client.upcoming_odds(leagues)
    df, unmatched = fixtures_frame(raw, client.participants(), fd_names)
    if len(df):
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        df = df[(df["kickoff"] > now) & (df["kickoff"] <= now + pd.Timedelta(days=days))]
    return df.reset_index(drop=True), unmatched


def find_fixture_id(client: OddsPapi, league: str, date, home: str, away: str,
                    fd_names: list[str]) -> tuple[str | None, str | None]:
    """Look up a finished fixture by football-data names. Returns (fixture_id, kickoff)."""
    d = pd.Timestamp(date)
    raw = client.finished_fixtures(league, d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))
    names = client.participants()
    for fx in raw:
        api = [names.get(str(fx["participant1Id"]), ""), names.get(str(fx["participant2Id"]), "")]
        mapping, _ = map_names(api, fd_names)
        if mapping.get(api[0]) == home and mapping.get(api[1]) == away:
            return fx["fixtureId"], fx["startTime"]
    return None, None
