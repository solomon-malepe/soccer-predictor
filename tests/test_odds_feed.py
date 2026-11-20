"""OddsPapi client tests. All API responses are mocked; no real requests are made."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictor.odds_feed import (OddsPapi, closing_from_history, extract_prices, fixtures_frame,
                                 live_fixtures)
from predictor.paper import add_picks, empty_log, fill_closing
from predictor.teams import map_names, match_team

DATA = Path(__file__).parent / "data"

MAIN = ["H", "D", "A", "O25", "U25"]


def _price(p, active=True, ts="2026-11-15T10:00:00Z"):
    return {"players": {"0": {"price": p, "active": active, "changedAt": ts}}}


def _markets(h, d, a, o, u):
    return {"101": {"outcomes": {"101": _price(h), "102": _price(d), "103": _price(a)}},
            "1010": {"outcomes": {"1010": _price(o), "1011": _price(u)}}}


PARTICIPANTS = {"17": "Manchester City", "43": "Fulham FC", "44": "Liverpool FC",
                "45": "Manchester United", "99": "Gotham Rovers"}

UPCOMING = [
    {"fixtureId": "f1", "participant1Id": 17, "participant2Id": 43, "tournamentId": 17, "statusId": 0,
     "startTime": "2026-11-21T12:30:00.000Z",
     "bookmakerOdds": {"pinnacle": {"markets": _markets(1.30, 6.0, 9.5, 1.40, 3.0)}}},
    {"fixtureId": "f2", "participant1Id": 44, "participant2Id": 45, "tournamentId": 17, "statusId": 0,
     "startTime": "2026-11-22T16:30:00.000Z",
     "bookmakerOdds": {"pinnacle": {"markets": _markets(1.769, 4.13, 4.31, 1.526, 2.59)}}},
    {"fixtureId": "f3", "participant1Id": 99, "participant2Id": 43, "tournamentId": 17, "statusId": 0,
     "startTime": "2026-11-22T14:00:00.000Z", "bookmakerOdds": {}},
]
FD_E0 = ["Man City", "Fulham", "Liverpool", "Man United", "Arsenal"]


def _snap(ts, p):
    return {"createdAt": ts, "price": p, "active": True}


HISTORY = {"fixtureId": "f2", "bookmakers": {"pinnacle": {"markets": {
    "101": {"outcomes": {
        "101": {"players": {"0": [_snap("2026-11-20T10:00:00Z", 1.80), _snap("2026-11-22T16:29:30Z", 1.75),
                                  _snap("2026-11-22T16:45:00Z", 1.20)]}},   # in-play, must be ignored
        "102": {"players": {"0": [_snap("2026-11-22T16:29:30Z", 4.20)]}},
        "103": {"players": {"0": [_snap("2026-11-22T16:29:30Z", 4.40)]}}}},
    "1010": {"outcomes": {
        "1010": {"players": {"0": [_snap("2026-11-22T16:29:00Z", 1.55)]}},
        "1011": {"players": {"0": [_snap("2026-11-22T16:29:00Z", 2.50)]}}}}}}}}


class FakeAPI:
    """Stands in for the HTTP layer; records every URL requested."""

    def __init__(self, responses):
        self.responses, self.urls = responses, []

    def __call__(self, url):
        self.urls.append(url)
        for key, resp in self.responses.items():
            if f"/{key}?" in url:
                return resp(url) if callable(resp) else resp
        return 404, {"error": {"code": "NOT_FOUND"}}


def _client(tmp_path, responses):
    fake = FakeAPI(responses)
    return OddsPapi(api_key="test", cache_dir=tmp_path, fetch=fake, sleep=lambda s: None), fake


# ---- team names ------------------------------------------------------------------------
def test_all_current_big5_teams_map_correctly():
    pairs = json.loads((DATA / "oddspapi_team_pairs.json").read_text())
    for league, expected in pairs.items():
        mapping, unmatched = map_names(list(expected), list(expected.values()))
        assert unmatched == [], league
        assert mapping == expected, league


def test_unknown_team_is_not_guessed():
    assert match_team("Gotham Rovers", FD_E0) is None
    # "Paris FC" must not be taken for PSG and vice versa
    assert match_team("Paris FC", ["Paris SG", "Paris FC"]) == "Paris FC"
    assert match_team("Paris Saint-Germain", ["Paris SG", "Paris FC"]) == "Paris SG"


# ---- parsing ---------------------------------------------------------------------------
def test_extract_prices_and_inactive_outcomes():
    m = _markets(1.77, 4.13, 4.31, 1.53, 2.59)
    p = extract_prices(m)
    assert {o: p[o] for o in MAIN} == pytest.approx({"H": 1.77, "D": 4.13, "A": 4.31, "O25": 1.53, "U25": 2.59})
    assert all(np.isnan(p[o]) for o in p if o not in MAIN)      # markets the fixture doesn't carry
    m["1010"]["outcomes"]["1011"] = _price(2.59, active=False)
    del m["101"]["outcomes"]["102"]
    p = extract_prices(m)
    assert np.isnan(p["U25"]) and np.isnan(p["D"]) and p["H"] == 1.77


def test_closing_price_is_last_before_kickoff():
    close = closing_from_history(HISTORY, "2026-11-22T16:30:00Z")
    assert {o: close[o] for o in MAIN} == pytest.approx({"H": 1.75, "D": 4.20, "A": 4.40, "O25": 1.55, "U25": 2.50})
    early = closing_from_history(HISTORY, "2026-11-21T00:00:00Z")
    assert early["H"] == 1.80 and np.isnan(early["D"])


def test_fixtures_frame_maps_names_and_flags_unknown_teams():
    df, unmatched = fixtures_frame(UPCOMING, PARTICIPANTS, {"E0": FD_E0})
    assert list(df[["home", "away"]].itertuples(index=False, name=None)) == \
        [("Man City", "Fulham"), ("Liverpool", "Man United")]
    assert unmatched == ["E0: Gotham Rovers"]
    row = df.iloc[1]
    assert (row["fixture_id"], row["pin_H"], row["pin_U25"]) == ("f2", 1.769, 2.59)
    assert np.isnan(row["soft_H"]) and row["softbook_H"] == ""
    for col in ("date", "league", "pinc_H", "soft_O25", "softbook_A"):
        assert col in df


# ---- client ----------------------------------------------------------------------------
def _by_tournament(url):
    return 200, (UPCOMING if "tournamentIds=17%2C" in url else [])


def test_client_batches_leagues_caches_and_counts(tmp_path):
    client, fake = _client(tmp_path, {"odds-by-tournaments": _by_tournament,
                                      "participants": (200, PARTICIPANTS)})
    df, _ = live_fixtures(client, ["E0", "SP1", "I1", "D1", "F1"], {"E0": FD_E0},
                          now="2026-11-20T00:00:00Z")
    assert len(df) == 2
    odds_calls = [u for u in fake.urls if "odds-by-tournaments" in u]
    assert len(odds_calls) == 2                                # 3 leagues + 2 leagues
    assert "tournamentIds=17%2C8%2C23" in odds_calls[0] and "bookmaker=pinnacle" in odds_calls[0]
    assert "apiKey=test" in odds_calls[0]
    assert client.month_usage() == 3                           # 2 odds + 1 participants

    live_fixtures(client, ["E0", "SP1", "I1", "D1", "F1"], {"E0": FD_E0}, now="2026-11-20T00:00:00Z")
    assert len(fake.urls) == 3                                 # second run served from cache
    assert len(list((tmp_path / "snapshots").glob("*.json.gz"))) == 2  # one per real fetch


def test_days_window_filters_fixtures(tmp_path):
    client, _ = _client(tmp_path, {"odds-by-tournaments": (200, UPCOMING),
                                   "participants": (200, PARTICIPANTS)})
    df, _ = live_fixtures(client, ["E0"], {"E0": FD_E0}, days=2, now="2026-11-20T00:00:00Z")
    assert list(df["home"]) == ["Man City"]                   # 21 Nov in, 22 Nov out
    df, _ = live_fixtures(client, ["E0"], {"E0": FD_E0}, days=7, now="2026-11-21T13:00:00Z")
    assert list(df["home"]) == ["Liverpool"]                  # already kicked off: dropped


def test_client_retries_after_rate_limit(tmp_path):
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        return (429, {"error": {"retryMs": 10}}) if calls["n"] == 1 else (200, PARTICIPANTS)

    client, _ = _client(tmp_path, {"participants": flaky})
    assert client.participants() == PARTICIPANTS
    assert calls["n"] == 2


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setattr("predictor.odds_feed.load_api_key", lambda: None)
    with pytest.raises(Exception, match="ODDS_API_KEY"):
        OddsPapi(cache_dir=tmp_path)


# ---- closing odds for the paper log -------------------------------------------------------
def test_fill_closing_fetches_once_and_keeps_typed_prices(tmp_path):
    finished = [{"fixtureId": "f2", "participant1Id": 44, "participant2Id": 45,
                 "startTime": "2026-11-22T16:30:00.000Z"}]
    client, fake = _client(tmp_path, {"historical-odds": (200, HISTORY), "fixtures": (200, finished),
                                      "participants": (200, PARTICIPANTS)})
    picks = pd.DataFrame([
        dict(league="E0", date="2026-11-22", home="Liverpool", away="Man United", market="1X2",
             outcome="D", odds=4.5, book="SportyBet"),                      # no fixture id: looked up
        dict(league="E0", date="2026-11-22", home="Man City", away="Fulham", market="OU25",
             outcome="U25", odds=3.2, book="Betway", fixture_id="f9",
             kickoff="2026-11-29T12:30:00Z"),                               # not kicked off yet
    ])
    log, _, _ = add_picks(empty_log(), picks)
    log.loc[0, "close_D"] = 4.0                                             # typed by hand
    now = pd.Timestamp("2026-11-23T00:00:00Z")
    log, n = fill_closing(log, client, {"E0": FD_E0}, now=now, log_fn=lambda m: None)
    assert n == 1
    assert log.loc[0, ["close_H", "close_D", "close_A"]].tolist() == [1.75, 4.0, 4.40]
    assert log.loc[0, "fixture_id"] == "f2"
    assert pd.isna(log.loc[1, "close_U25"])
    assert not any("f9" in u for u in fake.urls)

    log, n = fill_closing(log, client, {"E0": FD_E0}, now=now, log_fn=lambda m: None)
    assert n == 0 and sum("historical-odds" in u for u in fake.urls) == 1   # nothing re-fetched
