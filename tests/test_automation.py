"""Phases 4-6: targets, Supabase client, Telegram bot and the two scheduled jobs.
Everything external is faked: no real database, chat or odds API is touched."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from jobs import daily_picks, settle
from jobs.common import CFG, process_took
from predictor.db import Store, Supabase, picks_frame
from predictor.targets import match_targets, qualifies, round_up_price, target_price
from predictor.telegram_bot import (Bot, format_settled, format_targets, parse_took, split_message,
                                    took_commands)


# ---- targets -----------------------------------------------------------------------------
def test_target_price_is_the_stricter_of_both_rules():
    # model 40%, Pinnacle fair 38%: model needs 1.05/0.40=2.625, Pinnacle needs 1.02/0.38=2.684
    assert target_price(0.40, 0.38, CFG) == pytest.approx(1.02 / 0.38)
    assert target_price(0.36, 0.38, CFG) == pytest.approx(1.05 / 0.36)
    assert np.isnan(target_price(0.15, 0.17, CFG))            # above the 5.00 cap
    assert np.isnan(target_price(0.60, 0.40, CFG))            # 20-point disagreement
    assert target_price(0.95, 0.95, CFG) == 1.30              # floor at min odds


def test_round_up_price_never_rounds_below_target():
    assert round_up_price(2.6841) == 2.69
    assert round_up_price(3.01) == 3.05
    assert round_up_price(2.70) == 2.70


def test_match_targets_filters_unrealistic_and_sorts():
    row = {"pin_H": 1.769, "pin_D": 4.13, "pin_A": 4.31, "pin_O25": 1.526, "pin_U25": 2.59}
    model = {"H": 0.536, "D": 0.230, "A": 0.234, "O25": 0.629, "U25": 0.371}
    ts = match_targets(row, model, CFG)
    assert [t["outcome"] for t in ts] == ["A", "O25", "U25"]    # H and D need > 10% over Pinnacle
    assert all(t["target_odds"] / t["pin_odds"] - 1 <= 0.10 for t in ts)
    assert [t["over_pinnacle"] for t in ts] == sorted(t["over_pinnacle"] for t in ts)
    from predictor.targets import all_targets
    df = all_targets(pd.DataFrame([row]), {0: model}, CFG)       # posted: best 2 per match
    assert list(df["outcome"]) == ["A", "O25"]
    # a price at or above the target qualifies; below doesn't
    t = ts[1]
    assert qualifies(t["target_odds"], t["p_model"], t["p_pin"], CFG)
    assert not qualifies(t["target_odds"] - 0.05, t["p_model"], t["p_pin"], CFG)


# ---- Supabase client ---------------------------------------------------------------------
def test_supabase_requests_are_shaped_for_postgrest():
    calls = []

    def fake(method, url, headers, body):
        calls.append((method, url, headers, json.loads(body) if body else None))
        return 201, [{"id": 1, "name": "Arsenal"}]

    db = Supabase(url="https://x.supabase.co/", key="svc", fetch=fake)
    db.insert("teams", [{"name": "Arsenal", "league": "E0", "x": float("nan")}], on_conflict="name",
              ignore_duplicates=True)
    m, url, h, body = calls[-1]
    assert (m, url) == ("POST", "https://x.supabase.co/rest/v1/teams?on_conflict=name")
    assert h["apikey"] == "svc" and h["Authorization"] == "Bearer svc"
    assert "resolution=ignore-duplicates" in h["Prefer"]
    assert body == [{"name": "Arsenal", "league": "E0", "x": None}]   # NaN -> null

    new_style = Supabase(url="https://x.supabase.co", key="sb_secret_abc", fetch=fake)
    new_style.select("teams")
    assert calls[-1][2]["apikey"] == "sb_secret_abc" and "Authorization" not in calls[-1][2]

    db.update("picks", {"status": "won"}, id="eq.7")
    assert calls[-1][0] == "PATCH" and calls[-1][1].endswith("/picks?id=eq.7")
    Store(db).picks(status="neq.void")
    assert "select=*,matches(" in calls[-1][1] and "status=neq.void" in calls[-1][1]


def test_supabase_errors_are_raised():
    db = Supabase(url="https://x", key="k", fetch=lambda *a: (401, {"message": "bad key"}))
    with pytest.raises(Exception, match="401"):
        db.select("teams")


# ---- Telegram ----------------------------------------------------------------------------
def test_parse_took():
    assert parse_took("/took 57 2.90 sportybet") == (57, 2.9, "SportyBet")
    assert parse_took("/took@ValueBot #12 3,05 Betway") == (12, 3.05, "Betway")
    assert isinstance(parse_took("/took 57 2.90 hollywoodbets"), str)
    assert isinstance(parse_took("/took fifty"), str)


def test_took_commands_only_from_the_group():
    ups = [{"update_id": 5, "message": {"message_id": 1, "date": 1700000000, "chat": {"id": -100},
                                        "from": {"username": "sol"}, "text": "/took 3 2.5 betway"}},
           {"update_id": 6, "message": {"message_id": 2, "date": 1700000000, "chat": {"id": 999},
                                        "text": "/took 3 2.5 betway"}},
           {"update_id": 7, "message": {"message_id": 3, "date": 1700000000, "chat": {"id": -100},
                                        "text": "nice one"}}]
    cmds = took_commands(ups, "-100")
    assert len(cmds) == 1 and cmds[0]["user"] == "sol"


def test_split_message_respects_limit():
    parts = split_message("\n".join(f"line {i:04d} " + "x" * 50 for i in range(200)), limit=1000)
    assert all(len(p) <= 1000 for p in parts) and sum(p.count("\n") + 1 for p in parts) == 200


def test_bot_send_and_errors():
    sent = []

    def fake(url, payload):
        sent.append((url, payload))
        return 200, {"ok": True, "result": {"message_id": len(sent)}}

    bot = Bot(token="T", chat_id="-100", fetch=fake)
    assert bot.send("hi", reply_to=9) == [1]
    url, payload = sent[0]
    assert url == "https://api.telegram.org/botT/sendMessage"
    assert payload["chat_id"] == "-100" and payload["reply_parameters"]["message_id"] == 9
    bad = Bot(token="T", chat_id="-100", fetch=lambda u, p: (401, {"ok": False, "description": "Unauthorized"}))
    with pytest.raises(Exception, match="Unauthorized"):
        bad.send("hi")


def test_format_targets_and_empty_day():
    t = pd.DataFrame([{"id": 58, "kickoff": "2026-11-22T16:30:00Z", "home": "Liverpool",
                       "away": "Man United", "outcome": "U25", "target_odds": 2.84, "pin_odds": 2.60}])
    msg = format_targets(t, now="2026-11-22T06:00:00Z")
    assert "Sun 18:30  Liverpool v Man United" in msg            # SA time
    assert "#58  Under 2.5: bet if ≥ 2.84  (Pinnacle 2.60)" in msg
    assert "PAPER ONLY" in msg
    assert "No realistic targets" in format_targets(t.iloc[:0], now="2026-11-22T06:00:00Z")


# ---- fakes for the jobs --------------------------------------------------------------------
class FakeStore:
    def __init__(self):
        self.teams, self.matches, self.snaps, self.preds = {}, {}, [], []
        self.targets, self.picks_, self.kv = {}, {}, {}

    def team_ids(self, names_by_league):
        for names in names_by_league.values():
            for n in names:
                self.teams.setdefault(n, len(self.teams) + 1)
        return dict(self.teams)

    def upsert_matches(self, rows):
        out = {}
        for r in rows:
            m = next((m for m in self.matches.values() if m["oddspapi_id"] == r["oddspapi_id"]), None)
            if m is None:
                m = {**r, "id": len(self.matches) + 1, "home_goals": None, "away_goals": None}
                self.matches[m["id"]] = m
            out[r["oddspapi_id"]] = m["id"]
        return out

    def set_result(self, mid, hg, ag):
        self.matches[mid].update(home_goals=hg, away_goals=ag)

    def add_snapshots(self, rows):
        self.snaps += rows

    def add_predictions(self, rows):
        out = {}
        for r in rows:
            self.preds.append({**r, "id": len(self.preds) + 1})
            out[r["match_id"]] = len(self.preds)
        return out

    def upsert_targets(self, rows):
        out = []
        for r in rows:
            key = (r["match_id"], r["market"], r["outcome"])
            tid = next((i for i, t in self.targets.items()
                        if (t["match_id"], t["market"], t["outcome"]) == key), len(self.targets) + 1)
            self.targets[tid] = {**r, "id": tid}
            out.append(self.targets[tid])
        return out

    def _embed(self, mid):
        m = self.matches[mid]
        name = {v: k for k, v in self.teams.items()}
        return {**m, "home": {"name": name[m["home_id"]]}, "away": {"name": name[m["away_id"]]}}

    def target(self, tid):
        t = self.targets.get(tid)
        return {**t, "matches": self._embed(t["match_id"])} if t else None

    def add_pick(self, row):
        if any(p["match_id"] == row["match_id"] for p in self.picks_.values()):
            return None
        pid = len(self.picks_) + 1
        self.picks_[pid] = {**row, "id": pid, "closing_odds": None, "clv": None, "profit_units": None}
        return self.picks_[pid]

    def picks(self, **filters):
        rows = list(self.picks_.values())
        if filters.get("status") == "neq.void":
            rows = [p for p in rows if p["status"] != "void"]
        if "id" in filters:
            ids = {int(x) for x in filters["id"][4:-1].split(",")}
            rows = [p for p in rows if p["id"] in ids]
        return [{**p, "matches": self._embed(p["match_id"])} for p in rows]

    def update_pick(self, pid, values):
        self.picks_[pid].update(values)

    def state(self, key, default=None):
        return self.kv.get(key, default)

    def set_state(self, key, value):
        self.kv[key] = str(value)


class FakeBot:
    chat_id = "-100"

    def __init__(self, updates=()):
        self._updates, self.sent = list(updates), []

    def updates(self, offset=None):
        return [u for u in self._updates if offset is None or u["update_id"] >= offset]

    def send(self, text, chat_id=None, reply_to=None):
        self.sent.append((text, reply_to))
        return [len(self.sent)]


class FakeOdds:
    """Stands in for OddsPapi in the jobs."""

    def __init__(self, upcoming, history=None):
        self._upcoming, self._history, self.calls = upcoming, history or {}, 0

    def upcoming_odds(self, leagues, **kw):
        return self._upcoming

    def participants(self):
        return {"1": "Arsenal FC", "2": "Chelsea FC", "3": "Liverpool FC", "4": "Everton FC"}

    def history(self, fixture_id, bookmakers=("pinnacle",)):
        return self._history[fixture_id]

    def month_usage(self):
        return 0


def _odds(h, d, a, o, u):
    p = lambda x: {"players": {"0": {"price": x, "active": True}}}
    return {"101": {"outcomes": {"101": p(h), "102": p(d), "103": p(a)}},
            "1010": {"outcomes": {"1010": p(o), "1011": p(u)}}}


def _history_frames():
    """Two seasons where Arsenal and Liverpool are strong and both games are tight."""
    rng = np.random.default_rng(0)
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Spurs", "Leeds"]
    strength = {"Arsenal": 0.4, "Liverpool": 0.4, "Chelsea": 0.1, "Everton": -0.2, "Spurs": 0.0, "Leeds": -0.3}
    rows, day = [], pd.Timestamp("2025-08-01")
    for rnd in range(40):
        for i, h in enumerate(teams):
            for a in teams[i + 1:]:
                if rng.random() < 0.5:
                    h, a = a, h
                lam_h = np.exp(0.25 + strength[h] - strength[a] * 0.5)
                lam_a = np.exp(strength[a] - strength[h] * 0.5)
                rows.append({"date": day, "league": "E0", "season": "2526", "home": h, "away": a,
                             "hg": rng.poisson(lam_h), "ag": rng.poisson(lam_a)})
        day += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    return {"E0": (df.iloc[:0], df)}


UPCOMING = [
    {"fixtureId": "fx1", "participant1Id": 1, "participant2Id": 2, "tournamentId": 17, "statusId": 0,
     "startTime": "2026-11-21T15:00:00Z", "bookmakerOdds": {"pinnacle": {"markets": _odds(1.9, 3.7, 4.2, 1.8, 2.05)}}},
    {"fixtureId": "fx2", "participant1Id": 3, "participant2Id": 4, "tournamentId": 17, "statusId": 0,
     "startTime": "2026-11-21T17:30:00Z", "bookmakerOdds": {"pinnacle": {"markets": _odds(1.5, 4.4, 6.5, 1.6, 2.4)}}},
]


def _args(**kw):
    return argparse.Namespace(**{"league": ["E0"], "hours": 24, "offline": True, **kw})


# ---- daily job ---------------------------------------------------------------------------
def test_daily_job_saves_everything_and_posts_numbered_targets():
    store, bot = FakeStore(), FakeBot()
    msg = daily_picks.run(_args(), store=store, bot=bot, client=FakeOdds(UPCOMING),
                          history=_history_frames(), now="2026-11-21T06:00:00Z", log=lambda m: None)
    assert len(store.matches) == 2 and len(store.preds) == 2
    assert len(store.snaps) == 10                               # 5 Pinnacle prices x 2 matches
    assert store.preds[0]["model_version"] == "dc-v1-xi0019-r3"
    assert bot.sent and bot.sent[0][0] == msg
    for tid in store.targets:                                   # every saved target is in the message
        assert f"#{tid} " in msg
    # re-running the same morning doesn't duplicate matches or targets
    n_targets = len(store.targets)
    daily_picks.run(_args(), store=store, bot=bot, client=FakeOdds(UPCOMING), history=_history_frames(),
                    now="2026-11-21T06:00:00Z", log=lambda m: None)
    assert len(store.matches) == 2 and len(store.targets) == n_targets


def test_monday_run_also_sends_weekly_summary():
    store, bot = FakeStore(), FakeBot()
    daily_picks.run(_args(), store=store, bot=bot, client=FakeOdds([]), history=_history_frames(),
                    now="2026-11-23T06:00:00Z", log=lambda m: None)   # a Monday
    assert len(bot.sent) == 2 and "Weekly summary" in bot.sent[1][0]


# ---- /took ---------------------------------------------------------------------------------
def _store_with_target():
    store = FakeStore()
    store.team_ids({"E0": ["Liverpool", "Everton"]})
    mid = store.upsert_matches([{"league": "E0", "kickoff": "2026-11-21T17:30:00+00:00", "home_id": 1,
                                 "away_id": 2, "oddspapi_id": "fx2"}])["fx2"]
    store.upsert_targets([{"match_id": mid, "prediction_id": None, "market": "OU25", "outcome": "U25",
                           "target_odds": 2.60, "pin_odds": 2.40, "p_model": 0.42, "p_pinnacle": 0.40}])
    return store


def _took(uid, text, ts="2026-11-21T09:00:00Z", user="sol"):
    return {"update_id": uid, "message": {"message_id": 100 + uid, "date": int(pd.Timestamp(ts).timestamp()),
                                          "chat": {"id": -100}, "from": {"username": user}, "text": text}}


def test_took_logs_one_pick_per_match_and_rejects_bad_prices():
    store = _store_with_target()
    bot = FakeBot([_took(1, "/took 1 2.50 betway"),                      # below target
                   _took(2, "/took 1 2.75 sportybet"),                   # good
                   _took(3, "/took 1 2.80 betway", user="thabo"),        # second pick, same match
                   _took(4, "/took 9 3.00 betway"),                      # no such target
                   _took(5, "/took 1 2.90 betway", ts="2026-11-21T18:00:00Z")])  # after kickoff
    assert process_took(store, bot, log=lambda m: None) == 1
    (pick,) = store.picks_.values()
    assert (pick["odds"], pick["book"], pick["logged_by"], pick["outcome"]) == (2.75, "SportyBet", "sol", "U25")
    assert pick["edge"] == pytest.approx(0.42 * 2.75 - 1, abs=1e-4)
    replies = [t for t, _ in bot.sent]
    assert "below the target" in replies[0] and "Logged for sol" in replies[1]
    assert "already has a pick" in replies[2] and "no target #9" in replies[3]
    assert "already kicked off" in replies[4]
    assert store.kv["telegram_offset"] == "5"
    assert process_took(store, bot, log=lambda m: None) == 0             # nothing re-read


# ---- settle job ----------------------------------------------------------------------------
def test_settle_fetches_closing_settles_and_posts():
    store = _store_with_target()
    store.add_pick({"match_id": 1, "target_id": 1, "market": "OU25", "outcome": "U25", "book": "Betway",
                    "odds": 2.75, "p_model": 0.42, "p_pinnacle": 0.40, "stake_units": 1, "status": "open"})
    snap = lambda ts, p: {"createdAt": ts, "price": p}
    hist = {"fx2": {"bookmakers": {"pinnacle": {"markets": {"1010": {"outcomes": {
        "1010": {"players": {"0": [snap("2026-11-21T17:29:00Z", 1.62), snap("2026-11-21T18:00:00Z", 3.0)]}},
        "1011": {"players": {"0": [snap("2026-11-21T17:29:00Z", 2.35), snap("2026-11-21T18:00:00Z", 1.3)]}}}}}}}}}
    results = pd.DataFrame([{"date": pd.Timestamp("2026-11-21"), "league": "E0", "home": "Liverpool",
                             "away": "Everton", "hg": 1, "ag": 0}])
    bot = FakeBot()
    settled = settle.run(_args(), store, bot=bot, client=FakeOdds([], hist),
                         history={"E0": (results.iloc[:0], results)}, now="2026-11-22T21:30:00Z",
                         log=lambda m: None)
    pick = store.picks_[1]
    assert pick["closing_odds"] == 2.35                                    # last pre-kickoff price
    fair_u = (1 / 2.35) / (1 / 1.62 + 1 / 2.35)
    assert pick["clv"] == pytest.approx(2.75 * fair_u - 1, abs=1e-4)
    assert (pick["status"], pick["profit_units"]) == ("won", pytest.approx(1.75))
    assert store.matches[1]["home_goals"] == 1
    assert sum(s.get("is_closing", False) for s in store.snaps) == 2
    assert len(settled) == 1 and "🏁 Settled" in bot.sent[-1][0] and "+1.75u" in bot.sent[-1][0]


def test_format_settled_handles_void():
    rows = pd.DataFrame([{"status": "void", "home": "A", "away": "B", "home_goals": None, "away_goals": None,
                          "outcome": "H", "odds": 2.0, "profit_units": None, "clv": None}])
    assert "➖ A v B · Home win @ 2.00 · +0.00u" in format_settled(rows)


def test_picks_frame_flattens_embedded_match():
    df = picks_frame([{"id": 1, "status": "open", "matches": {"league": "E0", "kickoff": "k",
                       "oddspapi_id": "fx", "home": {"name": "A"}, "away": {"name": "B"}}}])
    assert df.loc[0, ["league", "home", "away", "fixture_id"]].tolist() == ["E0", "A", "B", "fx"]
