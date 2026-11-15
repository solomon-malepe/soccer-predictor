"""Supabase storage through its REST API (PostgREST), using the service key.

Server side only: the service key bypasses row-level security, so it lives in
.env locally and in GitHub repo secrets for the jobs, never in a browser.

Needs SUPABASE_URL and SUPABASE_SERVICE_KEY. Schema: supabase/migrations/001_init.sql.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import pandas as pd

Fetch = Callable[[str, str, dict, bytes | None], tuple[int, object]]

MATCH_EMBED = ("matches(id,league,kickoff,oddspapi_id,home_goals,away_goals,"
               "home:teams!matches_home_id_fkey(name),away:teams!matches_away_id_fkey(name))")


class DBError(RuntimeError):
    pass


def _http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, object]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else []
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def _clean(v):
    """JSON-safe values: NaN -> None, Timestamps -> ISO strings, numpy -> python."""
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


class Supabase:
    def __init__(self, url: str | None = None, key: str | None = None, fetch: Fetch | None = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.url = (url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not (self.url and self.key):
            raise DBError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set (in .env or the environment)")
        self._fetch = fetch or _http

    def _req(self, method: str, table: str, params: dict | None = None, body=None,
             prefer: str | None = None) -> list:
        url = f"{self.url}/rest/v1/{table}"
        if params:
            url += "?" + urllib.parse.urlencode(params, safe="(),!:*.")
        headers = {"apikey": self.key, "Content-Type": "application/json", "Accept": "application/json"}
        if not self.key.startswith("sb_"):
            # legacy service_role keys are JWTs and also go in Authorization;
            # new sb_secret_ keys are not JWTs and only go in apikey
            headers["Authorization"] = f"Bearer {self.key}"
        if prefer:
            headers["Prefer"] = prefer
        data = None
        if body is not None:
            if isinstance(body, list):
                body = [{k: _clean(v) for k, v in row.items()} for row in body]
            elif isinstance(body, dict):
                body = {k: _clean(v) for k, v in body.items()}
            data = json.dumps(body).encode("utf-8")
        status, out = self._fetch(method, url, headers, data)
        if status >= 300:
            raise DBError(f"Supabase {method} {table} failed (HTTP {status}): {out}")
        return out if isinstance(out, list) else ([out] if out else [])

    def select(self, table: str, select: str = "*", **filters) -> list[dict]:
        return self._req("GET", table, {"select": select, **filters})

    def insert(self, table: str, rows, on_conflict: str | None = None,
               ignore_duplicates: bool = False) -> list[dict]:
        rows = rows if isinstance(rows, list) else [rows]
        if not rows:
            return []
        prefer = "return=representation"
        params = {}
        if on_conflict:
            prefer += ",resolution=" + ("ignore-duplicates" if ignore_duplicates else "merge-duplicates")
            params["on_conflict"] = on_conflict
        return self._req("POST", table, params, rows, prefer)

    def update(self, table: str, values: dict, **filters) -> list[dict]:
        return self._req("PATCH", table, filters, values, "return=representation")


class Store:
    """The handful of operations the jobs need."""

    def __init__(self, db: Supabase):
        self.db = db

    # teams + matches
    def team_ids(self, names_by_league: dict[str, list[str]]) -> dict[str, int]:
        rows = [{"name": n, "league": lg} for lg, names in names_by_league.items() for n in names]
        self.db.insert("teams", rows, on_conflict="name", ignore_duplicates=True)
        wanted = {r["name"] for r in rows}
        return {t["name"]: t["id"] for t in self.db.select("teams", "id,name") if t["name"] in wanted}

    def upsert_matches(self, rows: list[dict]) -> dict[str, int]:
        """rows: league, season, kickoff, home_id, away_id, oddspapi_id. Returns oddspapi_id -> id."""
        out = self.db.insert("matches", rows, on_conflict="oddspapi_id")
        return {m["oddspapi_id"]: m["id"] for m in out}

    def set_result(self, match_id: int, hg: int, ag: int) -> None:
        self.db.update("matches", {"home_goals": hg, "away_goals": ag}, id=f"eq.{match_id}")

    # odds, predictions, targets
    def add_snapshots(self, rows: list[dict]) -> None:
        self.db.insert("odds_snapshots", rows)

    def add_predictions(self, rows: list[dict]) -> dict[int, int]:
        out = self.db.insert("predictions", rows)
        return {p["match_id"]: p["id"] for p in out}

    def upsert_targets(self, rows: list[dict]) -> list[dict]:
        return self.db.insert("targets", rows, on_conflict="match_id,market,outcome")

    def target(self, target_id: int) -> dict | None:
        rows = self.db.select("targets", f"*,{MATCH_EMBED}", id=f"eq.{target_id}")
        return rows[0] if rows else None

    # picks
    def add_pick(self, row: dict) -> dict | None:
        """Insert unless the match already has a pick (one pick per match)."""
        out = self.db.insert("picks", row, on_conflict="match_id", ignore_duplicates=True)
        return out[0] if out else None

    def picks(self, **filters) -> list[dict]:
        return self.db.select("picks", f"*,{MATCH_EMBED}", **filters)

    def update_pick(self, pick_id: int, values: dict) -> None:
        self.db.update("picks", values, id=f"eq.{pick_id}")

    # bot state
    def state(self, key: str, default: str | None = None) -> str | None:
        rows = self.db.select("bot_state", "value", key=f"eq.{key}")
        return rows[0]["value"] if rows else default

    def set_state(self, key: str, value) -> None:
        self.db.insert("bot_state", {"key": key, "value": str(value)}, on_conflict="key")


def picks_frame(picks: list[dict]) -> pd.DataFrame:
    """Flatten picks with embedded match info into one row per pick."""
    rows = []
    for p in picks:
        m = p.get("matches") or {}
        rows.append({**{k: v for k, v in p.items() if k != "matches"},
                     "league": m.get("league"), "kickoff": m.get("kickoff"),
                     "fixture_id": m.get("oddspapi_id"),
                     "home": (m.get("home") or {}).get("name"), "away": (m.get("away") or {}).get("name"),
                     "home_goals": m.get("home_goals"), "away_goals": m.get("away_goals")})
    return pd.DataFrame(rows)
