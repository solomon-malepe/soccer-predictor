"""Telegram delivery: post targets, read /took replies, send summaries.

Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (the friends' group). Optional
TELEGRAM_ADMIN_CHAT_ID gets failure alerts (your private chat with the bot).

The jobs run on a schedule, not as a always-on bot, so /took replies are read the
next time a job runs (morning and night). Telegram keeps unread messages 24 hours.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import numpy as np
import pandas as pd

from .targets import LABELS

Fetch = Callable[[str, dict], tuple[int, object]]
MAX_LEN = 3900          # Telegram's limit is 4096 characters per message
BOOKS = {"betway": "Betway", "sportybet": "SportyBet", "sporty": "SportyBet"}
SAST = "Africa/Johannesburg"


class TelegramError(RuntimeError):
    pass


def _http(url: str, payload: dict) -> tuple[int, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except json.JSONDecodeError:
            return e.code, {}


class Bot:
    def __init__(self, token: str | None = None, chat_id: str | None = None,
                 admin_chat_id: str | None = None, fetch: Fetch | None = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.admin_chat_id = admin_chat_id or os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
        if not (self.token and self.chat_id):
            raise TelegramError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        self._fetch = fetch or _http

    def _call(self, method: str, payload: dict):
        status, body = self._fetch(f"https://api.telegram.org/bot{self.token}/{method}", payload)
        if status != 200 or not (isinstance(body, dict) and body.get("ok")):
            desc = body.get("description") if isinstance(body, dict) else body
            raise TelegramError(f"Telegram {method} failed (HTTP {status}): {desc}")
        return body["result"]

    def send(self, text: str, chat_id: str | None = None, reply_to: int | None = None) -> list[int]:
        ids = []
        for part in split_message(text):
            payload = {"chat_id": chat_id or self.chat_id, "text": part,
                       "disable_web_page_preview": True}
            if reply_to:
                payload["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
            ids.append(self._call("sendMessage", payload)["message_id"])
        return ids

    def alert(self, text: str) -> None:
        """Failure alert to the admin chat, falling back to nothing (never raises)."""
        if not self.admin_chat_id:
            return
        try:
            self.send(text, chat_id=self.admin_chat_id)
        except Exception:
            pass

    def updates(self, offset: int | None = None) -> list[dict]:
        payload = {"timeout": 0, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        return self._call("getUpdates", payload)


def split_message(text: str, limit: int = MAX_LEN) -> list[str]:
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            parts.append(cur.rstrip("\n"))
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.rstrip("\n"))
    return parts


# ---- /took --------------------------------------------------------------------------
TOOK_RE = re.compile(r"^/took(?:@\w+)?\s+#?(\d+)\s+(\d+(?:[.,]\d+)?)\s+(\w+)\s*$", re.IGNORECASE)


def parse_took(text: str) -> tuple[int, float, str] | str:
    """'/took 57 2.90 sportybet' -> (57, 2.9, 'SportyBet'), or an error message."""
    m = TOOK_RE.match((text or "").strip())
    if not m:
        return "Format: /took <target number> <price> <betway|sportybet>, e.g. /took 57 2.90 sportybet"
    book = BOOKS.get(m.group(3).lower())
    if not book:
        return "Book must be betway or sportybet."
    return int(m.group(1)), float(m.group(2).replace(",", ".")), book


def took_commands(updates: list[dict], chat_id: str) -> list[dict]:
    """/took messages from the group: update_id, message_id, date, user, text."""
    out = []
    for u in updates:
        msg = u.get("message") or {}
        text = msg.get("text") or ""
        if str(msg.get("chat", {}).get("id")) != str(chat_id) or not text.lower().startswith("/took"):
            continue
        user = msg.get("from", {})
        out.append({"update_id": u["update_id"], "message_id": msg.get("message_id"),
                    "date": pd.Timestamp(msg.get("date", 0), unit="s", tz="UTC"),
                    "user": user.get("username") or user.get("first_name") or "someone", "text": text})
    return out


# ---- message formats ----------------------------------------------------------------
def _sast(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(SAST)


def format_targets(targets: pd.DataFrame, now=None) -> str:
    """Morning message. `targets` needs id, kickoff, home, away, outcome, target_odds, pin_odds."""
    now = _sast(now if now is not None else pd.Timestamp.now(tz="UTC"))
    head = f"⚽ Value targets · {now.strftime('%a %d %b')}"
    if targets.empty:
        return head + "\n\nNo realistic targets today. That's normal; most days have few or none."
    lines = [head, ""]
    for (ko, home, away), g in targets.sort_values(["kickoff", "home"]).groupby(
            ["kickoff", "home", "away"], sort=False):
        lines.append(f"{_sast(ko).strftime('%a %H:%M')}  {home} v {away}")
        for _, t in g.iterrows():
            lines.append(f"  #{int(t['id'])}  {LABELS[t['outcome']]}: bet if ≥ {t['target_odds']:.2f}"
                         f"  (Pinnacle {t['pin_odds']:.2f})")
    lines += ["",
              "If Betway or SportyBet offers the price or better: 1 unit, ONE bet per match.",
              "Then reply: /took <number> <price> <book>   e.g. /took 57 2.90 sportybet",
              "Most targets won't be on offer; a real one turns up about once a week.",
              "PAPER ONLY until 150 picks show positive CLV."]
    return "\n".join(lines)


def format_pick_logged(user: str, target: dict, odds: float, book: str) -> str:
    m = target["matches"]
    return (f"✅ Logged for {user}: {m['home']['name']} v {m['away']['name']} · "
            f"{LABELS[target['outcome']]} @ {odds:.2f} ({book}) · 1 unit (paper)")


def format_weekly_summary(picks: pd.DataFrame, all_picks: pd.DataFrame, gate: str) -> str:
    """Monday message: last 7 days plus all-time record."""
    def line(df: pd.DataFrame, label: str) -> str:
        done = df[df["status"].isin(["won", "lost"])]
        if done.empty:
            return f"{label}: no settled picks"
        units = pd.to_numeric(done["profit_units"], errors="coerce").sum()
        clv = pd.to_numeric(df["clv"], errors="coerce").dropna()
        won = (done["status"] == "won").sum()
        clv_txt = f", avg CLV {clv.mean():+.1%} ({len(clv)})" if len(clv) else ""
        return (f"{label}: {len(done)} settled, {won} won, {units:+.2f} units, "
                f"ROI {units / len(done):+.1%}{clv_txt}")

    open_n = int((all_picks["status"] == "open").sum()) if len(all_picks) else 0
    return "\n".join(["📊 Weekly summary", line(picks, "Last 7 days"), line(all_picks, "All time"),
                      f"Open picks: {open_n}", gate,
                      "CLV is what counts. Profit over a few dozen bets is mostly luck."])


def format_settled(rows: pd.DataFrame) -> str:
    lines = ["🏁 Settled"]
    for _, r in rows.iterrows():
        mark = {"won": "✅", "lost": "❌", "void": "➖"}.get(r["status"], "")
        clv = f" · CLV {float(r['clv']):+.1%}" if pd.notna(r.get("clv")) else ""
        score = (f" {int(r['home_goals'])}-{int(r['away_goals'])}"
                 if pd.notna(r.get("home_goals")) and pd.notna(r.get("away_goals")) else "")
        profit = float(r["profit_units"]) if pd.notna(r.get("profit_units")) else 0.0
        lines.append(f"{mark} {r['home']} v {r['away']}{score} · "
                     f"{LABELS[r['outcome']]} @ {float(r['odds']):.2f} · {profit:+.2f}u{clv}")
    return "\n".join(lines)
