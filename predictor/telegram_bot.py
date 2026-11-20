"""Telegram delivery: post targets, read /took replies, send summaries.

Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (the friends' group). Optional
TELEGRAM_ADMIN_CHAT_ID gets failure alerts (your private chat with the bot).

The jobs run on a schedule, not as a always-on bot, so /took replies are read the
next time a job runs (morning and night). Telegram keeps unread messages 24 hours.
"""

from __future__ import annotations

import json
from html import escape
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

import numpy as np
import pandas as pd

from .targets import LABELS
from .value import is_experimental

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

    def send(self, text: str, chat_id: str | None = None, reply_to: int | None = None,
             html: bool = False) -> list[int]:
        ids = []
        for part in split_message(text):
            payload = {"chat_id": chat_id or self.chat_id, "text": part,
                       "disable_web_page_preview": True}
            if html:
                payload["parse_mode"] = "HTML"
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


LEAGUE_HEADERS = {"E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 PREMIER LEAGUE", "SP1": "🇪🇸 LA LIGA", "I1": "🇮🇹 SERIE A",
                  "D1": "🇩🇪 BUNDESLIGA", "F1": "🇫🇷 LIGUE 1"}
LEAGUE_ORDER = ["E0", "SP1", "I1", "D1", "F1"]
SHORT = {"H": "Home", "D": "Draw", "A": "Away", "O25": "Over 2.5", "U25": "Under 2.5",
         "BTTS_Y": "BTTS Yes", "BTTS_N": "BTTS No", "O15": "Over 1.5", "U15": "Under 1.5",
         "O35": "Over 3.5", "U35": "Under 3.5"}
EXP = "🧪"   # marks experimental (untested) markets


def format_targets(targets: pd.DataFrame, now=None) -> str:
    """Morning message (Telegram HTML): grouped by day, then league, then match.
    `targets` needs id, league, kickoff, home, away, outcome, target_odds, pin_odds."""
    now = _sast(now if now is not None else pd.Timestamp.now(tz="UTC"))
    head = f"⚽ <b>VALUE TARGETS</b> · {now.strftime('%a %d %b')}"
    if targets.empty:
        return head + "\n\nNo realistic targets today. That's normal; most days have few or none."
    t = targets.copy()
    t["ko"] = t["kickoff"].map(_sast)
    t["day"] = t["ko"].map(lambda x: x.strftime("%A %d %b"))
    t["lg_order"] = t["league"].map(lambda lg: LEAGUE_ORDER.index(lg) if lg in LEAGUE_ORDER else 9)
    t["exp"] = t["market"].map(is_experimental) if "market" in t else False
    t = t.sort_values(["ko", "lg_order", "home", "exp", "over_pinnacle" if "over_pinnacle" in t else "id"])
    n_matches = t.groupby(["home", "away"]).ngroups
    n_exp = int(t["exp"].sum())
    count = f"{len(t) - n_exp} targets" + (f" + {n_exp} {EXP}" if n_exp else "")
    lines = [head, f"{count} · {n_matches} matches · paper only", ""]
    multi_day = t["day"].nunique() > 1
    for day, d in t.groupby("day", sort=False):
        if multi_day:
            lines += [f"📅 <b>{escape(day.upper())}</b>", ""]
        for lg, g in sorted(d.groupby("league"), key=lambda x: LEAGUE_ORDER.index(x[0])
                            if x[0] in LEAGUE_ORDER else 9):
            lines.append(f"<b>{LEAGUE_HEADERS.get(lg, lg)}</b>")
            for (ko, home, away), m in g.groupby(["ko", "home", "away"], sort=False):
                lines.append(f"<b>{ko.strftime('%H:%M')}</b>  {escape(home)} v {escape(away)}")
                for _, r in m.iterrows():
                    lines.append(f"<code>  #{int(r['id']):<4} {SHORT[r['outcome']]:<9} ≥ {r['target_odds']:>5.2f}"
                                 f"   Pin {r['pin_odds']:.2f}</code>" + (f" {EXP}" if r["exp"] else ""))
            lines.append("")
    lines += ["<b>How to use</b>",
              "1. Find the match on Betway or SportyBet.",
              "2. If the odds are the number after ≥ or higher, bet 1 unit.",
              "3. Reply here with the target number, the odds you got and the bookie:",
              "<code>/took 57 2.90 sportybet</code>",
              "<i>One bet per match. Most targets won't be on offer; a real one turns up about once a week.</i>"]
    if n_exp:
        lines += ["", f"{EXP} <i>= experimental (BTTS, Over/Under 1.5 and 3.5). Never tested on past seasons, "
                      "so these are scored separately and don't count towards the main record. "
                      f"You can take one {EXP} per match on top of the main bet.</i>"]
    return "\n".join(lines)


def format_pick_logged(user: str, target: dict, odds: float, book: str) -> str:
    m = target["matches"]
    exp = f" {EXP} experimental" if is_experimental(target.get("market", "")) else ""
    return (f"✅ Logged for {user}: {m['home']['name']} v {m['away']['name']} · "
            f"{LABELS[target['outcome']]} @ {odds:.2f} ({book}) · 1 unit (paper){exp}")


def format_weekly_summary(picks: pd.DataFrame, all_picks: pd.DataFrame, gate: str,
                          experimental: pd.DataFrame | None = None) -> str:
    """Monday message (Telegram HTML): last 7 days plus all-time record (main track),
    then the experimental track's all-time record if it has any picks."""
    def block(df: pd.DataFrame, label: str) -> list[str]:
        done = df[df["status"].isin(["won", "lost"])] if len(df) else df
        if done.empty:
            return [f"<b>{label}</b>", "  no settled picks yet"]
        units = pd.to_numeric(done["profit_units"], errors="coerce").sum()
        clv = pd.to_numeric(df["clv"], errors="coerce").dropna()
        won = int((done["status"] == "won").sum())
        rows = [f"<b>{label}</b>",
                f"<code>  Picks   {len(done):>4}   won {won}</code>",
                f"<code>  Units   {units:>+7.2f}</code>",
                f"<code>  ROI     {units / len(done):>+7.1%}</code>"]
        if len(clv):
            rows.append(f"<code>  CLV     {clv.mean():>+7.1%}   ({len(clv)} with close)</code>")
        return rows

    open_n = int((all_picks["status"] == "open").sum()) if len(all_picks) else 0
    exp = []
    if experimental is not None and len(experimental):
        exp = ["", *block(experimental, f"{EXP} Experimental, all time"),
               "<i>  BTTS and Over/Under 1.5/3.5. Not part of the main record.</i>"]
    return "\n".join(["📊 <b>WEEKLY SUMMARY</b>", "", *block(picks, "Last 7 days"), "",
                      *block(all_picks, "All time"), "", f"Open picks: {open_n}", escape(gate), *exp, "",
                      "<i>CLV is what counts. Profit over a few dozen bets is mostly luck.</i>"])


def format_settled(rows: pd.DataFrame) -> str:
    """Night message (Telegram HTML): one line per settled pick."""
    lines = ["🏁 <b>SETTLED</b>", ""]
    for _, r in rows.iterrows():
        mark = {"won": "✅", "lost": "❌", "void": "➖"}.get(r["status"], "")
        clv = f" · CLV {float(r['clv']):+.1%}" if pd.notna(r.get("clv")) else ""
        score = (f" {int(r['home_goals'])}-{int(r['away_goals'])}"
                 if pd.notna(r.get("home_goals")) and pd.notna(r.get("away_goals")) else "")
        profit = float(r["profit_units"]) if pd.notna(r.get("profit_units")) else 0.0
        lines.append(f"{mark} <b>{escape(str(r['home']))} v {escape(str(r['away']))}</b>{score}")
        exp = f" {EXP}" if is_experimental(str(r.get("market", ""))) else ""
        lines.append(f"     {LABELS[r['outcome']]} @ {float(r['odds']):.2f} · <b>{profit:+.2f}u</b>{clv}{exp}")
    return "\n".join(lines)
