"""Soccer value-betting dashboard (Streamlit).

    streamlit run dashboard/app.py

Reads Supabase with the keys in .env (or Streamlit secrets when deployed).
Tabs: today's targets (+ log a bet), picks, performance, system.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Streamlit Cloud: secrets -> environment, so the same code runs locally and deployed
try:
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

from jobs.common import GATE_PICKS_TEXT, log_pick  # noqa: E402
from predictor.db import Store, Supabase, picks_frame  # noqa: E402
from predictor.paper import GATE_PICKS  # noqa: E402
from predictor.stats import breakdown, clv_series, kpis, profit_curve  # noqa: E402
from predictor.targets import LABELS  # noqa: E402

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
SAST = "Africa/Johannesburg"

st.set_page_config(page_title="Value Bets", page_icon="⚽", layout="wide")

# Colours from the validated reference palette, stepped per theme
DARK = (getattr(getattr(st.context, "theme", None), "type", "light") == "dark")
BLUE = "#3987e5" if DARK else "#2a78d6"
RED = "#e66767" if DARK else "#e34948"
INK = "#f0efec" if DARK else "#0b0b0b"
GRID = "#383835" if DARK else "#e4e3df"
BASELINE = "#8a8984" if DARK else "#9b9a95"   # zero line: stronger than the grid, weaker than data


@st.cache_resource
def store() -> Store:
    return Store(Supabase())


@st.cache_data(ttl=300, show_spinner=False)
def load_picks() -> pd.DataFrame:
    df = picks_frame(store().picks())
    if len(df):
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_targets(days_back: int = 14) -> pd.DataFrame:
    rows = store().targets(since=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back))
    out = []
    for t in rows:
        m = t.get("matches") or {}
        out.append({"id": t["id"], "league": m.get("league"), "kickoff": m.get("kickoff"),
                    "home": (m.get("home") or {}).get("name"), "away": (m.get("away") or {}).get("name"),
                    "match_id": t["match_id"], "market": t["market"], "outcome": t["outcome"],
                    "target_odds": float(t["target_odds"]), "pin_odds": float(t["pin_odds"] or 0),
                    "p_model": float(t["p_model"]), "p_pin": float(t["p_pinnacle"])})
    df = pd.DataFrame(out)
    if len(df):
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_counts() -> dict:
    return store().counts()


def pct(v, signed=True) -> str:
    return "–" if pd.isna(v) else (f"{v:+.1%}" if signed else f"{v:.0%}")


def sast(ts) -> str:
    return pd.Timestamp(ts).tz_convert(SAST).strftime("%a %d %b %H:%M")


# ---- header -----------------------------------------------------------------------------
try:
    picks, targets = load_picks(), load_targets()
except Exception as e:  # missing keys, network
    st.error(f"Can't reach Supabase: {e}")
    st.stop()

k = kpis(picks)
st.title("⚽ Value Bets")
st.caption("Big-5 leagues · Betway & SportyBet vs model + Pinnacle · paper trading")

c = st.columns(6)
c[0].metric("Paper picks", k["picks"], help="Logged picks, voids excluded")
c[1].metric("Open", k["open"])
c[2].metric("Units", f"{k['units']:+.2f}", help="Flat 1 unit per settled pick")
c[3].metric("ROI", pct(k["roi"]))
c[4].metric("Average CLV", pct(k["avg_clv"]),
            help="Your price vs Pinnacle's closing fair price. Above 0 over hundreds of picks = real edge.")
c[5].metric("Gate", f"{k['with_clv']}/{GATE_PICKS}", help=GATE_PICKS_TEXT)
st.progress(k["gate_progress"], text=f"Phase 2 gate: {k['with_clv']} of {GATE_PICKS} picks with closing odds"
            + ("" if pd.isna(k["avg_clv"]) else f" · average CLV {pct(k['avg_clv'])}"))

tab_targets, tab_picks, tab_perf, tab_sys = st.tabs(["🎯 Targets", "📒 Picks", "📈 Performance", "⚙️ System"])

# ---- targets ------------------------------------------------------------------------------
with tab_targets:
    now = pd.Timestamp.now(tz="UTC")
    upcoming = targets[targets["kickoff"] > now].copy() if len(targets) else targets
    if upcoming.empty:
        st.info("No upcoming targets. The 08:00 job posts the day's targets.")
    else:
        taken = set(picks["match_id"]) if len(picks) else set()
        upcoming["Day"] = upcoming["kickoff"].dt.tz_convert(SAST).dt.strftime("%a %d %b")
        f1, f2 = st.columns([2, 3])
        day = f1.selectbox("Day", ["All days"] + list(dict.fromkeys(upcoming.sort_values("kickoff")["Day"])))
        lgs = f2.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES), format_func=LEAGUES.get)
        view = upcoming[upcoming["league"].isin(lgs)]
        if day != "All days":
            view = view[view["Day"] == day]
        view = view.sort_values(["kickoff", "home", "target_odds"])
        table = pd.DataFrame({
            "#": view["id"], "Kick-off (SAST)": view["kickoff"].map(sast),
            "League": view["league"].map(LEAGUES), "Match": view["home"] + " v " + view["away"],
            "Bet": view["outcome"].map(LABELS), "Bet if ≥": view["target_odds"],
            "Pinnacle": view["pin_odds"], "Model": view["p_model"], "Pinnacle fair": view["p_pin"],
            "Picked": view["match_id"].isin(taken).map({True: "✅", False: ""}),
        })
        st.caption(f"{len(view)} targets · {view['match_id'].nunique()} matches. "
                   "Bet only if Betway or SportyBet shows the target price or higher; one bet per match.")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Bet if ≥": st.column_config.NumberColumn(format="%.2f"),
            "Pinnacle": st.column_config.NumberColumn(format="%.2f"),
            "Model": st.column_config.NumberColumn(format="percent"),
            "Pinnacle fair": st.column_config.NumberColumn(format="percent"),
        })

        st.subheader("Log a bet")
        options = {int(r["id"]): f"#{int(r['id'])} · {r['home']} v {r['away']} · {LABELS[r['outcome']]} ≥ "
                   f"{r['target_odds']:.2f} ({sast(r['kickoff'])})" for _, r in view.iterrows()}
        with st.form("log_bet", clear_on_submit=True):
            a, b, cc, d = st.columns([4, 1.2, 1.4, 1.4])
            tid = a.selectbox("Target", list(options), format_func=options.get)
            price = b.number_input("Price you got", min_value=1.01, max_value=50.0, value=2.00, step=0.01,
                                   format="%.2f")
            book = cc.selectbox("Book", ["Betway", "SportyBet"])
            who = d.text_input("Logged by", value="me")
            if st.form_submit_button("Log paper pick", type="primary") and tid is not None:
                pick, msg = log_pick(store(), int(tid), float(price), book, who or "me", pd.Timestamp.now(tz="UTC"))
                (st.success if pick else st.warning)(msg)
                if pick:
                    load_picks.clear()

# ---- picks --------------------------------------------------------------------------------
with tab_picks:
    if picks.empty:
        st.info("No picks logged yet. Log one from the Targets tab, or reply /took in Telegram.")
    else:
        show = picks.sort_values("kickoff", ascending=False)
        status_icon = {"open": "⏳ open", "won": "✅ won", "lost": "❌ lost", "void": "➖ void"}
        table = pd.DataFrame({
            "Kick-off (SAST)": show["kickoff"].map(sast), "League": show["league"].map(LEAGUES),
            "Match": show["home"] + " v " + show["away"], "Bet": show["outcome"].map(LABELS),
            "Book": show["book"], "Odds": pd.to_numeric(show["odds"]),
            "Status": show["status"].map(status_icon),
            "Score": [f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) else ""
                      for h, a in zip(show["home_goals"], show["away_goals"])],
            "Units": pd.to_numeric(show["profit_units"]), "Close": pd.to_numeric(show["closing_odds"]),
            "CLV": pd.to_numeric(show["clv"]), "By": show["logged_by"],
        })
        only_open = st.toggle("Open picks only", value=False)
        if only_open:
            table = table[show["status"].values == "open"]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Close": st.column_config.NumberColumn(format="%.2f"),
            "Units": st.column_config.NumberColumn(format="%+.2f"),
            "CLV": st.column_config.NumberColumn(format="percent"),
        })

# ---- performance --------------------------------------------------------------------------
def zero_rule():
    return alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=BASELINE, strokeWidth=1).encode(y="y:Q")


with tab_perf:
    curve, clv = profit_curve(picks), clv_series(picks)
    if curve.empty and clv.empty:
        st.info("Charts appear once picks are settled. CLV shows up first (the night after kickoff); "
                "profit needs results.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Profit, units** (flat 1 unit per pick)")
            if curve.empty:
                st.caption("No settled picks yet.")
            else:
                base = alt.Chart(curve).encode(
                    x=alt.X("n:Q", title="Settled pick #", axis=alt.Axis(tickMinStep=1, grid=False)),
                    y=alt.Y("cum_units:Q", title=None, axis=alt.Axis(gridColor=GRID)),
                    tooltip=[alt.Tooltip("n:Q", title="Pick"), alt.Tooltip("match:N", title="Match"),
                             alt.Tooltip("result:N", title="Result"),
                             alt.Tooltip("profit:Q", title="Units", format="+.2f"),
                             alt.Tooltip("cum_units:Q", title="Running total", format="+.2f")])
                chart = zero_rule() + base.mark_line(color=BLUE, strokeWidth=2)
                if len(curve) <= 40:   # dots help with few picks, turn to noise with many
                    chart += base.mark_circle(color=BLUE, size=64, opacity=1)
                else:   # keep a generous hover target without drawing the dots
                    chart += base.mark_circle(size=200, opacity=0)
                st.altair_chart(chart.properties(height=280), width="stretch")
                with st.expander("Table"):
                    st.dataframe(curve, hide_index=True, width="stretch")
        with right:
            st.markdown("**CLV per pick**, with running average")
            if clv.empty:
                st.caption("No closing odds yet.")
            else:
                base = alt.Chart(clv).encode(x=alt.X("n:O", title="Pick # (with closing odds)",
                                                     axis=alt.Axis(labelAngle=0, grid=False)))
                bars = base.mark_bar(cornerRadiusEnd=4, size=10).encode(
                    y=alt.Y("clv:Q", title=None, axis=alt.Axis(format="%", gridColor=GRID)),
                    color=alt.Color("sign:N", title=None,
                                    scale=alt.Scale(domain=["beat the close", "worse than close"],
                                                    range=[BLUE, RED]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("match:N", title="Match"),
                             alt.Tooltip("clv:Q", title="CLV", format="+.1%"),
                             alt.Tooltip("running_avg:Q", title="Running average", format="+.1%")])
                line = base.mark_line(color=INK, strokeWidth=2, strokeDash=[4, 3]).encode(y="running_avg:Q")
                st.altair_chart((zero_rule() + bars + line).properties(height=280), width="stretch")
                st.caption("Dashed line = running average CLV. It matters far more than profit at this stage.")
                with st.expander("Table"):
                    st.dataframe(clv, hide_index=True, width="stretch")

        st.markdown("**Breakdown**")
        dim = st.radio("By", ["league", "outcome", "book"], horizontal=True,
                       format_func={"league": "League", "outcome": "Bet type", "book": "Bookmaker"}.get)
        bd = breakdown(picks, dim)
        if dim == "league":
            bd["league"] = bd["league"].map(LEAGUES)
        if dim == "outcome":
            bd["outcome"] = bd["outcome"].map(LABELS)
        st.dataframe(bd, hide_index=True, width="stretch", column_config={
            "units": st.column_config.NumberColumn("Units", format="%+.2f"),
            "roi": st.column_config.NumberColumn("ROI", format="percent"),
            "avg_clv": st.column_config.NumberColumn("Avg CLV", format="percent"),
        })

# ---- system -------------------------------------------------------------------------------
with tab_sys:
    st.markdown("**Locked strategy** · `both`: model edge ≥ 5% and Pinnacle edge ≥ 2%, odds 1.30–5.00, "
                "max 15-point model/Pinnacle disagreement, one pick per match, flat 1 unit.")
    st.markdown("**Schedule** · 08:00 SAST targets (Mondays + weekly summary) · 23:30 SAST closing odds + "
                "settling. Telegram `/took` replies are read at both runs; the form on the Targets tab logs instantly.")
    try:
        counts = load_counts()
        st.dataframe(pd.DataFrame({"Table": list(counts), "Rows": list(counts.values())}), hide_index=True)
    except Exception as e:
        st.warning(f"Couldn't count rows: {e}")
    if len(targets):
        st.caption(f"Latest targets cover kick-offs up to {sast(targets['kickoff'].max())} SAST.")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
