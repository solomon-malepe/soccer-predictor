"""SPOVA GANG PREDICTOR: the soccer value-betting dashboard (Streamlit).

    streamlit run dashboard/app.py

Reads Supabase with the keys in .env (or Streamlit secrets when deployed).
Set DASHBOARD_PASSWORD (in .env or Streamlit secrets) to ask for a password; unset = open.
Tabs: targets (with an "I took it" button), our bets, results, help.
Experimental markets (🧪) have their own record, kept apart from the main one.
"""

from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path
from urllib.parse import quote

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
from predictor.value import is_experimental  # noqa: E402

LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
SAST = "Africa/Johannesburg"

APP_NAME = "SPOVA GANG PREDICTOR"
st.set_page_config(page_title=APP_NAME, page_icon="⚽", layout="wide")

# ---- look: a floodlit pitch behind everything ------------------------------------------------
# Pitch markings (105 x 68 m, to scale) drawn faintly, over mown-grass stripes.
# Low opacity so tables and numbers in front stay easy to read.
PITCH_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='-10 -10 1070 700' fill='none'
 stroke='white' stroke-opacity='0.09' stroke-width='3'><rect x='0' y='0' width='1050' height='680'/>
<line x1='525' y1='0' x2='525' y2='680'/><circle cx='525' cy='340' r='91.5'/>
<circle cx='525' cy='340' r='4' fill='white' fill-opacity='0.09'/>
<rect x='0' y='138.5' width='165' height='403'/><rect x='885' y='138.5' width='165' height='403'/>
<rect x='0' y='248.5' width='55' height='183'/><rect x='995' y='248.5' width='55' height='183'/>
<circle cx='110' cy='340' r='4' fill='white' fill-opacity='0.09'/><circle cx='940' cy='340' r='4' fill='white' fill-opacity='0.09'/>
<path d='M165 266.9 A91.5 91.5 0 0 1 165 413.1'/><path d='M885 266.9 A91.5 91.5 0 0 0 885 413.1'/>
</svg>"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
.stApp {{
  background:
    radial-gradient(ellipse 70% 45% at 50% -5%, rgba(255,236,170,.16), transparent 70%),
    url("data:image/svg+xml,{quote(PITCH_SVG)}") center 55% / min(1200px, 94vw) auto no-repeat fixed,
    repeating-linear-gradient(90deg, rgba(255,255,255,.028) 0 90px, transparent 90px 180px),
    linear-gradient(180deg, #0e2d1c 0%, #0a1f14 55%, #07170f 100%);
  background-attachment: fixed;
}}
[data-testid="stHeader"] {{ background: transparent; }}
.spova-title {{
  font-family: 'Bebas Neue', Impact, sans-serif; letter-spacing: .06em; line-height: .95;
  font-size: clamp(2.6rem, 7vw, 4.6rem); margin: .2rem 0 .1rem;
  background: linear-gradient(180deg, #ffe58a 0%, #f2c230 55%, #c9971a 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  text-shadow: 0 2px 24px rgba(242,194,48,.18);
}}
.spova-sub {{ color: #b9cdbf; font-size: .95rem; margin-bottom: 1.1rem; }}
[data-testid="stMetric"] {{
  background: rgba(10,31,20,.72); border: 1px solid rgba(242,194,48,.22);
  border-radius: 14px; padding: .7rem .9rem; backdrop-filter: blur(4px);
}}
[data-testid="stMetricValue"] {{ color: #fff6d6; }}
[data-testid="stDataFrame"], [data-testid="stForm"], [data-testid="stExpander"] {{
  background: rgba(10,31,20,.80); border-radius: 12px;
}}
.steps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 0 0 1rem; }}
.steps div {{
  background: rgba(10,31,20,.72); border: 1px solid rgba(242,194,48,.22); border-radius: 12px;
  padding: .6rem .8rem; font-size: .92rem; color: #dfe9e1;
}}
.steps b:first-child {{
  display: inline-grid; place-items: center; width: 1.5rem; height: 1.5rem; margin-right: .4rem;
  border-radius: 50%; background: #f2c230; color: #0a1f14; font-weight: 700;
}}
@media (max-width: 640px) {{ .steps {{ grid-template-columns: 1fr; }} }}
.match {{ font-weight: 700; font-size: 1.08rem; color: #fff6d6; }}
.meta {{ color: #9fb7a7; font-size: .85rem; margin-bottom: .3rem; }}
.target {{ font-weight: 800; font-size: 1.25rem; color: #f2c230; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: rgba(10,31,20,.78); border-radius: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{
  background: rgba(10,31,20,.65); border-radius: 10px 10px 0 0; padding: .4rem 1rem;
}}
</style>
""", unsafe_allow_html=True)


def header(sub: str) -> None:
    st.markdown(f"<div class='spova-title'>⚽ {APP_NAME}</div><div class='spova-sub'>{sub}</div>",
                unsafe_allow_html=True)


# ---- password (only when DASHBOARD_PASSWORD is set) -------------------------------------------
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
if PASSWORD and not st.session_state.get("authed"):
    header("Members only. Enter the gang password.")
    with st.form("login"):
        given = st.text_input("Password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if hmac.compare_digest(given.encode(), PASSWORD.encode()):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

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



def track(df: pd.DataFrame) -> pd.Series:
    """'experimental' or 'both' (the main record) for each pick."""
    if "strategy" not in df:
        return pd.Series("both", index=df.index)
    return df["strategy"].fillna("both")


def bet_label(outcome: str, market: str) -> str:
    return ("🧪 " if is_experimental(market) else "") + LABELS.get(outcome, outcome)


# ---- data -----------------------------------------------------------------------------------
try:
    picks, targets = load_picks(), load_targets()
except Exception as e:  # missing keys, network
    st.error(f"Can't reach the database right now: {e}")
    st.stop()

main_picks = picks[track(picks) != "experimental"] if len(picks) else picks
exp_picks = picks[track(picks) == "experimental"] if len(picks) else picks
k = kpis(main_picks)

header("Value bets for the big-5 leagues · Betway &amp; SportyBet · paper money for now")
st.markdown("""
<div class='steps'>
  <div><b>1</b> Find a target below</div>
  <div><b>2</b> Open Betway or SportyBet. Odds the same or higher? Bet 1 unit</div>
  <div><b>3</b> Tap <b>I took it</b> and enter the odds you got</div>
</div>""", unsafe_allow_html=True)

if flash := st.session_state.pop("flash", None):
    st.success(flash)

tab_targets, tab_bets, tab_results, tab_help = st.tabs(["🎯 Targets", "📒 Our bets", "📈 Results", "❓ Help"])


# ---- "I took it" pop-up ----------------------------------------------------------------------
@st.dialog("Log your bet")
def take_dialog(r: dict) -> None:
    st.markdown(f"**{r['home']} v {r['away']}**  \n{bet_label(r['outcome'], r['market'])} · "
                f"target **{r['target_odds']:.2f}** or higher")
    if is_experimental(r["market"]):
        st.caption("🧪 Experimental market: scored separately, doesn't count towards the main record.")
    price = st.number_input("Odds you got", min_value=1.01, max_value=50.0, value=float(r["target_odds"]),
                            step=0.01, format="%.2f")
    book = st.radio("Bookie", ["Betway", "SportyBet"], horizontal=True)
    who = st.text_input("Your name", value=st.session_state.get("who", ""), placeholder="e.g. Sol")
    if price < r["target_odds"]:
        st.warning(f"{price:.2f} is below the target of {r['target_odds']:.2f}, so it isn't a value bet.")
    if st.button("Save bet", type="primary", width="stretch", disabled=price < r["target_odds"]):
        if not who.strip():
            st.warning("Add your name so the gang knows who took it.")
            return
        st.session_state["who"] = who.strip()
        pick, msg = log_pick(store(), int(r["id"]), float(price), book, who.strip(),
                             pd.Timestamp.now(tz="UTC"))
        if pick:
            load_picks.clear()
            st.session_state["flash"] = msg
            st.rerun()
        st.warning(msg)


# ---- targets ---------------------------------------------------------------------------------
with tab_targets:
    now = pd.Timestamp.now(tz="UTC")
    up = targets[targets["kickoff"] > now].copy() if len(targets) else targets
    if up.empty:
        st.info("No targets right now. New ones arrive every morning (usually 08:00, sometimes later).")
    else:
        up["exp"] = up["market"].map(is_experimental)
        today = now.tz_convert(SAST).normalize()
        up["day"] = up["kickoff"].dt.tz_convert(SAST).dt.normalize()

        def day_name(d) -> str:
            diff = (d - today).days
            return "Today" if diff == 0 else "Tomorrow" if diff == 1 else d.strftime("%A %d %b")

        up["day_name"] = up["day"].map(day_name)
        day_names = list(dict.fromkeys(up.sort_values("kickoff")["day_name"]))
        f1, f2 = st.columns([3, 2], vertical_alignment="bottom")
        day = f1.pills("Day", ["All days"] + day_names, default="All days") or "All days"
        show_exp = f2.toggle("Show 🧪 experimental", value=True,
                             help="BTTS and Over/Under 1.5 / 3.5. Never tested on past seasons, so they're "
                                  "tracked separately from the main record.")
        with st.expander("Filter leagues"):
            lgs = st.multiselect("Leagues", list(LEAGUES), default=list(LEAGUES), format_func=LEAGUES.get,
                                 label_visibility="collapsed")
        view = up[up["league"].isin(lgs)]
        if day != "All days":
            view = view[view["day_name"] == day]
        if not show_exp:
            view = view[~view["exp"]]

        # who already took what: (match, track) -> pick
        taken = {}
        for _, p in picks.iterrows():
            taken[(p["match_id"], "experimental" if p.get("strategy") == "experimental" else "both")] = p

        n_main, n_exp = int((~view["exp"]).sum()), int(view["exp"].sum())
        st.caption(f"{view['match_id'].nunique()} matches · {n_main} targets"
                   + (f" + {n_exp} 🧪" if n_exp else "")
                   + " · most won't be on offer, so check before you bet")
        view = view.sort_values(["kickoff", "home", "exp", "target_odds"])
        for (mid, home, away), g in view.groupby(["match_id", "home", "away"], sort=False):
            first = g.iloc[0]
            with st.container(border=True):
                st.markdown(f"<div class='match'>{home} v {away}</div>"
                            f"<div class='meta'>{sast(first['kickoff'])} · {LEAGUES.get(first['league'], '')}</div>",
                            unsafe_allow_html=True)
                for _, r in g.iterrows():
                    key = (mid, "experimental" if r["exp"] else "both")
                    a, b, c = st.columns([3, 3, 2], vertical_alignment="center")
                    a.markdown(f"**{bet_label(r['outcome'], r['market'])}**")
                    b.markdown(f"bet if odds ≥ <span class='target'>{r['target_odds']:.2f}</span>",
                               unsafe_allow_html=True)
                    p = taken.get(key)
                    if p is not None and int(p.get("target_id") or 0) == int(r["id"]):
                        c.markdown(f"✅ {p['logged_by']} @ {float(p['odds']):.2f}")
                    elif p is not None:
                        c.caption("Match already has a bet")
                    elif c.button("I took it", key=f"take_{int(r['id'])}", width="stretch",
                                  type="secondary" if r["exp"] else "primary"):
                        take_dialog(r.to_dict())


# ---- our bets --------------------------------------------------------------------------------
TRACKS = {"Main record": "both", "🧪 Experimental": "experimental"}


def pick_track(key: str) -> pd.DataFrame:
    choice = st.segmented_control("Record", list(TRACKS), default="Main record", key=key,
                                  label_visibility="collapsed") or "Main record"
    return main_picks if TRACKS[choice] == "both" else exp_picks


def kpi_row(df: pd.DataFrame, main: bool) -> None:
    kk = kpis(df)
    c = st.columns(4)
    c[0].metric("Bets", kk["picks"])
    c[1].metric("Waiting for result", kk["open"])
    c[2].metric("Profit", f"{kk['units']:+.2f} units", help="1 unit per bet. 1 unit = whatever stake you agree on.")
    c[3].metric("Beat the closing odds", pct(kk["avg_clv"]),
                help="How much better our odds were than Pinnacle's final price before kickoff. "
                     "Above 0% over many bets = a real edge, even through losing runs.")
    if main:
        st.progress(kk["gate_progress"], text=f"Real money unlocks after {GATE_PICKS} bets that beat the "
                    f"closing odds on average · {kk['with_clv']} of {GATE_PICKS} so far")


with tab_bets:
    sel = pick_track("bets_track")
    is_main = sel is main_picks
    kpi_row(sel, is_main)
    if sel.empty:
        st.info("No bets yet. Tap **I took it** on a target, or reply /took in the Telegram group.")
    else:
        show = sel.sort_values("kickoff", ascending=False)
        result = {"open": "⏳ waiting", "won": "✅ won", "lost": "❌ lost", "void": "➖ void"}
        table = pd.DataFrame({
            "Kick-off": show["kickoff"].map(sast),
            "Match": show["home"] + " v " + show["away"],
            "Bet": [bet_label(o, m) for o, m in zip(show["outcome"], show["market"])],
            "Odds": pd.to_numeric(show["odds"]), "Bookie": show["book"], "By": show["logged_by"],
            "Result": show["status"].map(result),
            "Score": [f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) else ""
                      for h, a in zip(show["home_goals"], show["away_goals"])],
            "Profit": pd.to_numeric(show["profit_units"]),
        })
        if st.toggle("Only bets waiting for a result", value=False):
            table = table[show["status"].values == "open"]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Odds": st.column_config.NumberColumn(format="%.2f"),
            "Profit": st.column_config.NumberColumn(format="%+.2f"),
        })


# ---- results ---------------------------------------------------------------------------------
def zero_rule():
    return alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=BASELINE, strokeWidth=1).encode(y="y:Q")


with tab_results:
    sel = pick_track("results_track")
    curve, clv = profit_curve(sel), clv_series(sel)
    if curve.empty and clv.empty:
        st.info("Charts appear once bets are settled. 'Beat the closing odds' shows up first (the night "
                "after kick-off); profit needs the final score.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Profit over time** (units)")
            if curve.empty:
                st.caption("No settled bets yet.")
            else:
                base = alt.Chart(curve).encode(
                    x=alt.X("n:Q", title="Bet number", axis=alt.Axis(tickMinStep=1, grid=False)),
                    y=alt.Y("cum_units:Q", title=None, axis=alt.Axis(gridColor=GRID)),
                    tooltip=[alt.Tooltip("n:Q", title="Bet"), alt.Tooltip("match:N", title="Match"),
                             alt.Tooltip("result:N", title="Result"),
                             alt.Tooltip("profit:Q", title="Units", format="+.2f"),
                             alt.Tooltip("cum_units:Q", title="Running total", format="+.2f")])
                chart = zero_rule() + base.mark_line(color=BLUE, strokeWidth=2)
                if len(curve) <= 40:   # dots help with few bets, turn to noise with many
                    chart += base.mark_circle(color=BLUE, size=64, opacity=1)
                else:   # keep a generous hover target without drawing the dots
                    chart += base.mark_circle(size=200, opacity=0)
                st.altair_chart(chart.properties(height=280), width="stretch")
        with right:
            st.markdown("**Did we beat the closing odds?** One bar per bet")
            if clv.empty:
                st.caption("No closing odds yet.")
            else:
                base = alt.Chart(clv).encode(x=alt.X("n:O", title="Bet number",
                                                     axis=alt.Axis(labelAngle=0, grid=False)))
                bars = base.mark_bar(cornerRadiusEnd=4, size=10).encode(
                    y=alt.Y("clv:Q", title=None, axis=alt.Axis(format="%", gridColor=GRID)),
                    color=alt.Color("sign:N", title=None,
                                    scale=alt.Scale(domain=["beat the close", "worse than close"],
                                                    range=[BLUE, RED]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("match:N", title="Match"),
                             alt.Tooltip("clv:Q", title="vs closing odds", format="+.1%"),
                             alt.Tooltip("running_avg:Q", title="Running average", format="+.1%")])
                line = base.mark_line(color=INK, strokeWidth=2, strokeDash=[4, 3]).encode(y="running_avg:Q")
                st.altair_chart((zero_rule() + bars + line).properties(height=280), width="stretch")
                st.caption("Dashed line = running average. Early on this matters far more than profit.")

        st.markdown("**Breakdown**")
        dim = st.radio("By", ["league", "outcome", "book"], horizontal=True,
                       format_func={"league": "League", "outcome": "Bet type", "book": "Bookie"}.get)
        bd = breakdown(sel, dim)
        if dim == "league":
            bd["league"] = bd["league"].map(LEAGUES)
        if dim == "outcome":
            bd["outcome"] = bd["outcome"].map(LABELS)
        st.dataframe(bd, hide_index=True, width="stretch", column_config={
            "units": st.column_config.NumberColumn("Units", format="%+.2f"),
            "roi": st.column_config.NumberColumn("Return", format="percent"),
            "avg_clv": st.column_config.NumberColumn("vs closing odds", format="percent"),
        })


# ---- help ------------------------------------------------------------------------------------
with tab_help:
    st.markdown(f"""
**What is a target?**
The lowest odds at which a bet is worth taking. Our model and Pinnacle (the sharpest bookie in the
world) both have to agree the bet is good value at that price. If Betway or SportyBet offers the
target or more, take it.

**Why are most targets not on offer?**
They need odds a bit better than Pinnacle's, which is rare. A real chance turns up about once a week.
That's normal. Don't bet below the target.

**How do I log a bet?**
Tap **I took it** on the target here, or reply in the Telegram group:
`/took 57 2.90 sportybet` (target number, odds you got, bookie). One bet per match on the main record,
plus one 🧪 if you like.

**What does 🧪 mean?**
Experimental markets: both teams to score, and Over/Under 1.5 and 3.5. They use the same rule but were
never tested on past seasons, so they have their own record and don't count towards real money.

**What is "beat the closing odds"?**
The best sign of a real edge. If our odds are better than Pinnacle's final price just before kick-off,
we're getting value, even through a losing run. Profit over a few dozen bets is mostly luck.

**When do we bet real money?**
{GATE_PICKS_TEXT}
""")
    st.divider()
    st.markdown("**Schedule** · targets every morning around 08:00 · results and closing odds every night "
                "around 23:30 (SA time). GitHub sometimes runs these a few hours late.")
    try:
        counts = load_counts()
        st.caption("Database: " + " · ".join(f"{t} {n}" for t, n in counts.items()))
    except Exception as e:
        st.caption(f"Couldn't count rows: {e}")
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
