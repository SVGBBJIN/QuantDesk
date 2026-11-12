"""tqqq-desk dashboard (Streamlit).

Reads the bot's state from a local checkout of the `data` branch:
    git fetch origin data && git worktree add state origin/data     # first time
    git -C state pull                                               # refresh
    streamlit run dashboard/app.py            # or TQQQ_STATE_DIR=/path streamlit run ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Reference categorical order (validated: adjacent-pair CVD dE >= 8, normal-vision dE >= 15).
# Colour follows the entity, so each strategy always keeps the same hue.
SERIES_COLORS = {
    "ML ensemble": "#2a78d6",
    "Rule regime (fallback)": "#eb6834",
    "B&H QQQ": "#1baf7a",
    "B&H TQQQ": "#eda100",
    "TQQQ if QQQ>SMA200": "#e87ba4",
}
LIVE_COLOR = "#2a78d6"

st.set_page_config(page_title="tqqq-desk", page_icon="📈", layout="wide")


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _read_csv(p: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(p, **kw)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _age(ts) -> str:
    if not ts:
        return "never"
    d = pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts).tz_convert("UTC")
    h = d.total_seconds() / 3600
    return f"{h:.1f} h ago" if h < 48 else f"{h / 24:.1f} days ago"


def line_chart(df: pd.DataFrame, y_title: str, log: bool = False, fmt: str = ".2f", axis_fmt: str | None = None):
    long = df.reset_index().melt(id_vars=df.index.name or "date", var_name="series", value_name="value").dropna()
    long = long.rename(columns={df.index.name or "date": "date"})
    names = [c for c in df.columns]
    colors = [SERIES_COLORS.get(n, LIVE_COLOR) for n in names]
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelOverlap=True)),
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=axis_fmt or fmt),
                scale=alt.Scale(type="log") if log else alt.Scale(zero=False)),
        # a single series needs no legend box: the axis title names it
        color=alt.Color("series:N", scale=alt.Scale(domain=names, range=colors),
                        legend=alt.Legend(orient="top", title=None) if len(names) > 1 else None),
    )
    lines = base.mark_line(strokeWidth=2)
    rule = alt.Chart(long).mark_rule(color="#9a9a96", strokeWidth=1).encode(
        x="date:T",
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("date:T")] + [alt.Tooltip(f"{n}:Q", format=fmt) for n in names],
    ).transform_pivot("series", value="value", groupby=["date"]).add_params(hover)
    pts = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return (lines + rule + pts).properties(height=320).configure_axis(grid=True, gridOpacity=0.25)


# ------------------------------------------------------------------------------------- load
default_dir = os.environ.get("TQQQ_STATE_DIR", "state")
state_dir = Path(st.sidebar.text_input("State directory (checkout of the `data` branch)", default_dir))
art = state_dir / "artifacts"
state = _read_json(state_dir / "state.json")
status = _read_json(art / "model_status.json")
if st.sidebar.button("Reload"):
    st.rerun()

mode = (state.get("mode") or "paper").upper()
if mode == "LIVE":
    st.markdown("<div style='background:#c62828;color:#fff;padding:14px;border-radius:8px;font-size:28px;"
                "font-weight:700;text-align:center'>🔴 LIVE TRADING — REAL MONEY</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='background:#1565c0;color:#fff;padding:14px;border-radius:8px;font-size:28px;"
                "font-weight:700;text-align:center'>📝 PAPER TRADING</div>", unsafe_allow_html=True)
st.title("tqqq-desk")

if not state:
    st.warning(f"No state.json found in `{state_dir}`. Check out the `data` branch there (see README).")

# ------------------------------------------------------------------------------- risk + model
ks = state.get("kill_switch", {})
model = state.get("model", {})
c1, c2 = st.columns(2)
with c1:
    st.subheader("Risk status")
    if ks.get("tripped"):
        st.error(f"⛔ KILL SWITCH TRIPPED ({ks.get('action')}) at {ks.get('tripped_at')}\n\n{ks.get('reason')}\n\n"
                 "Trading is halted until a manual run with `reset_kill_switch`.")
    elif state.get("last_decision", {}).get("halted"):
        st.warning("⚠️ Last run halted (not latched): " + "; ".join(state["last_decision"].get("halt_reasons", [])))
    else:
        st.success("✅ Kill switch armed, not tripped")
with c2:
    st.subheader("Model")
    m = model.get("mode") or status.get("mode") or "RULE"
    if m == "ML":
        st.success("✅ ML ensemble active: it passed the out-of-sample gate")
    else:
        st.warning("⚠️ RULE-BASED FALLBACK ACTIVE. " + (model.get("message") or status.get("message")
                                                        or "No validation has been run yet."))

# ------------------------------------------------------------------------------- decision
st.subheader("Today's decision")
d = state.get("last_decision", {})
exp = state.get("expected_positions", {})
tgt = d.get("target")
tgt_txt = "frozen (halt)" if tgt is None and d else (", ".join(f"{s} {w:.0%}" for s, w in (tgt or {}).items()) or "CASH")
fmt_p = lambda x: f"{x:.2f}" if x is not None else "n/a (rule mode)"
st.table(pd.DataFrame({
    "Signal session": [d.get("signal_date") or "—"], "Regime": [d.get("regime") or "—"],
    "P(up, next 5d)": [fmt_p(d.get("p_up"))], "P(high vol, next 5d)": [fmt_p(d.get("p_hivol"))],
    "Model": [d.get("model_mode") or "—"], "Target": [tgt_txt],
}).set_index("Signal session"))
pos_txt = ", ".join(f"{s}: {q:g} sh" for s, q in exp.items()) or "CASH"
st.markdown(f"**Position after pending orders fill:** {pos_txt}")
if d.get("orders"):
    st.markdown("**Orders submitted (market-on-open):** " +
                "; ".join(f"{o['side'].upper()} {o['qty']} {o['symbol']}" for o in d["orders"]))
st.info("**Why:** " + (d.get("reason") or "no decision yet"))

# ------------------------------------------------------------------------------- freshness
st.subheader("Freshness")
f = st.columns(4)
f[0].metric("Last run", _age(state.get("last_run_at")))
f[1].metric("Last daily bar", state.get("data_last_bar") or "—")
f[2].metric("Model validated", _age(model.get("validated_at") or status.get("validated_at")))
f[3].metric("Model trained through", model.get("trained_through") or "—")
if status.get("data_source") and status.get("data_source") != "ALPACA":
    st.warning(f"Validation artifacts come from {status['data_source']} data, so they are a plumbing check only.")

# ------------------------------------------------------------------------------- equity
st.subheader("Live account equity")
eq = _read_csv(state_dir / "equity.csv", parse_dates=["date"])
if not eq.empty:
    e = eq.drop_duplicates("date", keep="last").set_index("date")
    dd = (e["equity"] / e["equity"].cummax() - 1).rename("Drawdown from peak").to_frame()
    m = st.columns(3)
    m[0].metric("Equity", f"${e['equity'].iloc[-1]:,.0f}")
    m[1].metric("Day P&L", f"{e['equity'].iloc[-1] / e['last_equity'].iloc[-1] - 1:+.2%}")
    m[2].metric("Drawdown from peak", f"{dd.iloc[-1, 0]:.1%}")
    if len(e) >= 2:
        st.altair_chart(line_chart(e[["equity"]].rename(columns={"equity": "Account equity"}), "Account equity ($)",
                                   fmt=",.0f", axis_fmt="$,.0f"), width="stretch")
        st.altair_chart(line_chart(dd, "Drawdown from peak", fmt=".1%", axis_fmt=".0%"), width="stretch")
    else:
        st.caption("Charts appear once there are at least two daily equity points.")
else:
    st.caption("No live equity history yet.")

st.subheader("Backtest: out-of-sample equity vs benchmarks (after costs)")
bt = _read_csv(art / "backtest_equity.csv", index_col=0, parse_dates=True)
if not bt.empty:
    bt.index.name = "date"
    st.altair_chart(line_chart(bt / bt.iloc[0], "Growth of $1 (log scale)", log=True, axis_fmt="$.1f"), width="stretch")
    bdd = _read_csv(art / "backtest_drawdown.csv", index_col=0, parse_dates=True)
    if not bdd.empty:
        bdd.index.name = "date"
        st.altair_chart(line_chart(bdd, "Drawdown from peak", fmt=".1%", axis_fmt=".0%"), width="stretch")
    summ = _read_csv(art / "backtest_summary.csv", index_col=0)
    if not summ.empty:
        cols = [c for c in ["CAGR", "Sharpe", "Sharpe95_lo", "Sharpe95_hi", "Sortino", "MaxDD", "Turnover/yr",
                            "Exposure", "HitRate(days)"] if c in summ]
        pct = {"CAGR", "MaxDD", "Exposure", "HitRate(days)"}
        st.dataframe(summ[cols].style.format({c: ("{:.1%}" if c in pct else "{:.2f}") for c in cols}), width="stretch")
    rep = state_dir / "reports" / "backtest_report.md"
    if rep.exists():
        with st.expander("Full backtest report (gate, model quality, sensitivity grids)"):
            st.markdown(rep.read_text())
else:
    st.caption("No backtest artifacts yet. Run the `validate` workflow or `scripts/backtest_report.py`.")

# ------------------------------------------------------------------------------- trades
st.subheader("Recent trades")
tr = _read_csv(state_dir / "trades.csv")
if not tr.empty:
    st.dataframe(tr.tail(25).iloc[::-1], width="stretch", hide_index=True)
else:
    st.caption("No trades yet.")
st.subheader("Recent signals")
sg = _read_csv(state_dir / "signals.csv")
if not sg.empty:
    st.dataframe(sg.tail(15).iloc[::-1], width="stretch", hide_index=True)
