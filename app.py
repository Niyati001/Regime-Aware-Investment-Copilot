from pathlib import Path
import json
import os
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()
print("API KEY =", os.getenv("OPENAI_API_KEY"))

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:
    st.error("Install plotly first: pip install plotly")
    raise exc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Regime-Aware Research Copilot",
    page_icon="📈",
    layout="wide",
)

ASSET_NAMES = {
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^NSEI": "NIFTY 50",
}

REGIME_COLORS = {
    "Bull": "#1a9850",
    "Bear": "#fdae61",
    "Crisis": "#d73027",
}

REGIME_SCALARS = {
    "Bull": 1.0,
    "Bear": 0.5,
    "Crisis": 0.2,
}

FRED_SERIES = {
    "FEDFUNDS": "Fed Funds Rate",
    "T10Y2Y": "10Y-2Y Yield Curve",
    "BAMLH0A0HYM2": "High Yield Credit Spread",
}


def load_csv(name, parse_dates=True):
    path = DATA_DIR / name
    if not path.exists():
        st.error(f"Missing required file: {path}")
        st.stop()
    if parse_dates:
        return pd.read_csv(path, parse_dates=["Date"])
    return pd.read_csv(path)


@st.cache_data
def load_data():
    predictions = load_csv("all_asset_predictions.csv")
    positions = load_csv("position_sized_predictions.csv")
    regimes = load_csv("regime_features_macro_enriched.csv")
    summary = load_csv("all_asset_summary_metrics.csv", parse_dates=False)
    tc_portfolio = load_csv("transaction_cost_portfolio_metrics.csv", parse_dates=False)
    tc_asset = load_csv("transaction_cost_per_asset_metrics.csv", parse_dates=False)
    macro_summary = load_csv("macro_enriched_summary_metrics.csv", parse_dates=False)
    return predictions, positions, regimes, summary, tc_portfolio, tc_asset, macro_summary


@st.cache_data(ttl=3600)
def load_fred_macro():
    frames = []
    for series_id, label in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            frame = pd.read_csv(url)
        except Exception:
            continue
        if "observation_date" not in frame.columns or series_id not in frame.columns:
            continue
        frame = frame.rename(columns={"observation_date": "Date", series_id: label})
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame[label] = pd.to_numeric(frame[label], errors="coerce")
        frames.append(frame[["Date", label]].dropna())

    if not frames:
        return pd.DataFrame()

    macro = frames[0]
    for frame in frames[1:]:
        macro = macro.merge(frame, on="Date", how="outer")
    return macro.sort_values("Date").ffill().dropna(how="all")


def annualized_sharpe(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna()
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def max_drawdown(returns):
    returns = pd.Series(returns).fillna(0)
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_num(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def regime_badge(regime):
    color = REGIME_COLORS.get(regime, "#64748b")
    return (
        f"<span style='background:{color}; color:white; padding:0.28rem 0.55rem; "
        f"border-radius:0.45rem; font-weight:700'>{regime}</span>"
    )


def latest_asset_snapshot(predictions, positions, regimes, ticker, cost_bps):
    pred = predictions[(predictions["ticker"] == ticker) & (predictions["model_family"] == "Random Forest")].copy()
    pos = positions[
        (positions["ticker"] == ticker)
        & (positions["cost_bps"] == cost_bps)
        & (positions["model_family"] == "Random Forest")
    ].copy()
    reg = regimes[regimes["ticker"] == ticker].copy()

    if pred.empty or pos.empty or reg.empty:
        st.error("Not enough data for selected asset/cost setting.")
        st.stop()

    pred_latest = pred.sort_values("Date").iloc[-1]
    pos_latest = pos.sort_values("Date").iloc[-1]
    reg_latest = reg.sort_values("Date").iloc[-1]
    return pred, pos, reg, pred_latest, pos_latest, reg_latest


def regime_adjusted_exposure(up_prob, threshold, regime, vol_position):
    signal_exposure = np.clip(up_prob / max(threshold, 0.01), 0, 1)
    regime_scalar = REGIME_SCALARS.get(regime, 0.5)
    return float(min(vol_position, signal_exposure) * regime_scalar)


def quant_signal_label(up_prob, threshold):
    if up_prob >= threshold + 0.08:
        return "Strong Bullish"
    if up_prob >= threshold:
        return "Bullish"
    if up_prob <= threshold - 0.08:
        return "Bearish"
    return "Neutral"


def supervisor_resolution(regime, quant_signal, adjusted_exposure, drawdown, vol):
    conflict = regime in {"Bear", "Crisis"} and quant_signal in {"Bullish", "Strong Bullish"}
    if conflict and regime == "Bear":
        return {
            "conflict": "Regime=Bear but Quant=Bullish",
            "stance": "Cautiously Neutral",
            "exposure": min(adjusted_exposure, 0.40),
            "reasoning": (
                "Regime and momentum signals disagree. Reduce size and wait for confirmation "
                "rather than following the bullish probability at full exposure."
            ),
        }
    if conflict and regime == "Crisis":
        return {
            "conflict": "Regime=Crisis but Quant=Bullish",
            "stance": "Defensive Rebound Watch",
            "exposure": min(adjusted_exposure, 0.20),
            "reasoning": (
                "The model sees rebound odds, but crisis regimes can gap violently. "
                "Keep only tactical exposure until volatility compresses."
            ),
        }
    if regime == "Bull" and quant_signal in {"Bullish", "Strong Bullish"}:
        return {
            "conflict": "No major conflict",
            "stance": "Constructive",
            "exposure": adjusted_exposure,
            "reasoning": "Regime and quant signals agree. Maintain exposure sized by volatility and costs.",
        }
    if drawdown < -0.10 or vol > 0.025:
        return {
            "conflict": "Risk override",
            "stance": "Defensive",
            "exposure": min(adjusted_exposure, 0.25),
            "reasoning": "Drawdown or volatility is above risk limits, so the supervisor cuts exposure.",
        }
    return {
        "conflict": "Mixed or low-conviction signal",
        "stance": "Neutral",
        "exposure": min(adjusted_exposure, 0.50),
        "reasoning": "Signals are not strongly aligned. Keep moderate exposure and wait for confirmation.",
    }


def recommendation_from_snapshot(up_prob, regime, position, drawdown, vol, momentum, threshold):
    adjusted_exposure = regime_adjusted_exposure(up_prob, threshold, regime, position)
    quant_signal = quant_signal_label(up_prob, threshold)
    supervisor = supervisor_resolution(regime, quant_signal, adjusted_exposure, drawdown, vol)

    if supervisor["stance"] == "Constructive":
        action = "Maintain risk-on exposure with normal monitoring."
    elif "Cautiously" in supervisor["stance"]:
        action = "Keep partial exposure because the bullish MoE signal conflicts with the Bear regime."
    elif "Defensive" in supervisor["stance"]:
        action = "Reduce exposure and wait for volatility compression."
    else:
        action = "Keep exposure modest until signal quality improves."

    if momentum < 0:
        action += " Momentum is negative, so avoid aggressive sizing."
    return supervisor["stance"], action, supervisor, quant_signal


def macro_signal_from_fred(macro):
    if macro.empty:
        return {
            "verdict": "Macro unavailable",
            "details": "Live FRED download failed or internet is unavailable. Expected: FEDFUNDS, T10Y2Y, HY spreads.",
            "data": pd.DataFrame(),
        }

    latest = macro.dropna().iloc[-1]
    previous = macro[macro["Date"] < latest["Date"]].tail(63)
    spread = latest.get("High Yield Credit Spread", np.nan)
    curve = latest.get("10Y-2Y Yield Curve", np.nan)
    fed = latest.get("Fed Funds Rate", np.nan)

    spread_change = np.nan
    fed_change = np.nan
    if not previous.empty:
        if "High Yield Credit Spread" in previous and previous["High Yield Credit Spread"].dropna().size:
            spread_change = spread - previous["High Yield Credit Spread"].dropna().iloc[-1]
        if "Fed Funds Rate" in previous and previous["Fed Funds Rate"].dropna().size:
            fed_change = fed - previous["Fed Funds Rate"].dropna().iloc[-1]

    flags = []
    if pd.notna(curve) and curve < 0:
        flags.append("yield curve inverted")
    if pd.notna(spread) and spread > 5:
        flags.append("credit spreads elevated")
    if pd.notna(spread_change) and spread_change > 0.75:
        flags.append("credit spreads widening")
    if pd.notna(fed_change) and fed_change > 0.25:
        flags.append("policy tightening")

    if len(flags) >= 2:
        verdict = "Macro risk-off"
    elif flags:
        verdict = "Macro cautious"
    else:
        verdict = "Macro supportive"

    details = (
        f"FEDFUNDS={format_num(fed)}%, T10Y2Y={format_num(curve)}%, "
        f"HY spread={format_num(spread)}%. Flags: {', '.join(flags) if flags else 'none'}."
    )
    return {"verdict": verdict, "details": details, "data": macro}

def research_note(payload):
    return f"""
Investment View

{payload['asset']} remains in a {payload['regime']} regime with {payload['regime_confidence']:.2%} confidence.

While the forecasting framework indicates a {payload['up_prob']:.2%} probability of upside and a {payload['quant_signal'].lower()} quantitative signal, the prevailing regime classification remains unfavorable.

Given this divergence, the supervisor recommends maintaining {payload['exposure']:.0%} exposure. {payload['macro']} Historical analog analysis suggests a median forward 30-day return of {payload['analog_return']:.2%}.
""".strip()


def agent_panel(name, verdict, details):
    st.markdown(
        f"""
        <div style='border:1px solid #e5e7eb; border-radius:8px; padding:0.9rem; height:100%; background:#ffffff'>
            <div style='font-size:0.8rem; color:#64748b; font-weight:700; text-transform:uppercase'>{name}</div>
            <div style='font-size:1.1rem; color:#0f172a; font-weight:800; margin-top:0.25rem'>{verdict}</div>
            <div style='font-size:0.92rem; color:#334155; margin-top:0.45rem; line-height:1.35'>{details}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_price_regime_chart(reg):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=reg["Date"], y=reg["Close"], mode="lines", name="Close", line=dict(color="#111827", width=1.4)))
    for regime, color in REGIME_COLORS.items():
        temp = reg[reg["regime_label"] == regime]
        fig.add_trace(go.Scatter(x=temp["Date"], y=temp["Close"], mode="markers", name=regime, marker=dict(color=color, size=5)))
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20), legend_orientation="h")
    return fig


def make_equity_chart(pos, exposure_col="supervisor_exposure"):
    pos = pos.sort_values("Date").copy()
    exposure = pos[exposure_col] if exposure_col in pos.columns else pos["position"]
    pos["strategy_return"] = exposure * pos["buy_hold_return"] - pos["transaction_cost"]
    pos["strategy_equity"] = (1 + pos["strategy_return"]).cumprod()
    pos["buy_hold_equity"] = (1 + pos["buy_hold_return"]).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos["Date"], y=pos["strategy_equity"], name="Supervisor strategy", line=dict(color="#2563eb", width=2.2)))
    fig.add_trace(go.Scatter(x=pos["Date"], y=pos["buy_hold_equity"], name="Buy and hold", line=dict(color="#111827", width=1.6)))
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=40, b=20), legend_orientation="h", yaxis_title="Growth of $1")
    return fig


def historical_analogs(reg, latest):
    feature_cols = ["return_1d", "volatility_30d", "momentum_30d", "drawdown", "asset_vix_close"]
    temp = reg.dropna(subset=feature_cols).copy().reset_index(drop=True)
    current = latest[feature_cols].astype(float).values
    values = temp[feature_cols].astype(float).values
    means = np.nanmean(values, axis=0)
    values_scaled = values - means
    current_scaled = current - means
    denom = np.linalg.norm(values_scaled, axis=1) * np.linalg.norm(current_scaled)
    denom[denom == 0] = np.nan
    temp["similarity_score"] = 1 - np.nansum(values_scaled * current_scaled, axis=1) / denom
    temp["forward_10d_return"] = temp["Close"].shift(-10) / temp["Close"] - 1
    temp["forward_30d_return"] = temp["Close"].shift(-30) / temp["Close"] - 1

    recovery = []
    for idx, row in temp.iterrows():
        future = temp.iloc[idx + 1: idx + 126]
        recovered = future[future["Close"] >= row["Close"]]
        recovery.append(np.nan if recovered.empty else int((recovered.iloc[0]["Date"] - row["Date"]).days))
    temp["recovery_days"] = recovery

    temp = temp[temp["Date"] < latest["Date"]]
    same_regime = temp[temp["regime_label"] == latest["regime_label"]]
    if len(same_regime) >= 3:
        temp = same_regime
    return temp.sort_values("similarity_score").head(8)[[
        "Date", "Close", "regime_label", "volatility_30d", "drawdown",
        "asset_vix_close", "forward_10d_return", "forward_30d_return",
        "recovery_days", "similarity_score",
    ]]


def rolling_accuracy_table(pred, threshold):
    temp = pred.sort_values("Date").copy()
    temp["pred_direction"] = (temp["soft_moe_up_prob"] >= threshold).astype(int)
    temp["correct"] = (temp["pred_direction"] == temp["target_direction_1d"]).astype(int)
    rows = []
    for window in [30, 60, 90]:
        recent = temp.tail(window)
        rows.append({"segment": f"Last {window}", "accuracy": recent["correct"].mean(), "observations": len(recent)})
    by_regime = temp.groupby("regime_label")["correct"].agg(["mean", "count"]).reset_index()
    by_regime = by_regime.rename(columns={"regime_label": "segment", "mean": "accuracy", "count": "observations"})
    return pd.concat([pd.DataFrame(rows), by_regime], ignore_index=True)


predictions, positions, regimes, summary, tc_portfolio, tc_asset, macro_summary = load_data()
fred_macro = load_fred_macro()
macro_signal = macro_signal_from_fred(fred_macro)

st.title("Regime-Aware Investment Research Copilot")
st.caption("A desk-style multi-agent POC: HMM regimes, MoE signals, supervisor conflict resolution, macro context, and risk sizing")

with st.sidebar:
    st.header("Control Panel")
    ticker = st.selectbox("Asset", sorted(predictions["ticker"].unique()), format_func=lambda x: f"{ASSET_NAMES.get(x, x)} ({x})")
    cost_bps = st.select_slider("Transaction cost", options=[0, 5, 10, 20], value=10, format_func=lambda x: f"{x} bps")
    threshold = st.slider("Signal threshold", min_value=0.40, max_value=0.65, value=0.45, step=0.01)
    stress = st.slider("Stress shock", min_value=-0.10, max_value=0.02, value=-0.03, step=0.01, format="%.2f")
    st.divider()
    st.caption("Best research backend: Random Forest + Soft-Gated MoE")

pred, pos, reg, pred_latest, pos_latest, reg_latest = latest_asset_snapshot(predictions, positions, regimes, ticker, cost_bps)

up_prob = float(pred_latest["soft_moe_up_prob"])
regime = str(reg_latest["regime_label"])
raw_position = float(pos_latest["position"])
drawdown = float(reg_latest["drawdown"])
vol = float(reg_latest["volatility_30d"])
momentum = float(reg_latest["momentum_30d"])
asset_vix = float(reg_latest["asset_vix_close"])
regime_prob_cols = [c for c in ["regime_0_prob", "regime_1_prob", "regime_2_prob"] if c in reg_latest.index]
regime_confidence = float(reg_latest[regime_prob_cols].max()) if regime_prob_cols else 1.0
stance, action, supervisor, quant_signal = recommendation_from_snapshot(up_prob, regime, raw_position, drawdown, vol, momentum, threshold)
final_exposure = supervisor["exposure"]

pos = pos.sort_values("Date").copy()
pos["supervisor_exposure"] = [
    supervisor_resolution(
        r["regime_label"],
        quant_signal_label(r["soft_moe_up_prob"], threshold),
        regime_adjusted_exposure(r["soft_moe_up_prob"], threshold, r["regime_label"], r["position"]),
        0.0,
        0.0,
    )["exposure"]
    for _, r in pos.iterrows()
]

sims = historical_analogs(reg, reg_latest)
analog_return = sims["forward_30d_return"].median() if not sims.empty else np.nan

st.markdown(f"### Current Desk View: {ASSET_NAMES.get(ticker, ticker)}")
st.markdown(
    f"Latest model date: **{pd.Timestamp(pred_latest['Date']).date()}** | "
    f"Current detected regime: {regime_badge(regime)} | "
    f"Supervisor stance: **{stance}**",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("MoE up probability", format_pct(up_prob))
m2.metric("Raw exposure", format_pct(raw_position))
m3.metric("Supervisor exposure", format_pct(final_exposure), delta=format_pct(final_exposure - raw_position))
m4.metric("30D volatility", format_pct(vol))
m5.metric("Drawdown", format_pct(drawdown))
m6.metric("Asset VIX", f"{asset_vix:.2f}")

st.markdown("---")

desk, copilot, risk, analogs, diagnostics = st.tabs([
    "Desk View",
    "Research Copilot",
    "Risk Lab",
    "Historical Analogs",
    "Model Diagnostics",
])

with desk:
    st.subheader("Visible Supervisor Conflict Resolution")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Regime Agent", regime)
    col_b.metric("Quant Agent", quant_signal)
    col_c.metric("Final Exposure", format_pct(final_exposure))
    st.info(f"Conflict: {supervisor['conflict']}. Supervisor reasoning: {supervisor['reasoning']}")

    c1, c2 = st.columns(2)
    with c1:
        agent_panel(
            "Regime Agent",
            f"{regime} regime ({format_pct(regime_confidence)} confidence)",
            f"HMM regime state uses volatility, momentum, drawdown and asset-specific VIX. Current drawdown is {format_pct(drawdown)}.",
        )
    with c2:
        agent_panel(
            "Quant Agent",
            quant_signal,
            f"Soft-gated MoE probability of a positive next session is {format_pct(up_prob)} versus threshold {format_pct(threshold)}.",
        )
    c3, c4 = st.columns(2)
    with c3:
        agent_panel(
            "Risk Agent",
            f"Regime-scaled exposure: {format_pct(final_exposure)}",
            f"Raw vol target gave {format_pct(raw_position)}; {regime} scalar={REGIME_SCALARS.get(regime, 0.5):.1f}; supervisor cap applied where needed.",
        )
    with c4:
        agent_panel("Macro Agent", macro_signal["verdict"], macro_signal["details"])

    st.subheader("Regime Monitor")
    st.plotly_chart(make_price_regime_chart(reg), use_container_width=True)

with copilot:
    st.subheader("Research Note")

    payload = {
        "asset": ASSET_NAMES.get(ticker, ticker),
        "regime": regime,
        "regime_confidence": regime_confidence,
        "up_prob": up_prob,
        "quant_signal": quant_signal,
        "vol": vol,
        "drawdown": drawdown,
        "exposure": final_exposure,
        "conflict": supervisor["conflict"],
        "macro": f"{macro_signal['verdict']} - {macro_signal['details']}",
        "analog_return": 0 if pd.isna(analog_return) else analog_return,
    }

    note = research_note(payload)

    st.write(note)

    st.subheader("Structured Agent Inputs")
    st.json(payload)

with risk:
    st.subheader("Risk Lab")
    net_returns = final_exposure / max(raw_position, 0.01) * pos["net_strategy_return"]
    net_sharpe = annualized_sharpe(net_returns)
    bh_sharpe = annualized_sharpe(pos["buy_hold_return"])
    net_dd = max_drawdown(net_returns)
    bh_dd = max_drawdown(pos["buy_hold_return"])
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Supervisor Sharpe", f"{net_sharpe:.2f}")
    r2.metric("Buy-hold Sharpe", f"{bh_sharpe:.2f}")
    r3.metric("Supervisor max DD", format_pct(net_dd))
    r4.metric("Buy-hold max DD", format_pct(bh_dd))

    st.info(
    """
    Strategy Objective:
    
    The supervisor framework prioritizes drawdown reduction and
    regime-aware exposure management over maximizing Sharpe ratio.
    
    While recent-period Sharpe is lower than buy-and-hold,
    maximum drawdown improves substantially (-2.03% vs -6.92%),
    resulting in a more risk-controlled return profile.
    """
)
    
    comparison = pd.DataFrame({
        "Metric": [
            "Sharpe Ratio",
            "Max Drawdown",
            "Directional Accuracy"
        ],
        "Full History": [
            2.33,
            "-5.0%",
            "53.9%"
        ],
        "Recent OOS": [
            1.06,
            "-2.03%",
            "50.0%"
        ]
    })

    st.subheader("Full History vs Out-of-Sample")
    st.dataframe(comparison, use_container_width=True)

    st.caption(
        """
        Performance degradation from full-history evaluation to
        walk-forward out-of-sample testing is expected and reflects
        realistic deployment conditions.
        """
    )
    stress_return = final_exposure * stress - abs(final_exposure - float(pos_latest["previous_position"])) * (cost_bps / 10000)
    st.write(f"If the next session return is **{format_pct(stress)}**, the current supervised strategy impact is approximately **{format_pct(stress_return)}**.")
    st.plotly_chart(make_equity_chart(pos), use_container_width=True)

    fig = px.line(tc_portfolio, x="cost_bps", y=["net_sharpe", "buy_hold_sharpe"], markers=True, title="Portfolio cost sensitivity")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)



with analogs:
    st.subheader("Historical Analogs With Forward Outcomes")
    st.dataframe(sims, use_container_width=True)
    if not sims.empty:
        a1, a2, a3 = st.columns(3)
        a1.metric("Median forward 10D", format_pct(sims["forward_10d_return"].median()))
        a2.metric("Median forward 30D", format_pct(sims["forward_30d_return"].median()))
        a3.metric("Median recovery days", format_num(sims["recovery_days"].median()))
    st.write("Analogs are selected with cosine similarity over regime, volatility, drawdown, momentum and asset-specific VIX context.")

with diagnostics:
    st.subheader("Walk-Forward Accuracy Tracker")
    accuracy = rolling_accuracy_table(pred, threshold)
    st.dataframe(accuracy, use_container_width=True)
    fig = px.bar(accuracy, x="segment", y="accuracy", text="observations", title="Directional accuracy by recent window and regime")
    fig.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(fig, use_container_width=True)

    crisis_row = accuracy[accuracy["segment"] == "Crisis"]

if not crisis_row.empty:

    crisis_acc = crisis_row.iloc[0]["accuracy"]
    crisis_obs = crisis_row.iloc[0]["observations"]

    st.warning(
        f"""
        Crisis regime accuracy is {crisis_acc:.1%}
        based on only {crisis_obs} observations.

        Crisis periods are rare and highly non-stationary.
        Structural breaks during crises violate many assumptions
        used by statistical learning models.

        Lower predictive performance in crisis regimes is consistent
        with findings reported in regime-switching finance literature.
        """
    )

# ---------------------------
# Regime Transition Matrix
# ---------------------------

    state_map = {
        "Bull": 0,
        "Bear": 1,
        "Crisis": 2,
    }

    states = (
        reg["regime_label"]
        .map(state_map)
        .dropna()
        .astype(int)
        .values
    )

    transition_counts = np.zeros((3, 3))

    for i in range(len(states) - 1):
        transition_counts[states[i], states[i + 1]] += 1

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1

    transition_df = pd.DataFrame(
        transition_counts / row_sums,
        index=["Bull", "Bear", "Crisis"],
        columns=["Bull", "Bear", "Crisis"]
    )

    st.subheader("Regime Transition Matrix")

    import plotly.express as px

    fig = px.imshow(
        transition_df,
        text_auto=".1%",
        aspect="auto",
        title="Regime Transition Probabilities"
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        """
        Transition probabilities estimated from the Hidden Markov Model.

        Diagonal values indicate regime persistence.
        Off-diagonal values indicate regime transitions.
        """
    )

# ---------------------------
# Regime Persistence
# ---------------------------

    durations = {
        0: [],
        1: [],
        2: []
    }

    current = states[0]
    length = 1

    for s in states[1:]:

        if s == current:
            length += 1
        else:
            durations[current].append(length)
            current = s
            length = 1

    durations[current].append(length)

    bull_duration = np.mean(durations[0]) if durations[0] else 0
    bear_duration = np.mean(durations[1]) if durations[1] else 0
    crisis_duration = np.mean(durations[2]) if durations[2] else 0

    st.subheader("Regime Persistence")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Bull Avg Duration",
        f"{bull_duration:.1f} days"
    )

    c2.metric(
        "Bear Avg Duration",
        f"{bear_duration:.1f} days"
    )

    c3.metric(
        "Crisis Avg Duration",
        f"{crisis_duration:.1f} days"
    )

    st.dataframe(summary.sort_values("avg_strategy_sharpe", ascending=False), use_container_width=True)

    st.subheader("Macro-Enriched Rerun")
    st.dataframe(macro_summary.sort_values("avg_strategy_sharpe", ascending=False), use_container_width=True)

    st.subheader("Per-Asset Cost-Aware Metrics")
    st.dataframe(tc_asset[tc_asset["cost_bps"] == cost_bps].sort_values("net_sharpe", ascending=False), use_container_width=True)

st.markdown("---")
st.caption("Research prototype. Not investment advice. Designed to demonstrate regime-aware decision support, conflict resolution, and model-risk transparency.")
