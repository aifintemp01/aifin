from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import get_prices
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class MarketRegimeSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


# ---------------------------------------------------------------------------
# Dynamic VIX and benchmark selection based on ticker universe
# ---------------------------------------------------------------------------

def _detect_market(tickers: list[str]) -> dict:
    """
    Detect whether the ticker universe is Indian or US.
    Returns config dict with VIX symbol, benchmark symbol, and market label.
    """
    indian_tickers = [t for t in tickers if ":BSE" in t or ":NSE" in t]
    if indian_tickers:
        return {
            "market": "India",
            "vix_symbol": "INDIAVIX:NSE",
            "benchmark_symbol": "NIFTY:NSE",
            "vix_label": "India VIX",
            "benchmark_label": "Nifty 50",
        }
    return {
        "market": "US",
        "vix_symbol": "VIX",
        "benchmark_symbol": "SPY",
        "vix_label": "CBOE VIX",
        "benchmark_label": "S&P 500 (SPY)",
    }


def market_regime_agent(state: AgentState, agent_id: str = "market_regime_agent"):
    """
    Analyzes the macro market regime using VIX and benchmark price data.
    Dynamically selects India VIX + Nifty 50 for Indian ticker universes,
    or CBOE VIX + SPY for US ticker universes.

    Sub-analyses:
    - VIX level and trend: fear gauge — absolute level + 20D direction
    - VIX term structure proxy: short-term vs medium-term VIX comparison
    - Benchmark trend: price above/below 50D and 200D moving averages
    - Market breadth proxy: benchmark momentum vs volatility ratio

    Low VIX + uptrending benchmark = risk-on regime (bullish)
    High VIX + downtrending benchmark = risk-off regime (bearish)
    """
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    # Detect market context — India or US
    market_config = _detect_market(tickers)
    vix_symbol = market_config["vix_symbol"]
    benchmark_symbol = market_config["benchmark_symbol"]

    # Need 200 trading days for moving averages — use 300 calendar days buffer
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=300)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    market_regime_analysis: dict[str, dict] = {}

    # Market regime is a single analysis for the whole ticker universe
    # but we report it per ticker (each gets the same regime signal)
    progress.update_status(agent_id, None, f"Fetching {market_config['vix_label']} data")
    vix_prices = get_prices(vix_symbol, start_date, end_date, api_key=api_key)

    progress.update_status(agent_id, None, f"Fetching {market_config['benchmark_label']} data")
    benchmark_prices = get_prices(benchmark_symbol, start_date, end_date, api_key=api_key)

    # Sort oldest → newest
    if vix_prices:
        vix_prices = sorted(
            vix_prices,
            key=lambda p: p.time if hasattr(p, "time") else p["time"]
        )
    if benchmark_prices:
        benchmark_prices = sorted(
            benchmark_prices,
            key=lambda p: p.time if hasattr(p, "time") else p["time"]
        )

    # Run sub-analyses
    progress.update_status(agent_id, None, "Analyzing VIX level and trend")
    vix_analysis = _analyze_vix_level(vix_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing VIX term structure")
    vix_term_analysis = _analyze_vix_term_structure(vix_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing benchmark trend")
    benchmark_analysis = _analyze_benchmark_trend(benchmark_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing market breadth proxy")
    breadth_analysis = _analyze_market_breadth(benchmark_prices, vix_prices, market_config)

    # Aggregate score
    # VIX level is the most direct fear signal; benchmark trend confirms direction
    total_score = (
        vix_analysis["score"] * 0.35          # fear gauge level = primary regime signal
        + benchmark_analysis["score"] * 0.35  # price trend = regime confirmation
        + breadth_analysis["score"] * 0.20    # momentum vs vol = risk appetite
        + vix_term_analysis["score"] * 0.10   # term structure = forward fear
    )
    max_score = 10

    if total_score >= 6.5:
        regime_signal = "bullish"
    elif total_score <= 3.5:
        regime_signal = "bearish"
    else:
        regime_signal = "neutral"

    # Build regime data dict — same for all tickers in this run
    regime_data = {
        "signal": regime_signal,
        "score": total_score,
        "max_score": max_score,
        "market": market_config["market"],
        "vix_symbol": vix_symbol,
        "benchmark_symbol": benchmark_symbol,
        "vix_analysis": vix_analysis,
        "vix_term_structure": vix_term_analysis,
        "benchmark_trend": benchmark_analysis,
        "market_breadth": breadth_analysis,
        "weights": {
            "vix_level": 0.35,
            "benchmark_trend": 0.35,
            "market_breadth": 0.20,
            "vix_term_structure": 0.10,
        },
    }

    # Apply regime signal to each ticker
    for ticker in tickers:
        analysis_data[ticker] = regime_data

        progress.update_status(agent_id, ticker, "Generating market regime analysis")
        regime_output = _generate_market_regime_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        market_regime_analysis[ticker] = {
            "signal": regime_output.signal,
            "confidence": regime_output.confidence,
            "reasoning": regime_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=regime_output.reasoning)

    # Return to graph
    message = HumanMessage(content=json.dumps(market_regime_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(market_regime_analysis, "Market Regime Agent")

    state["data"]["analyst_signals"][agent_id] = market_regime_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub-analysis helpers
###############################################################################

def _safe_float(val) -> float | None:
    """Safely convert a value to float."""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_field(price_obj, field: str):
    """Get a field from a price object whether dict or Pydantic model."""
    if hasattr(price_obj, field):
        return getattr(price_obj, field)
    elif isinstance(price_obj, dict):
        return price_obj.get(field)
    return None


def _get_closes(prices: list) -> list[float]:
    """Extract close prices as floats."""
    closes = []
    for p in prices:
        c = _safe_float(_get_field(p, "close"))
        if c is not None:
            closes.append(c)
    return closes


def _moving_average(values: list[float], window: int) -> float | None:
    """Compute simple moving average over last `window` values."""
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


# ----- VIX Level and Trend --------------------------------------------------

def _analyze_vix_level(vix_prices: list, market_config: dict) -> dict:
    """
    Assess VIX level and 20-day trend.
    VIX thresholds differ slightly between India VIX and CBOE VIX:
    - India VIX typically runs higher (15–35 normal range vs 10–25 for CBOE)
    - Both use the same directional logic

    Low VIX = complacency / risk-on = bullish
    High VIX = fear / risk-off = bearish
    Rising VIX = regime deteriorating
    Falling VIX = regime improving
    """
    max_score = 6  # 4pts level + 2pts trend
    score = 0
    details: list[str] = []

    is_india = market_config["market"] == "India"
    vix_label = market_config["vix_label"]

    if not vix_prices or len(vix_prices) < 5:
        return {
            "score": 5,  # neutral default
            "max_score": max_score,
            "details": f"{vix_label}: insufficient data — defaulting to neutral",
            "latest_vix": None,
            "vix_20d_avg": None,
        }

    closes = _get_closes(vix_prices)
    if not closes:
        return {
            "score": 5,
            "max_score": max_score,
            "details": f"{vix_label}: no close data",
            "latest_vix": None,
            "vix_20d_avg": None,
        }

    latest_vix = closes[-1]
    vix_20d_avg = _moving_average(closes, 20) or latest_vix

    # VIX level scoring — India VIX runs ~1.5x higher than CBOE VIX
    if is_india:
        # India VIX thresholds
        if latest_vix < 14:
            score += 4
            details.append(f"{vix_label} very low: {latest_vix:.1f} — maximum risk-on regime")
        elif latest_vix < 18:
            score += 3
            details.append(f"{vix_label} low: {latest_vix:.1f} — risk-on regime")
        elif latest_vix < 24:
            score += 2
            details.append(f"{vix_label} moderate: {latest_vix:.1f} — neutral regime")
        elif latest_vix < 30:
            score += 1
            details.append(f"{vix_label} elevated: {latest_vix:.1f} — risk-off caution")
        else:
            details.append(f"{vix_label} high: {latest_vix:.1f} — fear regime, risk-off")
    else:
        # CBOE VIX thresholds
        if latest_vix < 13:
            score += 4
            details.append(f"{vix_label} very low: {latest_vix:.1f} — maximum risk-on regime")
        elif latest_vix < 18:
            score += 3
            details.append(f"{vix_label} low: {latest_vix:.1f} — risk-on regime")
        elif latest_vix < 25:
            score += 2
            details.append(f"{vix_label} moderate: {latest_vix:.1f} — neutral regime")
        elif latest_vix < 35:
            score += 1
            details.append(f"{vix_label} elevated: {latest_vix:.1f} — risk-off caution")
        else:
            details.append(f"{vix_label} high: {latest_vix:.1f} — fear regime, risk-off")

    # VIX trend: latest vs 20D average
    if vix_20d_avg > 0:
        vix_trend = (latest_vix - vix_20d_avg) / vix_20d_avg
        if vix_trend < -0.10:
            score += 2
            details.append(f"{vix_label} falling: {vix_trend:.1%} below 20D avg — fear receding")
        elif vix_trend < 0:
            score += 1
            details.append(f"{vix_label} slightly falling: {vix_trend:.1%} below 20D avg")
        elif vix_trend < 0.10:
            details.append(f"{vix_label} stable: {vix_trend:.1%} vs 20D avg")
        else:
            details.append(f"{vix_label} rising: {vix_trend:.1%} above 20D avg — fear increasing")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "latest_vix": round(latest_vix, 2),
        "vix_20d_avg": round(vix_20d_avg, 2),
    }


# ----- VIX Term Structure ---------------------------------------------------

def _analyze_vix_term_structure(vix_prices: list, market_config: dict) -> dict:
    """
    Proxy for VIX term structure using short-term vs medium-term VIX average.
    True term structure requires VIX futures (VX1, VX2) — unavailable here.
    Proxy: 5D avg VIX vs 30D avg VIX.
    Short-term < long-term (contango) = calm short-term = bullish
    Short-term > long-term (backwardation) = near-term fear spike = bearish
    """
    max_score = 3
    score = 0
    details: list[str] = []

    vix_label = market_config["vix_label"]

    if not vix_prices or len(vix_prices) < 30:
        return {
            "score": 5,  # neutral
            "max_score": max_score,
            "details": f"{vix_label} term structure: insufficient history — defaulting to neutral",
            "short_term_vix": None,
            "medium_term_vix": None,
        }

    closes = _get_closes(vix_prices)
    short_term = _moving_average(closes, 5)
    medium_term = _moving_average(closes, 30)

    if short_term is None or medium_term is None:
        return {
            "score": 5,
            "max_score": max_score,
            "details": f"{vix_label} term structure: insufficient data",
            "short_term_vix": None,
            "medium_term_vix": None,
        }

    ratio = short_term / medium_term if medium_term > 0 else 1.0

    if ratio < 0.90:
        score += 3
        details.append(
            f"{vix_label} contango: 5D avg {short_term:.1f} vs 30D avg {medium_term:.1f} "
            f"— near-term fear well below trend, bullish structure"
        )
    elif ratio < 1.0:
        score += 2
        details.append(
            f"{vix_label} mild contango: 5D {short_term:.1f} vs 30D {medium_term:.1f} — calm near-term"
        )
    elif ratio < 1.10:
        score += 1
        details.append(
            f"{vix_label} flat: 5D {short_term:.1f} vs 30D {medium_term:.1f} — neutral structure"
        )
    else:
        details.append(
            f"{vix_label} backwardation: 5D {short_term:.1f} vs 30D {medium_term:.1f} "
            f"— near-term fear spike, bearish structure"
        )

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "short_term_vix": round(short_term, 2),
        "medium_term_vix": round(medium_term, 2),
        "structure_ratio": round(ratio, 4),
    }


# ----- Benchmark Trend (50D and 200D MAs) -----------------------------------

def _analyze_benchmark_trend(benchmark_prices: list, market_config: dict) -> dict:
    """
    Assess benchmark trend via 50D and 200D moving averages.
    Price above both MAs = strong uptrend = bullish regime.
    Price below both MAs = downtrend = bearish regime.
    Golden cross (50D > 200D) = long-term bullish structure.
    Death cross (50D < 200D) = long-term bearish structure.
    """
    max_score = 6  # 3pts MA position + 3pts MA cross
    score = 0
    details: list[str] = []

    benchmark_label = market_config["benchmark_label"]

    if not benchmark_prices or len(benchmark_prices) < 50:
        return {
            "score": 5,  # neutral
            "max_score": max_score,
            "details": f"{benchmark_label}: insufficient history for MA analysis",
            "latest_price": None,
            "ma_50": None,
            "ma_200": None,
        }

    closes = _get_closes(benchmark_prices)
    if len(closes) < 50:
        return {
            "score": 5,
            "max_score": max_score,
            "details": f"{benchmark_label}: insufficient close data",
            "latest_price": None,
            "ma_50": None,
            "ma_200": None,
        }

    latest_price = closes[-1]
    ma_50 = _moving_average(closes, 50)
    ma_200 = _moving_average(closes, 200) if len(closes) >= 200 else None

    # Price vs MA position: 0–3 pts
    if ma_200 is not None:
        if latest_price > ma_50 and latest_price > ma_200:
            score += 3
            details.append(
                f"{benchmark_label} above both MAs: price {latest_price:.1f} > "
                f"50D {ma_50:.1f} > 200D {ma_200:.1f} — strong uptrend"
            )
        elif latest_price > ma_50:
            score += 2
            details.append(
                f"{benchmark_label} above 50D ({ma_50:.1f}) but below 200D ({ma_200:.1f}) — recovering"
            )
        elif latest_price > ma_200:
            score += 1
            details.append(
                f"{benchmark_label} below 50D ({ma_50:.1f}) but above 200D ({ma_200:.1f}) — pullback in uptrend"
            )
        else:
            details.append(
                f"{benchmark_label} below both MAs: price {latest_price:.1f} < "
                f"50D {ma_50:.1f}, 200D {ma_200:.1f} — downtrend"
            )

        # Golden/Death cross: 0–3 pts
        if ma_50 > ma_200:
            cross_pct = (ma_50 - ma_200) / ma_200
            if cross_pct > 0.05:
                score += 3
                details.append(f"Strong golden cross: 50D {cross_pct:.1%} above 200D — long-term bullish")
            elif cross_pct > 0.01:
                score += 2
                details.append(f"Golden cross: 50D {cross_pct:.1%} above 200D — bullish structure")
            else:
                score += 1
                details.append(f"Weak golden cross: 50D marginally above 200D — early bull")
        else:
            cross_pct = (ma_200 - ma_50) / ma_200
            details.append(f"Death cross: 50D {cross_pct:.1%} below 200D — bearish structure")

    else:
        # Only 50D available
        if latest_price > ma_50:
            score += 3
            details.append(f"{benchmark_label} above 50D MA ({ma_50:.1f}) — uptrend")
        else:
            details.append(f"{benchmark_label} below 50D MA ({ma_50:.1f}) — downtrend")
        details.append("200D MA unavailable — insufficient history")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "latest_price": round(latest_price, 2),
        "ma_50": round(ma_50, 2) if ma_50 else None,
        "ma_200": round(ma_200, 2) if ma_200 else None,
    }


# ----- Market Breadth Proxy (Momentum vs Volatility) -----------------------

def _analyze_market_breadth(
    benchmark_prices: list,
    vix_prices: list,
    market_config: dict
) -> dict:
    """
    Proxy for market breadth via benchmark momentum-to-volatility ratio.
    Formula: 20D benchmark return / VIX level.
    High positive momentum with low VIX = strong breadth = bullish.
    Negative momentum with high VIX = collapsing breadth = bearish.

    True breadth requires advance/decline data — not available via price API.
    This ratio captures the same risk-appetite concept.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    benchmark_label = market_config["benchmark_label"]
    vix_label = market_config["vix_label"]

    if not benchmark_prices or len(benchmark_prices) < 20:
        return {
            "score": 5,
            "max_score": max_score,
            "details": "Market breadth proxy: insufficient benchmark data",
            "momentum_20d": None,
            "momentum_vix_ratio": None,
        }

    bench_closes = _get_closes(benchmark_prices)
    vix_closes = _get_closes(vix_prices) if vix_prices else []

    if len(bench_closes) < 20:
        return {
            "score": 5,
            "max_score": max_score,
            "details": "Market breadth proxy: insufficient close data",
            "momentum_20d": None,
            "momentum_vix_ratio": None,
        }

    # 20D benchmark return
    momentum_20d = (bench_closes[-1] - bench_closes[-20]) / bench_closes[-20]

    # Momentum / VIX ratio (normalize VIX to 0–1 scale by dividing by 50)
    latest_vix = vix_closes[-1] if vix_closes else None
    mv_ratio = None

    if latest_vix and latest_vix > 0:
        mv_ratio = momentum_20d / (latest_vix / 20)  # normalize VIX around 20

        if mv_ratio > 0.10:
            score += 4
            details.append(
                f"Strong risk appetite: {benchmark_label} +{momentum_20d:.1%} 20D return "
                f"vs {vix_label} {latest_vix:.1f} — momentum/fear ratio {mv_ratio:.2f}"
            )
        elif mv_ratio > 0.02:
            score += 3
            details.append(
                f"Positive risk appetite: {momentum_20d:.1%} 20D return, {vix_label} {latest_vix:.1f}"
            )
        elif mv_ratio > -0.02:
            score += 2
            details.append(
                f"Neutral risk appetite: {momentum_20d:.1%} 20D return, {vix_label} {latest_vix:.1f}"
            )
        elif mv_ratio > -0.10:
            score += 1
            details.append(
                f"Negative risk appetite: {momentum_20d:.1%} 20D return, {vix_label} {latest_vix:.1f}"
            )
        else:
            details.append(
                f"Risk-off regime: {momentum_20d:.1%} 20D return, {vix_label} {latest_vix:.1f} — breadth collapsing"
            )
    else:
        # Score on momentum alone if VIX unavailable
        if momentum_20d > 0.05:
            score += 3
            details.append(f"Strong 20D momentum: {momentum_20d:.1%} — {benchmark_label}")
        elif momentum_20d > 0:
            score += 2
            details.append(f"Positive 20D momentum: {momentum_20d:.1%}")
        elif momentum_20d > -0.05:
            score += 1
            details.append(f"Slight 20D decline: {momentum_20d:.1%}")
        else:
            details.append(f"Negative 20D momentum: {momentum_20d:.1%} — breadth weak")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "momentum_20d": round(momentum_20d, 4),
        "momentum_vix_ratio": round(mv_ratio, 4) if mv_ratio is not None else None,
        "latest_vix": round(latest_vix, 2) if latest_vix else None,
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_market_regime_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MarketRegimeSignal:
    """Generate a market regime signal grounded in VIX and benchmark data."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined market regime analyst. Your mandate:
                - Market regime is the single most important overlay for all stock-level analysis
                - VIX level is the most direct fear gauge: below 18 is risk-on, above 25 is risk-off
                - Benchmark MA crossovers (golden cross / death cross) define the long-term regime
                - A rising VIX combined with a benchmark below both MAs = bear regime — do not fight it
                - A falling VIX with the benchmark in a golden cross = bull regime — lean into it
                - The regime signal should inform position sizing, not stock selection

                When providing your reasoning, be specific by:
                1. Leading with the VIX level — what does it signal about current fear?
                2. Assessing the VIX term structure — is near-term fear elevated vs medium-term?
                3. Evaluating the benchmark trend — are we above or below key moving averages?
                4. Commenting on market breadth proxy — is momentum confirming or diverging?
                5. Concluding with a clear regime verdict: bull, bear, or transitional
                """,
            ),
            (
                "human",
                """Based on the following market regime data, generate a regime signal for {ticker}:

                Analysis Data:
                {analysis_data}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke(
        {"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker}
    )

    def create_default_market_regime_signal():
        return MarketRegimeSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=MarketRegimeSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_market_regime_signal,
    )
