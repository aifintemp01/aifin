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
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def _detect_market(tickers: list[str]) -> dict:
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


def market_regime_agent(
    state: AgentState,
    agent_id: str = "market_regime_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes the macro market regime using VIX and benchmark price data.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    market_config = _detect_market(tickers)
    vix_symbol = market_config["vix_symbol"]
    benchmark_symbol = market_config["benchmark_symbol"]

    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=300)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    market_regime_analysis: dict[str, dict] = {}
    layer_context_updates: dict = {}

    # ── Gather upstream context (Layer 2+) — same for all tickers ────────────
    upstream_context: dict = {}
    if layer > 1 and upstream_agent_ids:
        lc = state.get("layer_context", {})
        for ticker in tickers:
            for upstream_id in upstream_agent_ids:
                ctx_key = f"{upstream_id}:{ticker}"
                if ctx_key in lc:
                    upstream_context[upstream_id] = lc[ctx_key]
        if upstream_context:
            print(f"[{agent_id}] Layer {layer} — injecting context from: {list(upstream_context.keys())}")

    progress.update_status(agent_id, None, f"Fetching {market_config['vix_label']} data")
    vix_prices = get_prices(vix_symbol, start_date, end_date, api_key=api_key)

    progress.update_status(agent_id, None, f"Fetching {market_config['benchmark_label']} data")
    benchmark_prices = get_prices(benchmark_symbol, start_date, end_date, api_key=api_key)

    if vix_prices:
        vix_prices = sorted(vix_prices, key=lambda p: p.time if hasattr(p, "time") else p["time"])
    if benchmark_prices:
        benchmark_prices = sorted(benchmark_prices, key=lambda p: p.time if hasattr(p, "time") else p["time"])

    progress.update_status(agent_id, None, "Analyzing VIX level and trend")
    vix_analysis = _analyze_vix_level(vix_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing VIX term structure")
    vix_term_analysis = _analyze_vix_term_structure(vix_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing benchmark trend")
    benchmark_analysis = _analyze_benchmark_trend(benchmark_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing market breadth proxy")
    breadth_analysis = _analyze_market_breadth(benchmark_prices, vix_prices, market_config)

    total_score = (
        vix_analysis["score"] * 0.35
        + benchmark_analysis["score"] * 0.35
        + breadth_analysis["score"] * 0.20
        + vix_term_analysis["score"] * 0.10
    )
    max_score = 10

    if total_score >= 6.5:
        regime_signal = "bullish"
    elif total_score <= 3.5:
        regime_signal = "bearish"
    else:
        regime_signal = "neutral"

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
        "weights": {"vix_level": 0.35, "benchmark_trend": 0.35, "market_breadth": 0.20, "vix_term_structure": 0.10},
    }

    for ticker in tickers:
        analysis_data[ticker] = regime_data

        progress.update_status(agent_id, ticker, "Generating market regime analysis")
        regime_output = _generate_market_regime_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        market_regime_analysis[ticker] = {
            "signal": regime_output.signal,
            "confidence": regime_output.confidence,
            "reasoning": regime_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": regime_output.signal,
            "confidence": regime_output.confidence,
            "key_findings": [
                f"Regime score: {total_score:.1f}/{max_score}",
                f"Market: {market_config['market']}",
                f"VIX: {vix_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": regime_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=regime_output.reasoning)

    message = HumanMessage(content=json.dumps(market_regime_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(market_regime_analysis, "Market Regime Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = market_regime_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Sub-analysis helpers (unchanged from original) ────────────────────────────

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _get_field(price_obj, field: str):
    if hasattr(price_obj, field):
        return getattr(price_obj, field)
    elif isinstance(price_obj, dict):
        return price_obj.get(field)
    return None


def _get_closes(prices: list) -> list[float]:
    closes = []
    for p in prices:
        c = _safe_float(_get_field(p, "close"))
        if c is not None:
            closes.append(c)
    return closes


def _moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _analyze_vix_level(vix_prices: list, market_config: dict) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []
    is_india = market_config["market"] == "India"
    vix_label = market_config["vix_label"]

    if not vix_prices or len(vix_prices) < 5:
        return {"score": 5, "max_score": max_score, "details": f"{vix_label}: insufficient data", "latest_vix": None, "vix_20d_avg": None}

    closes = _get_closes(vix_prices)
    if not closes:
        return {"score": 5, "max_score": max_score, "details": f"{vix_label}: no close data", "latest_vix": None, "vix_20d_avg": None}

    latest_vix = closes[-1]
    vix_20d_avg = _moving_average(closes, 20) or latest_vix

    if is_india:
        if latest_vix < 14: score += 4; details.append(f"{vix_label} very low: {latest_vix:.1f} — maximum risk-on")
        elif latest_vix < 18: score += 3; details.append(f"{vix_label} low: {latest_vix:.1f} — risk-on")
        elif latest_vix < 24: score += 2; details.append(f"{vix_label} moderate: {latest_vix:.1f}")
        elif latest_vix < 30: score += 1; details.append(f"{vix_label} elevated: {latest_vix:.1f}")
        else: details.append(f"{vix_label} high: {latest_vix:.1f} — fear regime")
    else:
        if latest_vix < 13: score += 4; details.append(f"{vix_label} very low: {latest_vix:.1f} — maximum risk-on")
        elif latest_vix < 18: score += 3; details.append(f"{vix_label} low: {latest_vix:.1f} — risk-on")
        elif latest_vix < 25: score += 2; details.append(f"{vix_label} moderate: {latest_vix:.1f}")
        elif latest_vix < 35: score += 1; details.append(f"{vix_label} elevated: {latest_vix:.1f}")
        else: details.append(f"{vix_label} high: {latest_vix:.1f} — fear regime")

    if vix_20d_avg > 0:
        vix_trend = (latest_vix - vix_20d_avg) / vix_20d_avg
        if vix_trend < -0.10: score += 2; details.append(f"{vix_label} falling: {vix_trend:.1%} below 20D avg")
        elif vix_trend < 0: score += 1; details.append(f"{vix_label} slightly falling: {vix_trend:.1%}")
        elif vix_trend < 0.10: details.append(f"{vix_label} stable: {vix_trend:.1%}")
        else: details.append(f"{vix_label} rising: {vix_trend:.1%} above 20D avg")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details), "latest_vix": round(latest_vix, 2), "vix_20d_avg": round(vix_20d_avg, 2)}


def _analyze_vix_term_structure(vix_prices: list, market_config: dict) -> dict:
    max_score = 3
    score = 0
    details: list[str] = []
    vix_label = market_config["vix_label"]

    if not vix_prices or len(vix_prices) < 30:
        return {"score": 5, "max_score": max_score, "details": f"{vix_label} term structure: insufficient history", "short_term_vix": None, "medium_term_vix": None}

    closes = _get_closes(vix_prices)
    short_term = _moving_average(closes, 5)
    medium_term = _moving_average(closes, 30)

    if short_term is None or medium_term is None:
        return {"score": 5, "max_score": max_score, "details": f"{vix_label} term structure: insufficient data", "short_term_vix": None, "medium_term_vix": None}

    ratio = short_term / medium_term if medium_term > 0 else 1.0

    if ratio < 0.90: score += 3; details.append(f"{vix_label} contango: 5D {short_term:.1f} vs 30D {medium_term:.1f} — bullish structure")
    elif ratio < 1.0: score += 2; details.append(f"{vix_label} mild contango: 5D {short_term:.1f} vs 30D {medium_term:.1f}")
    elif ratio < 1.10: score += 1; details.append(f"{vix_label} flat: 5D {short_term:.1f} vs 30D {medium_term:.1f}")
    else: details.append(f"{vix_label} backwardation: 5D {short_term:.1f} vs 30D {medium_term:.1f} — bearish structure")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details), "short_term_vix": round(short_term, 2), "medium_term_vix": round(medium_term, 2), "structure_ratio": round(ratio, 4)}


def _analyze_benchmark_trend(benchmark_prices: list, market_config: dict) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []
    benchmark_label = market_config["benchmark_label"]

    if not benchmark_prices or len(benchmark_prices) < 50:
        return {"score": 5, "max_score": max_score, "details": f"{benchmark_label}: insufficient history", "latest_price": None, "ma_50": None, "ma_200": None}

    closes = _get_closes(benchmark_prices)
    if len(closes) < 50:
        return {"score": 5, "max_score": max_score, "details": f"{benchmark_label}: insufficient close data", "latest_price": None, "ma_50": None, "ma_200": None}

    latest_price = closes[-1]
    ma_50 = _moving_average(closes, 50)
    ma_200 = _moving_average(closes, 200) if len(closes) >= 200 else None

    if ma_200 is not None:
        if latest_price > ma_50 and latest_price > ma_200:
            score += 3; details.append(f"{benchmark_label} above both MAs: {latest_price:.1f} > 50D {ma_50:.1f} > 200D {ma_200:.1f}")
        elif latest_price > ma_50:
            score += 2; details.append(f"{benchmark_label} above 50D but below 200D — recovering")
        elif latest_price > ma_200:
            score += 1; details.append(f"{benchmark_label} below 50D but above 200D — pullback in uptrend")
        else:
            details.append(f"{benchmark_label} below both MAs — downtrend")

        if ma_50 > ma_200:
            cross_pct = (ma_50 - ma_200) / ma_200
            if cross_pct > 0.05: score += 3; details.append(f"Strong golden cross: 50D {cross_pct:.1%} above 200D")
            elif cross_pct > 0.01: score += 2; details.append(f"Golden cross: 50D {cross_pct:.1%} above 200D")
            else: score += 1; details.append("Weak golden cross: 50D marginally above 200D")
        else:
            cross_pct = (ma_200 - ma_50) / ma_200
            details.append(f"Death cross: 50D {cross_pct:.1%} below 200D")
    else:
        if latest_price > ma_50: score += 3; details.append(f"{benchmark_label} above 50D MA ({ma_50:.1f})")
        else: details.append(f"{benchmark_label} below 50D MA ({ma_50:.1f})")
        details.append("200D MA unavailable")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details), "latest_price": round(latest_price, 2), "ma_50": round(ma_50, 2) if ma_50 else None, "ma_200": round(ma_200, 2) if ma_200 else None}


def _analyze_market_breadth(benchmark_prices: list, vix_prices: list, market_config: dict) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []
    benchmark_label = market_config["benchmark_label"]
    vix_label = market_config["vix_label"]

    if not benchmark_prices or len(benchmark_prices) < 20:
        return {"score": 5, "max_score": max_score, "details": "Market breadth proxy: insufficient data", "momentum_20d": None, "momentum_vix_ratio": None}

    bench_closes = _get_closes(benchmark_prices)
    vix_closes = _get_closes(vix_prices) if vix_prices else []

    if len(bench_closes) < 20:
        return {"score": 5, "max_score": max_score, "details": "Market breadth proxy: insufficient close data", "momentum_20d": None, "momentum_vix_ratio": None}

    momentum_20d = (bench_closes[-1] - bench_closes[-20]) / bench_closes[-20]
    latest_vix = vix_closes[-1] if vix_closes else None
    mv_ratio = None

    if latest_vix and latest_vix > 0:
        mv_ratio = momentum_20d / (latest_vix / 20)
        if mv_ratio > 0.10: score += 4; details.append(f"Strong risk appetite: {benchmark_label} +{momentum_20d:.1%} vs {vix_label} {latest_vix:.1f}")
        elif mv_ratio > 0.02: score += 3; details.append(f"Positive risk appetite: {momentum_20d:.1%}, {vix_label} {latest_vix:.1f}")
        elif mv_ratio > -0.02: score += 2; details.append(f"Neutral: {momentum_20d:.1%}, {vix_label} {latest_vix:.1f}")
        elif mv_ratio > -0.10: score += 1; details.append(f"Negative risk appetite: {momentum_20d:.1%}, {vix_label} {latest_vix:.1f}")
        else: details.append(f"Risk-off: {momentum_20d:.1%}, {vix_label} {latest_vix:.1f}")
    else:
        if momentum_20d > 0.05: score += 3; details.append(f"Strong 20D momentum: {momentum_20d:.1%}")
        elif momentum_20d > 0: score += 2; details.append(f"Positive 20D momentum: {momentum_20d:.1%}")
        elif momentum_20d > -0.05: score += 1; details.append(f"Slight 20D decline: {momentum_20d:.1%}")
        else: details.append(f"Negative 20D momentum: {momentum_20d:.1%}")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details), "momentum_20d": round(momentum_20d, 4), "momentum_vix_ratio": round(mv_ratio, 4) if mv_ratio is not None else None, "latest_vix": round(latest_vix, 2) if latest_vix else None}


def _generate_market_regime_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> MarketRegimeSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your regime assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined market regime analyst. Your mandate:
            - VIX below 18 = risk-on; above 25 = risk-off
            - Benchmark MA crossovers define the long-term regime
            - Rising VIX + benchmark below both MAs = bear regime
            - Falling VIX + golden cross = bull regime

            Reasoning: VIX level → VIX term structure → benchmark trend → breadth → regime verdict.""",
        ),
        (
            "human",
            """Based on the following market regime data, generate a regime signal for {ticker}:

Analysis Data:
{analysis_data}
{upstream_context}
Return the trading signal in the following JSON format exactly:
{{
  "signal": "bullish" | "bearish" | "neutral",
  "confidence": float between 0 and 100,
  "reasoning": "string"
}}""",
        ),
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker,
        "upstream_context": upstream_section,
    })

    def create_default():
        return MarketRegimeSignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=MarketRegimeSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )