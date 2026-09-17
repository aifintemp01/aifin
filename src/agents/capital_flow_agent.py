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


class CapitalFlowSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


INDIA_SECTORS = {
    "IT":         "NIFTYIT:NSE",
    "Financials": "BANKNIFTY:NSE",
    "FMCG":       "NIFTYFMCG:NSE",
    "Pharma":     "NIFTYPHARMA:NSE",
    "Energy":     "NIFTYENERGY:NSE",
}

US_SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy":     "XLE",
    "ConsDisc":   "XLY",
}

INDIA_SAFE_HAVEN = "GOLDBEES:NSE"
US_SAFE_HAVEN = "GLD"


def _detect_market(tickers: list[str]) -> dict:
    indian_tickers = [t for t in tickers if ":BSE" in t or ":NSE" in t]
    if indian_tickers:
        return {
            "market": "India",
            "sectors": INDIA_SECTORS,
            "benchmark": "NIFTY:NSE",
            "safe_haven": INDIA_SAFE_HAVEN,
            "safe_haven_label": "Gold ETF (GOLDBEES)",
            "benchmark_label": "Nifty 50",
        }
    return {
        "market": "US",
        "sectors": US_SECTORS,
        "benchmark": "SPY",
        "safe_haven": US_SAFE_HAVEN,
        "safe_haven_label": "Gold (GLD)",
        "benchmark_label": "S&P 500 (SPY)",
    }


def capital_flow_agent(
    state: AgentState,
    agent_id: str = "capital_flow_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes capital flow patterns using sector rotation and risk appetite proxies.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    market_config = _detect_market(tickers)
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=60)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    capital_flow_analysis: dict[str, dict] = {}
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

    progress.update_status(agent_id, None, f"Fetching {market_config['benchmark_label']} benchmark")
    benchmark_prices = get_prices(market_config["benchmark"], start_date, end_date, api_key=api_key)

    progress.update_status(agent_id, None, f"Fetching {market_config['safe_haven_label']}")
    safe_haven_prices = get_prices(market_config["safe_haven"], start_date, end_date, api_key=api_key)

    sector_price_data: dict[str, list] = {}
    for sector_name, sector_symbol in market_config["sectors"].items():
        progress.update_status(agent_id, None, f"Fetching {sector_name} sector data")
        prices = get_prices(sector_symbol, start_date, end_date, api_key=api_key)
        if prices:
            sector_price_data[sector_name] = sorted(
                prices, key=lambda p: p.time if hasattr(p, "time") else p["time"]
            )

    if benchmark_prices:
        benchmark_prices = sorted(
            benchmark_prices, key=lambda p: p.time if hasattr(p, "time") else p["time"]
        )
    if safe_haven_prices:
        safe_haven_prices = sorted(
            safe_haven_prices, key=lambda p: p.time if hasattr(p, "time") else p["time"]
        )

    progress.update_status(agent_id, None, "Analyzing cyclical vs defensive rotation")
    rotation_analysis = _analyze_sector_rotation(sector_price_data, market_config)

    progress.update_status(agent_id, None, "Analyzing risk-on vs risk-off asset flow")
    risk_appetite_analysis = _analyze_risk_appetite(benchmark_prices, safe_haven_prices, market_config)

    progress.update_status(agent_id, None, "Analyzing sector breadth")
    breadth_analysis = _analyze_sector_breadth(sector_price_data, market_config)

    progress.update_status(agent_id, None, "Analyzing benchmark momentum")
    momentum_analysis = _analyze_benchmark_momentum(benchmark_prices, market_config)

    total_score = (
        rotation_analysis["score"] * 0.35
        + risk_appetite_analysis["score"] * 0.30
        + breadth_analysis["score"] * 0.20
        + momentum_analysis["score"] * 0.15
    )
    max_score = 10

    if total_score >= 6.5:
        flow_signal = "bullish"
    elif total_score <= 3.5:
        flow_signal = "bearish"
    else:
        flow_signal = "neutral"

    flow_data = {
        "signal": flow_signal,
        "score": total_score,
        "max_score": max_score,
        "market": market_config["market"],
        "sectors_analyzed": list(sector_price_data.keys()),
        "sectors_missing": [s for s in market_config["sectors"] if s not in sector_price_data],
        "data_note": "FII/DII institutional flow data not available. Sector rotation used as proxy.",
        "sector_rotation": rotation_analysis,
        "risk_appetite": risk_appetite_analysis,
        "sector_breadth": breadth_analysis,
        "benchmark_momentum": momentum_analysis,
        "weights": {"sector_rotation": 0.35, "risk_appetite": 0.30, "sector_breadth": 0.20, "benchmark_momentum": 0.15},
    }

    for ticker in tickers:
        analysis_data[ticker] = flow_data

        progress.update_status(agent_id, ticker, "Generating capital flow analysis")
        flow_output = _generate_capital_flow_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        capital_flow_analysis[ticker] = {
            "signal": flow_output.signal,
            "confidence": flow_output.confidence,
            "reasoning": flow_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": flow_output.signal,
            "confidence": flow_output.confidence,
            "key_findings": [
                f"Flow score: {total_score:.1f}/{max_score}",
                f"Market: {market_config['market']}",
                f"Rotation: {rotation_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": flow_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=flow_output.reasoning)

    message = HumanMessage(content=json.dumps(capital_flow_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(capital_flow_analysis, "Capital Flow Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = capital_flow_analysis
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


def _compute_return(prices: list, lookback_days: int = 20) -> float | None:
    closes = _get_closes(prices)
    if len(closes) < lookback_days:
        return None
    base = closes[-lookback_days]
    if base == 0:
        return None
    return (closes[-1] - base) / base


CYCLICAL_SECTORS = {"IT", "Technology", "Financials", "ConsDisc", "Energy"}
DEFENSIVE_SECTORS = {"FMCG", "Healthcare", "Pharma"}


def _analyze_sector_rotation(sector_price_data: dict[str, list], market_config: dict) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    if not sector_price_data:
        return {"score": 5, "max_score": max_score, "details": "Sector rotation: no sector data available", "cyclical_avg_return": None, "defensive_avg_return": None}

    cyclical_returns, defensive_returns, sector_returns = [], [], {}

    for sector_name, prices in sector_price_data.items():
        ret = _compute_return(prices, 20)
        if ret is not None:
            sector_returns[sector_name] = ret
            if sector_name in CYCLICAL_SECTORS:
                cyclical_returns.append(ret)
            elif sector_name in DEFENSIVE_SECTORS:
                defensive_returns.append(ret)

    if sector_returns:
        best = max(sector_returns, key=sector_returns.get)
        worst = min(sector_returns, key=sector_returns.get)
        details.append(f"Best 20D: {best} ({sector_returns[best]:.1%}), Worst: {worst} ({sector_returns[worst]:.1%})")

    avg_cyclical = sum(cyclical_returns) / len(cyclical_returns) if cyclical_returns else None
    avg_defensive = sum(defensive_returns) / len(defensive_returns) if defensive_returns else None

    if avg_cyclical is not None and avg_defensive is not None:
        rotation_spread = avg_cyclical - avg_defensive
        if rotation_spread > 0.05:
            score += 4
            details.append(f"Strong cyclical rotation: {avg_cyclical:.1%} vs {avg_defensive:.1%} ({rotation_spread:.1%} spread)")
        elif rotation_spread > 0.02:
            score += 3
            details.append(f"Cyclical outperformance: {rotation_spread:.1%} spread")
        elif rotation_spread > -0.02:
            score += 2
            details.append(f"Rotation neutral: {rotation_spread:.1%} spread")
        elif rotation_spread > -0.05:
            score += 1
            details.append(f"Defensive tilt: {rotation_spread:.1%} spread")
        else:
            details.append(f"Strong defensive rotation: {rotation_spread:.1%} spread")
        abs_spread = abs(rotation_spread)
        if abs_spread > 0.08:
            score += 2
            details.append(f"High conviction rotation: {abs_spread:.1%} absolute spread")
        elif abs_spread > 0.04:
            score += 1
            details.append(f"Moderate conviction: {abs_spread:.1%} absolute spread")
    elif avg_cyclical is not None:
        score += 3 if avg_cyclical > 0.02 else (2 if avg_cyclical > 0 else 1)
        details.append(f"Cyclicals: {avg_cyclical:.1%} 20D avg")
    elif avg_defensive is not None:
        score += 3 if avg_defensive < 0 else 1
        details.append(f"Defensives: {avg_defensive:.1%}")
    else:
        details.append("Sector rotation: insufficient data")
        score = 5

    return {
        "score": min((score / max_score) * 10, 10),
        "max_score": max_score,
        "details": "; ".join(details),
        "cyclical_avg_return": round(avg_cyclical, 4) if avg_cyclical is not None else None,
        "defensive_avg_return": round(avg_defensive, 4) if avg_defensive is not None else None,
        "sector_returns": {k: round(v, 4) for k, v in sector_returns.items()},
    }


def _analyze_risk_appetite(benchmark_prices: list, safe_haven_prices: list, market_config: dict) -> dict:
    max_score = 5
    score = 0
    details: list[str] = []
    benchmark_label = market_config["benchmark_label"]
    safe_haven_label = market_config["safe_haven_label"]

    bench_return = _compute_return(benchmark_prices, 20) if benchmark_prices else None
    gold_return = _compute_return(safe_haven_prices, 20) if safe_haven_prices else None

    if bench_return is None and gold_return is None:
        return {"score": 5, "max_score": max_score, "details": "Risk appetite: insufficient data", "benchmark_return": None, "safe_haven_return": None}

    if bench_return is not None and gold_return is not None:
        spread = bench_return - gold_return
        if spread > 0.08:
            score += 5
            details.append(f"Strong risk-on: {benchmark_label} {bench_return:.1%} vs {safe_haven_label} {gold_return:.1%}")
        elif spread > 0.04:
            score += 4
            details.append(f"Risk-on: equity outperforming gold by {spread:.1%}")
        elif spread > 0:
            score += 3
            details.append(f"Mild risk-on: {spread:.1%} spread")
        elif spread > -0.04:
            score += 2
            details.append(f"Mild risk-off: gold {abs(spread):.1%} ahead")
        elif spread > -0.08:
            score += 1
            details.append(f"Risk-off: gold outperforming by {abs(spread):.1%}")
        else:
            details.append(f"Strong risk-off: {safe_haven_label} {gold_return:.1%} vs {benchmark_label} {bench_return:.1%}")
    elif bench_return is not None:
        score += 3 if bench_return > 0.03 else (2 if bench_return > 0 else 1)
        details.append(f"{benchmark_label} {bench_return:.1%} 20D")
    elif gold_return is not None:
        score += 1 if gold_return > 0.03 else 2
        details.append(f"{safe_haven_label} {gold_return:.1%}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "benchmark_return": round(bench_return, 4) if bench_return is not None else None,
        "safe_haven_return": round(gold_return, 4) if gold_return is not None else None,
        "spread": round(bench_return - gold_return, 4) if (bench_return is not None and gold_return is not None) else None,
    }


def _analyze_sector_breadth(sector_price_data: dict[str, list], market_config: dict) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    if not sector_price_data:
        return {"score": 5, "max_score": max_score, "details": "Sector breadth: no sector data", "positive_sectors": 0, "total_sectors": 0, "breadth_ratio": None}

    positive_sectors, total_sectors = 0, 0
    positive_names, negative_names = [], []

    for sector_name, prices in sector_price_data.items():
        ret = _compute_return(prices, 20)
        if ret is not None:
            total_sectors += 1
            if ret > 0:
                positive_sectors += 1
                positive_names.append(sector_name)
            else:
                negative_names.append(sector_name)

    if total_sectors == 0:
        return {"score": 5, "max_score": max_score, "details": "Sector breadth: no return data", "positive_sectors": 0, "total_sectors": 0, "breadth_ratio": None}

    breadth_ratio = positive_sectors / total_sectors
    if breadth_ratio >= 0.80:
        score += 4
        details.append(f"Strong breadth: {positive_sectors}/{total_sectors} sectors positive")
    elif breadth_ratio >= 0.60:
        score += 3
        details.append(f"Good breadth: {positive_sectors}/{total_sectors} sectors positive")
    elif breadth_ratio >= 0.40:
        score += 2
        details.append(f"Mixed breadth: {positive_sectors}/{total_sectors} sectors positive")
    elif breadth_ratio >= 0.20:
        score += 1
        details.append(f"Weak breadth: only {positive_sectors}/{total_sectors} sectors positive")
    else:
        details.append(f"Poor breadth: {positive_sectors}/{total_sectors} sectors positive")

    if positive_names:
        details.append(f"Positive: {', '.join(positive_names)}")
    if negative_names:
        details.append(f"Negative: {', '.join(negative_names)}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "positive_sectors": positive_sectors,
        "total_sectors": total_sectors,
        "breadth_ratio": round(breadth_ratio, 4),
    }


def _analyze_benchmark_momentum(benchmark_prices: list, market_config: dict) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []
    benchmark_label = market_config["benchmark_label"]

    if not benchmark_prices or len(benchmark_prices) < 5:
        return {"score": 5, "max_score": max_score, "details": f"{benchmark_label}: insufficient data", "return_5d": None, "return_20d": None, "return_40d": None}

    ret_5d = _compute_return(benchmark_prices, 5)
    ret_20d = _compute_return(benchmark_prices, 20)
    ret_40d = _compute_return(benchmark_prices, 40)

    if ret_20d is not None:
        if ret_20d > 0.05:
            score += 2
            details.append(f"{benchmark_label} strong 20D: +{ret_20d:.1%}")
        elif ret_20d > 0.02:
            score += 1
            details.append(f"{benchmark_label} positive 20D: +{ret_20d:.1%}")
        elif ret_20d > -0.02:
            details.append(f"{benchmark_label} flat 20D: {ret_20d:.1%}")
        else:
            details.append(f"{benchmark_label} negative 20D: {ret_20d:.1%}")

    if ret_5d is not None and ret_20d is not None:
        ann_5d = ret_5d * (252 / 5)
        ann_20d = ret_20d * (252 / 20)
        acceleration = ann_5d - ann_20d
        if acceleration > 0.20:
            score += 2
            details.append(f"Momentum accelerating: 5D ann {ann_5d:.0%} vs 20D {ann_20d:.0%}")
        elif acceleration > 0:
            score += 1
            details.append(f"Momentum stable: {acceleration:.0%} delta")
        else:
            details.append(f"Momentum decelerating: {acceleration:.0%} delta")

    if ret_5d is not None:
        details.append(f"5D: {ret_5d:.1%}")
    if ret_40d is not None:
        details.append(f"40D: {ret_40d:.1%}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "return_5d": round(ret_5d, 4) if ret_5d is not None else None,
        "return_20d": round(ret_20d, 4) if ret_20d is not None else None,
        "return_40d": round(ret_40d, 4) if ret_40d is not None else None,
    }


def _generate_capital_flow_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> CapitalFlowSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your capital flow assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined capital flow analyst. Your mandate:
            - Capital flows are the most honest signal of institutional intent
            - Sector rotation from defensives to cyclicals = clearest risk-on flow signal
            - Equity outperforming gold = institutional capital seeking risk
            - Broad sector participation confirms sustained flow
            - Note: Direct FII/DII data unavailable — sector rotation proxy used

            Reasoning should cover: sector rotation, equity vs gold, breadth, benchmark momentum.""",
        ),
        (
            "human",
            """Based on the following capital flow data, generate a flow signal for {ticker}:

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
        return CapitalFlowSignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=CapitalFlowSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )