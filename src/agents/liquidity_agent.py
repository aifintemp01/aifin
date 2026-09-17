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


class LiquiditySignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def liquidity_agent(
    state: AgentState,
    agent_id: str = "liquidity_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using a comprehensive liquidity framework.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    liquidity_analysis: dict[str, dict] = {}
    layer_context_updates: dict = {}

    for ticker in tickers:
        # ── Gather upstream context (Layer 2+) ───────────────────────────────
        upstream_context: dict = {}
        if layer > 1 and upstream_agent_ids:
            lc = state.get("layer_context", {})
            for upstream_id in upstream_agent_ids:
                ctx_key = f"{upstream_id}:{ticker}"
                if ctx_key in lc:
                    upstream_context[upstream_id] = lc[ctx_key]
            if upstream_context:
                print(f"[{agent_id}] Layer {layer} — injecting context from: {list(upstream_context.keys())}")

        progress.update_status(agent_id, ticker, "Fetching price and volume data")
        prices = get_prices(ticker, start_date, end_date, api_key=api_key)

        if not prices or len(prices) < 5:
            progress.update_status(agent_id, ticker, "Insufficient price history — skipping")
            liquidity_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "Insufficient price history to compute liquidity metrics.",
            }
            layer_context_updates[f"{agent_id}:{ticker}"] = {
                "agent": agent_id, "ticker": ticker, "layer": layer,
                "signal": "neutral", "confidence": 0.0,
                "key_findings": ["Insufficient price history"],
                "data_summary": "No liquidity data available.",
            }
            continue

        sorted_prices = sorted(prices, key=lambda p: p.time if hasattr(p, "time") else p["time"])

        progress.update_status(agent_id, ticker, "Analyzing volume metrics")
        volume_analysis = _analyze_volume(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing traded value")
        traded_value_analysis = _analyze_traded_value(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing Amihud illiquidity")
        illiquidity_analysis = _analyze_amihud_illiquidity(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing impact cost proxy")
        impact_cost_analysis = _analyze_impact_cost(sorted_prices)

        total_score = (
            illiquidity_analysis["score"] * 0.35
            + traded_value_analysis["score"] * 0.30
            + volume_analysis["score"] * 0.20
            + impact_cost_analysis["score"] * 0.15
        )
        max_score = 10

        if total_score >= 6.5:
            signal = "bullish"
        elif total_score <= 3.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "volume_analysis": volume_analysis,
            "traded_value_analysis": traded_value_analysis,
            "illiquidity_analysis": illiquidity_analysis,
            "impact_cost_analysis": impact_cost_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating liquidity analysis")
        liquidity_output = _generate_liquidity_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        liquidity_analysis[ticker] = {
            "signal": liquidity_output.signal,
            "confidence": liquidity_output.confidence,
            "reasoning": liquidity_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": liquidity_output.signal,
            "confidence": liquidity_output.confidence,
            "key_findings": [
                f"Liquidity score: {total_score:.1f}/{max_score}",
                f"Amihud: {illiquidity_analysis.get('details', 'N/A')[:80]}",
                f"Traded value: {traded_value_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": liquidity_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=liquidity_output.reasoning)

    message = HumanMessage(content=json.dumps(liquidity_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(liquidity_analysis, "Liquidity Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = liquidity_analysis
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


def _analyze_volume(sorted_prices: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    volumes = [_safe_float(_get_field(p, "volume")) for p in sorted_prices]
    volumes = [v for v in volumes if v is not None and v > 0]

    if not volumes:
        return {"score": 0, "max_score": max_score, "details": "Volume data unavailable", "latest_volume": None, "avg_30d_volume": None}

    latest_volume = volumes[-1]
    avg_30d = sum(volumes[-30:]) / len(volumes[-30:]) if len(volumes) >= 5 else sum(volumes) / len(volumes)

    if avg_30d >= 5_000_000:
        score += 2
        details.append(f"High 30D avg volume: {avg_30d:,.0f} shares")
    elif avg_30d >= 500_000:
        score += 1
        details.append(f"Moderate 30D avg volume: {avg_30d:,.0f} shares")
    else:
        details.append(f"Low 30D avg volume: {avg_30d:,.0f} shares — liquidity risk")

    if avg_30d > 0:
        vol_ratio = latest_volume / avg_30d
        if vol_ratio >= 1.5:
            score += 2
            details.append(f"Volume surge: {vol_ratio:.1f}x 30D average")
        elif vol_ratio >= 0.8:
            score += 1
            details.append(f"Normal volume: {vol_ratio:.1f}x 30D average")
        else:
            details.append(f"Below-average volume: {vol_ratio:.1f}x 30D average")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "latest_volume": int(latest_volume),
        "avg_30d_volume": round(avg_30d, 0),
    }


def _analyze_traded_value(sorted_prices: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    traded_values = []
    for p in sorted_prices:
        vol = _safe_float(_get_field(p, "volume"))
        close = _safe_float(_get_field(p, "close"))
        if vol is not None and close is not None and vol > 0 and close > 0:
            traded_values.append(vol * close)

    if not traded_values:
        return {"score": 0, "max_score": max_score, "details": "Traded value data unavailable", "latest_traded_value": None, "avg_30d_traded_value": None}

    latest_tv = traded_values[-1]
    avg_30d_tv = sum(traded_values[-30:]) / len(traded_values[-30:]) if len(traded_values) >= 5 else sum(traded_values) / len(traded_values)

    if avg_30d_tv >= 50_000_000:
        score += 2
        details.append(f"Excellent 30D avg traded value: ${avg_30d_tv / 1e6:.1f}M — institutional grade")
    elif avg_30d_tv >= 10_000_000:
        score += 1
        details.append(f"Adequate 30D avg traded value: ${avg_30d_tv / 1e6:.1f}M")
    else:
        details.append(f"Low 30D avg traded value: ${avg_30d_tv / 1e6:.2f}M — below institutional threshold")

    if avg_30d_tv > 0:
        tv_ratio = latest_tv / avg_30d_tv
        if tv_ratio >= 1.5:
            score += 2
            details.append(f"Traded value surge: {tv_ratio:.1f}x 30D average")
        elif tv_ratio >= 0.7:
            score += 1
            details.append(f"Normal traded value: {tv_ratio:.1f}x 30D average")
        else:
            details.append(f"Declining traded value: {tv_ratio:.1f}x 30D average")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "latest_traded_value": round(latest_tv, 2),
        "avg_30d_traded_value": round(avg_30d_tv, 2),
    }


def _analyze_amihud_illiquidity(sorted_prices: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    ratios = []
    for i in range(1, len(sorted_prices)):
        prev_close = _safe_float(_get_field(sorted_prices[i - 1], "close"))
        curr_close = _safe_float(_get_field(sorted_prices[i], "close"))
        vol = _safe_float(_get_field(sorted_prices[i], "volume"))
        if prev_close and curr_close and vol and prev_close > 0 and vol > 0:
            daily_return = abs((curr_close - prev_close) / prev_close)
            traded_value = curr_close * vol
            if traded_value > 0:
                ratios.append(daily_return / traded_value)

    if not ratios:
        return {"score": 0, "max_score": max_score, "details": "Amihud illiquidity: insufficient data", "amihud_ratio": None}

    amihud = (sum(ratios) / len(ratios)) * 1e6

    if amihud < 0.01:
        score += 6
        details.append(f"Exceptional liquidity — Amihud: {amihud:.4f}")
    elif amihud < 0.05:
        score += 5
        details.append(f"High liquidity — Amihud: {amihud:.4f}")
    elif amihud < 0.20:
        score += 3
        details.append(f"Moderate liquidity — Amihud: {amihud:.4f}")
    elif amihud < 0.50:
        score += 1
        details.append(f"Low liquidity — Amihud: {amihud:.4f}")
    else:
        details.append(f"Illiquid — Amihud: {amihud:.4f} — significant market impact risk")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "amihud_ratio": round(amihud, 6),
    }


def _analyze_impact_cost(sorted_prices: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    spreads = []
    recent = sorted_prices[-30:] if len(sorted_prices) >= 30 else sorted_prices

    for p in recent:
        high = _safe_float(_get_field(p, "high"))
        low = _safe_float(_get_field(p, "low"))
        close = _safe_float(_get_field(p, "close"))
        if high and low and close and close > 0:
            spreads.append((high - low) / close)

    if not spreads:
        return {"score": 0, "max_score": max_score, "details": "Impact cost proxy: insufficient data", "avg_hl_spread_pct": None}

    avg_spread = sum(spreads) / len(spreads)

    if avg_spread < 0.01:
        score += 4
        details.append(f"Tight spread: {avg_spread:.2%} — minimal execution cost")
    elif avg_spread < 0.02:
        score += 3
        details.append(f"Narrow spread: {avg_spread:.2%} — low execution cost")
    elif avg_spread < 0.04:
        score += 2
        details.append(f"Moderate spread: {avg_spread:.2%}")
    elif avg_spread < 0.07:
        score += 1
        details.append(f"Wide spread: {avg_spread:.2%} — elevated execution cost")
    else:
        details.append(f"Very wide spread: {avg_spread:.2%} — high slippage risk")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_hl_spread_pct": round(avg_spread, 6),
    }


def _generate_liquidity_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> LiquiditySignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your liquidity assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined liquidity analyst. Your mandate:
            - The Amihud illiquidity ratio is the most academically robust measure; weight it heavily
            - Daily traded value above $10M is the minimum institutional threshold
            - Volume surge vs 30D average signals accumulation or distribution
            - Wide high-low spreads indicate hidden transaction costs
            - Illiquid stocks carry permanent execution risk

            Reasoning: Amihud ratio → traded value → volume trend → spread → verdict.""",
        ),
        (
            "human",
            """Based on the following data, generate a liquidity signal for {ticker}:

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
        return LiquiditySignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=LiquiditySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )