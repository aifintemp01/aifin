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
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def liquidity_agent(state: AgentState, agent_id: str = "liquidity_agent"):
    """
    Analyzes stocks using a comprehensive liquidity framework.
    Focuses on volume metrics (daily volume, 30D avg volume), traded value,
    Amihud illiquidity ratio, and impact cost proxy via high-low spread.
    High liquidity reduces execution risk and signals institutional confidence.
    Low liquidity stocks carry hidden transaction costs that erode returns.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    # 30D avg volume + Amihud needs at least 30 trading days (~45 calendar days)
    # Use 90 calendar days to ensure sufficient trading days after weekends/holidays
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=90)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    liquidity_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw price + volume data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching price and volume data")
        prices = get_prices(ticker, start_date, end_date, api_key=api_key)

        if not prices or len(prices) < 5:
            progress.update_status(agent_id, ticker, "Insufficient price history — skipping")
            liquidity_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "Insufficient price history to compute liquidity metrics.",
            }
            continue

        # Sort oldest → newest
        sorted_prices = sorted(
            prices,
            key=lambda p: p.time if hasattr(p, "time") else p["time"]
        )

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing volume metrics")
        volume_analysis = _analyze_volume(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing traded value")
        traded_value_analysis = _analyze_traded_value(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing Amihud illiquidity")
        illiquidity_analysis = _analyze_amihud_illiquidity(sorted_prices)

        progress.update_status(agent_id, ticker, "Analyzing impact cost proxy")
        impact_cost_analysis = _analyze_impact_cost(sorted_prices)

        # ------------------------------------------------------------------
        # Aggregate score
        # Liquidity weights: Amihud and traded value are most informative;
        # volume and impact cost confirm
        # ------------------------------------------------------------------
        total_score = (
            illiquidity_analysis["score"] * 0.35    # Amihud = truest liquidity signal
            + traded_value_analysis["score"] * 0.30  # traded value = institutional accessibility
            + volume_analysis["score"] * 0.20        # raw volume = market interest
            + impact_cost_analysis["score"] * 0.15   # impact cost = execution risk
        )
        max_score = 10

        if total_score >= 6.5:
            signal = "bullish"
        elif total_score <= 3.5:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect for LLM
        # ------------------------------------------------------------------
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
        )

        liquidity_analysis[ticker] = {
            "signal": liquidity_output.signal,
            "confidence": liquidity_output.confidence,
            "reasoning": liquidity_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=liquidity_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(liquidity_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(liquidity_analysis, "Liquidity Agent")

    state["data"]["analyst_signals"][agent_id] = liquidity_analysis

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


# ----- Volume Metrics (Daily Volume, 30D Avg Volume) ------------------------

def _analyze_volume(sorted_prices: list) -> dict:
    """
    Assess trading volume: latest daily volume vs 30D average.
    Rising volume relative to average signals growing market interest.
    Very low absolute volume flags liquidity risk regardless of trend.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    volumes = [_safe_float(_get_field(p, "volume")) for p in sorted_prices]
    volumes = [v for v in volumes if v is not None and v > 0]

    if not volumes:
        return {"score": 0, "max_score": max_score, "details": "Volume data unavailable",
                "latest_volume": None, "avg_30d_volume": None}

    latest_volume = volumes[-1]
    avg_30d = sum(volumes[-30:]) / len(volumes[-30:]) if len(volumes) >= 5 else sum(volumes) / len(volumes)

    # Score absolute 30D average volume — proxy for institutional accessibility
    if avg_30d >= 5_000_000:
        score += 2
        details.append(f"High 30D avg volume: {avg_30d:,.0f} shares — strong institutional accessibility")
    elif avg_30d >= 500_000:
        score += 1
        details.append(f"Moderate 30D avg volume: {avg_30d:,.0f} shares")
    else:
        details.append(f"Low 30D avg volume: {avg_30d:,.0f} shares — liquidity risk")

    # Score latest volume relative to 30D average
    if avg_30d > 0:
        vol_ratio = latest_volume / avg_30d
        if vol_ratio >= 1.5:
            score += 2
            details.append(f"Volume surge: {vol_ratio:.1f}x 30D average — elevated interest")
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


# ----- Traded Value Metrics (Daily Traded Value, 30D Avg Traded Value) ------

def _analyze_traded_value(sorted_prices: list) -> dict:
    """
    Assess traded value: volume × close price.
    Traded value in USD is the clearest measure of real liquidity —
    it accounts for both volume and price level.
    Institutional minimum is typically $10M+ daily traded value.
    """
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
        return {"score": 0, "max_score": max_score, "details": "Traded value data unavailable",
                "latest_traded_value": None, "avg_30d_traded_value": None}

    latest_tv = traded_values[-1]
    avg_30d_tv = sum(traded_values[-30:]) / len(traded_values[-30:]) if len(traded_values) >= 5 else sum(traded_values) / len(traded_values)

    # Score 30D average traded value — institutional threshold
    if avg_30d_tv >= 50_000_000:
        score += 2
        details.append(f"Excellent 30D avg traded value: ${avg_30d_tv / 1e6:.1f}M — institutional grade")
    elif avg_30d_tv >= 10_000_000:
        score += 1
        details.append(f"Adequate 30D avg traded value: ${avg_30d_tv / 1e6:.1f}M")
    else:
        details.append(f"Low 30D avg traded value: ${avg_30d_tv / 1e6:.2f}M — below institutional threshold")

    # Score latest vs 30D average
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


# ----- Amihud Illiquidity ---------------------------------------------------

def _analyze_amihud_illiquidity(sorted_prices: list) -> dict:
    """
    Compute Amihud (2002) illiquidity ratio: avg(|daily return| / daily traded value).
    Lower values = more liquid.
    This is the gold standard illiquidity measure used in academic finance.
    Multiplied by 1e6 for readability (raw values are extremely small).
    """
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
        return {"score": 0, "max_score": max_score,
                "details": "Amihud illiquidity: insufficient data",
                "amihud_ratio": None}

    amihud = (sum(ratios) / len(ratios)) * 1e6  # scale for readability

    # Lower Amihud = more liquid = higher score
    if amihud < 0.01:
        score += 6
        details.append(f"Exceptional liquidity — Amihud ratio: {amihud:.4f} (very low price impact)")
    elif amihud < 0.05:
        score += 5
        details.append(f"High liquidity — Amihud ratio: {amihud:.4f}")
    elif amihud < 0.20:
        score += 3
        details.append(f"Moderate liquidity — Amihud ratio: {amihud:.4f}")
    elif amihud < 0.50:
        score += 1
        details.append(f"Low liquidity — Amihud ratio: {amihud:.4f} — elevated price impact")
    else:
        details.append(f"Illiquid — Amihud ratio: {amihud:.4f} — significant market impact risk")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "amihud_ratio": round(amihud, 6),
    }


# ----- Impact Cost Proxy (High-Low Spread × Volume Sensitivity) -------------

def _analyze_impact_cost(sorted_prices: list) -> dict:
    """
    Estimate impact cost using high-low spread as a bid-ask proxy.
    Formula: avg((high - low) / close) over last 30 days.
    A tight high-low spread relative to price indicates low transaction costs.
    Combined with volume, this approximates real-world execution slippage.
    """
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
        return {"score": 0, "max_score": max_score,
                "details": "Impact cost proxy: insufficient data",
                "avg_hl_spread_pct": None}

    avg_spread = sum(spreads) / len(spreads)

    # Lower spread = lower impact cost = higher score
    if avg_spread < 0.01:
        score += 4
        details.append(f"Tight high-low spread: {avg_spread:.2%} — minimal execution cost")
    elif avg_spread < 0.02:
        score += 3
        details.append(f"Narrow high-low spread: {avg_spread:.2%} — low execution cost")
    elif avg_spread < 0.04:
        score += 2
        details.append(f"Moderate high-low spread: {avg_spread:.2%}")
    elif avg_spread < 0.07:
        score += 1
        details.append(f"Wide high-low spread: {avg_spread:.2%} — elevated execution cost")
    else:
        details.append(f"Very wide high-low spread: {avg_spread:.2%} — high slippage risk")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_hl_spread_pct": round(avg_spread, 6),
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_liquidity_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> LiquiditySignal:
    """Generate a liquidity-focused signal grounded strictly in the computed metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined liquidity analyst. Your mandate:
                - Liquidity is not just a risk filter — it is a signal of market confidence and institutional participation
                - The Amihud illiquidity ratio is the most academically robust measure; weight it heavily
                - Daily traded value above $10M is the minimum institutional threshold — below this, execution risk is real
                - A volume surge relative to 30D average signals accumulation or distribution — context matters
                - Wide high-low spreads indicate hidden transaction costs that erode stated returns
                - Illiquid stocks may appear cheap but carry permanent execution risk — flag this clearly

                When providing your reasoning, be specific by:
                1. Leading with the Amihud illiquidity ratio — what does it say about price impact?
                2. Citing 30D average traded value — is this institutionally accessible?
                3. Commenting on volume trend — accumulation or distribution signals
                4. Noting the high-low spread — what is the real cost to execute?
                5. Concluding with a clear, liquidity-anchored stance
                """,
            ),
            (
                "human",
                """Based on the following data, generate a liquidity signal for {ticker}:

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

    def create_default_liquidity_signal():
        return LiquiditySignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=LiquiditySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_liquidity_signal,
    )
