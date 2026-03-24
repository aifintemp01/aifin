from __future__ import annotations

import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class ValuePackSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def value_pack_agent(state: AgentState, agent_id: str = "value_pack_agent"):
    """
    Analyzes stocks using a comprehensive value investing framework.
    Focuses on intrinsic value metrics: FCF yield, PE, PB, EV/EBITDA,
    ROIC, debt-to-equity, margin of safety, and normalized FCF.
    Designed as a pure value-oriented philosophy agent.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    value_pack_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(
            ticker, end_date, period="annual", limit=5, api_key=api_key
        )

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "earnings_per_share",
                "free_cash_flow",
                "capital_expenditure",
                "ebitda",
                "operating_income",
                "return_on_invested_capital",
                "total_debt",
                "cash_and_equivalents",
                "shareholders_equity",
                "total_assets",
                "outstanding_shares",
                "book_value_per_share",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing earnings and price ratios")
        ratio_analysis = _analyze_price_ratios(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing cash flow quality")
        cashflow_analysis = _analyze_cash_flow(line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing capital efficiency")
        capital_analysis = _analyze_capital_efficiency(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet strength")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status(agent_id, ticker, "Calculating margin of safety")
        margin_of_safety_analysis = _analyze_margin_of_safety(line_items, market_cap)

        # ------------------------------------------------------------------
        # Aggregate score
        # Value investing weights: cash flow and margin of safety dominate
        # ------------------------------------------------------------------
        total_score = (
            cashflow_analysis["score"] * 0.30          # FCF yield + normalized FCF
            + margin_of_safety_analysis["score"] * 0.25  # intrinsic value vs price
            + ratio_analysis["score"] * 0.20            # PE, PB, EV/EBITDA
            + capital_analysis["score"] * 0.15          # ROIC
            + balance_sheet_analysis["score"] * 0.10    # D/E, leverage
        )
        max_score = 10

        if total_score >= 7.0:
            signal = "bullish"
        elif total_score <= 4.0:
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
            "ratio_analysis": ratio_analysis,
            "cashflow_analysis": cashflow_analysis,
            "capital_analysis": capital_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "margin_of_safety_analysis": margin_of_safety_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating value pack analysis")
        value_pack_output = _generate_value_pack_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        value_pack_analysis[ticker] = {
            "signal": value_pack_output.signal,
            "confidence": value_pack_output.confidence,
            "reasoning": value_pack_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=value_pack_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(value_pack_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(value_pack_analysis, "Value Pack Agent")

    state["data"]["analyst_signals"][agent_id] = value_pack_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub-analysis helpers
###############################################################################

def _latest(line_items: list):
    """Return the most recent line-item object or None."""
    return line_items[0] if line_items else None


def _safe_get(obj, attr: str):
    """Safely get an attribute from an object, returning None if missing."""
    return getattr(obj, attr, None) if obj is not None else None


# ----- Price Ratios (PE, PB, EV/EBITDA) ------------------------------------

def _analyze_price_ratios(metrics: list, line_items: list, market_cap: float | None) -> dict:
    """
    Assess valuation multiples: PE ratio, PB ratio, and EV/EBITDA.
    Lower multiples signal potential undervaluation.
    """
    max_score = 6  # 2pts PE + 2pts PB + 2pts EV/EBITDA
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    # PE Ratio
    net_income = _safe_get(latest_item, "net_income")
    if net_income and net_income > 0 and market_cap:
        pe = market_cap / net_income
        if pe < 15:
            score += 2
            details.append(f"Attractive PE ratio: {pe:.1f}x")
        elif pe < 25:
            score += 1
            details.append(f"Moderate PE ratio: {pe:.1f}x")
        else:
            details.append(f"Elevated PE ratio: {pe:.1f}x")
    else:
        details.append("PE ratio: insufficient data or negative earnings")

    # PB Ratio
    book_value_per_share = _safe_get(latest_item, "book_value_per_share")
    shares = _safe_get(latest_item, "outstanding_shares")
    if book_value_per_share and book_value_per_share > 0 and shares and market_cap:
        book_value_total = book_value_per_share * shares
        pb = market_cap / book_value_total
        if pb < 1.5:
            score += 2
            details.append(f"Undervalued on book value: PB {pb:.2f}x")
        elif pb < 3.0:
            score += 1
            details.append(f"Reasonable PB ratio: {pb:.2f}x")
        else:
            details.append(f"Premium to book value: PB {pb:.2f}x")
    else:
        details.append("PB ratio: insufficient data")

    # EV/EBITDA
    ebitda = _safe_get(latest_item, "ebitda")
    total_debt = _safe_get(latest_item, "total_debt")
    cash = _safe_get(latest_item, "cash_and_equivalents")
    if ebitda and ebitda > 0 and market_cap and total_debt is not None and cash is not None:
        ev = market_cap + total_debt - cash
        ev_ebitda = ev / ebitda
        if ev_ebitda < 8:
            score += 2
            details.append(f"Cheap on EV/EBITDA: {ev_ebitda:.1f}x")
        elif ev_ebitda < 14:
            score += 1
            details.append(f"Fair EV/EBITDA: {ev_ebitda:.1f}x")
        else:
            details.append(f"Rich EV/EBITDA: {ev_ebitda:.1f}x")
    else:
        details.append("EV/EBITDA: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- Cash Flow (FCF Yield, Normalized FCF) --------------------------------

def _analyze_cash_flow(line_items: list, market_cap: float | None) -> dict:
    """
    Assess free cash flow quality: FCF yield and normalized FCF trend.
    These are the most reliable indicators of true earning power.
    """
    max_score = 7  # 4pts FCF yield + 3pts normalized FCF trend
    score = 0
    details: list[str] = []

    fcf_values = [
        _safe_get(item, "free_cash_flow")
        for item in line_items
        if _safe_get(item, "free_cash_flow") is not None
    ]

    if not fcf_values:
        return {"score": 0, "max_score": max_score, "details": "FCF data unavailable", "fcf_yield": None, "normalized_fcf": None}

    # Normalized FCF: average of up to 5 years
    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))

    # FCF Yield
    if market_cap and market_cap > 0 and normalized_fcf > 0:
        fcf_yield = normalized_fcf / market_cap
        if fcf_yield >= 0.10:
            score += 4
            details.append(f"Exceptional FCF yield: {fcf_yield:.1%}")
        elif fcf_yield >= 0.07:
            score += 3
            details.append(f"Strong FCF yield: {fcf_yield:.1%}")
        elif fcf_yield >= 0.05:
            score += 2
            details.append(f"Adequate FCF yield: {fcf_yield:.1%}")
        elif fcf_yield >= 0.03:
            score += 1
            details.append(f"Thin FCF yield: {fcf_yield:.1%}")
        else:
            details.append(f"Weak FCF yield: {fcf_yield:.1%}")
    else:
        fcf_yield = None
        details.append("FCF yield: non-positive FCF or missing market cap")

    # Normalized FCF trend: recent 3 vs older 3
    if len(fcf_values) >= 4:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
        if older_avg != 0:
            fcf_growth = (recent_avg - older_avg) / abs(older_avg)
            if fcf_growth > 0.15:
                score += 3
                details.append(f"Growing FCF trend: {fcf_growth:.1%}")
            elif fcf_growth > 0:
                score += 2
                details.append(f"Stable FCF trend: {fcf_growth:.1%}")
            else:
                score += 0
                details.append(f"Declining FCF trend: {fcf_growth:.1%}")
    else:
        details.append("Insufficient FCF history for trend analysis")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf,
    }


# ----- Capital Efficiency (ROIC) --------------------------------------------

def _analyze_capital_efficiency(metrics: list, line_items: list) -> dict:
    """
    Assess return on invested capital (ROIC).
    High and consistent ROIC is the clearest signal of a durable competitive advantage.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    # Pull ROIC from line items across periods
    roic_values = [
        _safe_get(item, "return_on_invested_capital")
        for item in line_items
        if _safe_get(item, "return_on_invested_capital") is not None
    ]

    if not roic_values:
        # Fall back to financial metrics object
        latest_metrics = metrics[0] if metrics else None
        roic_from_metrics = _safe_get(latest_metrics, "return_on_invested_capital")
        if roic_from_metrics is not None:
            roic_values = [roic_from_metrics]

    if not roic_values:
        return {"score": 0, "max_score": max_score, "details": "ROIC data unavailable"}

    avg_roic = sum(roic_values) / len(roic_values)
    high_roic_periods = sum(1 for r in roic_values if r > 0.15)
    consistency_ratio = high_roic_periods / len(roic_values)

    if avg_roic > 0.20 and consistency_ratio >= 0.8:
        score += 4
        details.append(f"Exceptional ROIC: avg {avg_roic:.1%}, consistent across {high_roic_periods}/{len(roic_values)} periods")
    elif avg_roic > 0.15 and consistency_ratio >= 0.6:
        score += 3
        details.append(f"Strong ROIC: avg {avg_roic:.1%}, {high_roic_periods}/{len(roic_values)} periods above 15%")
    elif avg_roic > 0.10:
        score += 2
        details.append(f"Adequate ROIC: avg {avg_roic:.1%}")
    elif avg_roic > 0:
        score += 1
        details.append(f"Weak ROIC: avg {avg_roic:.1%}")
    else:
        details.append(f"Negative ROIC: avg {avg_roic:.1%} — capital destructive")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance Sheet (Debt-to-Equity) ---------------------------------------

def _analyze_balance_sheet(metrics: list, line_items: list) -> dict:
    """
    Assess leverage via debt-to-equity ratio and net cash position.
    Low leverage preserves optionality and reduces downside risk.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    # Debt-to-Equity
    debt = _safe_get(latest_item, "total_debt")
    equity = _safe_get(latest_item, "shareholders_equity")

    de_ratio = None
    if debt is not None and equity and equity > 0:
        de_ratio = debt / equity
        if de_ratio < 0.3:
            score += 2
            details.append(f"Conservative leverage: D/E {de_ratio:.2f}")
        elif de_ratio < 0.8:
            score += 1
            details.append(f"Moderate leverage: D/E {de_ratio:.2f}")
        else:
            details.append(f"High leverage: D/E {de_ratio:.2f}")
    else:
        details.append("D/E ratio: insufficient data")

    # Net cash position
    cash = _safe_get(latest_item, "cash_and_equivalents")
    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0:
            score += 2
            details.append(f"Net cash position: ${net_cash:,.0f}")
        else:
            details.append(f"Net debt position: ${net_cash:,.0f}")
    else:
        details.append("Net cash position: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details), "de_ratio": de_ratio}


# ----- Margin of Safety / Intrinsic Value -----------------------------------

def _analyze_margin_of_safety(line_items: list, market_cap: float | None) -> dict:
    """
    Estimate intrinsic value via a simple DCF on normalized FCF.
    Compares intrinsic value to current market cap to derive margin of safety.
    Conservative (8x), reasonable (12x), and optimistic (16x) FCF multiples used.
    """
    max_score = 6
    score = 0
    details: list[str] = []

    if not line_items or not market_cap or market_cap <= 0:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Insufficient data for margin of safety calculation",
            "intrinsic_value_range": None,
            "margin_of_safety": None,
        }

    fcf_values = [
        _safe_get(item, "free_cash_flow")
        for item in line_items
        if _safe_get(item, "free_cash_flow") is not None
    ]

    if not fcf_values or sum(fcf_values[:3]) / max(len(fcf_values[:3]), 1) <= 0:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Non-positive normalized FCF — margin of safety not calculable",
            "intrinsic_value_range": None,
            "margin_of_safety": None,
        }

    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))

    # Simple multiple-based intrinsic value range
    conservative_value = normalized_fcf * 8
    reasonable_value   = normalized_fcf * 12
    optimistic_value   = normalized_fcf * 16

    margin_of_safety = (reasonable_value - market_cap) / reasonable_value

    # Score based on upside to reasonable value
    if margin_of_safety >= 0.40:
        score += 6
        details.append(f"Deep value: {margin_of_safety:.1%} margin of safety vs reasonable value")
    elif margin_of_safety >= 0.25:
        score += 5
        details.append(f"Strong margin of safety: {margin_of_safety:.1%}")
    elif margin_of_safety >= 0.10:
        score += 3
        details.append(f"Moderate margin of safety: {margin_of_safety:.1%}")
    elif margin_of_safety >= 0:
        score += 1
        details.append(f"Slim margin of safety: {margin_of_safety:.1%}")
    else:
        details.append(f"No margin of safety: trading {-margin_of_safety:.1%} above reasonable value")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value,
        },
        "margin_of_safety": margin_of_safety,
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_value_pack_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> ValuePackSignal:
    """Generate a value-investing signal grounded strictly in the computed metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined value investing analyst. Your mandate:
                - Base every decision on hard metrics: FCF yield, PE, PB, EV/EBITDA, ROIC, D/E, and margin of safety
                - Intrinsic value vs current price is the primary question — not momentum, not narrative
                - Demand a margin of safety before turning bullish; be skeptical of expensive multiples
                - ROIC above 15% consistently is the clearest sign of a durable business
                - Leverage destroys value in downturns — penalize high D/E ratios
                - Normalized FCF is more reliable than reported earnings — trust it more

                When providing your reasoning, be specific by:
                1. Leading with the margin of safety figure and what it implies
                2. Citing FCF yield as the core earning power metric
                3. Commenting on ROIC trend — is capital being deployed wisely?
                4. Flagging any valuation multiple that is stretched or attractive
                5. Noting leverage risk if D/E is elevated
                6. Concluding with a clear, number-anchored stance
                """,
            ),
            (
                "human",
                """Based on the following data, generate a value investing signal for {ticker}:

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

    def create_default_value_pack_signal():
        return ValuePackSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=ValuePackSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_value_pack_signal,
    )
