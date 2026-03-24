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


class QualityPackSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def quality_pack_agent(state: AgentState, agent_id: str = "quality_pack_agent"):
    """
    Analyzes stocks using a comprehensive quality investing framework.
    Focuses on the durability and consistency of financial performance:
    ROCE, ROE, operating margins, revenue and EPS CAGR stability,
    FCF conversion efficiency, and operating cash flow consistency.
    A high-quality business earns high returns on capital consistently
    and converts earnings to cash reliably — this agent finds them.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    quality_pack_analysis: dict[str, dict] = {}

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
                "ebit",
                "net_income",
                "earnings_per_share",
                "operating_income",
                "operating_margin",
                "gross_margin",
                "free_cash_flow",
                "capital_expenditure",
                "total_debt",
                "cash_and_equivalents",
                "shareholders_equity",
                "return_on_equity",
                "return_on_invested_capital",
                "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=7,  # Quality requires longer history for CAGR and stability
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing profitability metrics")
        profitability_analysis = _analyze_profitability(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing growth stability")
        growth_stability_analysis = _analyze_growth_stability(line_items)

        progress.update_status(agent_id, ticker, "Analyzing cash flow quality")
        cashflow_quality_analysis = _analyze_cashflow_quality(line_items)

        # ------------------------------------------------------------------
        # Aggregate score
        # Quality investing weights: profitability and cash flow dominate
        # Growth consistency validates durability
        # ------------------------------------------------------------------
        total_score = (
            profitability_analysis["score"] * 0.40    # ROCE, ROE, operating margins
            + cashflow_quality_analysis["score"] * 0.35  # FCF conversion, OCF stability
            + growth_stability_analysis["score"] * 0.25  # revenue + EPS CAGR stability
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
            "profitability_analysis": profitability_analysis,
            "growth_stability_analysis": growth_stability_analysis,
            "cashflow_quality_analysis": cashflow_quality_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating quality pack analysis")
        quality_pack_output = _generate_quality_pack_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        quality_pack_analysis[ticker] = {
            "signal": quality_pack_output.signal,
            "confidence": quality_pack_output.confidence,
            "reasoning": quality_pack_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=quality_pack_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(quality_pack_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(quality_pack_analysis, "Quality Pack Agent")

    state["data"]["analyst_signals"][agent_id] = quality_pack_analysis

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


# ----- Profitability (ROCE, ROE, Operating Margin) --------------------------

def _analyze_profitability(metrics: list, line_items: list) -> dict:
    """
    Assess core profitability via ROCE, ROE, and operating margin.
    High and stable returns on capital signal a durable competitive advantage.
    ROCE is the primary metric — it captures how well the entire capital base
    is deployed, not just equity.
    """
    max_score = 9  # 3pts ROCE + 3pts ROE + 3pts operating margin
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    # ROCE: EBIT / (Total Equity + Total Debt - Cash)
    ebit = _safe_get(latest_item, "ebit")
    total_debt = _safe_get(latest_item, "total_debt")
    cash = _safe_get(latest_item, "cash_and_equivalents")
    equity = _safe_get(latest_item, "shareholders_equity")

    if ebit is not None and equity is not None and total_debt is not None and cash is not None:
        capital_employed = equity + total_debt - cash
        if capital_employed > 0:
            roce = ebit / capital_employed
            if roce > 0.25:
                score += 3
                details.append(f"Exceptional ROCE: {roce:.1%}")
            elif roce > 0.15:
                score += 2
                details.append(f"Strong ROCE: {roce:.1%}")
            elif roce > 0.08:
                score += 1
                details.append(f"Adequate ROCE: {roce:.1%}")
            else:
                details.append(f"Weak ROCE: {roce:.1%}")
        else:
            details.append("ROCE: negative capital employed — unusual capital structure")
    else:
        details.append("ROCE: insufficient data")

    # ROE: Net Income / Shareholders' Equity
    net_income = _safe_get(latest_item, "net_income")
    roe_from_metrics = _safe_get(latest_metrics, "return_on_equity")

    if roe_from_metrics is not None:
        roe = roe_from_metrics
    elif net_income is not None and equity and equity > 0:
        roe = net_income / equity
    else:
        roe = None

    if roe is not None:
        if roe > 0.20:
            score += 3
            details.append(f"Excellent ROE: {roe:.1%}")
        elif roe > 0.12:
            score += 2
            details.append(f"Good ROE: {roe:.1%}")
        elif roe > 0.06:
            score += 1
            details.append(f"Moderate ROE: {roe:.1%}")
        else:
            details.append(f"Poor ROE: {roe:.1%}")
    else:
        details.append("ROE: insufficient data")

    # Operating Margin: consistency across periods matters more than one year
    # Calculate operating margin from operating_income and sales
    op_margins = []
    for item in line_items:
        operating_income = _safe_get(item, "operating_income")
        sales = _safe_get(item, "sales")
        if operating_income is not None and sales is not None and sales != 0:
            op_margins.append(operating_income / sales)

    if op_margins:
        avg_op_margin = sum(op_margins) / len(op_margins)
        # Also check stability: if margins are shrinking, penalize
        is_expanding = len(op_margins) >= 2 and op_margins[0] >= op_margins[-1]

        if avg_op_margin > 0.25:
            score += 3
            details.append(f"Premium operating margins: avg {avg_op_margin:.1%}" + (" (expanding)" if is_expanding else " (compressing)"))
        elif avg_op_margin > 0.15:
            score += 2
            details.append(f"Healthy operating margins: avg {avg_op_margin:.1%}" + (" (expanding)" if is_expanding else " (compressing)"))
        elif avg_op_margin > 0.08:
            score += 1
            details.append(f"Thin operating margins: avg {avg_op_margin:.1%}")
        else:
            details.append(f"Weak operating margins: avg {avg_op_margin:.1%}")
    else:
        details.append("Operating margin: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- Growth Stability (Revenue CAGR, EPS CAGR) ----------------------------

def _analyze_growth_stability(line_items: list) -> dict:
    """
    Assess revenue and EPS growth using CAGR over available history.
    Quality businesses grow consistently — we penalize erratic growth
    even if the average rate looks acceptable.
    """
    max_score = 6  # 3pts revenue CAGR + 3pts EPS CAGR
    score = 0
    details: list[str] = []

    # Revenue CAGR
    revenues = [
        _safe_get(item, "revenue")
        for item in line_items
        if _safe_get(item, "revenue") is not None
    ]

    if len(revenues) >= 2:
        latest_rev = revenues[0]
        oldest_rev = revenues[-1]
        n = len(revenues) - 1
        if oldest_rev > 0 and latest_rev > 0:
            rev_cagr = (latest_rev / oldest_rev) ** (1 / n) - 1
            if rev_cagr > 0.15:
                score += 3
                details.append(f"Strong revenue CAGR: {rev_cagr:.1%} over {n}Y")
            elif rev_cagr > 0.08:
                score += 2
                details.append(f"Healthy revenue CAGR: {rev_cagr:.1%} over {n}Y")
            elif rev_cagr > 0.03:
                score += 1
                details.append(f"Modest revenue CAGR: {rev_cagr:.1%} over {n}Y")
            else:
                details.append(f"Stagnant revenue CAGR: {rev_cagr:.1%} over {n}Y")
        else:
            details.append("Revenue CAGR: zero or negative base revenue")
    else:
        details.append("Revenue CAGR: insufficient history")

    # EPS CAGR
    eps_values = [
        _safe_get(item, "earnings_per_share")
        for item in line_items
        if _safe_get(item, "earnings_per_share") is not None
    ]

    if len(eps_values) >= 2:
        latest_eps = eps_values[0]
        oldest_eps = eps_values[-1]
        n = len(eps_values) - 1
        if oldest_eps > 0 and latest_eps > 0:
            eps_cagr = (latest_eps / oldest_eps) ** (1 / n) - 1
            if eps_cagr > 0.15:
                score += 3
                details.append(f"Strong EPS CAGR: {eps_cagr:.1%} over {n}Y")
            elif eps_cagr > 0.08:
                score += 2
                details.append(f"Healthy EPS CAGR: {eps_cagr:.1%} over {n}Y")
            elif eps_cagr > 0.03:
                score += 1
                details.append(f"Modest EPS CAGR: {eps_cagr:.1%} over {n}Y")
            else:
                details.append(f"Stagnant EPS CAGR: {eps_cagr:.1%} over {n}Y")
        else:
            details.append("EPS CAGR: negative or zero base EPS — earnings not yet established")
    else:
        details.append("EPS CAGR: insufficient history")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- Cash Flow Quality (FCF Conversion, OCF Stability) --------------------

def _analyze_cashflow_quality(line_items: list) -> dict:
    """
    Assess the quality of earnings via FCF conversion and OCF stability.
    FCF conversion > 1.0 means the business generates more cash than it reports
    in net income — a hallmark of high-quality accounting.
    OCF stability measures how consistent cash generation is year-over-year.
    """
    max_score = 7  # 4pts FCF conversion + 3pts OCF stability
    score = 0
    details: list[str] = []

    fcf_values = [
        _safe_get(item, "free_cash_flow")
        for item in line_items
        if _safe_get(item, "free_cash_flow") is not None
    ]
    net_income_values = [
        _safe_get(item, "net_income")
        for item in line_items
        if _safe_get(item, "net_income") is not None
    ]

    # FCF Conversion: FCF / Net Income — average across available periods
    conversions = []
    for item in line_items:
        fcf = _safe_get(item, "free_cash_flow")
        ni = _safe_get(item, "net_income")
        if fcf is not None and ni is not None and ni > 0:
            conversions.append(fcf / ni)

    if conversions:
        avg_conversion = sum(conversions) / len(conversions)
        if avg_conversion >= 1.10:
            score += 4
            details.append(f"Exceptional FCF conversion: {avg_conversion:.2f}x (cash exceeds reported earnings)")
        elif avg_conversion >= 0.90:
            score += 3
            details.append(f"Strong FCF conversion: {avg_conversion:.2f}x")
        elif avg_conversion >= 0.70:
            score += 2
            details.append(f"Adequate FCF conversion: {avg_conversion:.2f}x")
        elif avg_conversion >= 0.50:
            score += 1
            details.append(f"Weak FCF conversion: {avg_conversion:.2f}x — earnings quality questionable")
        else:
            details.append(f"Poor FCF conversion: {avg_conversion:.2f}x — significant earnings/cash divergence")
    else:
        details.append("FCF conversion: insufficient data")

    # OCF Stability: use FCF as proxy for operating cash flow
    # Measure year-over-year growth consistency
    if len(fcf_values) >= 3:
        positive_periods = sum(1 for f in fcf_values if f > 0)
        consistency_ratio = positive_periods / len(fcf_values)

        # Also check for trend direction
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]

        if consistency_ratio >= 0.90 and recent_avg >= older_avg:
            score += 3
            details.append(f"Highly stable and growing cash flows: positive in {positive_periods}/{len(fcf_values)} periods")
        elif consistency_ratio >= 0.75:
            score += 2
            details.append(f"Mostly stable cash flows: positive in {positive_periods}/{len(fcf_values)} periods")
        elif consistency_ratio >= 0.50:
            score += 1
            details.append(f"Inconsistent cash flows: positive in {positive_periods}/{len(fcf_values)} periods")
        else:
            details.append(f"Unreliable cash flows: positive in only {positive_periods}/{len(fcf_values)} periods")
    else:
        details.append("OCF stability: insufficient history (need 3+ periods)")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_fcf_conversion": sum(conversions) / len(conversions) if conversions else None,
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_quality_pack_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> QualityPackSignal:
    """Generate a quality investing signal grounded strictly in the computed metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined quality investing analyst. Your mandate:
                - Identify businesses that consistently earn high returns on capital (ROCE > 15%, ROE > 12%)
                - Prioritize earnings quality: FCF conversion above 1.0x means real cash backs reported profits
                - Consistent revenue and EPS CAGR over 5+ years proves the business model is durable
                - Expanding or stable operating margins signal pricing power and operational discipline
                - Volatile or declining cash flows disqualify a business regardless of other metrics
                - You are not looking for cheap stocks — you are looking for excellent businesses

                When providing your reasoning, be specific by:
                1. Leading with ROCE and ROE — are returns on capital sustainably high?
                2. Citing FCF conversion ratio — does cash match reported earnings?
                3. Discussing revenue and EPS CAGR — is growth consistent or erratic?
                4. Commenting on operating margin trend — expanding, stable, or compressing?
                5. Flagging any breakdown in cash flow consistency as a quality red flag
                6. Concluding with a clear, metric-anchored stance on business quality
                """,
            ),
            (
                "human",
                """Based on the following data, generate a quality investing signal for {ticker}:

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

    def create_default_quality_pack_signal():
        return QualityPackSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=QualityPackSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_quality_pack_signal,
    )
