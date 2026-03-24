from __future__ import annotations

import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_financial_metrics,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class FactorCompositeSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def factor_composite_agent(state: AgentState, agent_id: str = "factor_composite_agent"):
    """
    Analyzes stocks using a factor composite framework focused on financial risk.
    Combines two core dimensions:

    - Earnings Volatility: standard deviation of EPS growth over available history.
      High volatility signals cyclical, unpredictable, or structurally fragile earnings.

    - Balance Sheet Stress: (Debt/Equity) × (1 / Interest Coverage).
      This composite captures both the size of the debt burden and the company's
      ability to service it. High stress = fragile financial structure.

    Low earnings volatility + low balance sheet stress = a resilient, predictable
    business that survives downturns and compounds through cycles.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    factor_composite_analysis: dict[str, dict] = {}

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
                "earnings_per_share",
                "ebit",
                "operating_income",
                "interest_expense",
                "total_debt",
                "shareholders_equity",
                "net_income",
                "free_cash_flow",
            ],
            end_date,
            period="annual",
            limit=6,  # Need 6 periods for 5 YoY growth rates
            api_key=api_key,
        )

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing earnings volatility")
        earnings_volatility_analysis = _analyze_earnings_volatility(line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet stress")
        balance_sheet_stress_analysis = _analyze_balance_sheet_stress(metrics, line_items)

        # ------------------------------------------------------------------
        # Aggregate score
        # Both dimensions are equally critical — a company can fail from
        # either volatile earnings or an over-levered balance sheet
        # ------------------------------------------------------------------
        total_score = (
            earnings_volatility_analysis["score"] * 0.45   # earnings predictability
            + balance_sheet_stress_analysis["score"] * 0.55  # financial structure resilience
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
            "earnings_volatility_analysis": earnings_volatility_analysis,
            "balance_sheet_stress_analysis": balance_sheet_stress_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating factor composite analysis")
        factor_composite_output = _generate_factor_composite_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        factor_composite_analysis[ticker] = {
            "signal": factor_composite_output.signal,
            "confidence": factor_composite_output.confidence,
            "reasoning": factor_composite_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=factor_composite_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(factor_composite_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(factor_composite_analysis, "Factor Composite Agent")

    state["data"]["analyst_signals"][agent_id] = factor_composite_analysis

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


# ----- Earnings Volatility --------------------------------------------------

def _analyze_earnings_volatility(line_items: list) -> dict:
    """
    Measure earnings predictability via standard deviation of EPS YoY growth.
    Low volatility signals a stable, durable business model.
    High volatility signals cyclicality, structural fragility, or poor
    earnings quality — all of which compress valuation multiples.

    Formula: std dev of EPS YoY growth rates over available history.
    Interpretation:
    - < 0.10 (10%): highly stable earnings — premium quality
    - 0.10–0.25: moderate volatility — acceptable for most businesses
    - 0.25–0.50: high volatility — cyclical or structurally challenged
    - > 0.50: very high volatility — earnings are unreliable
    """
    max_score = 5
    score = 0
    details: list[str] = []

    eps_values = [
        _safe_get(item, "earnings_per_share")
        for item in line_items
        if _safe_get(item, "earnings_per_share") is not None
    ]

    if len(eps_values) < 3:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Earnings volatility: insufficient EPS history (need 3+ periods)",
            "eps_growth_std": None,
            "eps_growth_rates": None,
        }

    # Compute YoY EPS growth rates (oldest → newest order, list is newest first)
    growth_rates = []
    for i in range(len(eps_values) - 1):
        current = eps_values[i]
        prior = eps_values[i + 1]
        if prior is not None and prior != 0:
            growth_rates.append((current - prior) / abs(prior))

    if len(growth_rates) < 2:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Earnings volatility: insufficient growth rate history",
            "eps_growth_std": None,
            "eps_growth_rates": growth_rates,
        }

    n = len(growth_rates)
    mean_g = sum(growth_rates) / n
    variance = sum((g - mean_g) ** 2 for g in growth_rates) / (n - 1)
    eps_std = variance ** 0.5

    if eps_std < 0.10:
        score += 5
        details.append(f"Highly stable earnings: EPS growth std dev {eps_std:.2f} — premium predictability")
    elif eps_std < 0.20:
        score += 4
        details.append(f"Stable earnings: EPS growth std dev {eps_std:.2f} — consistent performance")
    elif eps_std < 0.35:
        score += 2
        details.append(f"Moderate earnings volatility: std dev {eps_std:.2f} — some cyclicality")
    elif eps_std < 0.55:
        score += 1
        details.append(f"High earnings volatility: std dev {eps_std:.2f} — unreliable earnings trend")
    else:
        details.append(f"Very high earnings volatility: std dev {eps_std:.2f} — earnings are unpredictable")

    # Additional context: avg EPS growth direction
    if mean_g > 0.05:
        details.append(f"Positive avg EPS growth: {mean_g:.1%}")
    elif mean_g < -0.05:
        details.append(f"Negative avg EPS growth: {mean_g:.1%} — declining earnings")
    else:
        details.append(f"Flat avg EPS growth: {mean_g:.1%}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "eps_growth_std": round(eps_std, 4),
        "avg_eps_growth": round(mean_g, 4),
        "eps_growth_rates": [round(g, 4) for g in growth_rates],
    }


# ----- Balance Sheet Stress -------------------------------------------------

def _analyze_balance_sheet_stress(metrics: list, line_items: list) -> dict:
    """
    Compute the balance sheet stress composite:
    Stress = (Debt/Equity) × (1 / Interest Coverage)

    This formula punishes both high leverage AND poor coverage simultaneously.
    A company with D/E of 2.0 but coverage of 20x is less stressed than
    one with D/E of 1.0 but coverage of 1.5x.

    Supporting metrics reported individually:
    - Debt-to-Equity ratio
    - Interest Coverage ratio (EBIT / interest expense)
    - Net debt position (cash vs debt)

    Stress score interpretation:
    - < 0.05: very low stress — fortress balance sheet
    - 0.05–0.20: low stress — manageable leverage
    - 0.20–0.50: moderate stress — leverage becoming a concern
    - 0.50–1.00: high stress — financial fragility
    - > 1.00: critical stress — distress risk
    """
    max_score = 5
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    # --- Debt-to-Equity ---
    de_ratio = _safe_get(latest_metrics, "debt_to_equity")
    if de_ratio is None:
        debt = _safe_get(latest_item, "total_debt")
        equity = _safe_get(latest_item, "shareholders_equity")
        if debt is not None and equity and equity > 0:
            de_ratio = debt / equity

    # --- Interest Coverage ---
    interest_coverage = _safe_get(latest_metrics, "interest_coverage")
    if interest_coverage is None:
        ebit = _safe_get(latest_item, "ebit") or _safe_get(latest_item, "operating_income")
        interest_expense = _safe_get(latest_item, "interest_expense")
        if ebit is not None and interest_expense and interest_expense > 0:
            interest_coverage = ebit / interest_expense

    # --- Composite Balance Sheet Stress Score ---
    stress_score = None
    if de_ratio is not None and interest_coverage is not None and interest_coverage > 0:
        stress_score = de_ratio / interest_coverage

        if stress_score < 0.05:
            score += 5
            details.append(f"Very low balance sheet stress: {stress_score:.3f} (D/E {de_ratio:.2f}, coverage {interest_coverage:.1f}x)")
        elif stress_score < 0.20:
            score += 4
            details.append(f"Low balance sheet stress: {stress_score:.3f} (D/E {de_ratio:.2f}, coverage {interest_coverage:.1f}x)")
        elif stress_score < 0.50:
            score += 2
            details.append(f"Moderate balance sheet stress: {stress_score:.3f} (D/E {de_ratio:.2f}, coverage {interest_coverage:.1f}x)")
        elif stress_score < 1.00:
            score += 1
            details.append(f"High balance sheet stress: {stress_score:.3f} (D/E {de_ratio:.2f}, coverage {interest_coverage:.1f}x)")
        else:
            details.append(f"Critical balance sheet stress: {stress_score:.3f} — distress risk (D/E {de_ratio:.2f}, coverage {interest_coverage:.1f}x)")

    elif de_ratio is not None:
        # Fallback: score only on D/E if coverage unavailable
        if de_ratio < 0.3:
            score += 4
            details.append(f"Low leverage: D/E {de_ratio:.2f} (interest coverage unavailable)")
        elif de_ratio < 0.8:
            score += 2
            details.append(f"Moderate leverage: D/E {de_ratio:.2f} (interest coverage unavailable)")
        else:
            score += 1
            details.append(f"High leverage: D/E {de_ratio:.2f} (interest coverage unavailable)")
    else:
        details.append("Balance sheet stress: insufficient data for D/E or interest coverage")

    # --- Net Cash Position as additional context ---
    debt = _safe_get(latest_item, "total_debt")
    cash = _safe_get(latest_item, "cash_and_equivalents") if hasattr(latest_item, "cash_and_equivalents") else None
    # cash_and_equivalents may not be in line_items for this agent — note gracefully
    if debt is not None and cash is not None:
        net_cash = cash - debt
        details.append(
            f"Net {'cash' if net_cash >= 0 else 'debt'} position: ${net_cash:,.0f}"
        )

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "debt_to_equity": round(de_ratio, 4) if de_ratio is not None else None,
        "interest_coverage": round(interest_coverage, 2) if interest_coverage is not None else None,
        "balance_sheet_stress_score": round(stress_score, 4) if stress_score is not None else None,
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_factor_composite_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> FactorCompositeSignal:
    """Generate a factor composite signal grounded strictly in the computed metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined factor risk analyst. Your mandate:
                - Identify companies with low financial risk across two dimensions: earnings stability and balance sheet resilience
                - Earnings volatility (std dev of EPS growth) is the primary predictor of multiple compression
                - Balance sheet stress = (D/E) × (1/interest coverage) — this single number captures fragility better than either metric alone
                - A company can survive a bad year if its balance sheet is strong; it cannot survive both volatile earnings AND high leverage
                - Low stress + low volatility = a business that compounds reliably through cycles
                - High stress + high volatility = a business that is one downturn away from distress

                When providing your reasoning, be specific by:
                1. Leading with the EPS growth standard deviation — is this business predictable?
                2. Citing the composite balance sheet stress score — is the financial structure resilient?
                3. Breaking down D/E and interest coverage separately — which is driving the stress?
                4. Commenting on avg EPS growth direction — predictable decline is worse than volatile growth
                5. Concluding with a clear financial risk verdict
                """,
            ),
            (
                "human",
                """Based on the following data, generate a factor composite signal for {ticker}:

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

    def create_default_factor_composite_signal():
        return FactorCompositeSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=FactorCompositeSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_factor_composite_signal,
    )
