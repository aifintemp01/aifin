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


class MacroExposureSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def macro_exposure_agent(state: AgentState, agent_id: str = "macro_exposure_agent"):
    """
    Analyzes stocks using a comprehensive macro exposure framework.
    Assesses a company's sensitivity to macroeconomic forces via
    financial statement proxies:
    - Interest rate sensitivity: debt load, interest coverage, and leverage
    - Inflation sensitivity: gross margin stability and capex intensity
    - FX dependency: international revenue exposure and geographic concentration
    Companies with low debt, stable margins, and domestic revenue are more
    resilient across macro regimes.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    macro_exposure_analysis: dict[str, dict] = {}

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
                "total_debt",
                "cash_and_equivalents",
                "shareholders_equity",
                "interest_expense",
                "operating_income",
                "ebit",
                "revenue",
                "gross_profit",
                "gross_margin",
                "capital_expenditure",
                "net_income",
                "total_assets",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing interest rate sensitivity")
        rate_sensitivity = _analyze_rate_sensitivity(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing inflation sensitivity")
        inflation_sensitivity = _analyze_inflation_sensitivity(line_items)

        progress.update_status(agent_id, ticker, "Analyzing FX dependency")
        fx_dependency = _analyze_fx_dependency(metrics, line_items)

        # ------------------------------------------------------------------
        # Composite macro exposure score
        # Equal weights across three dimensions as per MacroExposurePack spec.
        # Higher score = lower macro exposure = more resilient business.
        # ------------------------------------------------------------------
        total_score = (
            rate_sensitivity["score"] * 0.35      # debt/leverage dominates macro risk
            + inflation_sensitivity["score"] * 0.35  # margin stability is equally critical
            + fx_dependency["score"] * 0.30       # FX risk is real but manageable
        )
        max_score = 10

        # A low-macro-exposure business is more predictable → bullish
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
            "rate_sensitivity": rate_sensitivity,
            "inflation_sensitivity": inflation_sensitivity,
            "fx_dependency": fx_dependency,
            "composite_macro_exposure": {
                "description": "Weighted composite of rate, inflation, and FX exposure (higher = more resilient)",
                "score": round(total_score, 2),
                "weights": {"rate_sensitivity": 0.35, "inflation_sensitivity": 0.35, "fx_dependency": 0.30},
            },
        }

        progress.update_status(agent_id, ticker, "Generating macro exposure analysis")
        macro_output = _generate_macro_exposure_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        macro_exposure_analysis[ticker] = {
            "signal": macro_output.signal,
            "confidence": macro_output.confidence,
            "reasoning": macro_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=macro_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(macro_exposure_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(macro_exposure_analysis, "Macro Exposure Agent")

    state["data"]["analyst_signals"][agent_id] = macro_exposure_analysis

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


# ----- Interest Rate Sensitivity --------------------------------------------

def _analyze_rate_sensitivity(metrics: list, line_items: list) -> dict:
    """
    Proxy for interest rate sensitivity via balance sheet leverage metrics.
    Companies with high debt loads and low interest coverage are most vulnerable
    to rising rates. Net cash positions are rate-environment agnostic.

    Metrics used:
    - Debt-to-equity ratio (from metrics object)
    - Interest coverage ratio: EBIT / interest expense
    - Net debt position: total debt - cash
    """
    max_score = 6  # 2pts D/E + 2pts interest coverage + 2pts net cash position
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    # Debt-to-Equity
    de_ratio = _safe_get(latest_metrics, "debt_to_equity")
    if de_ratio is None:
        debt = _safe_get(latest_item, "total_debt")
        equity = _safe_get(latest_item, "shareholders_equity")
        if debt is not None and equity and equity > 0:
            de_ratio = debt / equity

    if de_ratio is not None:
        if de_ratio < 0.3:
            score += 2
            details.append(f"Low rate sensitivity: D/E {de_ratio:.2f} — minimal debt burden")
        elif de_ratio < 0.8:
            score += 1
            details.append(f"Moderate rate sensitivity: D/E {de_ratio:.2f}")
        else:
            details.append(f"High rate sensitivity: D/E {de_ratio:.2f} — vulnerable to rising rates")
    else:
        details.append("D/E ratio: insufficient data")

    # Interest Coverage: EBIT / Interest Expense
    ebit = _safe_get(latest_item, "ebit") or _safe_get(latest_item, "operating_income")
    interest_expense = _safe_get(latest_item, "interest_expense")

    interest_coverage = _safe_get(latest_metrics, "interest_coverage")
    if interest_coverage is None and ebit and interest_expense and interest_expense > 0:
        interest_coverage = ebit / interest_expense

    if interest_coverage is not None:
        if interest_coverage > 10:
            score += 2
            details.append(f"Strong interest coverage: {interest_coverage:.1f}x — rate-resilient")
        elif interest_coverage > 4:
            score += 1
            details.append(f"Adequate interest coverage: {interest_coverage:.1f}x")
        else:
            details.append(f"Weak interest coverage: {interest_coverage:.1f}x — rate-sensitive")
    else:
        details.append("Interest coverage: insufficient data")

    # Net cash position
    debt = _safe_get(latest_item, "total_debt")
    cash = _safe_get(latest_item, "cash_and_equivalents")
    if debt is not None and cash is not None:
        net_cash = cash - debt
        if net_cash > 0:
            score += 2
            details.append(f"Net cash position: ${net_cash:,.0f} — benefits from rising rates")
        else:
            details.append(f"Net debt position: ${net_cash:,.0f} — rate headwind")
    else:
        details.append("Net cash/debt: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- Inflation Sensitivity ------------------------------------------------

def _analyze_inflation_sensitivity(line_items: list) -> dict:
    """
    Proxy for inflation sensitivity via gross margin stability and capex intensity.
    Companies with stable or expanding gross margins can pass cost increases to
    customers — they are inflation-resilient. High capex intensity means assets
    must be replaced at inflated costs, hurting long-term returns.

    Metrics used:
    - Gross margin stability across periods
    - Capex as % of revenue (capex intensity)
    """
    max_score = 6  # 4pts gross margin stability + 2pts capex intensity
    score = 0
    details: list[str] = []

    # Gross Margin Stability
    gm_values = [
        _safe_get(item, "gross_margin")
        for item in line_items
        if _safe_get(item, "gross_margin") is not None
    ]

    if gm_values:
        avg_gm = sum(gm_values) / len(gm_values)
        # Stability: check if margins are holding or expanding
        is_stable = len(gm_values) < 2 or gm_values[0] >= gm_values[-1] * 0.95

        if avg_gm > 0.40 and is_stable:
            score += 4
            details.append(f"Strong pricing power: avg gross margin {avg_gm:.1%}, stable/expanding")
        elif avg_gm > 0.25 and is_stable:
            score += 3
            details.append(f"Decent pricing power: avg gross margin {avg_gm:.1%}, stable")
        elif avg_gm > 0.40:
            score += 2
            details.append(f"High but compressing margins: avg {avg_gm:.1%} — inflation risk rising")
        elif avg_gm > 0.15:
            score += 1
            details.append(f"Thin margins: avg {avg_gm:.1%} — vulnerable to cost inflation")
        else:
            details.append(f"Very thin margins: avg {avg_gm:.1%} — high inflation pass-through risk")
    else:
        details.append("Gross margin: insufficient data")

    # Capex Intensity: capex as % of revenue
    capex_to_rev = []
    for item in line_items:
        capex = _safe_get(item, "capital_expenditure")
        revenue = _safe_get(item, "revenue")
        if capex is not None and revenue and revenue > 0:
            capex_to_rev.append(abs(capex) / revenue)

    if capex_to_rev:
        avg_capex_intensity = sum(capex_to_rev) / len(capex_to_rev)
        if avg_capex_intensity < 0.03:
            score += 2
            details.append(f"Asset-light: capex {avg_capex_intensity:.1%} of revenue — low inflation replacement risk")
        elif avg_capex_intensity < 0.08:
            score += 1
            details.append(f"Moderate capex intensity: {avg_capex_intensity:.1%} of revenue")
        else:
            details.append(f"Capex heavy: {avg_capex_intensity:.1%} of revenue — inflation replaces assets at higher cost")
    else:
        details.append("Capex intensity: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


# ----- FX Dependency --------------------------------------------------------

def _analyze_fx_dependency(metrics: list, line_items: list) -> dict:
    """
    Proxy for FX dependency via revenue growth consistency and asset base.
    Note: direct international revenue split is not available in the current
    data model. We use revenue growth volatility as a proxy — companies with
    highly volatile revenue often have significant FX exposure.
    Asset-to-revenue ratio provides additional context on capital base stability.

    A future enhancement should use segment revenue data when available.
    """
    max_score = 4  # 2pts revenue stability + 2pts asset efficiency
    score = 0
    details: list[str] = []

    # Revenue growth volatility as FX proxy
    revenues = [
        _safe_get(item, "revenue")
        for item in line_items
        if _safe_get(item, "revenue") is not None
    ]

    if len(revenues) >= 3:
        # Compute year-over-year growth rates
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i + 1] and revenues[i + 1] > 0:
                growth_rates.append((revenues[i] - revenues[i + 1]) / revenues[i + 1])

        if growth_rates:
            n = len(growth_rates)
            mean_g = sum(growth_rates) / n
            variance = sum((g - mean_g) ** 2 for g in growth_rates) / max(n - 1, 1)
            rev_volatility = variance ** 0.5

            if rev_volatility < 0.05:
                score += 2
                details.append(f"Stable revenue: {rev_volatility:.1%} YoY volatility — low FX exposure likely")
            elif rev_volatility < 0.15:
                score += 1
                details.append(f"Moderate revenue volatility: {rev_volatility:.1%} — some FX exposure possible")
            else:
                details.append(f"High revenue volatility: {rev_volatility:.1%} — elevated FX exposure likely")
        else:
            details.append("Revenue growth volatility: insufficient data")
    else:
        details.append("Revenue stability: insufficient history")

    # Asset efficiency: total assets / revenue — lower = more capital efficient
    latest_item = _latest(line_items)
    total_assets = _safe_get(latest_item, "total_assets")
    revenue = _safe_get(latest_item, "revenue")

    if total_assets and revenue and revenue > 0:
        asset_to_rev = total_assets / revenue
        if asset_to_rev < 1.0:
            score += 2
            details.append(f"Asset-light business: assets {asset_to_rev:.2f}x revenue — lower FX asset risk")
        elif asset_to_rev < 2.0:
            score += 1
            details.append(f"Moderate asset base: {asset_to_rev:.2f}x revenue")
        else:
            details.append(f"Asset-heavy: {asset_to_rev:.2f}x revenue — FX translation risk on foreign assets")
    else:
        details.append("Asset efficiency: insufficient data")

    details.append("Note: direct international revenue split unavailable — proxies used")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


###############################################################################
# LLM generation
###############################################################################

def _generate_macro_exposure_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MacroExposureSignal:
    """Generate a macro exposure signal grounded strictly in the computed metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined macro exposure analyst. Your mandate:
                - Identify companies that are resilient or vulnerable across macro regimes
                - Interest rate sensitivity is primarily a function of debt load and interest coverage
                - Inflation resilience requires pricing power: stable or expanding gross margins
                - FX dependency creates earnings volatility that is hard to predict or hedge
                - A company with low debt, strong margins, and stable domestic revenue is a macro-resilient compounder
                - High macro exposure does not always mean bearish — it means the thesis depends on getting the macro right

                When providing your reasoning, be specific by:
                1. Assessing interest rate vulnerability — what happens to this company if rates rise 200bps?
                2. Evaluating inflation resilience — can this company protect margins in a cost-push environment?
                3. Estimating FX risk — how much earnings volatility comes from currency moves?
                4. Summarizing the composite macro exposure score and what it implies for the investment thesis
                5. Concluding with a clear stance: is macro exposure a headwind, tailwind, or neutral factor?
                """,
            ),
            (
                "human",
                """Based on the following data, generate a macro exposure signal for {ticker}:

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

    def create_default_macro_exposure_signal():
        return MacroExposureSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=MacroExposureSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_macro_exposure_signal,
    )
