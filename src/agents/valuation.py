from __future__ import annotations

"""Valuation Agent

Implements four complementary valuation methodologies and aggregates them with
configurable weights, then generates an LLM summary in analyst voice.
"""

import json
import statistics
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)


class ValuationSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def _safe_get(obj, attr: str):
    """Safely get an attribute from a Pydantic model or dict."""
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr)
    return None


def valuation_analyst_agent(state: AgentState, agent_id: str = "valuation_analyst_agent"):
    """Run valuation across tickers and write signals back to state."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    valuation_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")

        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )
        if not financial_metrics:
            progress.update_status(agent_id, ticker, "Failed: No financial metrics found")
            continue
        most_recent_metrics = financial_metrics[0]

        progress.update_status(agent_id, ticker, "Gathering comprehensive line items")
        line_items = search_line_items(
            ticker=ticker,
            line_items=[
                "free_cash_flow",
                "net_income",
                "depreciation_and_amortization",
                "capital_expenditure",
                "total_debt",
                "cash_and_equivalents",
                "interest_expense",
                "revenue",
                "operating_income",
                "ebit",
                "ebitda",
                "shareholders_equity",
                "current_assets",
                "current_liabilities",
            ],
            end_date=end_date,
            period="ttm",
            limit=8,
            api_key=api_key,
        )
        if len(line_items) < 2:
            progress.update_status(agent_id, ticker, "Failed: Insufficient financial line items")
            continue

        li_curr = line_items[0]
        li_prev = line_items[1]

        # Working capital change (computed from current/current liabilities)
        curr_assets = _safe_get(li_curr, "current_assets")
        curr_liab = _safe_get(li_curr, "current_liabilities")
        prev_assets = _safe_get(li_prev, "current_assets")
        prev_liab = _safe_get(li_prev, "current_liabilities")

        if all(v is not None for v in [curr_assets, curr_liab, prev_assets, prev_liab]):
            wc_change = (curr_assets - curr_liab) - (prev_assets - prev_liab)
        else:
            wc_change = 0

        # ------------------------------------------------------------------
        # Valuation models
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Calculating owner earnings value")
        owner_val = calculate_owner_earnings_value(
            net_income=_safe_get(li_curr, "net_income"),
            depreciation=_safe_get(li_curr, "depreciation_and_amortization"),
            capex=_safe_get(li_curr, "capital_expenditure"),
            working_capital_change=wc_change,
            growth_rate=most_recent_metrics.earnings_growth or 0.05,
        )

        progress.update_status(agent_id, ticker, "Calculating WACC and enhanced DCF")
        wacc = calculate_wacc(
            market_cap=most_recent_metrics.market_cap or 0,
            total_debt=_safe_get(li_curr, "total_debt"),
            cash=_safe_get(li_curr, "cash_and_equivalents"),
            interest_coverage=most_recent_metrics.interest_coverage,
            debt_to_equity=most_recent_metrics.debt_to_equity,
        )

        fcf_history = [
            _safe_get(li, "free_cash_flow")
            for li in line_items
            if _safe_get(li, "free_cash_flow") is not None
        ]

        dcf_results = calculate_dcf_scenarios(
            fcf_history=fcf_history,
            growth_metrics={
                "revenue_growth": most_recent_metrics.revenue_growth,
                "fcf_growth": most_recent_metrics.free_cash_flow_growth,
                "earnings_growth": most_recent_metrics.earnings_growth,
            },
            wacc=wacc,
            market_cap=most_recent_metrics.market_cap or 0,
            revenue_growth=most_recent_metrics.revenue_growth,
        )
        dcf_val = dcf_results["expected_value"]

        progress.update_status(agent_id, ticker, "Calculating EV/EBITDA value")
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics)

        progress.update_status(agent_id, ticker, "Calculating residual income value")
        rim_val = calculate_residual_income_value(
            market_cap=most_recent_metrics.market_cap,
            net_income=_safe_get(li_curr, "net_income"),
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=most_recent_metrics.book_value_growth or 0.03,
        )

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)
        if not market_cap:
            progress.update_status(agent_id, ticker, "Failed: Market cap unavailable")
            continue

        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        total_weight = sum(v["weight"] for v in method_values.values() if v["value"] > 0)
        if total_weight == 0:
            progress.update_status(agent_id, ticker, "Failed: All valuation methods zero")
            continue

        for v in method_values.values():
            v["gap"] = (v["value"] - market_cap) / market_cap if v["value"] > 0 else None

        weighted_gap = sum(
            v["weight"] * v["gap"] for v in method_values.values() if v["gap"] is not None
        ) / total_weight

        pre_signal = "bullish" if weighted_gap > 0.15 else "bearish" if weighted_gap < -0.15 else "neutral"
        pre_confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100))

        reasoning = {}
        for m, vals in method_values.items():
            if vals["value"] > 0:
                base_details = (
                    f"Value: ${vals['value']:,.2f}, Market Cap: ${market_cap:,.2f}, "
                    f"Gap: {vals['gap']:.1%}, Weight: {vals['weight']*100:.0f}%"
                )
                if m == "dcf":
                    enhanced_details = (
                        f"{base_details} | "
                        f"WACC: {wacc:.1%}, Bear: ${dcf_results['downside']:,.2f}, "
                        f"Bull: ${dcf_results['upside']:,.2f}"
                    )
                else:
                    enhanced_details = base_details

                reasoning[f"{m}_analysis"] = {
                    "signal": (
                        "bullish" if vals["gap"] and vals["gap"] > 0.15 else
                        "bearish" if vals["gap"] and vals["gap"] < -0.15 else "neutral"
                    ),
                    "details": enhanced_details,
                }

        reasoning["dcf_scenario_analysis"] = {
            "bear_case": f"${dcf_results['downside']:,.2f}",
            "base_case": f"${dcf_results['scenarios']['base']:,.2f}",
            "bull_case": f"${dcf_results['upside']:,.2f}",
            "wacc_used": f"{wacc:.1%}",
            "fcf_periods_analyzed": len(fcf_history),
        }

        reasoning["weighted_gap"] = f"{weighted_gap:.1%}"
        reasoning["pre_signal"] = pre_signal
        reasoning["pre_confidence"] = pre_confidence

        # ------------------------------------------------------------------
        # LLM summary
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Generating valuation analysis")
        valuation_output = generate_valuation_output(
            ticker=ticker,
            analysis_data=reasoning,
            state=state,
            agent_id=agent_id,
        )

        valuation_analysis[ticker] = {
            "signal": valuation_output.signal,
            "confidence": valuation_output.confidence,
            "reasoning": valuation_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=valuation_output.reasoning)

    msg = HumanMessage(content=json.dumps(valuation_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(valuation_analysis, "Valuation Analysis Agent")

    state["data"]["analyst_signals"][agent_id] = valuation_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [msg], "data": data}


# ---------------------------------------------------------------------------
# LLM Output Generator
# ---------------------------------------------------------------------------

def generate_valuation_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> ValuationSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined valuation analyst. Your decisions are driven purely by 
            intrinsic value vs current price. You use four methods: DCF, Owner Earnings, 
            EV/EBITDA, and Residual Income — each weighted by reliability.

            When providing reasoning:
            1. Lead with the weighted gap between intrinsic value and market cap
            2. Highlight which valuation method gives the strongest signal
            3. Reference WACC and DCF scenario range
            4. Note any method disagreements
            5. Conclude with a clear, number-anchored stance

            Return JSON only.""",
        ),
        (
            "human",
            """Based on the following valuation analysis for {ticker}, generate a signal.

            Analysis:
            {analysis_data}

            Return exactly:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}""",
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data, indent=2),
    })

    def default_signal():
        return ValuationSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in valuation analysis, defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=ValuationSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )


# ---------------------------------------------------------------------------
# Valuation Helper Functions
# ---------------------------------------------------------------------------

def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, working_capital_change]):
        return 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    pv = 0.0
    for yr in range(1, num_years + 1):
        future = owner_earnings * (1 + growth_rate) ** yr
        pv += future / (1 + required_return) ** yr

    terminal_growth = min(growth_rate, 0.03)
    term_val = (owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)) / (
        required_return - terminal_growth
    )
    pv_term = term_val / (1 + required_return) ** num_years

    return (pv + pv_term) * (1 - margin_of_safety)


def calculate_ev_ebitda_value(financial_metrics: list) -> float:
    if not financial_metrics:
        return 0
    m0 = financial_metrics[0]
    if not (m0.enterprise_value and m0.enterprise_value_to_ebitda_ratio):
        return 0
    if m0.enterprise_value_to_ebitda_ratio == 0:
        return 0

    ebitda_now = m0.enterprise_value / m0.enterprise_value_to_ebitda_ratio
    med_mult = statistics.median([
        m.enterprise_value_to_ebitda_ratio
        for m in financial_metrics
        if m.enterprise_value_to_ebitda_ratio
    ])
    ev_implied = med_mult * ebitda_now
    net_debt = (m0.enterprise_value or 0) - (m0.market_cap or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
) -> float:
    if not (market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0):
        return 0

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    term_ri = ri0 * (1 + book_value_growth) ** (num_years + 1) / (
        cost_of_equity - terminal_growth_rate
    )
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    return (book_val + pv_ri + pv_term) * 0.8


def calculate_wacc(
    market_cap: float,
    total_debt: float | None,
    cash: float | None,
    interest_coverage: float | None,
    debt_to_equity: float | None,
    beta_proxy: float = 1.0,
    risk_free_rate: float = 0.045,
    market_risk_premium: float = 0.06,
) -> float:
    cost_of_equity = risk_free_rate + beta_proxy * market_risk_premium

    if interest_coverage and interest_coverage > 0:
        cost_of_debt = max(risk_free_rate + 0.01, risk_free_rate + (10 / interest_coverage))
    else:
        cost_of_debt = risk_free_rate + 0.05

    net_debt = max((total_debt or 0) - (cash or 0), 0)
    total_value = market_cap + net_debt

    if total_value > 0:
        weight_equity = market_cap / total_value
        weight_debt = net_debt / total_value
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * 0.75)
    else:
        wacc = cost_of_equity

    return min(max(wacc, 0.06), 0.20)


def calculate_fcf_volatility(fcf_history: list[float]) -> float:
    if len(fcf_history) < 3:
        return 0.5

    positive_fcf = [fcf for fcf in fcf_history if fcf > 0]
    if len(positive_fcf) < 2:
        return 0.8

    try:
        mean_fcf = statistics.mean(positive_fcf)
        std_fcf = statistics.stdev(positive_fcf)
        return min(std_fcf / mean_fcf, 1.0) if mean_fcf > 0 else 0.8
    except Exception:
        return 0.5


def calculate_enhanced_dcf_value(
    fcf_history: list[float],
    growth_metrics: dict,
    wacc: float,
    market_cap: float,
    revenue_growth: float | None = None,
) -> float:
    if not fcf_history or fcf_history[0] <= 0:
        return 0

    fcf_current = fcf_history[0]
    fcf_avg_3yr = sum(fcf_history[:3]) / min(3, len(fcf_history))
    fcf_volatility = calculate_fcf_volatility(fcf_history)

    high_growth = min(revenue_growth or 0.05, 0.25) if revenue_growth else 0.05
    if market_cap > 50_000_000_000:
        high_growth = min(high_growth, 0.10)

    transition_growth = (high_growth + 0.03) / 2
    terminal_growth = min(0.03, high_growth * 0.6)

    pv = 0.0
    base_fcf = max(fcf_current, fcf_avg_3yr * 0.85)

    for year in range(1, 4):
        fcf_projected = base_fcf * (1 + high_growth) ** year
        pv += fcf_projected / (1 + wacc) ** year

    for year in range(4, 8):
        transition_rate = transition_growth * (8 - year) / 4
        fcf_projected = base_fcf * (1 + high_growth) ** 3 * (1 + transition_rate) ** (year - 3)
        pv += fcf_projected / (1 + wacc) ** year

    final_fcf = base_fcf * (1 + high_growth) ** 3 * (1 + transition_growth) ** 4
    if wacc <= terminal_growth:
        terminal_growth = wacc * 0.8
    terminal_value = (final_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** 7

    quality_factor = max(0.7, 1 - (fcf_volatility * 0.5))
    return (pv + pv_terminal) * quality_factor


def calculate_dcf_scenarios(
    fcf_history: list[float],
    growth_metrics: dict,
    wacc: float,
    market_cap: float,
    revenue_growth: float | None = None,
) -> dict:
    scenarios = {
        "bear": {"growth_adj": 0.5, "wacc_adj": 1.2},
        "base": {"growth_adj": 1.0, "wacc_adj": 1.0},
        "bull": {"growth_adj": 1.5, "wacc_adj": 0.9},
    }

    results = {}
    base_revenue_growth = revenue_growth or 0.05

    for scenario, adjustments in scenarios.items():
        adjusted_revenue_growth = base_revenue_growth * adjustments["growth_adj"]
        adjusted_wacc = wacc * adjustments["wacc_adj"]
        results[scenario] = calculate_enhanced_dcf_value(
            fcf_history=fcf_history,
            growth_metrics=growth_metrics,
            wacc=adjusted_wacc,
            market_cap=market_cap,
            revenue_growth=adjusted_revenue_growth,
        )

    expected_value = (
        results["bear"] * 0.2
        + results["base"] * 0.6
        + results["bull"] * 0.2
    )

    return {
        "scenarios": results,
        "expected_value": expected_value,
        "range": results["bull"] - results["bear"],
        "upside": results["bull"],
        "downside": results["bear"],
    }