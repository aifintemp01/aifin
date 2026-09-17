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


class CapitalAllocationSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def capital_allocation_agent(
    state: AgentState,
    agent_id: str = "capital_allocation_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using a comprehensive capital allocation framework.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    capital_allocation_analysis: dict[str, dict] = {}
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

        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            [
                "dividends_and_other_cash_distributions", "outstanding_shares",
                "issuance_or_purchase_of_equity_shares", "total_debt",
                "cash_and_equivalents", "shareholders_equity", "net_income",
                "free_cash_flow", "capital_expenditure", "operating_income",
                "ebit", "return_on_invested_capital",
            ],
            end_date,
            period="annual",
            limit=6,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing dividend policy")
        dividend_analysis = _analyze_dividend_policy(line_items)

        progress.update_status(agent_id, ticker, "Analyzing share dilution")
        dilution_analysis = _analyze_share_dilution(line_items)

        progress.update_status(agent_id, ticker, "Analyzing buyback activity")
        buyback_analysis = _analyze_buyback_yield(line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing return on incremental capital")
        roic_delta_analysis = _analyze_return_on_incremental_capital(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing debt management")
        debt_management_analysis = _analyze_debt_management(line_items)

        total_score = (
            roic_delta_analysis["score"] * 0.30
            + dilution_analysis["score"] * 0.25
            + buyback_analysis["score"] * 0.20
            + debt_management_analysis["score"] * 0.15
            + dividend_analysis["score"] * 0.10
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
            "dividend_analysis": dividend_analysis,
            "dilution_analysis": dilution_analysis,
            "buyback_analysis": buyback_analysis,
            "roic_delta_analysis": roic_delta_analysis,
            "debt_management_analysis": debt_management_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating capital allocation analysis")
        ca_output = _generate_capital_allocation_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        capital_allocation_analysis[ticker] = {
            "signal": ca_output.signal,
            "confidence": ca_output.confidence,
            "reasoning": ca_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": ca_output.signal,
            "confidence": ca_output.confidence,
            "key_findings": [
                f"CA score: {total_score:.1f}/{max_score}",
                f"ROIC: {roic_delta_analysis.get('details', 'N/A')[:80]}",
                f"Debt: {debt_management_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": ca_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=ca_output.reasoning)

    message = HumanMessage(content=json.dumps(capital_allocation_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(capital_allocation_analysis, "Capital Allocation Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = capital_allocation_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Sub-analysis helpers (unchanged from original) ────────────────────────────

def _latest(line_items: list):
    return line_items[0] if line_items else None


def _analyze_dividend_policy(line_items: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    dividends = [
        abs(item.dividends_and_other_cash_distributions or 0)
        for item in line_items
        if item.dividends_and_other_cash_distributions is not None
    ]
    paying_periods = [d for d in dividends if d > 0]

    if len(paying_periods) < 2:
        return {
            "score": 5,
            "max_score": max_score,
            "details": "No dividend history — company reinvests capital (neutral, not negative)",
            "dividend_cagr": None,
        }

    latest_div = paying_periods[0]
    oldest_div = paying_periods[-1]
    n = len(paying_periods) - 1

    if oldest_div > 0 and latest_div > 0:
        div_cagr = (latest_div / oldest_div) ** (1 / n) - 1
        if div_cagr > 0.10:
            score += 4
            details.append(f"Strong dividend CAGR: {div_cagr:.1%} over {n}Y")
        elif div_cagr > 0.05:
            score += 3
            details.append(f"Healthy dividend CAGR: {div_cagr:.1%} over {n}Y")
        elif div_cagr > 0:
            score += 2
            details.append(f"Modest dividend growth: {div_cagr:.1%} over {n}Y")
        elif div_cagr == 0:
            score += 1
            details.append("Flat dividend — maintained but not growing")
        else:
            details.append(f"Dividend cut: {div_cagr:.1%} CAGR")
    else:
        div_cagr = None
        details.append("Dividend CAGR: invalid base dividend value")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "dividend_cagr": round(div_cagr, 4) if div_cagr is not None else None,
    }


def _analyze_share_dilution(line_items: list) -> dict:
    max_score = 5
    score = 0
    details: list[str] = []

    share_counts = [
        item.outstanding_shares
        for item in line_items
        if item.outstanding_shares is not None
    ]

    dilution_rate = None
    if len(share_counts) >= 2:
        latest = share_counts[0]
        oldest = share_counts[-1]
        n = len(share_counts) - 1
        if oldest > 0 and latest > 0:
            dilution_rate = (latest / oldest) ** (1 / n) - 1
            if dilution_rate < -0.02:
                score += 3
                details.append(f"Active share reduction: {dilution_rate:.1%} CAGR")
            elif dilution_rate < 0:
                score += 2
                details.append(f"Slight share reduction: {dilution_rate:.1%} CAGR")
            elif dilution_rate < 0.01:
                score += 2
                details.append(f"Share count stable: {dilution_rate:.1%} CAGR")
            elif dilution_rate < 0.03:
                score += 1
                details.append(f"Modest dilution: {dilution_rate:.1%} CAGR")
            else:
                details.append(f"Significant dilution: {dilution_rate:.1%} CAGR")
        else:
            details.append("Share count CAGR: invalid base value")
    else:
        details.append("Share dilution: insufficient history")

    latest_item = _latest(line_items)
    equity_activity = latest_item.issuance_or_purchase_of_equity_shares if latest_item else None
    if equity_activity is not None:
        if equity_activity < 0:
            score += 2
            details.append(f"Active buybacks: ${abs(equity_activity):,.0f} returned last period")
        elif equity_activity > 0:
            details.append(f"Equity issuance: ${equity_activity:,.0f}")
        else:
            score += 1
            details.append("No equity issuance or buyback activity last period")
    else:
        details.append("Equity issuance/buyback: data unavailable")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "dilution_rate_cagr": round(dilution_rate, 4) if dilution_rate is not None else None,
    }


def _analyze_buyback_yield(line_items: list, market_cap: float | None) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    equity_activity = latest_item.issuance_or_purchase_of_equity_shares if latest_item else None

    buyback_yield = None
    if equity_activity is not None and equity_activity < 0 and market_cap and market_cap > 0:
        buyback_amount = abs(equity_activity)
        buyback_yield = buyback_amount / market_cap
        if buyback_yield > 0.05:
            score += 4
            details.append(f"High buyback yield: {buyback_yield:.1%}")
        elif buyback_yield > 0.02:
            score += 3
            details.append(f"Solid buyback yield: {buyback_yield:.1%}")
        elif buyback_yield > 0.005:
            score += 2
            details.append(f"Modest buyback yield: {buyback_yield:.1%}")
        else:
            score += 1
            details.append(f"Minimal buyback yield: {buyback_yield:.1%}")
    elif equity_activity is not None and equity_activity >= 0:
        details.append("No buyback activity in most recent period")
    elif market_cap is None:
        details.append("Buyback yield: market cap unavailable")
    else:
        details.append("Buyback yield: equity activity data unavailable")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "buyback_yield": round(buyback_yield, 4) if buyback_yield is not None else None,
    }


def _analyze_return_on_incremental_capital(metrics: list, line_items: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    roic_values = [
        item.return_on_invested_capital
        for item in line_items
        if item.return_on_invested_capital is not None
    ]

    if not roic_values and metrics:
        roic_from_metrics = metrics[0].return_on_invested_capital
        if roic_from_metrics is not None:
            roic_values = [roic_from_metrics]

    if roic_values:
        avg_roic = sum(roic_values) / len(roic_values)
        is_improving = len(roic_values) >= 2 and roic_values[0] >= roic_values[-1]
        trend_label = "improving" if is_improving else "declining"
        if avg_roic > 0.20:
            score += 3
            details.append(f"Exceptional ROIC: avg {avg_roic:.1%} ({trend_label})")
        elif avg_roic > 0.12:
            score += 2
            details.append(f"Strong ROIC: avg {avg_roic:.1%} ({trend_label})")
        elif avg_roic > 0.06:
            score += 1
            details.append(f"Adequate ROIC: avg {avg_roic:.1%} ({trend_label})")
        else:
            details.append(f"Poor ROIC: avg {avg_roic:.1%}")
    else:
        avg_roic = None
        details.append("ROIC: insufficient data")

    nopat_values = [
        item.ebit or item.operating_income
        for item in line_items
        if (item.ebit or item.operating_income) is not None
    ]
    ic_values = []
    for item in line_items:
        equity = item.shareholders_equity
        debt = item.total_debt
        cash = item.cash_and_equivalents
        if equity is not None and debt is not None and cash is not None:
            ic_values.append(equity + debt - cash)

    if len(nopat_values) >= 2 and len(ic_values) >= 2:
        delta_nopat = nopat_values[0] - nopat_values[-1]
        delta_ic = ic_values[0] - ic_values[-1]
        if delta_ic != 0:
            roic_incremental = delta_nopat / abs(delta_ic)
            if roic_incremental > 0.20:
                score += 3
                details.append(f"Excellent incremental ROIC: {roic_incremental:.1%}")
            elif roic_incremental > 0.10:
                score += 2
                details.append(f"Good incremental ROIC: {roic_incremental:.1%}")
            elif roic_incremental > 0:
                score += 1
                details.append(f"Positive incremental ROIC: {roic_incremental:.1%}")
            else:
                details.append(f"Negative incremental ROIC: {roic_incremental:.1%}")
        else:
            details.append("Incremental ROIC: no change in invested capital")
    else:
        details.append("Incremental ROIC: insufficient data")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_roic": round(avg_roic, 4) if avg_roic is not None else None,
    }


def _analyze_debt_management(line_items: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    net_debt_series = []
    for item in line_items:
        debt = item.total_debt
        cash = item.cash_and_equivalents
        if debt is not None and cash is not None:
            net_debt_series.append(debt - cash)

    if len(net_debt_series) < 2:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Net debt trend: insufficient data",
            "net_debt_change": None,
            "latest_net_debt": None,
        }

    latest_net_debt = net_debt_series[0]
    oldest_net_debt = net_debt_series[-1]
    net_debt_change = latest_net_debt - oldest_net_debt

    if latest_net_debt < 0:
        score += 4
        details.append(f"Net cash position: ${abs(latest_net_debt):,.0f}")
    elif net_debt_change < 0:
        reduction_pct = abs(net_debt_change) / abs(oldest_net_debt) if oldest_net_debt != 0 else 0
        if reduction_pct > 0.20:
            score += 3
            details.append(f"Significant debt reduction: net debt down {reduction_pct:.1%}")
        elif reduction_pct > 0.05:
            score += 2
            details.append(f"Moderate debt reduction: net debt down {reduction_pct:.1%}")
        else:
            score += 1
            details.append(f"Slight debt reduction: net debt down {reduction_pct:.1%}")
    else:
        increase_pct = net_debt_change / abs(oldest_net_debt) if oldest_net_debt != 0 else 0
        if increase_pct < 0.10:
            score += 1
            details.append(f"Stable net debt: increased only {increase_pct:.1%}")
        else:
            details.append(f"Rising net debt: up {increase_pct:.1%}")

    details.append(f"Latest net debt: ${latest_net_debt:,.0f}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "net_debt_change": round(net_debt_change, 2),
        "latest_net_debt": round(latest_net_debt, 2),
    }


def _generate_capital_allocation_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> CapitalAllocationSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your capital allocation assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined capital allocation analyst. Your mandate:
            - Return on incremental capital is the single most important metric
            - Share dilution is a silent tax; buybacks at fair value are the most efficient return mechanism
            - A growing dividend signals confidence; a cut signals distress
            - Net debt trend reveals whether management is building or eroding financial optionality
            - Great capital allocators are rare — reward them with conviction

            When providing your reasoning:
            1. Lead with return on incremental capital
            2. Assess the shareholder return policy
            3. Comment on the net debt trend
            4. Evaluate overall ROIC level
            5. Conclude with a clear capital allocation quality verdict""",
        ),
        (
            "human",
            """Based on the following data, generate a capital allocation signal for {ticker}:

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
        return CapitalAllocationSignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=CapitalAllocationSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )