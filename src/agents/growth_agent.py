from __future__ import annotations

"""Growth Agent

Implements a growth-focused analysis methodology with LLM summary.
"""

import json
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
    get_insider_trades,
)


class GrowthSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def growth_analyst_agent(state: AgentState, agent_id: str = "growth_analyst_agent"):
    """Run growth analysis across tickers and write signals back to state."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    growth_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial data")

        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=12,
            api_key=api_key,
        )
        if not financial_metrics or len(financial_metrics) < 4:
            progress.update_status(agent_id, ticker, "Failed: Not enough financial metrics")
            continue

        most_recent_metrics = financial_metrics[0]

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing growth trends")
        growth_trends = analyze_growth_trends(financial_metrics)

        progress.update_status(agent_id, ticker, "Analyzing valuation")
        valuation_metrics = analyze_valuation(most_recent_metrics)

        progress.update_status(agent_id, ticker, "Analyzing margin trends")
        margin_trends = analyze_margin_trends(financial_metrics)

        progress.update_status(agent_id, ticker, "Analyzing insider conviction")
        insider_conviction = analyze_insider_conviction(insider_trades)

        progress.update_status(agent_id, ticker, "Checking financial health")
        financial_health = check_financial_health(most_recent_metrics)

        # ------------------------------------------------------------------
        # Aggregate & signal
        # ------------------------------------------------------------------
        scores = {
            "growth": growth_trends["score"],
            "valuation": valuation_metrics["score"],
            "margins": margin_trends["score"],
            "insider": insider_conviction["score"],
            "health": financial_health["score"],
        }

        weights = {
            "growth": 0.40,
            "valuation": 0.25,
            "margins": 0.15,
            "insider": 0.10,
            "health": 0.10,
        }

        weighted_score = sum(scores[key] * weights[key] for key in scores)

        if weighted_score > 0.6:
            pre_signal = "bullish"
        elif weighted_score < 0.4:
            pre_signal = "bearish"
        else:
            pre_signal = "neutral"

        pre_confidence = round(abs(weighted_score - 0.5) * 2 * 100)

        reasoning = {
            "historical_growth": growth_trends,
            "growth_valuation": valuation_metrics,
            "margin_expansion": margin_trends,
            "insider_conviction": insider_conviction,
            "financial_health": financial_health,
            "pre_signal": pre_signal,
            "pre_confidence": pre_confidence,
            "weighted_score": round(weighted_score, 2),
        }

        # ------------------------------------------------------------------
        # LLM summary
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Generating growth analysis")
        growth_output = generate_growth_output(
            ticker=ticker,
            analysis_data=reasoning,
            state=state,
            agent_id=agent_id,
        )

        growth_analysis[ticker] = {
            "signal": growth_output.signal,
            "confidence": growth_output.confidence,
            "reasoning": growth_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=growth_output.reasoning)

    msg = HumanMessage(content=json.dumps(growth_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(growth_analysis, "Growth Analysis Agent")

    state["data"]["analyst_signals"][agent_id] = growth_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [msg], "data": data}


# ---------------------------------------------------------------------------
# LLM Output Generator
# ---------------------------------------------------------------------------

def generate_growth_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> GrowthSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a growth-focused equity analyst. Your decisions are driven by:
            - Revenue and EPS growth rates and acceleration
            - Margin expansion trends
            - Valuation relative to growth (PEG, P/S)
            - Financial health to sustain growth

            When providing reasoning:
            1. Lead with revenue and EPS growth rates and whether they are accelerating
            2. Comment on margin trends — expanding margins amplify growth quality
            3. Reference PEG and P/S ratios relative to growth
            4. Note financial health — high debt can derail growth stories
            5. Conclude with a clear growth thesis or concern

            Return JSON only.""",
        ),
        (
            "human",
            """Based on the following growth analysis for {ticker}, generate a signal.

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
        return GrowthSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in growth analysis, defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=GrowthSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _calculate_trend(data: list) -> float:
    """Calculates the slope of the trend line for the given data."""
    clean_data = [d for d in data if d is not None]
    if len(clean_data) < 2:
        return 0.0

    x = list(range(len(clean_data)))
    y = clean_data

    try:
        n = len(y)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(i * j for i, j in zip(x, y))
        sum_x2 = sum(i ** 2 for i in x)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    except ZeroDivisionError:
        return 0.0


def analyze_growth_trends(metrics: list) -> dict:
    """Analyzes historical growth trends from FinancialMetrics objects."""
    rev_growth = [m.revenue_growth for m in metrics]
    eps_growth = [m.earnings_per_share_growth for m in metrics]
    fcf_growth = [m.free_cash_flow_growth for m in metrics]

    rev_trend = _calculate_trend(rev_growth)
    eps_trend = _calculate_trend(eps_growth)
    fcf_trend = _calculate_trend(fcf_growth)

    score = 0.0

    if rev_growth[0] is not None:
        if rev_growth[0] > 0.20:
            score += 0.4
        elif rev_growth[0] > 0.10:
            score += 0.2
        if rev_trend > 0:
            score += 0.1

    if eps_growth[0] is not None:
        if eps_growth[0] > 0.20:
            score += 0.25
        elif eps_growth[0] > 0.10:
            score += 0.1
        if eps_trend > 0:
            score += 0.05

    if fcf_growth[0] is not None:
        if fcf_growth[0] > 0.15:
            score += 0.1

    return {
        "score": min(score, 1.0),
        "revenue_growth": rev_growth[0],
        "revenue_trend": rev_trend,
        "eps_growth": eps_growth[0],
        "eps_trend": eps_trend,
        "fcf_growth": fcf_growth[0],
        "fcf_trend": fcf_trend,
    }


def analyze_valuation(metrics) -> dict:
    """Analyzes valuation from a growth perspective."""
    peg_ratio = metrics.peg_ratio
    ps_ratio = metrics.price_to_sales_ratio

    score = 0.0

    if peg_ratio is not None:
        if peg_ratio < 1.0:
            score += 0.5
        elif peg_ratio < 2.0:
            score += 0.25

    if ps_ratio is not None:
        if ps_ratio < 2.0:
            score += 0.5
        elif ps_ratio < 5.0:
            score += 0.25

    return {
        "score": min(score, 1.0),
        "peg_ratio": peg_ratio,
        "price_to_sales_ratio": ps_ratio,
    }


def analyze_margin_trends(metrics: list) -> dict:
    """Analyzes historical margin trends from FinancialMetrics objects."""
    gross_margins = [m.gross_margin for m in metrics]
    operating_margins = [m.operating_margin for m in metrics]
    net_margins = [m.net_margin for m in metrics]

    gm_trend = _calculate_trend(gross_margins)
    om_trend = _calculate_trend(operating_margins)
    nm_trend = _calculate_trend(net_margins)

    score = 0.0

    if gross_margins[0] is not None:
        if gross_margins[0] > 0.5:
            score += 0.2
        if gm_trend > 0:
            score += 0.2

    if operating_margins[0] is not None:
        if operating_margins[0] > 0.15:
            score += 0.2
        if om_trend > 0:
            score += 0.2

    if nm_trend > 0:
        score += 0.2

    return {
        "score": min(score, 1.0),
        "gross_margin": gross_margins[0],
        "gross_margin_trend": gm_trend,
        "operating_margin": operating_margins[0],
        "operating_margin_trend": om_trend,
        "net_margin": net_margins[0],
        "net_margin_trend": nm_trend,
    }


def analyze_insider_conviction(trades: list) -> dict:
    """Analyzes insider trading activity safely."""
    if not trades:
        return {
            "score": 0.5,
            "net_flow_ratio": 0,
            "buys": 0,
            "sells": 0,
            "details": "No insider trade data available — defaulting to neutral",
        }

    buys = sum(
        getattr(t, "transaction_value", 0) or 0
        for t in trades
        if (getattr(t, "transaction_shares", None) or 0) > 0
        and (getattr(t, "transaction_value", None) is not None)
    )
    sells = sum(
        abs(getattr(t, "transaction_value", 0) or 0)
        for t in trades
        if (getattr(t, "transaction_shares", None) or 0) < 0
        and (getattr(t, "transaction_value", None) is not None)
    )

    total = buys + sells
    net_flow_ratio = (buys - sells) / total if total > 0 else 0

    if net_flow_ratio > 0.5:
        score = 1.0
    elif net_flow_ratio > 0.1:
        score = 0.7
    elif net_flow_ratio > -0.1:
        score = 0.5
    else:
        score = 0.2

    return {
        "score": score,
        "net_flow_ratio": net_flow_ratio,
        "buys": buys,
        "sells": sells,
    }


def check_financial_health(metrics) -> dict:
    """Checks the company's financial health from FinancialMetrics object."""
    debt_to_equity = metrics.debt_to_equity
    current_ratio = metrics.current_ratio

    score = 1.0

    if debt_to_equity is not None:
        if debt_to_equity > 1.5:
            score -= 0.5
        elif debt_to_equity > 0.8:
            score -= 0.2

    if current_ratio is not None:
        if current_ratio < 1.0:
            score -= 0.5
        elif current_ratio < 1.5:
            score -= 0.2

    return {
        "score": max(score, 0.0),
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
    }