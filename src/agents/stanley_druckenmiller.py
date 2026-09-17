from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items, get_company_news
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class StanleyDruckenmillerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def stanley_druckenmiller_agent(
    state: AgentState,
    agent_id: str = "stanley_druckenmiller_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Stanley Druckenmiller's macro-driven, momentum-focused approach.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    druckenmiller_analysis = {}
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
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue", "net_income", "operating_income", "free_cash_flow",
                "capital_expenditure", "total_debt", "shareholders_equity",
                "cash_and_equivalents", "outstanding_shares", "gross_margin",
                "operating_margin", "earnings_per_share",
            ],
            end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing momentum")
        momentum_analysis = analyze_momentum(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing earnings power")
        earnings_analysis = analyze_earnings_power(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing risk/reward")
        risk_analysis = analyze_risk_reward(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing macro sentiment")
        sentiment_analysis = analyze_sentiment(company_news)

        total_score = (
            momentum_analysis["score"] +
            earnings_analysis["score"] +
            risk_analysis["score"] +
            sentiment_analysis["score"]
        )
        max_possible_score = 20

        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "momentum_analysis": momentum_analysis,
            "earnings_analysis": earnings_analysis,
            "risk_analysis": risk_analysis,
            "sentiment_analysis": sentiment_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Druckenmiller analysis")
        druck_output = generate_druckenmiller_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        druckenmiller_analysis[ticker] = {
            "signal": druck_output.signal,
            "confidence": druck_output.confidence,
            "reasoning": druck_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": druck_output.signal,
            "confidence": druck_output.confidence,
            "key_findings": [
                f"Druckenmiller score: {total_score}/{max_possible_score}",
                f"Momentum: {momentum_analysis.get('details', 'N/A')[:80]}",
                f"Risk/Reward: {risk_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": druck_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=druck_output.reasoning)

    message = HumanMessage(content=json.dumps(druckenmiller_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(druckenmiller_analysis, "Stanley Druckenmiller Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = druckenmiller_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_momentum(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for momentum analysis"}

    revenue_values = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenue_values) >= 3:
        recent_growth = (revenue_values[0] - revenue_values[2]) / abs(revenue_values[2]) if revenue_values[2] else 0
        if recent_growth > 0.20:
            score += 5
            details.append(f"Strong revenue momentum: {recent_growth:.1%} over 2 periods")
        elif recent_growth > 0.10:
            score += 3
            details.append(f"Good revenue momentum: {recent_growth:.1%}")
        elif recent_growth > 0:
            score += 1
            details.append(f"Moderate revenue momentum: {recent_growth:.1%}")
        else:
            details.append(f"Negative revenue momentum: {recent_growth:.1%}")

    eps_values = [item.earnings_per_share for item in financial_line_items if item.earnings_per_share is not None]
    if len(eps_values) >= 2:
        eps_trend = eps_values[0] > eps_values[1] if len(eps_values) >= 2 else False
        if eps_trend:
            score += 3
            details.append("Improving EPS trend — earnings accelerating")
        else:
            details.append("EPS trend is flat or declining")

    return {"score": score, "details": "; ".join(details)}


def analyze_earnings_power(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics:
        return {"score": 0, "details": "Insufficient metrics data"}

    latest = metrics[0]

    if latest.return_on_equity and latest.return_on_equity > 0.20:
        score += 3
        details.append(f"Strong ROE: {latest.return_on_equity:.1%}")
    elif latest.return_on_equity and latest.return_on_equity > 0.10:
        score += 2
        details.append(f"Decent ROE: {latest.return_on_equity:.1%}")
    elif latest.return_on_equity:
        details.append(f"Weak ROE: {latest.return_on_equity:.1%}")

    if latest.operating_margin and latest.operating_margin > 0.20:
        score += 3
        details.append(f"Excellent operating margin: {latest.operating_margin:.1%}")
    elif latest.operating_margin and latest.operating_margin > 0.10:
        score += 2
        details.append(f"Good operating margin: {latest.operating_margin:.1%}")
    elif latest.operating_margin:
        details.append(f"Thin operating margin: {latest.operating_margin:.1%}")

    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if fcf_values and fcf_values[0] > 0:
        score += 2
        details.append(f"Positive FCF: {fcf_values[0]:,.0f}")

    return {"score": score, "details": "; ".join(details)}


def analyze_risk_reward(financial_line_items: list, market_cap: float) -> dict:
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data"}

    latest = financial_line_items[0]
    debt = latest.total_debt
    equity = latest.shareholders_equity
    cash = latest.cash_and_equivalents
    fcf = latest.free_cash_flow

    if debt is not None and equity is not None and equity > 0:
        de_ratio = debt / equity
        if de_ratio < 0.5:
            score += 3
            details.append(f"Low leverage: D/E = {de_ratio:.2f}")
        elif de_ratio < 1.5:
            score += 1
            details.append(f"Moderate leverage: D/E = {de_ratio:.2f}")
        else:
            details.append(f"High leverage: D/E = {de_ratio:.2f}")

    if market_cap and fcf and fcf > 0:
        fcf_yield = fcf / market_cap
        if fcf_yield > 0.06:
            score += 3
            details.append(f"Attractive FCF yield: {fcf_yield:.1%}")
        elif fcf_yield > 0.03:
            score += 1
            details.append(f"Decent FCF yield: {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield: {fcf_yield:.1%}")

    return {"score": score, "details": "; ".join(details)}


def analyze_sentiment(company_news: list) -> dict:
    if not company_news:
        return {"score": 2, "details": "No news data — neutral sentiment assumed"}

    positive = sum(1 for n in company_news if getattr(n, 'sentiment', None) == 'positive')
    negative = sum(1 for n in company_news if getattr(n, 'sentiment', None) == 'negative')
    total = len(company_news)

    if positive > negative * 2:
        score = 4
        details = f"Strongly positive news sentiment: {positive}/{total} positive"
    elif positive > negative:
        score = 3
        details = f"Positive news sentiment: {positive}/{total} positive"
    elif negative > positive:
        score = 1
        details = f"Negative news sentiment: {negative}/{total} negative"
    else:
        score = 2
        details = f"Mixed news sentiment: {positive} positive, {negative} negative"

    return {"score": score, "details": details}


def generate_druckenmiller_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> StanleyDruckenmillerSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your macro-driven assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Stanley Druckenmiller AI agent, making decisions using his principles:
            1. Macro-first: understand the big picture economic context
            2. Momentum matters: follow earnings and revenue acceleration
            3. Asymmetric risk/reward: maximize upside, minimize downside
            4. Concentrate in high-conviction positions
            5. Cut losses quickly; let winners run
            6. Liquidity and FCF generation are key quality signals

            Be specific about momentum, earnings acceleration, and risk/reward dynamics.
            Return a rational recommendation with confidence (0-100) and reasoning.""",
        ),
        (
            "human",
            """Based on the following analysis, create a Druckenmiller-style investment signal:

Analysis Data for {ticker}:
{analysis_data}
{upstream_context}
Return JSON exactly:
{{
  "signal": "bullish" or "bearish" or "neutral",
  "confidence": float (0-100),
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
        return StanleyDruckenmillerSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=StanleyDruckenmillerSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )