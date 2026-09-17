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


class PeterLynchSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def peter_lynch_agent(
    state: AgentState,
    agent_id: str = "peter_lynch_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Peter Lynch's GARP (Growth At a Reasonable Price) principles.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    lynch_analysis = {}
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
                "revenue", "net_income", "earnings_per_share", "free_cash_flow",
                "total_debt", "shareholders_equity", "operating_income",
                "capital_expenditure", "outstanding_shares", "gross_margin",
            ],
            end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing PEG ratio")
        peg_analysis = analyze_peg_ratio(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing earnings growth")
        earnings_analysis = analyze_earnings_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balance_analysis = analyze_balance_sheet(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing Lynch story")
        story_analysis = analyze_lynch_story(metrics, financial_line_items, company_news)

        total_score = (
            peg_analysis["score"] +
            earnings_analysis["score"] +
            balance_analysis["score"] +
            story_analysis["score"]
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
            "peg_analysis": peg_analysis,
            "earnings_analysis": earnings_analysis,
            "balance_analysis": balance_analysis,
            "story_analysis": story_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Peter Lynch analysis")
        lynch_output = generate_peter_lynch_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        lynch_analysis[ticker] = {
            "signal": lynch_output.signal,
            "confidence": lynch_output.confidence,
            "reasoning": lynch_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": lynch_output.signal,
            "confidence": lynch_output.confidence,
            "key_findings": [
                f"Lynch score: {total_score}/{max_possible_score}",
                f"PEG: {peg_analysis.get('details', 'N/A')[:80]}",
                f"Earnings: {earnings_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": lynch_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=lynch_output.reasoning)

    message = HumanMessage(content=json.dumps(lynch_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(lynch_analysis, "Peter Lynch Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = lynch_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_peg_ratio(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for PEG analysis"}

    latest_metrics = metrics[0]
    pe_ratio = latest_metrics.price_to_earnings_ratio

    eps_values = [item.earnings_per_share for item in financial_line_items if item.earnings_per_share is not None]
    earnings_growth_rate = None

    if len(eps_values) >= 2 and eps_values[-1] and eps_values[-1] > 0:
        years = len(eps_values) - 1
        earnings_growth_rate = ((eps_values[0] / eps_values[-1]) ** (1 / years) - 1) * 100

    if pe_ratio and earnings_growth_rate and earnings_growth_rate > 0:
        peg_ratio = pe_ratio / earnings_growth_rate
        details.append(f"PEG Ratio: {peg_ratio:.2f} (P/E: {pe_ratio:.1f}, Growth: {earnings_growth_rate:.1f}%)")
        if peg_ratio < 0.5:
            score += 5
            details.append("Excellent: PEG < 0.5, significantly undervalued relative to growth")
        elif peg_ratio < 1.0:
            score += 4
            details.append("Good: PEG < 1.0, undervalued relative to growth")
        elif peg_ratio < 1.5:
            score += 2
            details.append("Fair: PEG between 1.0-1.5, reasonably valued")
        elif peg_ratio < 2.0:
            score += 1
            details.append("Caution: PEG between 1.5-2.0, slightly overvalued")
        else:
            details.append(f"Overvalued: PEG > 2.0 ({peg_ratio:.2f})")
    elif pe_ratio:
        details.append(f"P/E: {pe_ratio:.1f}, but insufficient earnings growth data for PEG")
        if pe_ratio < 15:
            score += 2
            details.append("Low P/E suggests potential value")
        elif pe_ratio > 30:
            details.append("High P/E without growth confirmation is risky")
    else:
        details.append("Unable to calculate PEG ratio — insufficient P/E or growth data")

    return {"score": score, "details": "; ".join(details)}


def analyze_earnings_growth(financial_line_items: list) -> dict:
    score = 0
    details = []

    if not financial_line_items or len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for earnings growth analysis"}

    eps_values = [item.earnings_per_share for item in financial_line_items if item.earnings_per_share is not None]
    revenue_values = [item.revenue for item in financial_line_items if item.revenue is not None]

    if len(eps_values) >= 3:
        positive_eps = sum(1 for e in eps_values if e > 0)
        if positive_eps == len(eps_values):
            score += 3
            details.append("Consistently positive earnings across all periods")
        elif positive_eps >= len(eps_values) * 0.8:
            score += 2
            details.append("Mostly positive earnings with minor exceptions")
        else:
            details.append(f"Inconsistent earnings: positive in only {positive_eps}/{len(eps_values)} periods")

        if len(eps_values) >= 2 and eps_values[-1] and eps_values[-1] > 0:
            years = len(eps_values) - 1
            eps_cagr = ((eps_values[0] / eps_values[-1]) ** (1 / years) - 1) * 100
            if eps_cagr > 25:
                score += 4
                details.append(f"Exceptional EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 15:
                score += 3
                details.append(f"Strong EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 10:
                score += 2
                details.append(f"Good EPS CAGR: {eps_cagr:.1f}%")
            elif eps_cagr > 5:
                score += 1
                details.append(f"Moderate EPS CAGR: {eps_cagr:.1f}%")
            else:
                details.append(f"Slow EPS CAGR: {eps_cagr:.1f}%")

    if len(revenue_values) >= 3:
        if revenue_values[-1] and revenue_values[-1] > 0:
            years = len(revenue_values) - 1
            rev_cagr = ((revenue_values[0] / revenue_values[-1]) ** (1 / years) - 1) * 100
            if rev_cagr > 15:
                score += 2
                details.append(f"Strong revenue CAGR: {rev_cagr:.1f}%")
            elif rev_cagr > 8:
                score += 1
                details.append(f"Good revenue CAGR: {rev_cagr:.1f}%")
            else:
                details.append(f"Slow revenue growth: {rev_cagr:.1f}%")

    return {"score": score, "details": "; ".join(details)}


def analyze_balance_sheet(financial_line_items: list) -> dict:
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for balance sheet analysis"}

    latest = financial_line_items[0]
    total_debt = latest.total_debt
    equity = latest.shareholders_equity
    fcf = latest.free_cash_flow

    if total_debt is not None and equity is not None and equity > 0:
        de_ratio = total_debt / equity
        if de_ratio < 0.3:
            score += 3
            details.append(f"Excellent balance sheet: D/E = {de_ratio:.2f}")
        elif de_ratio < 0.8:
            score += 2
            details.append(f"Good balance sheet: D/E = {de_ratio:.2f}")
        elif de_ratio < 1.5:
            score += 1
            details.append(f"Moderate leverage: D/E = {de_ratio:.2f}")
        else:
            details.append(f"High leverage: D/E = {de_ratio:.2f}")
    else:
        details.append("Insufficient debt/equity data")

    if fcf is not None:
        if fcf > 0:
            score += 2
            details.append(f"Positive FCF: {fcf:,.0f}")
        else:
            details.append(f"Negative FCF: {fcf:,.0f}")
    else:
        details.append("FCF data not available")

    return {"score": score, "details": "; ".join(details)}


def analyze_lynch_story(metrics: list, financial_line_items: list, company_news: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for Lynch story analysis"}

    latest_metrics = metrics[0]

    if latest_metrics.gross_margin and latest_metrics.gross_margin > 0.4:
        score += 2
        details.append(f"High gross margin ({latest_metrics.gross_margin:.1%}) suggests strong business model")
    elif latest_metrics.gross_margin and latest_metrics.gross_margin > 0.2:
        score += 1
        details.append(f"Decent gross margin ({latest_metrics.gross_margin:.1%})")

    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"Strong ROE ({latest_metrics.return_on_equity:.1%})")
    elif latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.10:
        score += 1
        details.append(f"Decent ROE ({latest_metrics.return_on_equity:.1%})")

    if company_news:
        details.append(f"Recent news available ({len(company_news)} articles)")
        score += 1

    return {"score": score, "details": "; ".join(details) if details else "Limited story analysis data"}


def generate_peter_lynch_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> PeterLynchSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your Lynch-style GARP assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Peter Lynch AI agent, making investment decisions using his GARP principles:
            1. PEG ratio is key — prefer PEG < 1.0 for growth at a reasonable price
            2. Invest in what you know — understand the business model
            3. Look for consistent earnings growth, ideally 15-30% annually
            4. Avoid companies with excessive debt
            5. Categorize companies: slow growers, stalwarts, fast growers, cyclicals, turnarounds, asset plays
            6. "Ten-bagger" potential matters — look for room to grow

            Be specific about PEG ratio, earnings growth rates, and Lynch's stock categories.
            Return a rational recommendation with confidence (0-100) and reasoning.""",
        ),
        (
            "human",
            """Based on the following analysis, create a Lynch-style investment signal:

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
        return PeterLynchSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=PeterLynchSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )