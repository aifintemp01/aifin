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


class PhilFisherSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def phil_fisher_agent(
    state: AgentState,
    agent_id: str = "phil_fisher_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Phil Fisher's growth investing principles.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    fisher_analysis = {}
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
                "revenue", "net_income", "research_and_development",
                "selling_general_administrative", "operating_income",
                "gross_margin", "operating_margin", "free_cash_flow",
                "total_debt", "shareholders_equity", "capital_expenditure",
                "outstanding_shares",
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

        progress.update_status(agent_id, ticker, "Analyzing growth potential")
        growth_analysis = analyze_growth_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing R&D and innovation")
        innovation_analysis = analyze_innovation(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        management_analysis = analyze_management(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing profitability")
        profitability_analysis = analyze_profitability(metrics, financial_line_items)

        total_score = (
            growth_analysis["score"] +
            innovation_analysis["score"] +
            management_analysis["score"] +
            profitability_analysis["score"]
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
            "growth_analysis": growth_analysis,
            "innovation_analysis": innovation_analysis,
            "management_analysis": management_analysis,
            "profitability_analysis": profitability_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Phil Fisher analysis")
        fisher_output = generate_fisher_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        fisher_analysis[ticker] = {
            "signal": fisher_output.signal,
            "confidence": fisher_output.confidence,
            "reasoning": fisher_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": fisher_output.signal,
            "confidence": fisher_output.confidence,
            "key_findings": [
                f"Fisher score: {total_score}/{max_possible_score}",
                f"Growth: {growth_analysis.get('details', 'N/A')[:80]}",
                f"Innovation: {innovation_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": fisher_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=fisher_output.reasoning)

    message = HumanMessage(content=json.dumps(fisher_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fisher_analysis, "Phil Fisher Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = fisher_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyze_growth_potential(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for growth analysis"}

    revenue_values = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenue_values) >= 3 and revenue_values[-1] and revenue_values[-1] > 0:
        years = len(revenue_values) - 1
        rev_cagr = ((revenue_values[0] / revenue_values[-1]) ** (1 / years) - 1) * 100
        if rev_cagr > 20:
            score += 5
            details.append(f"Exceptional revenue CAGR: {rev_cagr:.1f}%")
        elif rev_cagr > 12:
            score += 4
            details.append(f"Strong revenue CAGR: {rev_cagr:.1f}%")
        elif rev_cagr > 7:
            score += 2
            details.append(f"Moderate revenue CAGR: {rev_cagr:.1f}%")
        else:
            details.append(f"Slow revenue CAGR: {rev_cagr:.1f}%")
    else:
        details.append("Insufficient revenue data for CAGR")

    latest_metrics = metrics[0]
    if latest_metrics.price_to_earnings_ratio and latest_metrics.price_to_earnings_ratio > 0:
        net_income_values = [item.net_income for item in financial_line_items if item.net_income is not None and item.net_income > 0]
        if len(net_income_values) >= 2 and net_income_values[-1] > 0:
            years = len(net_income_values) - 1
            ni_cagr = ((net_income_values[0] / net_income_values[-1]) ** (1 / years) - 1) * 100
            peg = latest_metrics.price_to_earnings_ratio / ni_cagr if ni_cagr > 0 else None
            if peg and peg < 1.5:
                score += 3
                details.append(f"Attractive growth valuation: PEG = {peg:.2f}")
            elif peg:
                details.append(f"Expensive growth valuation: PEG = {peg:.2f}")

    return {"score": score, "details": "; ".join(details)}


def analyze_innovation(financial_line_items: list) -> dict:
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for innovation analysis"}

    rd_values = [item.research_and_development for item in financial_line_items if item.research_and_development is not None]
    revenue_values = [item.revenue for item in financial_line_items if item.revenue is not None]

    if rd_values and revenue_values:
        rd_to_revenue = []
        for i in range(min(len(rd_values), len(revenue_values))):
            if revenue_values[i] and revenue_values[i] > 0:
                rd_to_revenue.append(abs(rd_values[i]) / revenue_values[i])

        if rd_to_revenue:
            avg_rd_ratio = sum(rd_to_revenue) / len(rd_to_revenue)
            if avg_rd_ratio > 0.10:
                score += 5
                details.append(f"Heavy R&D investment: {avg_rd_ratio:.1%} of revenue")
            elif avg_rd_ratio > 0.05:
                score += 3
                details.append(f"Moderate R&D investment: {avg_rd_ratio:.1%} of revenue")
            elif avg_rd_ratio > 0.02:
                score += 1
                details.append(f"Light R&D investment: {avg_rd_ratio:.1%} of revenue")
            else:
                details.append(f"Minimal R&D investment: {avg_rd_ratio:.1%} of revenue")

            if len(rd_values) >= 3 and all(rd_values[i] >= rd_values[i+1] for i in range(len(rd_values)-1)):
                score += 2
                details.append("Consistently growing R&D investment")
        else:
            details.append("Could not calculate R&D to revenue ratio")
    else:
        details.append("No R&D data available — may not be innovation-driven")

    return {"score": score, "details": "; ".join(details)}


def analyze_management(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for management analysis"}

    roe_values = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roe_values:
        avg_roe = sum(roe_values) / len(roe_values)
        if avg_roe > 0.20:
            score += 3
            details.append(f"Excellent management efficiency: avg ROE {avg_roe:.1%}")
        elif avg_roe > 0.12:
            score += 2
            details.append(f"Good ROE: {avg_roe:.1%}")
        elif avg_roe > 0:
            score += 1
            details.append(f"Positive but modest ROE: {avg_roe:.1%}")
        else:
            details.append(f"Negative ROE: {avg_roe:.1%}")

    sga_values = [item.selling_general_administrative for item in financial_line_items if item.selling_general_administrative is not None]
    rev_values = [item.revenue for item in financial_line_items if item.revenue is not None]

    if sga_values and rev_values:
        sga_ratios = []
        for i in range(min(len(sga_values), len(rev_values))):
            if rev_values[i] and rev_values[i] > 0:
                sga_ratios.append(abs(sga_values[i]) / rev_values[i])
        if sga_ratios:
            avg_sga = sum(sga_ratios) / len(sga_ratios)
            if avg_sga < 0.15:
                score += 2
                details.append(f"Lean operations: SG&A only {avg_sga:.1%} of revenue")
            elif avg_sga < 0.25:
                score += 1
                details.append(f"Moderate overhead: SG&A {avg_sga:.1%} of revenue")
            else:
                details.append(f"High overhead: SG&A {avg_sga:.1%} of revenue")

    return {"score": score, "details": "; ".join(details)}


def analyze_profitability(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for profitability analysis"}

    margins = [item.gross_margin for item in financial_line_items if item.gross_margin is not None]
    if margins:
        avg_margin = sum(margins) / len(margins)
        if avg_margin > 0.50:
            score += 3
            details.append(f"Excellent gross margins: {avg_margin:.1%}")
        elif avg_margin > 0.30:
            score += 2
            details.append(f"Good gross margins: {avg_margin:.1%}")
        elif avg_margin > 0.15:
            score += 1
            details.append(f"Decent gross margins: {avg_margin:.1%}")
        else:
            details.append(f"Thin gross margins: {avg_margin:.1%}")

    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if fcf_values:
        positive_fcf = sum(1 for f in fcf_values if f > 0)
        if positive_fcf == len(fcf_values):
            score += 3
            details.append("Consistently positive free cash flow")
        elif positive_fcf >= len(fcf_values) * 0.7:
            score += 1
            details.append(f"Mostly positive FCF: {positive_fcf}/{len(fcf_values)} periods")
        else:
            details.append(f"Inconsistent FCF: positive in only {positive_fcf}/{len(fcf_values)} periods")

    return {"score": score, "details": "; ".join(details)}


def generate_fisher_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> PhilFisherSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your Fisher-style growth assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Phil Fisher AI agent, making investment decisions using his growth principles:
            1. Look for companies with above-average long-term sales and profit growth
            2. Prioritize innovation — companies investing heavily in R&D for future products
            3. Management must be outstanding, honest, and focused on long-term growth
            4. Prefer companies with strong profit margins that are improving
            5. Evaluate the "scuttlebutt" — what do customers, competitors, employees say?
            6. Only sell if fundamentals deteriorate, not on price decline

            Focus on long-term growth quality, R&D investment, and management excellence.
            Return a rational recommendation with confidence (0-100) and reasoning.""",
        ),
        (
            "human",
            """Based on the following analysis, create a Fisher-style investment signal:

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
        return PhilFisherSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=PhilFisherSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )