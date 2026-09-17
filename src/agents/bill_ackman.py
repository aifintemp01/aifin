from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(
    state: AgentState,
    agent_id: str = "bill_ackman_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Bill Ackman's investing principles.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    ackman_analysis = {}
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

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue", "operating_margin", "debt_to_equity", "free_cash_flow",
                "total_assets", "total_liabilities",
                "dividends_and_other_cash_distributions", "outstanding_shares",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)

        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
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
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "key_findings": [
                f"Ackman score: {total_score}/{max_possible_score}",
                f"Quality: {quality_analysis.get('details', 'N/A')[:80]}",
                f"Valuation MOS: {valuation_analysis.get('margin_of_safety', 'N/A')}",
            ],
            "data_summary": ackman_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=ackman_output.reasoning)

    message = HumanMessage(content=json.dumps(ackman_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = ackman_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged from original) ────────────────────────────────

def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze business quality"}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        initial, final = revenues[-1], revenues[0]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:
                score += 2
                details.append(f"Revenue grew by {growth_rate*100:.1f}% over the full period (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({growth_rate*100:.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")

    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = []
    for item in financial_line_items:
        oi = item.operating_income
        rev = item.revenue
        if oi is not None and rev is not None and rev != 0:
            op_margin_vals.append(oi / rev)

    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15%.")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")

    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")

    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating a competitive advantage.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
    else:
        details.append("ROE data not available.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze financial discipline"}

    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods.")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods.")
    else:
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")

    dividends_list = [
        item.dividends_and_other_cash_distributions
        for item in financial_line_items
        if item.dividends_and_other_cash_distributions is not None
    ]
    if dividends_list:
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid.")
    else:
        details.append("No dividend data found across periods.")

    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        if shares[0] < shares[-1]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")

    return {"score": score, "details": "; ".join(details)}


def analyze_activism_potential(financial_line_items: list) -> dict:
    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data for activism potential"}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    op_margins = []
    for item in financial_line_items:
        oi = item.operating_income
        rev = item.revenue
        if oi is not None and rev is not None and rev != 0:
            op_margins.append(oi / rev)

    if len(revenues) < 2 or not op_margins:
        return {"score": 0, "details": "Not enough data to assess activism potential."}

    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else 0
    avg_margin = sum(op_margins) / len(op_margins)

    score = 0
    details = []
    if revenue_growth > 0.15 and avg_margin < 0.10:
        score += 2
        details.append(
            f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low "
            f"(avg {avg_margin*100:.1f}%). Activism could unlock margin improvements."
        )
    else:
        details.append("No clear sign of activism opportunity.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow or 0

    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / ((1 + discount_rate) ** projection_years)

    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1

    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}",
    ]
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety,
    }


def generate_ackman_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> BillAckmanSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your Ackman-style activist assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:
            1. Seek high-quality businesses with durable competitive advantages (moats).
            2. Prioritize consistent free cash flow and growth potential over the long term.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value with a margin of safety.
            5. Consider activism where management or operational improvements can unlock substantial upside.
            6. Concentrate on a few high-conviction investments.
            Return your final recommendation with a 0-100 confidence and thorough reasoning.""",
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

Analysis Data for {ticker}:
{analysis_data}
{upstream_context}
Return your output in strictly valid JSON:
{{
  "signal": "bullish" | "bearish" | "neutral",
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

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=BillAckmanSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_bill_ackman_signal,
    )