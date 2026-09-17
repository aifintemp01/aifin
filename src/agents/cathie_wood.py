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


class CathieWoodSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def cathie_wood_agent(
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Cathie Wood's disruptive innovation investing principles.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    cw_analysis = {}
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
                "revenue", "gross_margin", "operating_margin", "debt_to_equity",
                "free_cash_flow", "total_assets", "total_liabilities",
                "dividends_and_other_cash_distributions", "outstanding_shares",
                "research_and_development", "capital_expenditure", "operating_expense",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing disruptive potential")
        disruptive_analysis = analyze_disruptive_potential(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing innovation-driven growth")
        innovation_analysis = analyze_innovation_growth(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating valuation & high-growth scenario")
        valuation_analysis = analyze_cathie_wood_valuation(financial_line_items, market_cap)

        total_score = disruptive_analysis["score"] + innovation_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15

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
            "disruptive_analysis": disruptive_analysis,
            "innovation_analysis": innovation_analysis,
            "valuation_analysis": valuation_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Cathie Wood analysis")
        cw_output = generate_cathie_wood_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        cw_analysis[ticker] = {
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "reasoning": cw_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": cw_output.signal,
            "confidence": cw_output.confidence,
            "key_findings": [
                f"CW score: {total_score:.1f}/{max_possible_score}",
                f"Disruptive: {disruptive_analysis.get('details', 'N/A')[:80]}",
                f"Innovation: {innovation_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": cw_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=cw_output.reasoning)

    message = HumanMessage(content=json.dumps(cw_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(cw_analysis, agent_id)

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = cw_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged from original) ────────────────────────────────

def analyze_disruptive_potential(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze disruptive potential"}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i] and revenues[i + 1]:
                gr = (revenues[i] - revenues[i + 1]) / abs(revenues[i + 1]) if revenues[i + 1] != 0 else 0
                growth_rates.append(gr)
        if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[-1]:
            score += 2
            details.append(f"Revenue growth accelerating: {growth_rates[0]*100:.1f}% vs {growth_rates[-1]*100:.1f}%")
        latest_growth = growth_rates[0] if growth_rates else 0
        if latest_growth > 1.0:
            score += 3
            details.append(f"Exceptional revenue growth: {latest_growth*100:.1f}%")
        elif latest_growth > 0.5:
            score += 2
            details.append(f"Strong revenue growth: {latest_growth*100:.1f}%")
        elif latest_growth > 0.2:
            score += 1
            details.append(f"Moderate revenue growth: {latest_growth*100:.1f}%")
    else:
        details.append("Insufficient revenue data")

    gross_margins = [item.gross_margin for item in financial_line_items if item.gross_margin is not None]
    if len(gross_margins) >= 2:
        margin_trend = gross_margins[0] - gross_margins[-1]
        if margin_trend > 0.05:
            score += 2
            details.append(f"Expanding gross margins: +{margin_trend*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append(f"Slightly improving gross margins: +{margin_trend*100:.1f}%")
        if gross_margins[0] > 0.50:
            score += 2
            details.append(f"High gross margin: {gross_margins[0]*100:.1f}%")
    else:
        details.append("Insufficient gross margin data")

    operating_expenses = [item.operating_expense for item in financial_line_items if item.operating_expense is not None]
    if len(revenues) >= 2 and len(operating_expenses) >= 2:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        opex_growth = (operating_expenses[0] - operating_expenses[-1]) / abs(operating_expenses[-1])
        if rev_growth > opex_growth:
            score += 2
            details.append("Positive operating leverage: Revenue growing faster than expenses")
    else:
        details.append("Insufficient data for operating leverage analysis")

    rd_expenses = [item.research_and_development for item in financial_line_items if item.research_and_development is not None]
    if rd_expenses and revenues:
        rd_intensity = rd_expenses[0] / revenues[0]
        if rd_intensity > 0.15:
            score += 3
            details.append(f"High R&D investment: {rd_intensity*100:.1f}% of revenue")
        elif rd_intensity > 0.08:
            score += 2
            details.append(f"Moderate R&D investment: {rd_intensity*100:.1f}% of revenue")
        elif rd_intensity > 0.05:
            score += 1
            details.append(f"Some R&D investment: {rd_intensity*100:.1f}% of revenue")
    else:
        details.append("No R&D data available")

    max_possible_score = 12
    normalized_score = (score / max_possible_score) * 5
    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_innovation_growth(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze innovation-driven growth"}

    rd_expenses = [item.research_and_development for item in financial_line_items if item.research_and_development is not None]
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]

    if rd_expenses and revenues and len(rd_expenses) >= 2:
        rd_growth = (rd_expenses[0] - rd_expenses[-1]) / abs(rd_expenses[-1]) if rd_expenses[-1] != 0 else 0
        if rd_growth > 0.5:
            score += 3
            details.append(f"Strong R&D investment growth: +{rd_growth*100:.1f}%")
        elif rd_growth > 0.2:
            score += 2
            details.append(f"Moderate R&D investment growth: +{rd_growth*100:.1f}%")
        rd_intensity_start = rd_expenses[-1] / revenues[-1] if revenues[-1] else 0
        rd_intensity_end = rd_expenses[0] / revenues[0] if revenues[0] else 0
        if rd_intensity_end > rd_intensity_start:
            score += 2
            details.append(f"Increasing R&D intensity: {rd_intensity_end*100:.1f}% vs {rd_intensity_start*100:.1f}%")
    else:
        details.append("Insufficient R&D data for trend analysis")

    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if fcf_vals and len(fcf_vals) >= 2:
        fcf_growth = (fcf_vals[0] - fcf_vals[-1]) / abs(fcf_vals[-1]) if fcf_vals[-1] != 0 else 0
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if fcf_growth > 0.3 and positive_fcf_count == len(fcf_vals):
            score += 3
            details.append("Strong and consistent FCF growth")
        elif positive_fcf_count >= len(fcf_vals) * 0.75:
            score += 2
            details.append("Consistent positive FCF")
        elif positive_fcf_count > len(fcf_vals) * 0.5:
            score += 1
            details.append("Moderately consistent FCF")
    else:
        details.append("Insufficient FCF data")

    op_margin_vals = []
    for item in financial_line_items:
        oi = item.operating_income
        rev = item.revenue
        if oi is not None and rev is not None and rev != 0:
            op_margin_vals.append(oi / rev)

    if op_margin_vals and len(op_margin_vals) >= 2:
        margin_trend = op_margin_vals[0] - op_margin_vals[-1]
        if op_margin_vals[0] > 0.15 and margin_trend > 0:
            score += 3
            details.append(f"Strong and improving operating margin: {op_margin_vals[0]*100:.1f}%")
        elif op_margin_vals[0] > 0.10:
            score += 2
            details.append(f"Healthy operating margin: {op_margin_vals[0]*100:.1f}%")
        elif margin_trend > 0:
            score += 1
            details.append("Improving operating efficiency")
    else:
        details.append("Insufficient operating margin data")

    capex = [item.capital_expenditure for item in financial_line_items if item.capital_expenditure is not None]
    if capex and revenues and len(capex) >= 2:
        capex_intensity = abs(capex[0]) / revenues[0] if revenues[0] else 0
        capex_growth = (abs(capex[0]) - abs(capex[-1])) / abs(capex[-1]) if capex[-1] != 0 else 0
        if capex_intensity > 0.10 and capex_growth > 0.2:
            score += 2
            details.append("Strong investment in growth infrastructure")
        elif capex_intensity > 0.05:
            score += 1
            details.append("Moderate investment in growth infrastructure")
    else:
        details.append("Insufficient CAPEX data")

    dividends = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if dividends and fcf_vals and fcf_vals[0] != 0:
        latest_payout_ratio = dividends[0] / fcf_vals[0]
        if latest_payout_ratio < 0.2:
            score += 2
            details.append("Strong focus on reinvestment over dividends")
        elif latest_payout_ratio < 0.4:
            score += 1
            details.append("Moderate focus on reinvestment over dividends")
    else:
        details.append("Insufficient dividend data")

    max_possible_score = 15
    normalized_score = (score / max_possible_score) * 5
    return {"score": normalized_score, "details": "; ".join(details), "raw_score": score, "max_score": max_possible_score}


def analyze_cathie_wood_valuation(financial_line_items: list, market_cap: float) -> dict:
    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data for valuation"}

    latest = financial_line_items[0]
    fcf = latest.free_cash_flow or 0
    if fcf <= 0:
        return {"score": 0, "details": f"No positive FCF for valuation; FCF = {fcf}", "intrinsic_value": None}

    growth_rate = 0.20
    discount_rate = 0.15
    terminal_multiple = 25
    projection_years = 5

    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv

    terminal_value = (fcf * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap

    score = 0
    if margin_of_safety > 0.5:
        score += 3
    elif margin_of_safety > 0.2:
        score += 1

    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}",
    ]
    return {"score": score, "details": "; ".join(details), "intrinsic_value": intrinsic_value, "margin_of_safety": margin_of_safety}


def generate_cathie_wood_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "cathie_wood_agent",
    upstream_context: dict = None,
) -> CathieWoodSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your disruptive innovation assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Cathie Wood AI agent, making investment decisions using her principles:
            1. Seek companies leveraging disruptive innovation.
            2. Emphasize exponential growth potential, large TAM.
            3. Focus on technology, healthcare, or other future-facing sectors.
            4. Consider multi-year time horizons for potential breakthroughs.
            5. Accept higher volatility in pursuit of high returns.
            6. Evaluate management's vision and ability to invest in R&D.

            When providing your reasoning:
            1. Identify the specific disruptive technologies the company is leveraging
            2. Highlight growth metrics indicating exponential potential
            3. Discuss long-term vision over 5+ year horizons
            4. Address R&D investment and innovation pipeline
            5. Use Cathie Wood's optimistic, future-focused, conviction-driven voice""",
        ),
        (
            "human",
            """Based on the following analysis, create a Cathie Wood-style investment signal.

Analysis Data for {ticker}:
{analysis_data}
{upstream_context}
Return the trading signal in this JSON format:
{{
  "signal": "bullish/bearish/neutral",
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

    def create_default_cathie_wood_signal():
        return CathieWoodSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=CathieWoodSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_cathie_wood_signal,
    )
# source: https://ark-invest.com