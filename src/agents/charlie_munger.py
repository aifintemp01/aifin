from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class CharlieMungerSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int
    reasoning: str


def charlie_munger_agent(
    state: AgentState,
    agent_id: str = "charlie_munger_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using Charlie Munger's investing principles and mental models.
    Focuses on moat strength, management quality, predictability, and valuation.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    munger_analysis = {}
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
                "revenue", "net_income", "operating_income", "return_on_invested_capital",
                "gross_margin", "operating_margin", "free_cash_flow", "capital_expenditure",
                "cash_and_equivalents", "total_debt", "shareholders_equity", "outstanding_shares",
                "research_and_development", "goodwill", "intangible_assets",
            ],
            end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=100, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing moat strength")
        moat_analysis = analyze_moat_strength(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        management_analysis = analyze_management_quality(financial_line_items, insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing business predictability")
        predictability_analysis = analyze_predictability(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating Munger-style valuation")
        valuation_analysis = calculate_munger_valuation(financial_line_items, market_cap)

        total_score = (
            moat_analysis["score"] * 0.35 +
            management_analysis["score"] * 0.25 +
            predictability_analysis["score"] * 0.25 +
            valuation_analysis["score"] * 0.15
        )
        max_possible_score = 10

        if total_score >= 7.5:
            signal = "bullish"
        elif total_score <= 5.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "moat_analysis": moat_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
            "valuation_analysis": valuation_analysis,
            "news_sentiment": analyze_news_sentiment(company_news) if company_news else "No news data available",
        }

        progress.update_status(agent_id, ticker, "Generating Charlie Munger analysis")
        munger_output = generate_munger_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
            confidence_hint=compute_confidence(analysis_data[ticker], signal),
            upstream_context=upstream_context,
        )

        munger_analysis[ticker] = {
            "signal": munger_output.signal,
            "confidence": munger_output.confidence,
            "reasoning": munger_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": munger_output.signal,
            "confidence": munger_output.confidence,
            "key_findings": [
                f"Moat score: {moat_analysis.get('score', 0):.1f}/10",
                f"Predictability score: {predictability_analysis.get('score', 0):.1f}/10",
                f"FCF yield: {valuation_analysis.get('fcf_yield', 0):.1%}" if valuation_analysis.get('fcf_yield') else "FCF yield: N/A",
            ],
            "data_summary": munger_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=munger_output.reasoning)

    message = HumanMessage(content=json.dumps(munger_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(munger_analysis, "Charlie Munger Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = munger_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged) ─────────────────────────────────────────────

def analyze_moat_strength(metrics: list, financial_line_items: list) -> dict:
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze moat strength"}

    roic_values = [item.return_on_invested_capital for item in financial_line_items
                   if item.return_on_invested_capital is not None]
    if roic_values:
        high_roic_count = sum(1 for r in roic_values if r > 0.15)
        if high_roic_count >= len(roic_values) * 0.8:
            score += 3
            details.append(f"Excellent ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count >= len(roic_values) * 0.5:
            score += 2
            details.append(f"Good ROIC: >15% in {high_roic_count}/{len(roic_values)} periods")
        elif high_roic_count > 0:
            score += 1
            details.append(f"Mixed ROIC: >15% in only {high_roic_count}/{len(roic_values)} periods")
        else:
            details.append("Poor ROIC: Never exceeds 15% threshold")
    else:
        details.append("No ROIC data available")

    gross_margins = [item.gross_margin for item in financial_line_items if item.gross_margin is not None]
    if gross_margins and len(gross_margins) >= 3:
        margin_trend = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
        if margin_trend >= len(gross_margins) * 0.7:
            score += 2
            details.append("Strong pricing power: Gross margins consistently improving")
        elif sum(gross_margins) / len(gross_margins) > 0.3:
            score += 1
            details.append(f"Good pricing power: Average gross margin {sum(gross_margins)/len(gross_margins):.1%}")
        else:
            details.append("Limited pricing power: Low or declining gross margins")
    else:
        details.append("Insufficient gross margin data")

    if len(financial_line_items) >= 3:
        capex_to_revenue = []
        for item in financial_line_items:
            capex = item.capital_expenditure
            revenue = item.revenue
            if capex is not None and revenue is not None and revenue > 0:
                capex_to_revenue.append(abs(capex) / revenue)
        if capex_to_revenue:
            avg_capex_ratio = sum(capex_to_revenue) / len(capex_to_revenue)
            if avg_capex_ratio < 0.05:
                score += 2
                details.append(f"Low capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            elif avg_capex_ratio < 0.10:
                score += 1
                details.append(f"Moderate capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
            else:
                details.append(f"High capital requirements: Avg capex {avg_capex_ratio:.1%} of revenue")
        else:
            details.append("No capital expenditure data available")
    else:
        details.append("Insufficient data for capital intensity analysis")

    r_and_d = [item.research_and_development for item in financial_line_items if item.research_and_development is not None]
    goodwill = [item.goodwill for item in financial_line_items if item.goodwill is not None]
    intangible_assets = [item.intangible_assets for item in financial_line_items if item.intangible_assets is not None]

    if r_and_d and sum(r_and_d) > 0:
        score += 1
        details.append("Invests in R&D, building intellectual property")
    if goodwill or intangible_assets:
        score += 1
        details.append("Significant goodwill/intangible assets, suggesting brand value or IP")

    final_score = min(10, score * 10 / 9)
    return {"score": final_score, "details": "; ".join(details)}


def analyze_management_quality(financial_line_items: list, insider_trades: list) -> dict:
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "Insufficient data to analyze management quality"}

    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    net_income_values = [item.net_income for item in financial_line_items if item.net_income is not None]

    if fcf_values and net_income_values and len(fcf_values) == len(net_income_values):
        fcf_to_ni_ratios = [fcf_values[i] / net_income_values[i]
                            for i in range(len(fcf_values))
                            if net_income_values[i] and net_income_values[i] > 0]
        if fcf_to_ni_ratios:
            avg_ratio = sum(fcf_to_ni_ratios) / len(fcf_to_ni_ratios)
            if avg_ratio > 1.1:
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.9:
                score += 2
                details.append(f"Good cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            elif avg_ratio > 0.7:
                score += 1
                details.append(f"Moderate cash conversion: FCF/NI ratio of {avg_ratio:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI ratio of only {avg_ratio:.2f}")
        else:
            details.append("Could not calculate FCF to Net Income ratios")
    else:
        details.append("Missing FCF or Net Income data")

    debt_values = [item.total_debt for item in financial_line_items if item.total_debt is not None]
    equity_values = [item.shareholders_equity for item in financial_line_items if item.shareholders_equity is not None]

    recent_de_ratio = None
    if debt_values and equity_values and len(debt_values) == len(equity_values):
        recent_de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float('inf')
        if recent_de_ratio < 0.3:
            score += 3
            details.append(f"Conservative debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 0.7:
            score += 2
            details.append(f"Prudent debt management: D/E ratio of {recent_de_ratio:.2f}")
        elif recent_de_ratio < 1.5:
            score += 1
            details.append(f"Moderate debt level: D/E ratio of {recent_de_ratio:.2f}")
        else:
            details.append(f"High debt level: D/E ratio of {recent_de_ratio:.2f}")
    else:
        details.append("Missing debt or equity data")

    cash_values = [item.cash_and_equivalents for item in financial_line_items if item.cash_and_equivalents is not None]
    revenue_values = [item.revenue for item in financial_line_items if item.revenue is not None]

    cash_to_revenue = None
    if cash_values and revenue_values and revenue_values[0] and revenue_values[0] > 0:
        cash_to_revenue = cash_values[0] / revenue_values[0]
        if 0.1 <= cash_to_revenue <= 0.25:
            score += 2
            details.append(f"Prudent cash management: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif 0.05 <= cash_to_revenue < 0.1 or 0.25 < cash_to_revenue <= 0.4:
            score += 1
            details.append(f"Acceptable cash position: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        elif cash_to_revenue > 0.4:
            details.append(f"Excess cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
        else:
            details.append(f"Low cash reserves: Cash/Revenue ratio of {cash_to_revenue:.2f}")
    else:
        details.append("Insufficient cash or revenue data")

    insider_buy_ratio = None
    if insider_trades and len(insider_trades) > 0:
        buys = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and t.transaction_type.lower() in ['buy', 'purchase'])
        sells = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and t.transaction_type.lower() in ['sell', 'sale'])
        total_trades = buys + sells
        if total_trades > 0:
            insider_buy_ratio = buys / total_trades
            if insider_buy_ratio > 0.7:
                score += 2
                details.append(f"Strong insider buying: {buys}/{total_trades} purchases")
            elif insider_buy_ratio > 0.4:
                score += 1
                details.append(f"Balanced insider trading: {buys}/{total_trades} purchases")
            elif insider_buy_ratio < 0.1 and sells > 5:
                score -= 1
                details.append(f"Concerning insider selling: {sells}/{total_trades} sales")
            else:
                details.append(f"Mixed insider activity: {buys}/{total_trades} purchases")
        else:
            details.append("No recorded insider transactions")
    else:
        details.append("No insider trading data available")

    share_counts = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    share_count_trend = "unknown"
    if share_counts and len(share_counts) >= 3:
        if share_counts[0] < share_counts[-1] * 0.95:
            share_count_trend = "decreasing"
            score += 2
            details.append("Shareholder-friendly: Reducing share count over time")
        elif share_counts[0] < share_counts[-1] * 1.05:
            share_count_trend = "stable"
            score += 1
            details.append("Stable share count: Limited dilution")
        elif share_counts[0] > share_counts[-1] * 1.2:
            share_count_trend = "increasing"
            score -= 1
            details.append("Concerning dilution: Share count increased significantly")
        else:
            share_count_trend = "increasing"
            details.append("Moderate share count increase over time")
    else:
        details.append("Insufficient share count data")

    final_score = max(0, min(10, score * 10 / 12))
    return {
        "score": final_score,
        "details": "; ".join(details),
        "insider_buy_ratio": insider_buy_ratio,
        "recent_de_ratio": recent_de_ratio,
        "cash_to_revenue": cash_to_revenue,
        "share_count_trend": share_count_trend,
    }


def analyze_predictability(financial_line_items: list) -> dict:
    score = 0
    details = []

    if not financial_line_items or len(financial_line_items) < 5:
        return {"score": 0, "details": "Insufficient data to analyze business predictability (need 5+ years)"}

    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if revenues and len(revenues) >= 5:
        growth_rates = []
        for i in range(len(revenues)-1):
            if revenues[i+1] != 0:
                growth_rates.append(revenues[i] / revenues[i+1] - 1)
        if not growth_rates:
            details.append("Cannot calculate revenue growth: zero revenue values found")
        else:
            avg_growth = sum(growth_rates) / len(growth_rates)
            growth_volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
            if avg_growth > 0.05 and growth_volatility < 0.1:
                score += 3
                details.append(f"Highly predictable revenue: {avg_growth:.1%} avg growth with low volatility")
            elif avg_growth > 0 and growth_volatility < 0.2:
                score += 2
                details.append(f"Moderately predictable revenue: {avg_growth:.1%} avg growth")
            elif avg_growth > 0:
                score += 1
                details.append(f"Growing but less predictable revenue: {avg_growth:.1%} avg growth")
            else:
                details.append(f"Declining or unpredictable revenue: {avg_growth:.1%} avg growth")
    else:
        details.append("Insufficient revenue history for predictability analysis")

    op_income = [item.operating_income for item in financial_line_items if item.operating_income is not None]
    if op_income and len(op_income) >= 5:
        positive_periods = sum(1 for income in op_income if income > 0)
        if positive_periods == len(op_income):
            score += 3
            details.append("Highly predictable operations: Operating income positive in all periods")
        elif positive_periods >= len(op_income) * 0.8:
            score += 2
            details.append(f"Predictable operations: positive in {positive_periods}/{len(op_income)} periods")
        elif positive_periods >= len(op_income) * 0.6:
            score += 1
            details.append(f"Somewhat predictable: positive in {positive_periods}/{len(op_income)} periods")
        else:
            details.append(f"Unpredictable: positive in only {positive_periods}/{len(op_income)} periods")
    else:
        details.append("Insufficient operating income history")

    op_margins = []
    for item in financial_line_items:
        oi = item.operating_income
        rev = item.revenue
        if oi is not None and rev is not None and rev != 0:
            op_margins.append(oi / rev)

    if op_margins and len(op_margins) >= 5:
        avg_margin = sum(op_margins) / len(op_margins)
        margin_volatility = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
        if margin_volatility < 0.03:
            score += 2
            details.append(f"Highly predictable margins: {avg_margin:.1%} avg with minimal volatility")
        elif margin_volatility < 0.07:
            score += 1
            details.append(f"Moderately predictable margins: {avg_margin:.1%} avg")
        else:
            details.append(f"Unpredictable margins: {avg_margin:.1%} avg with high volatility")
    else:
        details.append("Insufficient margin history")

    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if fcf_values and len(fcf_values) >= 5:
        positive_fcf_periods = sum(1 for fcf in fcf_values if fcf > 0)
        if positive_fcf_periods == len(fcf_values):
            score += 2
            details.append("Highly predictable cash generation: Positive FCF in all periods")
        elif positive_fcf_periods >= len(fcf_values) * 0.8:
            score += 1
            details.append(f"Predictable cash: Positive FCF in {positive_fcf_periods}/{len(fcf_values)} periods")
        else:
            details.append(f"Unpredictable cash: Positive FCF in only {positive_fcf_periods}/{len(fcf_values)} periods")
    else:
        details.append("Insufficient free cash flow history")

    final_score = min(10, score * 10 / 10)
    return {"score": final_score, "details": "; ".join(details)}


def calculate_munger_valuation(financial_line_items: list, market_cap: float) -> dict:
    score = 0
    details = []

    if not financial_line_items or market_cap is None:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    fcf_values = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    if not fcf_values or len(fcf_values) < 3:
        return {"score": 0, "details": "Insufficient free cash flow data for valuation"}

    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    if normalized_fcf <= 0:
        return {"score": 0, "details": f"Negative or zero normalized FCF", "intrinsic_value": None}

    if market_cap <= 0:
        return {"score": 0, "details": f"Invalid market cap"}

    fcf_yield = normalized_fcf / market_cap
    if fcf_yield > 0.08:
        score += 4
        details.append(f"Excellent value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.05:
        score += 3
        details.append(f"Good value: {fcf_yield:.1%} FCF yield")
    elif fcf_yield > 0.03:
        score += 1
        details.append(f"Fair value: {fcf_yield:.1%} FCF yield")
    else:
        details.append(f"Expensive: Only {fcf_yield:.1%} FCF yield")

    conservative_value = normalized_fcf * 10
    reasonable_value = normalized_fcf * 15
    optimistic_value = normalized_fcf * 20
    margin_of_safety_vs_fair_value = (reasonable_value - market_cap) / market_cap

    if margin_of_safety_vs_fair_value > 0.3:
        score += 3
        details.append(f"Large margin of safety: {margin_of_safety_vs_fair_value:.1%} upside")
    elif margin_of_safety_vs_fair_value > 0.1:
        score += 2
        details.append(f"Moderate margin of safety: {margin_of_safety_vs_fair_value:.1%} upside")
    elif margin_of_safety_vs_fair_value > -0.1:
        score += 1
        details.append(f"Fair price: Within 10% of reasonable value")
    else:
        details.append(f"Expensive: {-margin_of_safety_vs_fair_value:.1%} premium to reasonable value")

    if len(fcf_values) >= 3:
        recent_avg = sum(fcf_values[:3]) / 3
        older_avg = sum(fcf_values[-3:]) / 3 if len(fcf_values) >= 6 else fcf_values[-1]
        if recent_avg > older_avg * 1.2:
            score += 3
            details.append("Growing FCF trend adds to intrinsic value")
        elif recent_avg > older_avg:
            score += 2
            details.append("Stable to growing FCF supports valuation")
        else:
            details.append("Declining FCF trend is concerning")

    final_score = min(10, score * 10 / 10)
    return {
        "score": final_score,
        "details": "; ".join(details),
        "intrinsic_value_range": {
            "conservative": conservative_value,
            "reasonable": reasonable_value,
            "optimistic": optimistic_value,
        },
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf,
        "margin_of_safety_vs_fair_value": margin_of_safety_vs_fair_value,
    }


def analyze_news_sentiment(news_items: list) -> str:
    if not news_items:
        return "No news data available"
    return f"Qualitative review of {len(news_items)} recent news items would be needed"


def _r(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return None


def make_munger_facts_bundle(analysis: dict) -> dict:
    moat = analysis.get("moat_analysis") or {}
    mgmt = analysis.get("management_analysis") or {}
    pred = analysis.get("predictability_analysis") or {}
    val  = analysis.get("valuation_analysis") or {}
    ivr  = val.get("intrinsic_value_range") or {}

    moat_score = _r(moat.get("score"), 2) or 0
    mgmt_score = _r(mgmt.get("score"), 2) or 0
    pred_score = _r(pred.get("score"), 2) or 0
    val_score  = _r(val.get("score"), 2) or 0

    flags = {
        "moat_strong": moat_score >= 7,
        "predictable": pred_score >= 7,
        "owner_aligned": (mgmt_score >= 7) or ((mgmt.get("insider_buy_ratio") or 0) >= 0.6),
        "low_leverage": (mgmt.get("recent_de_ratio") is not None and mgmt.get("recent_de_ratio") < 0.7),
        "sensible_cash": (mgmt.get("cash_to_revenue") is not None and 0.1 <= mgmt.get("cash_to_revenue") <= 0.25),
        "mos_positive": (val.get("mos_to_reasonable") or 0) > 0.0,
        "fcf_yield_ok": (val.get("fcf_yield") or 0) >= 0.05,
        "share_count_friendly": (mgmt.get("share_count_trend") == "decreasing"),
    }

    return {
        "pre_signal": analysis.get("signal"),
        "score": _r(analysis.get("score"), 2),
        "max_score": _r(analysis.get("max_score"), 2),
        "moat_score": moat_score,
        "mgmt_score": mgmt_score,
        "predictability_score": pred_score,
        "valuation_score": val_score,
        "fcf_yield": _r(val.get("fcf_yield"), 4),
        "normalized_fcf": _r(val.get("normalized_fcf"), 0),
        "reasonable_value": _r(ivr.get("reasonable"), 0),
        "margin_of_safety_vs_fair_value": _r(val.get("margin_of_safety_vs_fair_value"), 3),
        "insider_buy_ratio": _r(mgmt.get("insider_buy_ratio"), 2),
        "recent_de_ratio": _r(mgmt.get("recent_de_ratio"), 2),
        "cash_to_revenue": _r(mgmt.get("cash_to_revenue"), 2),
        "share_count_trend": mgmt.get("share_count_trend"),
        "flags": flags,
        "notes": {
            "moat": (moat.get("details") or "")[:120],
            "mgmt": (mgmt.get("details") or "")[:120],
            "predictability": (pred.get("details") or "")[:120],
            "valuation": (val.get("details") or "")[:120],
        },
    }


def compute_confidence(analysis: dict, signal: str) -> int:
    moat = float((analysis.get("moat_analysis") or {}).get("score") or 0)
    mgmt = float((analysis.get("management_analysis") or {}).get("score") or 0)
    pred = float((analysis.get("predictability_analysis") or {}).get("score") or 0)
    val  = float((analysis.get("valuation_analysis") or {}).get("score") or 0)

    quality = 0.35 * moat + 0.25 * mgmt + 0.25 * pred
    quality_pct = 100 * (quality / 8.5) if quality > 0 else 0

    mos = (analysis.get("valuation_analysis") or {}).get("margin_of_safety_vs_fair_value")
    mos = float(mos) if mos is not None else 0.0
    val_adj = max(-10.0, min(10.0, mos * 100.0 / 3.0))

    base = 0.85 * quality_pct + 0.15 * (val * 10)
    base = base + val_adj

    if signal == "bullish":
        upper = 100 if mos > 0 else 69
        lower = 50 if quality_pct >= 55 else 30
    elif signal == "bearish":
        lower = 10 if mos < -0.05 else 30
        upper = 49
    else:
        lower, upper = 50, 69

    return max(10, min(100, int(round(max(lower, min(upper, base))))))


def generate_munger_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    confidence_hint: int,
    upstream_context: dict = None,
) -> CharlieMungerSignal:

    facts_bundle = make_munger_facts_bundle(analysis_data)

    upstream_section = ""
    if upstream_context:
        upstream_section = "\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "Incorporate this context into your Munger-style assessment.\n"

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are Charlie Munger. Decide bullish, bearish, or neutral using only the facts. "
         "Return JSON only. Keep reasoning under 120 characters. "
         "Use the provided confidence exactly; do not change it."),
        ("human",
         "Ticker: {ticker}\n"
         "Facts:\n{facts}\n"
         "{upstream_context}"
         "Confidence: {confidence}\n"
         "Return exactly:\n"
         "{{\n"
         '  "signal": "bullish" | "bearish" | "neutral",\n'
         f'  "confidence": {confidence_hint},\n'
         '  "reasoning": "short justification"\n'
         "}}")
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "facts": json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
        "upstream_context": upstream_section,
        "confidence": confidence_hint,
    })

    def _default():
        return CharlieMungerSignal(signal="neutral", confidence=confidence_hint, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=CharlieMungerSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )