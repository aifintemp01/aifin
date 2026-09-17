from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
from typing_extensions import Literal
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


def _safe_get(obj, attr: str):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    elif isinstance(obj, dict):
        return obj.get(attr)
    return None


def warren_buffett_agent(
    state: AgentState,
    agent_id: str = "warren_buffett_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data = {}
    buffett_analysis = {}
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
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
                "shareholders_equity",
                "dividends_and_other_cash_distributions",
                "issuance_or_purchase_of_equity_shares",
                "gross_profit",
                "revenue",
                "free_cash_flow",
            ],
            end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing fundamentals")
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status(agent_id, ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing competitive moat")
        moat_analysis = analyze_moat(metrics)

        progress.update_status(agent_id, ticker, "Analyzing pricing power")
        pricing_power_analysis = analyze_pricing_power(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing book value growth")
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        progress.update_status(agent_id, ticker, "Analyzing management quality")
        mgmt_analysis = analyze_management_quality(financial_line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        total_score = (
            fundamental_analysis["score"] +
            consistency_analysis["score"] +
            moat_analysis["score"] +
            mgmt_analysis["score"] +
            pricing_power_analysis["score"] +
            book_value_analysis["score"]
        )

        max_possible_score = (
            10 +
            moat_analysis["max_score"] +
            mgmt_analysis["max_score"] +
            5 +
            5
        )

        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        analysis_data[ticker] = {
            "ticker": ticker,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "pricing_power_analysis": pricing_power_analysis,
            "book_value_analysis": book_value_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status(agent_id, ticker, "Generating Warren Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "key_findings": [
                f"Score: {total_score}/{max_possible_score}",
                f"Margin of safety: {margin_of_safety:.1%}" if margin_of_safety is not None else "Margin of safety: N/A",
                f"Moat: {moat_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": buffett_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=buffett_output.reasoning)

    message = HumanMessage(content=json.dumps(buffett_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, agent_id)

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = buffett_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged) ─────────────────────────────────────────────

def analyze_fundamentals(metrics: list) -> dict:
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}
    latest_metrics = metrics[0]
    score = 0
    reasoning = []

    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict:
    if len(financial_line_items) < 4:
        return {"score": 0, "details": "Insufficient historical data"}
    score = 0
    reasoning = []

    earnings_values = [_safe_get(item, "net_income") for item in financial_line_items if _safe_get(item, "net_income")]
    if len(earnings_values) >= 4:
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))
        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_moat(metrics: list) -> dict:
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if len(historical_roes) >= 5:
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        if roe_consistency >= 0.8:
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%})")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE performance: {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
    else:
        reasoning.append("Insufficient ROE history for moat analysis")

    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_avg = sum(historical_margins[:3]) / 3
        older_avg = sum(historical_margins[-3:]) / 3
        if avg_margin > 0.2 and recent_avg >= older_avg:
            moat_score += 1
            reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%})")
        elif avg_margin > 0.15:
            reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%})")
        else:
            reasoning.append(f"Low operating margins (avg: {avg_margin:.1%})")

    if len(metrics) >= 5:
        asset_turnovers = [m.asset_turnover for m in metrics if hasattr(m, 'asset_turnover') and m.asset_turnover is not None]
        if len(asset_turnovers) >= 3 and any(t > 1.0 for t in asset_turnovers):
            moat_score += 1
            reasoning.append("Efficient asset utilization suggests operational moat")

    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((r - roe_avg) ** 2 for r in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0
        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((m - margin_avg) ** 2 for m in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0
        overall_stability = (roe_stability + margin_stability) / 2
        if overall_stability > 0.7:
            moat_score += 1
            reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong competitive moat")

    moat_score = min(moat_score, max_score)
    return {"score": moat_score, "max_score": max_score, "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available"}


def analyze_management_quality(financial_line_items: list) -> dict:
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}
    reasoning = []
    mgmt_score = 0
    latest = financial_line_items[0]

    issuance = _safe_get(latest, "issuance_or_purchase_of_equity_shares")
    if issuance and issuance < 0:
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
    elif issuance and issuance > 0:
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    dividends = _safe_get(latest, "dividends_and_other_cash_distributions")
    if dividends and dividends < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {"score": mgmt_score, "max_score": 2, "details": "; ".join(reasoning)}


def calculate_owner_earnings(financial_line_items: list) -> dict:
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []
    net_income = _safe_get(latest, "net_income")
    depreciation = _safe_get(latest, "depreciation_and_amortization")
    capex = _safe_get(latest, "capital_expenditure")

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None: missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None: missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    maintenance_capex = estimate_maintenance_capex(financial_line_items)
    working_capital_change = 0

    if len(financial_line_items) >= 2:
        try:
            ca_curr = _safe_get(latest, 'current_assets')
            cl_curr = _safe_get(latest, 'current_liabilities')
            prev = financial_line_items[1]
            ca_prev = _safe_get(prev, 'current_assets')
            cl_prev = _safe_get(prev, 'current_liabilities')
            if all([ca_curr, cl_curr, ca_prev, cl_prev]):
                wc_curr = ca_curr - cl_curr
                wc_prev = ca_prev - cl_prev
                working_capital_change = wc_curr - wc_prev
                details.append(f"Working capital change: ${working_capital_change:,.0f}")
        except:
            pass

    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change
    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0,
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    if not financial_line_items:
        return 0

    capex_ratios = []
    for item in financial_line_items[:5]:
        capex = _safe_get(item, 'capital_expenditure')
        revenue = _safe_get(item, 'revenue')
        if capex and revenue and revenue > 0:
            capex_ratios.append(abs(capex) / revenue)

    latest_depreciation = _safe_get(financial_line_items[0], "depreciation_and_amortization") or 0
    latest_capex = abs(_safe_get(financial_line_items[0], "capital_expenditure") or 0)

    method_1 = latest_capex * 0.85
    method_2 = latest_depreciation

    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = _safe_get(financial_line_items[0], "revenue") or 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0
        return sorted([method_1, method_2, method_3])[1]
    else:
        return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list) -> dict:
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    shares_outstanding = _safe_get(financial_line_items[0], "outstanding_shares")
    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    details = []
    historical_earnings = [_safe_get(item, 'net_income') for item in financial_line_items[:5] if _safe_get(item, 'net_income')]

    if len(historical_earnings) >= 3:
        oldest = historical_earnings[-1]
        latest = historical_earnings[0]
        years = len(historical_earnings) - 1
        if oldest > 0:
            historical_growth = ((latest / oldest) ** (1 / years)) - 1
            historical_growth = max(-0.05, min(historical_growth, 0.15))
            conservative_growth = historical_growth * 0.7
        else:
            conservative_growth = 0.03
    else:
        conservative_growth = 0.03

    stage1_growth = min(conservative_growth, 0.08)
    stage2_growth = min(conservative_growth * 0.5, 0.04)
    terminal_growth = 0.025
    discount_rate = 0.10

    stage1_pv = sum(owner_earnings * (1 + stage1_growth) ** y / (1 + discount_rate) ** y for y in range(1, 6))
    stage1_final = owner_earnings * (1 + stage1_growth) ** 5
    stage2_pv = sum(stage1_final * (1 + stage2_growth) ** y / (1 + discount_rate) ** (5 + y) for y in range(1, 6))
    final_earnings = stage1_final * (1 + stage2_growth) ** 5
    terminal_pv = (final_earnings * (1 + terminal_growth) / (discount_rate - terminal_growth)) / (1 + discount_rate) ** 10

    intrinsic_value = (stage1_pv + stage2_pv + terminal_pv) * 0.85
    details.append(f"Conservative IV (15% haircut): ${intrinsic_value:,.0f}")

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
        },
        "details": details,
    }


def analyze_book_value_growth(financial_line_items: list) -> dict:
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}

    book_values = []
    for item in financial_line_items:
        eq = _safe_get(item, 'shareholders_equity')
        sh = _safe_get(item, 'outstanding_shares')
        if eq and sh:
            book_values.append(eq / sh)

    if len(book_values) < 3:
        return {"score": 0, "details": "Insufficient book value data for growth analysis"}

    score = 0
    reasoning = []
    growth_periods = sum(1 for i in range(len(book_values) - 1) if book_values[i] > book_values[i + 1])
    growth_rate = growth_periods / (len(book_values) - 1)

    if growth_rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value per share growth")
    elif growth_rate >= 0.6:
        score += 2
        reasoning.append("Good book value per share growth pattern")
    elif growth_rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value per share growth")
    else:
        reasoning.append("Inconsistent book value per share growth")

    cagr_score, cagr_reason = _calculate_book_value_cagr(book_values)
    score += cagr_score
    reasoning.append(cagr_reason)

    return {"score": score, "details": "; ".join(reasoning)}


def _calculate_book_value_cagr(book_values: list) -> tuple:
    if len(book_values) < 2:
        return 0, "Insufficient data for CAGR calculation"
    oldest_bv, latest_bv = book_values[-1], book_values[0]
    years = len(book_values) - 1
    if oldest_bv > 0 and latest_bv > 0:
        cagr = ((latest_bv / oldest_bv) ** (1 / years)) - 1
        if cagr > 0.15: return 2, f"Excellent book value CAGR: {cagr:.1%}"
        elif cagr > 0.1: return 1, f"Good book value CAGR: {cagr:.1%}"
        else: return 0, f"Book value CAGR: {cagr:.1%}"
    elif oldest_bv < 0 < latest_bv:
        return 3, "Excellent: Company improved from negative to positive book value"
    elif oldest_bv > 0 > latest_bv:
        return 0, "Warning: Company declined from positive to negative book value"
    else:
        return 0, "Unable to calculate meaningful book value CAGR"


def analyze_pricing_power(financial_line_items: list, metrics: list) -> dict:
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}

    score = 0
    reasoning = []

    gross_margins = [_safe_get(item, 'gross_margin') for item in financial_line_items if _safe_get(item, 'gross_margin') is not None]

    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]
        if recent_avg > older_avg + 0.02:
            score += 3
            reasoning.append("Expanding gross margins indicate strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving gross margins suggest good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:
            score += 1
            reasoning.append("Stable gross margins")
        else:
            reasoning.append("Declining gross margins may indicate pricing pressure")

    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%})")
        elif avg_margin > 0.3:
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%})")

    return {"score": score, "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available"}


def generate_buffett_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str = "warren_buffett_agent",
    upstream_context: dict = None,
) -> WarrenBuffettSignal:

    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),
        "consistency": analysis_data.get("consistency_analysis", {}).get("details"),
        "moat": analysis_data.get("moat_analysis", {}).get("details"),
        "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),
        "book_value": analysis_data.get("book_value_analysis", {}).get("details"),
        "management": analysis_data.get("management_analysis", {}).get("details"),
        "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),
        "market_cap": analysis_data.get("market_cap"),
        "margin_of_safety": analysis_data.get("margin_of_safety"),
    }

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your Buffett-style assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Warren Buffett. Decide bullish, bearish, or neutral using only the provided facts.\n"
            "\n"
            "Checklist for decision:\n"
            "- Circle of competence\n"
            "- Competitive moat\n"
            "- Management quality\n"
            "- Financial strength\n"
            "- Valuation vs intrinsic value\n"
            "- Long-term prospects\n"
            "\n"
            "Signal rules:\n"
            "- Bullish: strong business AND margin_of_safety > 0.\n"
            "- Bearish: poor business OR clearly overvalued.\n"
            "- Neutral: good business but margin_of_safety <= 0, or mixed evidence.\n"
            "\n"
            "Keep reasoning under 120 characters. Do not invent data. Return JSON only."
        ),
        (
            "human",
            "Ticker: {ticker}\nFacts:\n{facts}\n{upstream_context}\n\n"
            "Return exactly:\n"
            "{{\n"
            '  "signal": "bullish" | "bearish" | "neutral",\n'
            '  "confidence": int,\n'
            '  "reasoning": "short justification"\n'
            "}}"
        ),
    ])

    prompt = template.invoke({
        "facts": json.dumps(facts, separators=(",", ":"), ensure_ascii=False),
        "ticker": ticker,
        "upstream_context": upstream_section,
    })

    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=50, reasoning="Insufficient data")

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )