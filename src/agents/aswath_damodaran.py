from __future__ import annotations
import json
from typing_extensions import Literal
from pydantic import BaseModel
from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.tools.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class AswathDamodaranSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def aswath_damodaran_agent(
    state: AgentState,
    agent_id: str = "aswath_damodaran_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyze equities through Aswath Damodaran's intrinsic-value lens.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    analysis_data: dict[str, dict] = {}
    damodaran_signals: dict[str, dict] = {}
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
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            [
                "revenue",
                "free_cash_flow",
                "ebit",
                "interest_expense",
                "capital_expenditure",
                "depreciation_and_amortization",
                "outstanding_shares",
                "net_income",
                "total_debt",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing growth and reinvestment")
        growth_analysis = analyze_growth_and_reinvestment(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing risk profile")
        risk_analysis = analyze_risk_profile(metrics, line_items)

        progress.update_status(agent_id, ticker, "Calculating intrinsic value (DCF)")
        intrinsic_val_analysis = calculate_intrinsic_value_dcf(metrics, line_items, risk_analysis)

        progress.update_status(agent_id, ticker, "Assessing relative valuation")
        relative_val_analysis = analyze_relative_valuation(metrics)

        total_score = (
            growth_analysis["score"]
            + risk_analysis["score"]
            + relative_val_analysis["score"]
        )
        max_score = (
            growth_analysis["max_score"]
            + risk_analysis["max_score"]
            + relative_val_analysis["max_score"]
        )

        intrinsic_value = intrinsic_val_analysis["intrinsic_value"]
        margin_of_safety = (
            (intrinsic_value - market_cap) / market_cap
            if intrinsic_value and market_cap else None
        )

        if margin_of_safety is not None and margin_of_safety >= 0.25:
            signal = "bullish"
        elif margin_of_safety is not None and margin_of_safety <= -0.25:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "margin_of_safety": margin_of_safety,
            "growth_analysis": growth_analysis,
            "risk_analysis": risk_analysis,
            "relative_val_analysis": relative_val_analysis,
            "intrinsic_val_analysis": intrinsic_val_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating Damodaran analysis")
        damodaran_output = generate_damodaran_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        damodaran_signals[ticker] = damodaran_output.model_dump()

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": damodaran_output.signal,
            "confidence": damodaran_output.confidence,
            "key_findings": [
                f"Margin of safety: {margin_of_safety:.1%}" if margin_of_safety is not None else "MOS: N/A",
                f"DCF value: {intrinsic_value:,.0f}" if intrinsic_value else "DCF: N/A",
                f"Growth score: {growth_analysis.get('score', 0)}/{growth_analysis.get('max_score', 0)}",
            ],
            "data_summary": damodaran_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=damodaran_output.reasoning)

    message = HumanMessage(content=json.dumps(damodaran_signals), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(damodaran_signals, "Aswath Damodaran Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = damodaran_signals
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged from original, bugs fixed) ───────────────────

def analyze_growth_and_reinvestment(metrics: list, line_items: list) -> dict:
    max_score = 4
    if len(metrics) < 2:
        return {"score": 0, "max_score": max_score, "details": "Insufficient history"}

    revs = [
        li.revenue for li in reversed(line_items)
        if li.revenue is not None
    ]
    if len(revs) >= 2 and revs[0] > 0:
        cagr = (revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1
    else:
        cagr = None

    score, details = 0, []
    if cagr is not None:
        if cagr > 0.08:
            score += 2
            details.append(f"Revenue CAGR {cagr:.1%} (> 8%)")
        elif cagr > 0.03:
            score += 1
            details.append(f"Revenue CAGR {cagr:.1%} (> 3%)")
        else:
            details.append(f"Sluggish revenue CAGR {cagr:.1%}")
    else:
        details.append("Revenue data incomplete")

    fcfs = [
        li.free_cash_flow for li in reversed(line_items)
        if li.free_cash_flow is not None
    ]
    if len(fcfs) >= 2 and fcfs[-1] > fcfs[0]:
        score += 1
        details.append("Positive FCFF growth")
    else:
        details.append("Flat or declining FCFF")

    latest = metrics[0]
    if latest.return_on_invested_capital and latest.return_on_invested_capital > 0.10:
        score += 1
        details.append(f"ROIC {latest.return_on_invested_capital:.1%} (> 10%)")

    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "metrics": latest.model_dump(),
    }


def analyze_risk_profile(metrics: list, line_items: list) -> dict:
    max_score = 3
    if not metrics:
        return {"score": 0, "max_score": max_score, "details": "No metrics"}

    latest = metrics[0]
    score, details = 0, []

    beta = getattr(latest, "beta", None)
    if beta is not None:
        if beta < 1.3:
            score += 1
            details.append(f"Beta {beta:.2f}")
        else:
            details.append(f"High beta {beta:.2f}")
    else:
        details.append("Beta NA")

    dte = latest.debt_to_equity
    if dte is not None:
        if dte < 1:
            score += 1
            details.append(f"D/E {dte:.1f}")
        else:
            details.append(f"High D/E {dte:.1f}")
    else:
        details.append("D/E NA")

    latest_li = line_items[0] if line_items else None
    ebit = latest_li.ebit if latest_li else None
    interest = latest_li.interest_expense if latest_li else None
    if ebit and interest and interest != 0:
        coverage = ebit / abs(interest)
        if coverage > 3:
            score += 1
            details.append(f"Interest coverage x{coverage:.1f}")
        else:
            details.append(f"Weak coverage x{coverage:.1f}")
    else:
        details.append("Interest coverage NA")

    cost_of_equity = estimate_cost_of_equity(beta)
    return {
        "score": score,
        "max_score": max_score,
        "details": "; ".join(details),
        "beta": beta,
        "cost_of_equity": cost_of_equity,
    }


def analyze_relative_valuation(metrics: list) -> dict:
    max_score = 1
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": max_score, "details": "Insufficient P/E history"}

    pes = [m.price_to_earnings_ratio for m in metrics if m.price_to_earnings_ratio]
    if len(pes) < 5:
        return {"score": 0, "max_score": max_score, "details": "P/E data sparse"}

    ttm_pe = pes[0]
    median_pe = sorted(pes)[len(pes) // 2]

    if ttm_pe < 0.7 * median_pe:
        score, desc = 1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (cheap)"
    elif ttm_pe > 1.3 * median_pe:
        score, desc = -1, f"P/E {ttm_pe:.1f} vs. median {median_pe:.1f} (expensive)"
    else:
        score, desc = 0, "P/E inline with history"

    return {"score": score, "max_score": max_score, "details": desc}


def calculate_intrinsic_value_dcf(metrics: list, line_items: list, risk_analysis: dict) -> dict:
    if not metrics or len(metrics) < 2 or not line_items:
        return {"intrinsic_value": None, "details": ["Insufficient data"]}

    latest_li = line_items[0]
    fcff0 = latest_li.free_cash_flow
    shares = latest_li.outstanding_shares

    if fcff0 is None or shares is None or shares == 0:
        return {"intrinsic_value": None, "details": ["Missing FCFF or share count"]}

    revs = [
        li.revenue for li in reversed(line_items)
        if li.revenue is not None
    ]
    if len(revs) >= 2 and revs[0] > 0:
        base_growth = min((revs[-1] / revs[0]) ** (1 / (len(revs) - 1)) - 1, 0.12)
    else:
        base_growth = 0.04

    terminal_growth = 0.025
    years = 10
    discount = risk_analysis.get("cost_of_equity") or 0.09

    pv_sum = 0.0
    g = base_growth
    g_step = (terminal_growth - base_growth) / (years - 1)
    for yr in range(1, years + 1):
        fcff_t = fcff0 * (1 + g)
        pv = fcff_t / (1 + discount) ** yr
        pv_sum += pv
        g += g_step

    tv = (
        fcff0
        * (1 + terminal_growth)
        / (discount - terminal_growth)
        / (1 + discount) ** years
    )
    equity_value = pv_sum + tv
    intrinsic_per_share = equity_value / shares

    return {
        "intrinsic_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share,
        "assumptions": {
            "base_fcff": fcff0,
            "base_growth": base_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount,
            "projection_years": years,
        },
        "details": ["FCFF DCF completed"],
    }


def estimate_cost_of_equity(beta: float | None) -> float:
    risk_free = 0.04
    erp = 0.05
    beta = beta if beta is not None else 1.0
    return risk_free + beta * erp


def generate_damodaran_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> AswathDamodaranSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your valuation narrative.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are Aswath Damodaran, Professor of Finance at NYU Stern.
            Use your valuation framework to issue trading signals.
            Speak with your usual clear, data-driven tone:
              - Start with the company "story" (qualitatively)
              - Connect that story to key numerical drivers: revenue growth, margins, reinvestment, risk
              - Conclude with value: FCFF DCF estimate, margin of safety, and relative valuation checks
              - Highlight major uncertainties and how they affect value
            Return ONLY the JSON specified below.""",
        ),
        (
            "human",
            """Ticker: {ticker}
Analysis data:
{analysis_data}
{upstream_context}
Respond EXACTLY in this JSON schema:
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

    def default_signal():
        return AswathDamodaranSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error; defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=AswathDamodaranSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )