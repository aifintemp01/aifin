from __future__ import annotations

import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_financial_metrics,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class MacroExposureSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def macro_exposure_agent(
    state: AgentState,
    agent_id: str = "macro_exposure_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes stocks using a comprehensive macro exposure framework.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    analysis_data: dict[str, dict] = {}
    macro_exposure_analysis: dict[str, dict] = {}
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
                "total_debt", "cash_and_equivalents", "shareholders_equity",
                "interest_expense", "operating_income", "ebit", "revenue",
                "gross_profit", "gross_margin", "capital_expenditure",
                "net_income", "total_assets",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing interest rate sensitivity")
        rate_sensitivity = _analyze_rate_sensitivity(metrics, line_items)

        progress.update_status(agent_id, ticker, "Analyzing inflation sensitivity")
        inflation_sensitivity = _analyze_inflation_sensitivity(line_items)

        progress.update_status(agent_id, ticker, "Analyzing FX dependency")
        fx_dependency = _analyze_fx_dependency(metrics, line_items)

        total_score = (
            rate_sensitivity["score"] * 0.35
            + inflation_sensitivity["score"] * 0.35
            + fx_dependency["score"] * 0.30
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
            "rate_sensitivity": rate_sensitivity,
            "inflation_sensitivity": inflation_sensitivity,
            "fx_dependency": fx_dependency,
            "composite_macro_exposure": {
                "description": "Weighted composite of rate, inflation, and FX exposure (higher = more resilient)",
                "score": round(total_score, 2),
                "weights": {"rate_sensitivity": 0.35, "inflation_sensitivity": 0.35, "fx_dependency": 0.30},
            },
        }

        progress.update_status(agent_id, ticker, "Generating macro exposure analysis")
        macro_output = _generate_macro_exposure_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        macro_exposure_analysis[ticker] = {
            "signal": macro_output.signal,
            "confidence": macro_output.confidence,
            "reasoning": macro_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": macro_output.signal,
            "confidence": macro_output.confidence,
            "key_findings": [
                f"Macro score: {total_score:.1f}/{max_score}",
                f"Rate sensitivity: {rate_sensitivity.get('details', 'N/A')[:80]}",
                f"Inflation: {inflation_sensitivity.get('details', 'N/A')[:80]}",
            ],
            "data_summary": macro_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=macro_output.reasoning)

    message = HumanMessage(content=json.dumps(macro_exposure_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(macro_exposure_analysis, "Macro Exposure Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = macro_exposure_analysis
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


def _analyze_rate_sensitivity(metrics: list, line_items: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    latest_item = _latest(line_items)
    latest_metrics = metrics[0] if metrics else None

    de_ratio = latest_metrics.debt_to_equity if latest_metrics else None
    if de_ratio is None and latest_item:
        debt = latest_item.total_debt
        equity = latest_item.shareholders_equity
        if debt is not None and equity and equity > 0:
            de_ratio = debt / equity

    if de_ratio is not None:
        if de_ratio < 0.3:
            score += 2
            details.append(f"Low rate sensitivity: D/E {de_ratio:.2f}")
        elif de_ratio < 0.8:
            score += 1
            details.append(f"Moderate rate sensitivity: D/E {de_ratio:.2f}")
        else:
            details.append(f"High rate sensitivity: D/E {de_ratio:.2f}")
    else:
        details.append("D/E ratio: insufficient data")

    ebit = (latest_item.ebit or latest_item.operating_income) if latest_item else None
    interest_expense = latest_item.interest_expense if latest_item else None
    interest_coverage = latest_metrics.interest_coverage if latest_metrics else None
    if interest_coverage is None and ebit and interest_expense and interest_expense > 0:
        interest_coverage = ebit / interest_expense

    if interest_coverage is not None:
        if interest_coverage > 10:
            score += 2
            details.append(f"Strong interest coverage: {interest_coverage:.1f}x")
        elif interest_coverage > 4:
            score += 1
            details.append(f"Adequate interest coverage: {interest_coverage:.1f}x")
        else:
            details.append(f"Weak interest coverage: {interest_coverage:.1f}x")
    else:
        details.append("Interest coverage: insufficient data")

    if latest_item:
        debt = latest_item.total_debt
        cash = latest_item.cash_and_equivalents
        if debt is not None and cash is not None:
            net_cash = cash - debt
            if net_cash > 0:
                score += 2
                details.append(f"Net cash position: ${net_cash:,.0f}")
            else:
                details.append(f"Net debt position: ${net_cash:,.0f}")
        else:
            details.append("Net cash/debt: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


def _analyze_inflation_sensitivity(line_items: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    gm_values = [item.gross_margin for item in line_items if item.gross_margin is not None]

    if gm_values:
        avg_gm = sum(gm_values) / len(gm_values)
        is_stable = len(gm_values) < 2 or gm_values[0] >= gm_values[-1] * 0.95
        if avg_gm > 0.40 and is_stable:
            score += 4
            details.append(f"Strong pricing power: avg gross margin {avg_gm:.1%}, stable/expanding")
        elif avg_gm > 0.25 and is_stable:
            score += 3
            details.append(f"Decent pricing power: avg gross margin {avg_gm:.1%}, stable")
        elif avg_gm > 0.40:
            score += 2
            details.append(f"High but compressing margins: avg {avg_gm:.1%}")
        elif avg_gm > 0.15:
            score += 1
            details.append(f"Thin margins: avg {avg_gm:.1%}")
        else:
            details.append(f"Very thin margins: avg {avg_gm:.1%}")
    else:
        details.append("Gross margin: insufficient data")

    capex_to_rev = []
    for item in line_items:
        capex = item.capital_expenditure
        revenue = item.revenue
        if capex is not None and revenue and revenue > 0:
            capex_to_rev.append(abs(capex) / revenue)

    if capex_to_rev:
        avg_capex_intensity = sum(capex_to_rev) / len(capex_to_rev)
        if avg_capex_intensity < 0.03:
            score += 2
            details.append(f"Asset-light: capex {avg_capex_intensity:.1%} of revenue")
        elif avg_capex_intensity < 0.08:
            score += 1
            details.append(f"Moderate capex intensity: {avg_capex_intensity:.1%} of revenue")
        else:
            details.append(f"Capex heavy: {avg_capex_intensity:.1%} of revenue")
    else:
        details.append("Capex intensity: insufficient data")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


def _analyze_fx_dependency(metrics: list, line_items: list) -> dict:
    max_score = 4
    score = 0
    details: list[str] = []

    revenues = [item.revenue for item in line_items if item.revenue is not None]

    if len(revenues) >= 3:
        growth_rates = []
        for i in range(len(revenues) - 1):
            if revenues[i + 1] and revenues[i + 1] > 0:
                growth_rates.append((revenues[i] - revenues[i + 1]) / revenues[i + 1])
        if growth_rates:
            n = len(growth_rates)
            mean_g = sum(growth_rates) / n
            variance = sum((g - mean_g) ** 2 for g in growth_rates) / max(n - 1, 1)
            rev_volatility = variance ** 0.5
            if rev_volatility < 0.05:
                score += 2
                details.append(f"Stable revenue: {rev_volatility:.1%} YoY volatility")
            elif rev_volatility < 0.15:
                score += 1
                details.append(f"Moderate revenue volatility: {rev_volatility:.1%}")
            else:
                details.append(f"High revenue volatility: {rev_volatility:.1%}")
        else:
            details.append("Revenue growth volatility: insufficient data")
    else:
        details.append("Revenue stability: insufficient history")

    latest_item = _latest(line_items)
    if latest_item:
        total_assets = latest_item.total_assets
        revenue = latest_item.revenue
        if total_assets and revenue and revenue > 0:
            asset_to_rev = total_assets / revenue
            if asset_to_rev < 1.0:
                score += 2
                details.append(f"Asset-light: {asset_to_rev:.2f}x revenue")
            elif asset_to_rev < 2.0:
                score += 1
                details.append(f"Moderate asset base: {asset_to_rev:.2f}x revenue")
            else:
                details.append(f"Asset-heavy: {asset_to_rev:.2f}x revenue")
        else:
            details.append("Asset efficiency: insufficient data")

    details.append("Note: direct international revenue split unavailable — proxies used")

    return {"score": (score / max_score) * 10, "max_score": max_score, "details": "; ".join(details)}


def _generate_macro_exposure_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> MacroExposureSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your macro exposure assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined macro exposure analyst. Your mandate:
            - Interest rate sensitivity is primarily a function of debt load and interest coverage
            - Inflation resilience requires pricing power: stable or expanding gross margins
            - FX dependency creates earnings volatility that is hard to predict or hedge
            - A company with low debt, strong margins, and stable revenue is a macro-resilient compounder

            Reasoning: rate sensitivity → inflation resilience → FX risk → composite verdict.""",
        ),
        (
            "human",
            """Based on the following data, generate a macro exposure signal for {ticker}:

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
        return MacroExposureSignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=MacroExposureSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )