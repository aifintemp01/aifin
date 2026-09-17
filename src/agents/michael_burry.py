from __future__ import annotations
from datetime import datetime, timedelta
import json
from typing_extensions import Literal
from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class MichaelBurrySignal(BaseModel):
    """Schema returned by the LLM."""
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def michael_burry_agent(
    state: AgentState,
    agent_id: str = "michael_burry_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """Analyse stocks using Michael Burry's deep-value, contrarian framework."""
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}
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

        progress.update_status(agent_id, ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow", "net_income", "total_debt",
                "cash_and_equivalents", "total_assets", "total_liabilities",
                "outstanding_shares", "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status(agent_id, ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Analyzing deep value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status(agent_id, ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(line_items)

        progress.update_status(agent_id, ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades, line_items)

        progress.update_status(agent_id, ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        total_score = (
            value_analysis["score"]
            + balance_sheet_analysis["score"]
            + insider_analysis["score"]
            + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"]
            + balance_sheet_analysis["max_score"]
            + insider_analysis["max_score"]
            + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status(agent_id, ticker, "Generating LLM output")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "key_findings": [
                f"Burry score: {total_score}/{max_score}",
                f"Value: {value_analysis.get('details', 'N/A')[:80]}",
                f"Contrarian: {contrarian_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": burry_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=burry_output.reasoning)

    message = HumanMessage(content=json.dumps(burry_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = burry_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


# ── Analysis helpers (unchanged from original) ────────────────────────────────

def _analyze_value(metrics: list, line_items: list, market_cap: float | None) -> dict:
    score = 0
    max_score = 6
    details: list[str] = []

    if not metrics or not line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data for value analysis"}

    latest = line_items[0]
    latest_metrics = metrics[0]

    # FCF yield
    fcf = latest.free_cash_flow
    if fcf is not None and fcf > 0:
        fcf_yield = fcf / market_cap
        if fcf_yield > 0.12:
            score += 3
            details.append(f"Exceptional FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.08:
            score += 2
            details.append(f"Strong FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0.05:
            score += 1
            details.append(f"Decent FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Weak FCF yield {fcf_yield:.1%}")
    else:
        details.append("Negative or missing FCF")

    # EV/EBIT proxy via P/E
    pe = latest_metrics.price_to_earnings_ratio
    if pe is not None and pe > 0:
        if pe < 8:
            score += 2
            details.append(f"Very cheap P/E {pe:.1f}x")
        elif pe < 15:
            score += 1
            details.append(f"Reasonable P/E {pe:.1f}x")
        else:
            details.append(f"Expensive P/E {pe:.1f}x")

    # EV/EBITDA
    ev_ebitda = latest_metrics.enterprise_value_to_ebitda_ratio
    if ev_ebitda is not None and ev_ebitda > 0:
        if ev_ebitda < 6:
            score += 1
            details.append(f"Low EV/EBITDA {ev_ebitda:.1f}x")
        elif ev_ebitda > 20:
            details.append(f"High EV/EBITDA {ev_ebitda:.1f}x")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_balance_sheet(line_items: list) -> dict:
    score = 0
    max_score = 4
    details: list[str] = []

    if not line_items:
        return {"score": 0, "max_score": max_score, "details": "Insufficient data"}

    latest = line_items[0]
    debt = latest.total_debt
    cash = latest.cash_and_equivalents
    assets = latest.total_assets
    liabilities = latest.total_liabilities

    if debt is not None and cash is not None:
        net_debt = debt - cash
        if net_debt < 0:
            score += 2
            details.append(f"Net cash position ${-net_debt:,.0f}")
        elif assets and assets > 0:
            nd_ratio = net_debt / assets
            if nd_ratio < 0.2:
                score += 1
                details.append(f"Low net debt/assets {nd_ratio:.2f}")
            else:
                details.append(f"Elevated net debt/assets {nd_ratio:.2f}")

    if assets and liabilities and assets > 0:
        liability_ratio = liabilities / assets
        if liability_ratio < 0.4:
            score += 2
            details.append(f"Conservative leverage {liability_ratio:.2f}")
        elif liability_ratio < 0.6:
            score += 1
            details.append(f"Moderate leverage {liability_ratio:.2f}")
        else:
            details.append(f"High leverage {liability_ratio:.2f}")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_insider_activity(insider_trades: list, line_items: list) -> dict:
    score = 0
    max_score = 3
    details: list[str] = []

    # Check buybacks via issuance_or_purchase_of_equity_shares
    if line_items:
        latest = line_items[0]
        buyback = latest.issuance_or_purchase_of_equity_shares
        if buyback is not None and buyback < 0:
            score += 1
            details.append(f"Share buybacks: ${abs(buyback):,.0f}")

    if not insider_trades:
        details.append("No insider trade data available")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    buys = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and
               t.transaction_type.lower() in ['buy', 'purchase'])
    sells = sum(1 for t in insider_trades if getattr(t, 'transaction_type', None) and
                t.transaction_type.lower() in ['sell', 'sale'])

    if buys > sells * 2:
        score += 2
        details.append(f"Strong insider buying: {buys} buys vs {sells} sells")
    elif buys > sells:
        score += 1
        details.append(f"Mild insider buying: {buys} buys vs {sells} sells")
    elif sells > buys * 2:
        details.append(f"Heavy insider selling: {sells} sells vs {buys} buys")
    else:
        details.append(f"Mixed insider activity: {buys} buys, {sells} sells")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _analyze_contrarian_sentiment(news: list) -> dict:
    score = 0
    max_score = 2
    details: list[str] = []

    if not news:
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    sentiment_negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )
    if sentiment_negative_count >= 5:
        score += 1
        details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> MichaelBurrySignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "Incorporate this context into your contrarian assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI agent emulating Dr. Michael J. Burry. Your mandate:
            - Hunt for deep value in equities using hard numbers (free cash flow, EV/EBIT, balance sheet)
            - Be contrarian: hatred in the press can be your friend if fundamentals are solid
            - Focus on downside first – avoid leveraged balance sheets
            - Look for hard catalysts such as insider buying, buybacks, or asset sales
            - Communicate in Burry's terse, data-driven style
            When providing your reasoning, be thorough and specific by:
            1. Start with the key metric(s) that drove your decision
            2. Cite concrete numbers (e.g. "FCF yield 14.7%", "EV/EBIT 5.3")
            3. Highlight risk factors and why they are acceptable (or not)
            4. Mention relevant insider activity or contrarian opportunities
            5. Use Burry's direct, number-focused communication style with minimal words""",
        ),
        (
            "human",
            """Based on the following data, create the investment signal as Michael Burry would:

Analysis Data for {ticker}:
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

    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="Parsing error – defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=MichaelBurrySignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_michael_burry_signal,
    )