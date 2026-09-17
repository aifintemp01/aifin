from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_insider_trades,
    get_company_news,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class GovernanceSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def governance_agent(
    state: AgentState,
    agent_id: str = "governance_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyzes corporate governance quality using proxy-based signals.
    Layer-aware: reads upstream context when in Layer 2+.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    governance_analysis: dict[str, dict] = {}
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

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date, limit=100, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=50, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            ["outstanding_shares", "free_cash_flow", "net_income", "revenue", "issuance_or_purchase_of_equity_shares"],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing insider ownership trend")
        insider_analysis = _analyze_insider_ownership(insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing share dilution trend")
        dilution_analysis = _analyze_share_dilution(line_items)

        progress.update_status(agent_id, ticker, "Analyzing earnings integrity")
        integrity_analysis = _analyze_earnings_integrity(line_items)

        progress.update_status(agent_id, ticker, "Analyzing governance news flags")
        news_analysis = _analyze_governance_news(company_news)

        total_score = (
            insider_analysis["score"] * 0.30
            + integrity_analysis["score"] * 0.30
            + dilution_analysis["score"] * 0.25
            + news_analysis["score"] * 0.15
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
            "insider_ownership_analysis": insider_analysis,
            "share_dilution_analysis": dilution_analysis,
            "earnings_integrity_analysis": integrity_analysis,
            "governance_news_analysis": news_analysis,
            "data_note": "Promoter holding %, pledge %, auditor history, and RPT unavailable. All metrics are proxy-based.",
        }

        progress.update_status(agent_id, ticker, "Generating governance analysis")
        governance_output = _generate_governance_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
            upstream_context=upstream_context,
        )

        governance_analysis[ticker] = {
            "signal": governance_output.signal,
            "confidence": governance_output.confidence,
            "reasoning": governance_output.reasoning,
        }

        # ── Build context JSON for downstream layers ──────────────────────────
        layer_context_updates[f"{agent_id}:{ticker}"] = {
            "agent": agent_id,
            "ticker": ticker,
            "layer": layer,
            "signal": governance_output.signal,
            "confidence": governance_output.confidence,
            "key_findings": [
                f"Governance score: {total_score:.1f}/{max_score}",
                f"Integrity: {integrity_analysis.get('details', 'N/A')[:80]}",
                f"News: {news_analysis.get('details', 'N/A')[:80]}",
            ],
            "data_summary": governance_output.reasoning[:300],
        }

        progress.update_status(agent_id, ticker, "Done", analysis=governance_output.reasoning)

    message = HumanMessage(content=json.dumps(governance_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(governance_analysis, "Governance Agent")

    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = governance_analysis
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


def _analyze_insider_ownership(insider_trades: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    if not insider_trades:
        return {"score": 5, "max_score": max_score, "details": "No insider trade data — defaulting to neutral", "net_shares": None, "buy_ratio": None}

    shares_bought = sum(
        getattr(t, "transaction_shares", 0) or 0
        for t in insider_trades
        if (getattr(t, "transaction_shares", 0) or 0) > 0
    )
    shares_sold = abs(sum(
        getattr(t, "transaction_shares", 0) or 0
        for t in insider_trades
        if (getattr(t, "transaction_shares", 0) or 0) < 0
    ))
    net_shares = shares_bought - shares_sold
    total_transactions = sum(
        1 for t in insider_trades
        if getattr(t, "transaction_shares", None) is not None
        and getattr(t, "transaction_shares", 0) != 0
    )

    if net_shares > 0:
        net_ratio = net_shares / max(shares_sold, 1)
        if net_ratio > 2.0:
            score += 3
            details.append(f"Strong net insider buying: {net_shares:,.0f} net shares")
        elif net_ratio > 0.5:
            score += 2
            details.append(f"Moderate net insider buying: {net_shares:,.0f} net shares")
        else:
            score += 1
            details.append(f"Slight net insider buying: {net_shares:,.0f} net shares")
    else:
        details.append(f"Net insider selling: {net_shares:,.0f} net shares")

    buy_ratio = None
    if total_transactions > 0:
        buy_count = sum(1 for t in insider_trades if (getattr(t, "transaction_shares", 0) or 0) > 0)
        buy_ratio = buy_count / total_transactions
        if buy_ratio > 0.65:
            score += 2
            details.append(f"High buy ratio: {buy_ratio:.0%}")
        elif buy_ratio > 0.40:
            score += 1
            details.append(f"Moderate buy ratio: {buy_ratio:.0%}")
        else:
            details.append(f"Low buy ratio: {buy_ratio:.0%}")

    board_buys = sum(1 for t in insider_trades if getattr(t, "is_board_director", False) is True and (getattr(t, "transaction_shares", 0) or 0) > 0)
    board_sells = sum(1 for t in insider_trades if getattr(t, "is_board_director", False) is True and (getattr(t, "transaction_shares", 0) or 0) < 0)

    if board_buys > board_sells and board_buys > 0:
        score += 1
        details.append(f"Board directors buying: {board_buys} transactions")
    elif board_sells > board_buys and board_sells > 0:
        details.append(f"Board directors selling: {board_sells} transactions")
    else:
        details.append("Board director activity: neutral or no data")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "net_shares": round(net_shares, 0),
        "buy_ratio": round(buy_ratio, 4) if buy_ratio is not None else None,
        "board_buys": board_buys,
        "board_sells": board_sells,
    }


def _analyze_share_dilution(line_items: list) -> dict:
    max_score = 5
    score = 0
    details: list[str] = []

    share_counts = [item.outstanding_shares for item in line_items if item.outstanding_shares is not None]

    if len(share_counts) >= 2:
        latest_shares = share_counts[0]
        oldest_shares = share_counts[-1]
        n = len(share_counts) - 1
        if oldest_shares > 0 and latest_shares > 0:
            share_cagr = (latest_shares / oldest_shares) ** (1 / n) - 1
            if share_cagr < -0.01:
                score += 3
                details.append(f"Share count shrinking: {share_cagr:.1%} CAGR")
            elif share_cagr < 0.01:
                score += 2
                details.append(f"Share count stable: {share_cagr:.1%} CAGR")
            elif share_cagr < 0.03:
                score += 1
                details.append(f"Modest dilution: {share_cagr:.1%} CAGR")
            else:
                details.append(f"Significant dilution: {share_cagr:.1%} CAGR")
        else:
            share_cagr = None
            details.append("Share count CAGR: invalid base value")
    else:
        share_cagr = None
        details.append("Share count trend: insufficient history")

    latest_item = _latest(line_items)
    equity_activity = latest_item.issuance_or_purchase_of_equity_shares if latest_item else None

    if equity_activity is not None:
        if equity_activity < 0:
            score += 2
            details.append(f"Active buybacks: ${abs(equity_activity):,.0f} returned")
        elif equity_activity > 0:
            details.append(f"Equity issuance: ${equity_activity:,.0f}")
        else:
            score += 1
            details.append("No equity issuance or buyback activity")
    else:
        details.append("Equity issuance/buyback: data unavailable")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "share_cagr": round(share_cagr, 4) if share_cagr is not None else None,
    }


def _analyze_earnings_integrity(line_items: list) -> dict:
    max_score = 6
    score = 0
    details: list[str] = []

    conversions = []
    for item in line_items:
        fcf = item.free_cash_flow
        ni = item.net_income
        if fcf is not None and ni is not None and ni > 0:
            conversions.append(fcf / ni)

    if not conversions:
        return {"score": 0, "max_score": max_score, "details": "Earnings integrity: insufficient FCF/net income data", "avg_fcf_conversion": None}

    avg_conversion = sum(conversions) / len(conversions)

    if avg_conversion >= 1.10:
        score += 4
        details.append(f"Exceptional earnings integrity: {avg_conversion:.2f}x FCF conversion")
    elif avg_conversion >= 0.90:
        score += 3
        details.append(f"Strong earnings integrity: {avg_conversion:.2f}x FCF conversion")
    elif avg_conversion >= 0.70:
        score += 2
        details.append(f"Adequate earnings integrity: {avg_conversion:.2f}x FCF conversion")
    elif avg_conversion >= 0.50:
        score += 1
        details.append(f"Weak earnings integrity: {avg_conversion:.2f}x FCF conversion")
    else:
        details.append(f"Poor earnings integrity: {avg_conversion:.2f}x FCF conversion")

    positive_periods = sum(1 for c in conversions if c > 0)
    consistency = positive_periods / len(conversions)

    if consistency >= 0.90:
        score += 2
        details.append(f"Highly consistent: {positive_periods}/{len(conversions)} periods positive")
    elif consistency >= 0.70:
        score += 1
        details.append(f"Mostly consistent: {positive_periods}/{len(conversions)} periods positive")
    else:
        details.append(f"Inconsistent: {positive_periods}/{len(conversions)} periods positive")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_fcf_conversion": round(avg_conversion, 4),
        "conversion_consistency": round(consistency, 4),
    }


def _analyze_governance_news(company_news: list) -> dict:
    max_score = 3
    score = 0
    details: list[str] = []

    GOVERNANCE_FLAGS = {
        "tier1": ["fraud", "embezzlement", "bribery", "sec investigation", "criminal", "indicted", "arrested", "money laundering"],
        "tier2": ["lawsuit", "litigation", "regulatory action", "misconduct", "insider trading", "accounting irregularities", "restatement", "whistleblower", "class action"],
        "tier3": ["resignation", "dispute", "conflict of interest", "related party", "nepotism", "board disagreement"],
    }

    if not company_news:
        return {"score": 5, "max_score": max_score, "details": "No recent news — defaulting to neutral", "flags_found": []}

    flags_found: list[str] = []
    tier1_hits = tier2_hits = tier3_hits = 0

    for article in company_news:
        title_lower = (article.title or "").lower()
        for keyword in GOVERNANCE_FLAGS["tier1"]:
            if keyword in title_lower:
                tier1_hits += 1
                flags_found.append(f"[CRITICAL] '{keyword}' in: {article.title[:60]}")
        for keyword in GOVERNANCE_FLAGS["tier2"]:
            if keyword in title_lower:
                tier2_hits += 1
                flags_found.append(f"[SERIOUS] '{keyword}' in: {article.title[:60]}")
        for keyword in GOVERNANCE_FLAGS["tier3"]:
            if keyword in title_lower:
                tier3_hits += 1
                flags_found.append(f"[WATCH] '{keyword}' in: {article.title[:60]}")

    if tier1_hits > 0:
        score = 0
        details.append(f"Critical governance flags: {tier1_hits} article(s)")
    elif tier2_hits > 0:
        score = 1
        details.append(f"Serious governance flags: {tier2_hits} article(s)")
    elif tier3_hits > 0:
        score = 2
        details.append(f"Minor governance flags: {tier3_hits} article(s)")
    else:
        score = 3
        details.append(f"Clean governance press: no red flags in {len(company_news)} articles")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "flags_found": flags_found[:10],
        "tier1_hits": tier1_hits,
        "tier2_hits": tier2_hits,
        "tier3_hits": tier3_hits,
    }


def _generate_governance_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    upstream_context: dict = None,
) -> GovernanceSignal:

    upstream_section = ""
    if upstream_context:
        upstream_section = "\n\nContext from upstream agents:\n"
        for upstream_id, ctx in upstream_context.items():
            upstream_section += f"{ctx.get('agent', upstream_id)}: {json.dumps(ctx, indent=2)}\n"
        upstream_section += "\nIncorporate this context into your governance assessment.\n"

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a disciplined corporate governance analyst. Your mandate:
            - Insider buying signals alignment; heavy selling signals extraction
            - Share dilution is a slow tax on shareholders
            - FCF conversion above 1.0x = reported earnings are real
            - Governance red flags in news often precede financial deterioration
            - Note: promoter holdings, pledge data, auditor history, RPT unavailable — proxy-based analysis

            Reasoning should cover: insider ownership, dilution, earnings integrity, news flags.""",
        ),
        (
            "human",
            """Based on the following data, generate a governance quality signal for {ticker}:

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
        return GovernanceSignal(signal="neutral", confidence=0.0, reasoning="Parsing error — defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=GovernanceSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default,
    )