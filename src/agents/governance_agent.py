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
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def governance_agent(state: AgentState, agent_id: str = "governance_agent"):
    """
    Analyzes stocks using a comprehensive corporate governance framework.
    Since promoter holdings, pledge data, auditor history, and related party
    transactions are not available in the current data model, governance quality
    is assessed via four available proxies:

    - Insider ownership trend: net insider buying/selling as skin-in-the-game proxy
    - Share dilution trend: outstanding shares growth as shareholder alignment proxy
    - Earnings integrity: FCF conversion quality as accounting quality proxy
    - News-based governance flags: negative press around governance red flags

    Good governance compounds returns. Poor governance destroys them silently.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    # Insider trades and news need a lookback window
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    governance_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(
            ticker,
            end_date=end_date,
            start_date=start_date,
            limit=100,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(
            ticker,
            end_date=end_date,
            start_date=start_date,
            limit=50,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Fetching financial line items")
        line_items = search_line_items(
            ticker,
            [
                "outstanding_shares",
                "free_cash_flow",
                "net_income",
                "revenue",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
            period="annual",
            limit=5,
            api_key=api_key,
        )

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing insider ownership trend")
        insider_analysis = _analyze_insider_ownership(insider_trades)

        progress.update_status(agent_id, ticker, "Analyzing share dilution trend")
        dilution_analysis = _analyze_share_dilution(line_items)

        progress.update_status(agent_id, ticker, "Analyzing earnings integrity")
        integrity_analysis = _analyze_earnings_integrity(line_items)

        progress.update_status(agent_id, ticker, "Analyzing governance news flags")
        news_analysis = _analyze_governance_news(company_news)

        # ------------------------------------------------------------------
        # Aggregate score
        # Governance weights: insider ownership and earnings integrity are the
        # most reliable signals; dilution and news provide confirmation
        # ------------------------------------------------------------------
        total_score = (
            insider_analysis["score"] * 0.30        # skin in the game = alignment
            + integrity_analysis["score"] * 0.30    # FCF vs earnings = accounting quality
            + dilution_analysis["score"] * 0.25     # share count = shareholder respect
            + news_analysis["score"] * 0.15         # governance red flags in press
        )
        max_score = 10

        if total_score >= 6.5:
            signal = "bullish"
        elif total_score <= 3.5:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect for LLM
        # ------------------------------------------------------------------
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "insider_ownership_analysis": insider_analysis,
            "share_dilution_analysis": dilution_analysis,
            "earnings_integrity_analysis": integrity_analysis,
            "governance_news_analysis": news_analysis,
            "data_note": (
                "Promoter holding %, pledge %, auditor history, and RPT unavailable "
                "in current data model. All metrics are proxy-based."
            ),
        }

        progress.update_status(agent_id, ticker, "Generating governance analysis")
        governance_output = _generate_governance_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        governance_analysis[ticker] = {
            "signal": governance_output.signal,
            "confidence": governance_output.confidence,
            "reasoning": governance_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=governance_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(governance_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(governance_analysis, "Governance Agent")

    state["data"]["analyst_signals"][agent_id] = governance_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub-analysis helpers
###############################################################################

def _safe_get(obj, attr: str):
    """Safely get an attribute from an object, returning None if missing."""
    return getattr(obj, attr, None) if obj is not None else None


def _latest(line_items: list):
    """Return the most recent line-item object or None."""
    return line_items[0] if line_items else None


# ----- Insider Ownership Trend (Promoter Holding Proxy) ---------------------

def _analyze_insider_ownership(insider_trades: list) -> dict:
    """
    Proxy for promoter/management skin-in-the-game via insider trade activity.
    Net buying signals confidence and alignment with shareholders.
    Net selling is ambiguous but heavy selling is a governance red flag.
    Board director trades carry more weight than other insiders.

    Metrics:
    - Net shares bought/sold over the lookback period
    - Buy ratio (buys / total transactions)
    - Board director buy ratio (higher weight — these are the stewards)
    """
    max_score = 6  # 3pts net direction + 2pts buy ratio + 1pt board director signal
    score = 0
    details: list[str] = []

    if not insider_trades:
        return {
            "score": 5,  # neutral default — absence of data ≠ bad governance
            "max_score": max_score,
            "details": "No insider trade data available — defaulting to neutral",
            "net_shares": None,
            "buy_ratio": None,
        }

    shares_bought = sum(
        _safe_get(t, "transaction_shares") or 0
        for t in insider_trades
        if (_safe_get(t, "transaction_shares") or 0) > 0
    )
    shares_sold = abs(sum(
        _safe_get(t, "transaction_shares") or 0
        for t in insider_trades
        if (_safe_get(t, "transaction_shares") or 0) < 0
    ))
    net_shares = shares_bought - shares_sold
    total_transactions = sum(
        1 for t in insider_trades
        if _safe_get(t, "transaction_shares") is not None
        and _safe_get(t, "transaction_shares") != 0
    )

    # Net direction: 0–3 pts
    if net_shares > 0:
        net_ratio = net_shares / max(shares_sold, 1)
        if net_ratio > 2.0:
            score += 3
            details.append(f"Strong net insider buying: {net_shares:,.0f} net shares — high alignment")
        elif net_ratio > 0.5:
            score += 2
            details.append(f"Moderate net insider buying: {net_shares:,.0f} net shares")
        else:
            score += 1
            details.append(f"Slight net insider buying: {net_shares:,.0f} net shares")
    else:
        details.append(f"Net insider selling: {net_shares:,.0f} net shares — potential misalignment")

    # Buy ratio: 0–2 pts
    buy_ratio = None
    if total_transactions > 0:
        buy_count = sum(
            1 for t in insider_trades
            if (_safe_get(t, "transaction_shares") or 0) > 0
        )
        buy_ratio = buy_count / total_transactions
        if buy_ratio > 0.65:
            score += 2
            details.append(f"High buy ratio: {buy_ratio:.0%} of transactions are purchases")
        elif buy_ratio > 0.40:
            score += 1
            details.append(f"Moderate buy ratio: {buy_ratio:.0%}")
        else:
            details.append(f"Low buy ratio: {buy_ratio:.0%} — mostly selling")

    # Board director signal: 0–1 pt
    board_buys = sum(
        1 for t in insider_trades
        if _safe_get(t, "is_board_director") is True
        and (_safe_get(t, "transaction_shares") or 0) > 0
    )
    board_sells = sum(
        1 for t in insider_trades
        if _safe_get(t, "is_board_director") is True
        and (_safe_get(t, "transaction_shares") or 0) < 0
    )
    if board_buys > board_sells and board_buys > 0:
        score += 1
        details.append(f"Board directors buying: {board_buys} purchase transactions — stewardship signal")
    elif board_sells > board_buys and board_sells > 0:
        details.append(f"Board directors selling: {board_sells} sale transactions — governance caution")
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


# ----- Share Dilution Trend (Shareholder Alignment Proxy) -------------------

def _analyze_share_dilution(line_items: list) -> dict:
    """
    Assess shareholder alignment via outstanding share count trend.
    Companies that consistently issue shares dilute existing holders —
    a form of governance failure even when not malicious.
    Buybacks are the strongest signal of management prioritizing shareholders.

    Metrics:
    - Share count CAGR over available history
    - Issuance vs purchase of equity shares (most recent period)
    """
    max_score = 5  # 3pts share count trend + 2pts equity issuance/buyback
    score = 0
    details: list[str] = []

    # Share count CAGR
    share_counts = [
        _safe_get(item, "outstanding_shares")
        for item in line_items
        if _safe_get(item, "outstanding_shares") is not None
    ]

    if len(share_counts) >= 2:
        latest_shares = share_counts[0]
        oldest_shares = share_counts[-1]
        n = len(share_counts) - 1
        if oldest_shares > 0 and latest_shares > 0:
            share_cagr = (latest_shares / oldest_shares) ** (1 / n) - 1
            if share_cagr < -0.01:
                score += 3
                details.append(f"Share count shrinking: {share_cagr:.1%} CAGR — buyback-driven shareholder returns")
            elif share_cagr < 0.01:
                score += 2
                details.append(f"Share count stable: {share_cagr:.1%} CAGR — no dilution")
            elif share_cagr < 0.03:
                score += 1
                details.append(f"Modest dilution: {share_cagr:.1%} CAGR — minor shareholder impact")
            else:
                details.append(f"Significant dilution: {share_cagr:.1%} CAGR — shareholder value erosion")
        else:
            details.append("Share count CAGR: invalid base value")
    else:
        details.append("Share count trend: insufficient history")

    # Equity issuance vs buyback (most recent period)
    latest_item = _latest(line_items)
    equity_activity = _safe_get(latest_item, "issuance_or_purchase_of_equity_shares")

    if equity_activity is not None:
        if equity_activity < 0:
            # Negative = cash outflow = buyback
            score += 2
            details.append(f"Active buybacks: ${abs(equity_activity):,.0f} returned to shareholders")
        elif equity_activity > 0:
            details.append(f"Equity issuance: ${equity_activity:,.0f} — dilutive activity")
        else:
            score += 1
            details.append("No equity issuance or buyback activity")
    else:
        details.append("Equity issuance/buyback: data unavailable")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "share_cagr": round(
            (share_counts[0] / share_counts[-1]) ** (1 / (len(share_counts) - 1)) - 1, 4
        ) if len(share_counts) >= 2 and share_counts[-1] > 0 else None,
    }


# ----- Earnings Integrity (Accounting Quality / Auditor Proxy) --------------

def _analyze_earnings_integrity(line_items: list) -> dict:
    """
    Proxy for auditor quality and related party transaction risk via
    FCF-to-net-income conversion ratio.
    Companies that consistently report earnings but generate little cash
    are either aggressive in accounting or destroying capital invisibly.
    High and consistent FCF conversion is the strongest accounting integrity signal.

    Metrics:
    - FCF conversion ratio: FCF / net income (averaged across periods)
    - Consistency: how many periods show positive FCF conversion
    """
    max_score = 6  # 4pts avg conversion + 2pts consistency
    score = 0
    details: list[str] = []

    conversions = []
    for item in line_items:
        fcf = _safe_get(item, "free_cash_flow")
        ni = _safe_get(item, "net_income")
        if fcf is not None and ni is not None and ni > 0:
            conversions.append(fcf / ni)

    if not conversions:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Earnings integrity: insufficient FCF/net income data",
            "avg_fcf_conversion": None,
        }

    avg_conversion = sum(conversions) / len(conversions)

    # Average FCF conversion: 0–4 pts
    if avg_conversion >= 1.10:
        score += 4
        details.append(f"Exceptional earnings integrity: {avg_conversion:.2f}x FCF conversion — cash exceeds reported profits")
    elif avg_conversion >= 0.90:
        score += 3
        details.append(f"Strong earnings integrity: {avg_conversion:.2f}x FCF conversion")
    elif avg_conversion >= 0.70:
        score += 2
        details.append(f"Adequate earnings integrity: {avg_conversion:.2f}x FCF conversion")
    elif avg_conversion >= 0.50:
        score += 1
        details.append(f"Weak earnings integrity: {avg_conversion:.2f}x FCF conversion — earnings quality questionable")
    else:
        details.append(f"Poor earnings integrity: {avg_conversion:.2f}x FCF conversion — significant earnings/cash divergence")

    # Consistency: periods with positive conversion: 0–2 pts
    positive_periods = sum(1 for c in conversions if c > 0)
    consistency = positive_periods / len(conversions)

    if consistency >= 0.90:
        score += 2
        details.append(f"Highly consistent cash generation: {positive_periods}/{len(conversions)} periods positive")
    elif consistency >= 0.70:
        score += 1
        details.append(f"Mostly consistent: {positive_periods}/{len(conversions)} periods positive")
    else:
        details.append(f"Inconsistent cash generation: {positive_periods}/{len(conversions)} periods positive")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "avg_fcf_conversion": round(avg_conversion, 4),
        "conversion_consistency": round(consistency, 4),
    }


# ----- Governance News Flags ------------------------------------------------

def _analyze_governance_news(company_news: list) -> dict:
    """
    Scan recent news for governance red flag keywords.
    Fraud, litigation, related party abuse, regulatory action, and
    management misconduct are the most common governance warning signs
    that appear in press before they appear in financial statements.

    Keywords are organized by severity tier:
    - Tier 1 (critical): fraud, embezzlement, SEC investigation, bribery
    - Tier 2 (serious): lawsuit, regulatory, misconduct, insider trading
    - Tier 3 (watch): resignation, dispute, conflict of interest
    """
    max_score = 3
    score = 0
    details: list[str] = []

    GOVERNANCE_FLAGS = {
        "tier1": [
            "fraud", "embezzlement", "bribery", "sec investigation",
            "criminal", "indicted", "arrested", "money laundering",
        ],
        "tier2": [
            "lawsuit", "litigation", "regulatory action", "misconduct",
            "insider trading", "accounting irregularities", "restatement",
            "whistleblower", "class action",
        ],
        "tier3": [
            "resignation", "dispute", "conflict of interest",
            "related party", "nepotism", "board disagreement",
        ],
    }

    if not company_news:
        return {
            "score": 5,  # neutral — no news ≠ bad governance
            "max_score": max_score,
            "details": "No recent news available — defaulting to neutral",
            "flags_found": [],
        }

    flags_found: list[str] = []
    tier1_hits = 0
    tier2_hits = 0
    tier3_hits = 0

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

    # Score: clean press = full marks, each tier deducts
    if tier1_hits > 0:
        score = 0
        details.append(f"Critical governance flags: {tier1_hits} article(s) — fraud/investigation risk")
    elif tier2_hits > 0:
        score = 1
        details.append(f"Serious governance flags: {tier2_hits} article(s) — litigation/misconduct risk")
    elif tier3_hits > 0:
        score = 2
        details.append(f"Minor governance flags: {tier3_hits} article(s) — worth monitoring")
    else:
        score = 3
        details.append(f"Clean governance press: no red flags in {len(company_news)} recent articles")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "flags_found": flags_found[:10],  # cap at 10 for LLM context window efficiency
        "tier1_hits": tier1_hits,
        "tier2_hits": tier2_hits,
        "tier3_hits": tier3_hits,
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_governance_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> GovernanceSignal:
    """Generate a governance quality signal grounded strictly in the computed proxies."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined corporate governance analyst. Your mandate:
                - Good governance is the foundation of sustainable shareholder value creation
                - Insider buying signals alignment; heavy selling without explanation signals extraction
                - Share dilution is a slow tax on shareholders — consistent buybacks are the opposite
                - FCF conversion above 1.0x is the clearest sign that reported earnings are real
                - Governance red flags in news often precede financial deterioration by 1-2 years
                - Note: promoter holdings, pledge data, auditor history, and RPT are unavailable —
                  all analysis is proxy-based; communicate this limitation clearly in your reasoning

                When providing your reasoning, be specific by:
                1. Assessing insider ownership trend — are insiders buying or exiting?
                2. Evaluating share dilution — are shareholders being respected or diluted?
                3. Commenting on earnings integrity — does cash match reported profits?
                4. Flagging any governance red flags found in recent news
                5. Acknowledging the proxy limitations and adjusting confidence accordingly
                6. Concluding with a clear governance quality verdict
                """,
            ),
            (
                "human",
                """Based on the following data, generate a governance quality signal for {ticker}:

                Analysis Data:
                {analysis_data}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke(
        {"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker}
    )

    def create_default_governance_signal():
        return GovernanceSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=GovernanceSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_governance_signal,
    )
