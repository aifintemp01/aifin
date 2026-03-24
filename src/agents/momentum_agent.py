from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import get_prices
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.utils.api_key import get_api_key_from_state


class MomentumSignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


def momentum_agent(state: AgentState, agent_id: str = "momentum_agent"):
    """
    Analyzes stocks using a comprehensive price momentum framework.
    Focuses on multi-period returns (1D, 1W, 1M, 3M, 6M, 12M), rolling CAGR,
    maximum drawdown, 12M return excluding last 1M (classic momentum factor),
    RSI, and volatility-adjusted return (Sharpe-like ratio).
    A stock with strong, consistent, risk-adjusted momentum across multiple
    timeframes is a high-conviction candidate.
    """
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    data = state["data"]
    end_date: str = data["end_date"]
    tickers: list[str] = data["tickers"]

    # Need ~14 months of daily prices to cover all metrics:
    # 12M-excl-1M requires 252 + 21 trading days (~273 trading days / ~385 calendar days)
    start_date = (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=430)
    ).strftime("%Y-%m-%d")

    analysis_data: dict[str, dict] = {}
    momentum_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw price data
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Fetching price data")
        prices = get_prices(ticker, start_date, end_date, api_key=api_key)

        if not prices or len(prices) < 21:
            progress.update_status(agent_id, ticker, "Insufficient price history — skipping")
            momentum_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 0.0,
                "reasoning": "Insufficient price history to compute momentum metrics.",
            }
            continue

        # Convert to ordered list of close prices (oldest → newest)
        close = _extract_close_prices(prices)

        # ------------------------------------------------------------------
        # Run sub-analyses
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Analyzing multi-period returns")
        returns_analysis = _analyze_returns(close)

        progress.update_status(agent_id, ticker, "Analyzing trend strength")
        trend_analysis = _analyze_trend(close)

        progress.update_status(agent_id, ticker, "Analyzing RSI")
        rsi_analysis = _analyze_rsi(close)

        progress.update_status(agent_id, ticker, "Analyzing volatility-adjusted return")
        risk_adjusted_analysis = _analyze_risk_adjusted(close)

        # ------------------------------------------------------------------
        # Aggregate score
        # Momentum weights: multi-period returns and trend dominate;
        # RSI and risk-adjusted quality act as confirmation filters
        # ------------------------------------------------------------------
        total_score = (
            returns_analysis["score"] * 0.35          # 1M, 3M, 6M, 12M returns
            + trend_analysis["score"] * 0.30          # rolling CAGR, 12M-excl-1M, max drawdown
            + risk_adjusted_analysis["score"] * 0.20  # volatility-adjusted return
            + rsi_analysis["score"] * 0.15            # RSI overbought/oversold filter
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
            "returns_analysis": returns_analysis,
            "trend_analysis": trend_analysis,
            "rsi_analysis": rsi_analysis,
            "risk_adjusted_analysis": risk_adjusted_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating momentum analysis")
        momentum_output = _generate_momentum_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
            agent_id=agent_id,
        )

        momentum_analysis[ticker] = {
            "signal": momentum_output.signal,
            "confidence": momentum_output.confidence,
            "reasoning": momentum_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=momentum_output.reasoning)

    # ----------------------------------------------------------------------
    # Return to graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(momentum_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(momentum_analysis, "Momentum Agent")

    state["data"]["analyst_signals"][agent_id] = momentum_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Price data helpers
###############################################################################

def _extract_close_prices(prices) -> list[float]:
    """
    Convert API price objects into an ordered list of close prices.
    Sorted oldest → newest so index[-1] is always the most recent close.
    """
    sorted_prices = sorted(
        prices,
        key=lambda p: p.date if hasattr(p, "date") else p["date"]
    )
    close = []
    for p in sorted_prices:
        val = p.close if hasattr(p, "close") else p.get("close")
        if val is not None:
            close.append(float(val))
    return close


def _period_return(close: list[float], trading_days: int) -> float | None:
    """Return the simple return over the last `trading_days` from the most recent close."""
    if len(close) < trading_days + 1:
        return None
    return (close[-1] / close[-(trading_days + 1)]) - 1


###############################################################################
# Sub-analysis helpers
###############################################################################

# ----- Multi-period Returns (1D, 1W, 1M, 3M, 6M, 12M) ----------------------

def _analyze_returns(close: list[float]) -> dict:
    """
    Score multi-period price returns across 1D, 1W, 1M, 3M, 6M, and 12M.
    Medium-to-long term returns (3M, 6M) carry the most predictive weight.
    Short-term (1D, 1W, 1M) are included for context and confirmation only.
    Breadth bonus awarded when all medium/long periods are simultaneously positive.
    """
    max_score = 9  # 3pts 3M + 3pts 6M + 1pt 12M + 1pt 1M + 1pt breadth bonus
    score = 0
    details: list[str] = []

    periods = {
        "1D":  1,
        "1W":  5,
        "1M":  21,
        "3M":  63,
        "6M":  126,
        "12M": 252,
    }

    computed: dict[str, float | None] = {
        label: _period_return(close, days)
        for label, days in periods.items()
    }

    # 3M return — primary signal: 0–3 pts
    r3m = computed["3M"]
    if r3m is not None:
        if r3m > 0.15:
            score += 3
            details.append(f"Strong 3M return: {r3m:.1%}")
        elif r3m > 0.05:
            score += 2
            details.append(f"Positive 3M return: {r3m:.1%}")
        elif r3m > 0:
            score += 1
            details.append(f"Slight 3M return: {r3m:.1%}")
        else:
            details.append(f"Negative 3M return: {r3m:.1%}")
    else:
        details.append("3M return: insufficient data")

    # 6M return — secondary signal: 0–3 pts
    r6m = computed["6M"]
    if r6m is not None:
        if r6m > 0.20:
            score += 3
            details.append(f"Strong 6M return: {r6m:.1%}")
        elif r6m > 0.08:
            score += 2
            details.append(f"Positive 6M return: {r6m:.1%}")
        elif r6m > 0:
            score += 1
            details.append(f"Slight 6M return: {r6m:.1%}")
        else:
            details.append(f"Negative 6M return: {r6m:.1%}")
    else:
        details.append("6M return: insufficient data")

    # 12M return — long-term context: 0–1 pt
    r12m = computed["12M"]
    if r12m is not None:
        if r12m > 0.10:
            score += 1
            details.append(f"Positive 12M return: {r12m:.1%}")
        else:
            details.append(f"12M return: {r12m:.1%}")
    else:
        details.append("12M return: insufficient data")

    # 1M return — short-term confirmation: 0–1 pt
    r1m = computed["1M"]
    if r1m is not None:
        if r1m > 0.03:
            score += 1
            details.append(f"Positive 1M return: {r1m:.1%}")
        else:
            details.append(f"1M return: {r1m:.1%}")
    else:
        details.append("1M return: insufficient data")

    # Breadth bonus: all medium/long periods positive — 0–1 pt
    medium_long = [r for r in [r3m, r6m, r12m] if r is not None]
    if medium_long and all(r > 0 for r in medium_long):
        score += 1
        details.append("All medium/long-term returns positive — broad momentum confirmed")

    # Attach 1D and 1W for LLM context (not scored)
    for label in ("1D", "1W"):
        val = computed[label]
        if val is not None:
            details.append(f"{label} return: {val:.1%}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "returns": {
            k: round(v, 4) if v is not None else None
            for k, v in computed.items()
        },
    }


# ----- Trend Strength (Rolling CAGR, 12M-excl-1M, Max Drawdown) -------------

def _analyze_trend(close: list[float]) -> dict:
    """
    Assess structural trend strength via three lenses:
    - Rolling CAGR over full history — is the long-run direction positive?
    - 12M-excl-1M return — the classic academic momentum factor that avoids
      short-term mean reversion noise (Jegadeesh & Titman, 1993)
    - Maximum drawdown — a deep drawdown signals structural weakness even if
      recent returns look acceptable
    """
    max_score = 7  # 2pts rolling CAGR + 3pts 12M-excl-1M + 2pts max drawdown
    score = 0
    details: list[str] = []

    # Rolling CAGR over full available history: 0–2 pts
    rolling_cagr = None
    if len(close) >= 2 and close[0] > 0:
        n_years = len(close) / 252.0
        rolling_cagr = (close[-1] / close[0]) ** (1 / n_years) - 1
        if rolling_cagr > 0.20:
            score += 2
            details.append(f"Strong rolling CAGR: {rolling_cagr:.1%}")
        elif rolling_cagr > 0.08:
            score += 1
            details.append(f"Positive rolling CAGR: {rolling_cagr:.1%}")
        else:
            details.append(f"Weak rolling CAGR: {rolling_cagr:.1%}")
    else:
        details.append("Rolling CAGR: insufficient data")

    # 12M-excl-1M momentum factor: 0–3 pts
    r12m_excl_1m = None
    if len(close) >= 252 + 21:
        price_1m_ago = close[-21]
        price_12m_ago = close[-252]
        if price_12m_ago > 0:
            r12m_excl_1m = (price_1m_ago / price_12m_ago) - 1
            if r12m_excl_1m > 0.25:
                score += 3
                details.append(f"Very strong 12M-excl-1M momentum: {r12m_excl_1m:.1%}")
            elif r12m_excl_1m > 0.10:
                score += 2
                details.append(f"Positive 12M-excl-1M momentum: {r12m_excl_1m:.1%}")
            elif r12m_excl_1m > 0:
                score += 1
                details.append(f"Slight 12M-excl-1M momentum: {r12m_excl_1m:.1%}")
            else:
                details.append(f"Negative 12M-excl-1M momentum: {r12m_excl_1m:.1%}")
    else:
        details.append("12M-excl-1M momentum: insufficient data")

    # Maximum drawdown over full history: 0–2 pts
    peak = close[0]
    max_dd = 0.0
    for price in close:
        if price > peak:
            peak = price
        dd = (price / peak) - 1
        if dd < max_dd:
            max_dd = dd

    if max_dd > -0.10:
        score += 2
        details.append(f"Shallow max drawdown: {max_dd:.1%}")
    elif max_dd > -0.20:
        score += 1
        details.append(f"Moderate max drawdown: {max_dd:.1%}")
    else:
        details.append(f"Deep max drawdown: {max_dd:.1%}")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "rolling_cagr": round(rolling_cagr, 4) if rolling_cagr is not None else None,
        "return_12m_excl_1m": round(r12m_excl_1m, 4) if r12m_excl_1m is not None else None,
        "max_drawdown": round(max_dd, 4),
    }


# ----- RSI (Relative Strength Index) ----------------------------------------

def _analyze_rsi(close: list[float]) -> dict:
    """
    Compute 14-period RSI using Wilder smoothing and score it as a momentum filter.
    RSI 50–70: healthy uptrend — bullish confirmation.
    RSI > 70: overbought — momentum exists but mean reversion risk rises.
    RSI 40–50: momentum weakening.
    RSI < 30: oversold — bearish momentum dominant.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    if len(close) < 15:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "RSI: insufficient data (need 15+ days)",
            "rsi": None,
        }

    # Compute daily price changes
    deltas = [close[i] - close[i - 1] for i in range(1, len(close))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [abs(d) if d < 0 else 0.0 for d in deltas]

    # Seed with simple 14-period average
    period = 14
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for all remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    if 50 <= rsi <= 70:
        score += 4
        details.append(f"RSI {rsi:.1f} — healthy uptrend, ideal momentum zone")
    elif rsi > 70:
        score += 2
        details.append(f"RSI {rsi:.1f} — overbought, strong momentum but mean reversion risk elevated")
    elif 40 <= rsi < 50:
        score += 1
        details.append(f"RSI {rsi:.1f} — momentum weakening, approaching neutral territory")
    else:
        details.append(f"RSI {rsi:.1f} — below 40, bearish momentum dominant")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "rsi": round(rsi, 2),
    }


# ----- Volatility-Adjusted Return (Sharpe-like) -----------------------------

def _analyze_risk_adjusted(close: list[float]) -> dict:
    """
    Compute the volatility-adjusted return: 12M return / annualized volatility.
    This is a Sharpe-like ratio using price returns only (no risk-free rate).
    High values mean the stock delivered strong returns without excessive noise.
    Volatility is annualized from daily returns over the full available history.
    """
    max_score = 4
    score = 0
    details: list[str] = []

    if len(close) < 252:
        return {
            "score": 0,
            "max_score": max_score,
            "details": "Volatility-adjusted return: need 252 days of price history",
            "vol_adjusted_return": None,
            "annualized_volatility": None,
            "return_12m": None,
        }

    # 12M return
    r12m = (close[-1] / close[-252]) - 1

    # Annualized volatility from daily returns over full history
    daily_returns = [(close[i] / close[i - 1]) - 1 for i in range(1, len(close))]
    n = len(daily_returns)
    mean_r = sum(daily_returns) / n
    variance = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
    ann_vol = (variance ** 0.5) * (252 ** 0.5)

    vol_adj_return = None
    if ann_vol > 0:
        vol_adj_return = r12m / ann_vol

        if vol_adj_return > 1.0:
            score += 4
            details.append(f"Exceptional vol-adjusted return: {vol_adj_return:.2f} (12M {r12m:.1%} / vol {ann_vol:.1%})")
        elif vol_adj_return > 0.5:
            score += 3
            details.append(f"Strong vol-adjusted return: {vol_adj_return:.2f} (12M {r12m:.1%} / vol {ann_vol:.1%})")
        elif vol_adj_return > 0:
            score += 2
            details.append(f"Positive vol-adjusted return: {vol_adj_return:.2f} (12M {r12m:.1%} / vol {ann_vol:.1%})")
        elif vol_adj_return > -0.5:
            score += 1
            details.append(f"Slightly negative vol-adjusted return: {vol_adj_return:.2f}")
        else:
            details.append(f"Poor vol-adjusted return: {vol_adj_return:.2f} (12M {r12m:.1%} / vol {ann_vol:.1%})")
    else:
        details.append("Volatility-adjusted return: zero volatility detected — unusual data")

    return {
        "score": (score / max_score) * 10,
        "max_score": max_score,
        "details": "; ".join(details),
        "vol_adjusted_return": round(vol_adj_return, 4) if vol_adj_return is not None else None,
        "annualized_volatility": round(ann_vol, 4),
        "return_12m": round(r12m, 4),
    }


###############################################################################
# LLM generation
###############################################################################

def _generate_momentum_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> MomentumSignal:
    """Generate a momentum investing signal grounded strictly in the computed price metrics."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a disciplined momentum investing analyst. Your mandate:
                - Follow price — stocks that have been going up tend to keep going up
                - The 12M-excl-1M return is the most academically robust momentum signal; weight it heavily
                - Multi-period confirmation matters: momentum across 3M, 6M, and 12M is stronger than any single period
                - RSI between 50–70 is the ideal zone — healthy trend without overbought risk
                - Deep drawdowns disqualify a stock regardless of recent recovery
                - Volatility-adjusted return separates true momentum from noisy price spikes
                - You do not care about fundamentals — price action is your only input

                When providing your reasoning, be specific by:
                1. Leading with the 12M-excl-1M momentum factor and what it signals
                2. Citing 3M and 6M returns to confirm or deny trend persistence
                3. Commenting on RSI — is the trend in a healthy zone or overextended?
                4. Noting the max drawdown — does the trend have structural integrity?
                5. Referencing the volatility-adjusted return — is momentum efficient or noisy?
                6. Concluding with a clear, price-anchored stance
                """,
            ),
            (
                "human",
                """Based on the following data, generate a momentum investing signal for {ticker}:

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

    def create_default_momentum_signal():
        return MomentumSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Parsing error — defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=MomentumSignal,
        agent_name=agent_id,
        state=state,
        default_factory=create_default_momentum_signal,
    )
