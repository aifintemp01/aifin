from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news


class SentimentSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(
            sentiment == "negative", "bearish",
            np.where(sentiment == "positive", "bullish", "neutral")
        ).tolist()

        progress.update_status(agent_id, ticker, "Combining signals")
        insider_weight = 0.3
        news_weight = 0.7

        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            pre_signal = "bullish"
        elif bearish_signals > bullish_signals:
            pre_signal = "bearish"
        else:
            pre_signal = "neutral"

        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        pre_confidence = 0.0
        if total_weighted_signals > 0:
            pre_confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)

        reasoning = {
            "insider_trading": {
                "signal": (
                    "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else
                    "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral"
                ),
                "confidence": round(
                    (max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100
                ),
                "metrics": {
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                },
            },
            "news_sentiment": {
                "signal": (
                    "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else
                    "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral"
                ),
                "confidence": round(
                    (max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100
                ),
                "metrics": {
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                },
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": (
                    f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} "
                    f"based on weighted signal comparison"
                ),
            },
            "pre_signal": pre_signal,
            "pre_confidence": pre_confidence,
        }

        # ------------------------------------------------------------------
        # LLM summary
        # ------------------------------------------------------------------
        progress.update_status(agent_id, ticker, "Generating sentiment analysis")
        sentiment_output = generate_sentiment_output(
            ticker=ticker,
            analysis_data=reasoning,
            state=state,
            agent_id=agent_id,
        )

        sentiment_analysis[ticker] = {
            "signal": sentiment_output.signal,
            "confidence": sentiment_output.confidence,
            "reasoning": sentiment_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=sentiment_output.reasoning)

    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    state["data"]["analyst_signals"][agent_id] = sentiment_analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }


# ---------------------------------------------------------------------------
# LLM Output Generator
# ---------------------------------------------------------------------------

def generate_sentiment_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
) -> SentimentSignal:
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a market sentiment analyst. You assess crowd psychology and 
            information flow around a stock using insider trading activity and news sentiment.

            When providing reasoning:
            1. Lead with the news sentiment breakdown — how many bullish vs bearish articles
            2. Comment on insider trading activity — are insiders buying or selling
            3. Explain the weighted combination (news 70%, insider 30%)
            4. Note if the two signals agree or contradict each other
            5. Conclude with an overall sentiment assessment

            Keep reasoning concise and data-driven. Return JSON only.""",
        ),
        (
            "human",
            """Based on the following sentiment analysis for {ticker}, generate a signal.

            Analysis:
            {analysis_data}

            Return exactly:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}""",
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "analysis_data": json.dumps(analysis_data, indent=2),
    })

    def default_signal():
        return SentimentSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in sentiment analysis, defaulting to neutral",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=SentimentSignal,
        agent_name=agent_id,
        state=state,
        default_factory=default_signal,
    )