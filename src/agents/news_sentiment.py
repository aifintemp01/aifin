from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from src.data.models import CompanyNews
import pandas as pd
import numpy as np
import json

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress
from typing import List, Optional
from typing_extensions import Literal


class Sentiment(BaseModel):
    """Represents the sentiment of a news article."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: int = Field(description="Confidence 0-100")


def _build_news_context_json(
    agent_id: str,
    ticker: str,
    layer: int,
    overall_signal: str,
    confidence: float,
    company_news: list,
    bullish_signals: int,
    bearish_signals: int,
    neutral_signals: int,
    total_signals: int,
) -> dict:
    """
    Build a context JSON summary for downstream layer agents.
    Generated programmatically from the news analysis — no extra LLM call.
    """
    # Extract top article titles by sentiment for key_findings / risk_factors
    positive_articles = [n for n in company_news if n.sentiment == "positive"][:3]
    negative_articles = [n for n in company_news if n.sentiment == "negative"][:3]

    key_findings = [a.title[:120] for a in positive_articles]
    risk_factors = [a.title[:120] for a in negative_articles]

    # Fallback if no articles extracted
    if not key_findings:
        key_findings = [f"{bullish_signals} positive news signals detected"]
    if not risk_factors and bearish_signals > 0:
        risk_factors = [f"{bearish_signals} negative news signals detected"]

    total = total_signals or 1
    data_summary = (
        f"News sentiment for {ticker} is {overall_signal} with {confidence:.1f}% confidence. "
        f"Analysed {total_signals} recent articles: "
        f"{bullish_signals} positive ({bullish_signals/total*100:.0f}%), "
        f"{bearish_signals} negative ({bearish_signals/total*100:.0f}%), "
        f"{neutral_signals} neutral ({neutral_signals/total*100:.0f}%)."
    )

    result = {
        "agent": agent_id,
        "ticker": ticker,
        "layer": layer,
        "signal": overall_signal,
        "confidence": confidence,
        "key_findings": key_findings,
        "data_summary": data_summary,
    }
    if risk_factors:
        result["risk_factors"] = risk_factors

    return result


def news_sentiment_agent(
    state: AgentState,
    agent_id: str = "news_sentiment_agent",
    layer: int = 1,
    is_last_hidden: bool = True,
    upstream_agent_ids: list = None,
):
    """
    Analyses news sentiment for a list of tickers and generates trading signals.

    Layer behaviour:
    - is_last_hidden=True  → writes to analyst_signals (PM sees this)
    - is_last_hidden=False → writes to layer_context only (intermediate layer)
    In both cases a context JSON is stored in layer_context for downstream agents.
    """
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "TWELVE_DATA_API_KEY")

    sentiment_analysis = {}
    layer_context_updates: dict = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(
            ticker=ticker,
            end_date=end_date,
            limit=100,
            api_key=api_key,
        )

        news_signals = []
        sentiment_confidences = {}
        sentiments_classified_by_llm = 0

        if company_news:
            recent_articles = company_news[:10]
            articles_without_sentiment = [n for n in recent_articles if n.sentiment is None]

            if articles_without_sentiment:
                num_to_analyze = 5
                articles_to_analyze = articles_without_sentiment[:num_to_analyze]
                progress.update_status(agent_id, ticker, f"Analysing sentiment for {len(articles_to_analyze)} articles")

                for idx, news in enumerate(articles_to_analyze):
                    progress.update_status(agent_id, ticker, f"Analysing article {idx + 1} of {len(articles_to_analyze)}")
                    prompt = (
                        f"Analyse the sentiment of the following news headline "
                        f"for the stock {ticker}. "
                        f"Determine if sentiment is 'positive', 'negative', or 'neutral'. "
                        f"Provide a confidence score 0-100. "
                        f"Respond in JSON format.\n\nHeadline: {news.title}"
                    )
                    response = call_llm(prompt, Sentiment, agent_name=agent_id, state=state)
                    if response:
                        news.sentiment = response.sentiment.lower()
                        sentiment_confidences[id(news)] = response.confidence
                    else:
                        news.sentiment = "neutral"
                        sentiment_confidences[id(news)] = 0
                    sentiments_classified_by_llm += 1

            sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
            news_signals = np.where(
                sentiment == "negative", "bearish",
                np.where(sentiment == "positive", "bullish", "neutral")
            ).tolist()

        progress.update_status(agent_id, ticker, "Aggregating signals")

        bullish_signals = news_signals.count("bullish")
        bearish_signals = news_signals.count("bearish")
        neutral_signals = news_signals.count("neutral")
        total_signals = len(news_signals)

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        confidence = _calculate_confidence_score(
            sentiment_confidences=sentiment_confidences,
            company_news=company_news,
            overall_signal=overall_signal,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            total_signals=total_signals,
        )

        reasoning = {
            "news_sentiment": {
                "signal": overall_signal,
                "confidence": confidence,
                "metrics": {
                    "total_articles": total_signals,
                    "bullish_articles": bullish_signals,
                    "bearish_articles": bearish_signals,
                    "neutral_articles": neutral_signals,
                    "articles_classified_by_llm": sentiments_classified_by_llm,
                },
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        # ── Build context JSON for downstream layers ───────────────────────
        context_json = _build_news_context_json(
            agent_id=agent_id,
            ticker=ticker,
            layer=layer,
            overall_signal=overall_signal,
            confidence=confidence,
            company_news=company_news,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            neutral_signals=neutral_signals,
            total_signals=total_signals,
        )
        layer_context_updates[f"{agent_id}:{ticker}"] = context_json

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentiment_analysis, "News Sentiment Analysis Agent")

    if "analyst_signals" not in state["data"]:
        state["data"]["analyst_signals"] = {}

    # ── Write to analyst_signals only if this is the last hidden layer ─────
    # Intermediate layer agents pass context forward via layer_context instead
    if is_last_hidden:
        state["data"]["analyst_signals"][agent_id] = sentiment_analysis
    else:
        print(f"[{agent_id}] Layer {layer} intermediate — writing to layer_context only")

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": state["data"],
        "layer_context": layer_context_updates,
    }


def _calculate_confidence_score(
    sentiment_confidences: dict,
    company_news: list,
    overall_signal: str,
    bullish_signals: int,
    bearish_signals: int,
    total_signals: int,
) -> float:
    if total_signals == 0:
        return 0.0

    if sentiment_confidences:
        matching_articles = [
            news for news in company_news
            if news.sentiment and (
                (overall_signal == "bullish" and news.sentiment == "positive") or
                (overall_signal == "bearish" and news.sentiment == "negative") or
                (overall_signal == "neutral" and news.sentiment == "neutral")
            )
        ]
        llm_confidences = [
            sentiment_confidences[id(news)]
            for news in matching_articles
            if id(news) in sentiment_confidences
        ]
        if llm_confidences:
            avg_llm_confidence = sum(llm_confidences) / len(llm_confidences)
            signal_proportion = (max(bullish_signals, bearish_signals) / total_signals) * 100
            return round(0.7 * avg_llm_confidence + 0.3 * signal_proportion, 2)

    return round((max(bullish_signals, bearish_signals) / total_signals) * 100, 2)