"""Constants and utilities related to analysts configuration."""

from src.agents import portfolio_manager
from src.agents.aswath_damodaran import aswath_damodaran_agent
from src.agents.ben_graham import ben_graham_agent
from src.agents.bill_ackman import bill_ackman_agent
from src.agents.cathie_wood import cathie_wood_agent
from src.agents.charlie_munger import charlie_munger_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.michael_burry import michael_burry_agent
from src.agents.phil_fisher import phil_fisher_agent
from src.agents.peter_lynch import peter_lynch_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.stanley_druckenmiller import stanley_druckenmiller_agent
from src.agents.technicals import technical_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.warren_buffett import warren_buffett_agent
from src.agents.rakesh_jhunjhunwala import rakesh_jhunjhunwala_agent
from src.agents.mohnish_pabrai import mohnish_pabrai_agent
from src.agents.news_sentiment import news_sentiment_agent
from src.agents.growth_agent import growth_analyst_agent
from src.agents.value_agent import value_pack_agent
from src.agents.quality_agent import quality_pack_agent
from src.agents.momentum_agent import momentum_agent
from src.agents.liquidity_agent import liquidity_agent
from src.agents.macro_exposure_agent import macro_exposure_agent
from src.agents.governance_agent import governance_agent
from src.agents.factorcomposite_agent import factor_composite_agent
from src.agents.capital_allocation import capital_allocation_agent
from src.agents.market_regime_agent import market_regime_agent
from src.agents.capital_flow_agent import capital_flow_agent

# Define analyst configuration - single source of truth
ANALYST_CONFIG = {
    "capital_flow": {
        "display_name": "Capital Flow Analyst",
        "description": "Capital Flow Specialist",
        "investing_style": "Analyzes capital flow patterns using sector rotation and risk appetite proxies. Since direct FII/DII institutional flow data is not available via Twelve Data, this agent uses sector rotation analysis as a proven capital flow proxy: Cyclical vs defensive sector performance differential, Risk-on vs risk-off asset rotation (equities vs gold), Sector breadth: how many sectors are participating in the trend, Relative strength of benchmark vs safe haven. Institutional capital flows into cyclicals and out of defensives in risk-on regimes. The reverse signals institutional risk reduction and potential market weakness. Dynamically selects India sector indices or US sector ETFs based on ticker universe.",
        "agent_func": capital_flow_agent,
        "type": "analyst",
        "order": 26,
    },
    "market_regime": {
        "display_name": "Market Regime Analyst",
        "description": "Market Regime Specialist",
        "investing_style": "Analyzes the macro market regime using VIX and benchmark price data. Dynamically selects India VIX + Nifty 50 for Indian ticker universes, or CBOE VIX + SPY for US ticker universes. Sub-analyses: VIX level and trend (fear gauge), VIX term structure proxy, benchmark trend (50D and 200D moving averages), market breadth proxy. Low VIX + uptrending benchmark = risk-on regime (bullish); High VIX + downtrending benchmark = risk-off regime (bearish).",
        "agent_func": market_regime_agent,
        "type": "analyst",
        "order": 25,
    },
    "capital_allocation": {
        "display_name": "Capital Allocation Analyst",
        "description": "Capital Allocation Specialist",
        "investing_style": "Analyzes stocks using a comprehensive capital allocation framework. Focuses on how management deploys the cash the business generates: Dividend growth (Dividend CAGR) — is capital returned consistently? Share dilution rate — are shareholders being respected or taxed? Buyback yield — is capital returned via repurchases? Return on incremental capital (ROIC delta) — is new capital deployed wisely? Net debt change trend — is the balance sheet strengthening or deteriorating? Great capital allocators compound wealth. Poor ones destroy it quietly.",
        "agent_func": capital_allocation_agent,
        "type": "analyst",
        "order": 24,
    },
    "factor_composite": {
        "display_name": "Factor Composite Analyst",
        "description": "Comprehensive Factor Risk Analysis Specialist",
        "investing_style": "Analyzes stocks using a factor composite framework focused on financial risk. Combines two core dimensions: earnings volatility (standard deviation of EPS growth) and balance sheet stress ((Debt/Equity) × (1 / Interest Coverage)). Low volatility + low stress = resilient, predictable businesses that survive downturns and compound through cycles.",
        "agent_func": factor_composite_agent,
        "type": "analyst",
        "order": 23,
    },
    "aswath_damodaran": {
        "display_name": "Aswath Damodaran",
        "description": "The Dean of Valuation",
        "investing_style": "Focuses on intrinsic value and financial metrics to assess investment opportunities through rigorous valuation analysis.",
        "agent_func": aswath_damodaran_agent,
        "type": "analyst",
        "order": 0,
    },
    "ben_graham": {
        "display_name": "Ben Graham",
        "description": "The Father of Value Investing",
        "investing_style": "Emphasizes a margin of safety and invests in undervalued companies with strong fundamentals through systematic value analysis.",
        "agent_func": ben_graham_agent,
        "type": "analyst",
        "order": 1,
    },
    "bill_ackman": {
        "display_name": "Bill Ackman",
        "description": "The Activist Investor",
        "investing_style": "Seeks to influence management and unlock value through strategic activism and contrarian investment positions.",
        "agent_func": bill_ackman_agent,
        "type": "analyst",
        "order": 2,
    },
    "cathie_wood": {
        "display_name": "Cathie Wood",
        "description": "The Queen of Growth Investing",
        "investing_style": "Focuses on disruptive innovation and growth, investing in companies that are leading technological advancements and market disruption.",
        "agent_func": cathie_wood_agent,
        "type": "analyst",
        "order": 3,
    },
    "charlie_munger": {
        "display_name": "Charlie Munger",
        "description": "The Rational Thinker",
        "investing_style": "Advocates for value investing with a focus on quality businesses and long-term growth through rational decision-making.",
        "agent_func": charlie_munger_agent,
        "type": "analyst",
        "order": 4,
    },
    "michael_burry": {
        "display_name": "Michael Burry",
        "description": "The Big Short Contrarian",
        "investing_style": "Makes contrarian bets, often shorting overvalued markets and investing in undervalued assets through deep fundamental analysis.",
        "agent_func": michael_burry_agent,
        "type": "analyst",
        "order": 5,
    },
    "mohnish_pabrai": {
        "display_name": "Mohnish Pabrai",
        "description": "The Dhandho Investor",
        "investing_style": "Focuses on value investing and long-term growth through fundamental analysis and a margin of safety.",
        "agent_func": mohnish_pabrai_agent,
        "type": "analyst",
        "order": 6,
    },
    "peter_lynch": {
        "display_name": "Peter Lynch",
        "description": "The 10-Bagger Investor",
        "investing_style": "Invests in companies with understandable business models and strong growth potential using the 'buy what you know' strategy.",
        "agent_func": peter_lynch_agent,
        "type": "analyst",
        "order": 6,
    },
    "phil_fisher": {
        "display_name": "Phil Fisher",
        "description": "The Scuttlebutt Investor",
        "investing_style": "Emphasizes investing in companies with strong management and innovative products, focusing on long-term growth through scuttlebutt research.",
        "agent_func": phil_fisher_agent,
        "type": "analyst",
        "order": 7,
    },
    "rakesh_jhunjhunwala": {
        "display_name": "Rakesh Jhunjhunwala",
        "description": "The Big Bull Of India",
        "investing_style": "Leverages macroeconomic insights to invest in high-growth sectors, particularly within emerging markets and domestic opportunities.",
        "agent_func": rakesh_jhunjhunwala_agent,
        "type": "analyst",
        "order": 8,
    },
    "stanley_druckenmiller": {
        "display_name": "Stanley Druckenmiller",
        "description": "The Macro Investor",
        "investing_style": "Focuses on macroeconomic trends, making large bets on currencies, commodities, and interest rates through top-down analysis.",
        "agent_func": stanley_druckenmiller_agent,
        "type": "analyst",
        "order": 9,
    },
    "warren_buffett": {
        "display_name": "Warren Buffett",
        "description": "The Oracle of Omaha",
        "investing_style": "Seeks companies with strong fundamentals and competitive advantages through value investing and long-term ownership.",
        "agent_func": warren_buffett_agent,
        "type": "analyst",
        "order": 10,
    },
    "technical_analyst": {
        "display_name": "Technical Analyst",
        "description": "Chart Pattern Specialist",
        "investing_style": "Focuses on chart patterns and market trends to make investment decisions, often using technical indicators and price action analysis.",
        "agent_func": technical_analyst_agent,
        "type": "analyst",
        "order": 11,
    },
    "fundamentals_analyst": {
        "display_name": "Fundamentals Analyst",
        "description": "Financial Statement Specialist",
        "investing_style": "Delves into financial statements and economic indicators to assess the intrinsic value of companies through fundamental analysis.",
        "agent_func": fundamentals_analyst_agent,
        "type": "analyst",
        "order": 12,
    },
    "growth_analyst": {
        "display_name": "Growth Analyst",
        "description": "Growth Specialist",
        "investing_style": "Analyzes growth trends and valuation to identify growth opportunities through growth analysis.",
        "agent_func": growth_analyst_agent,
        "type": "analyst",
        "order": 13,
    },
    "news_sentiment_analyst": {
        "display_name": "News Sentiment Analyst",
        "description": "News Sentiment Specialist",
        "investing_style": "Analyzes news sentiment to predict market movements and identify opportunities through news analysis.",
        "agent_func": news_sentiment_agent,
        "type": "analyst",
        "order": 14,
    },
    "sentiment_analyst": {
        "display_name": "Sentiment Analyst",
        "description": "Market Sentiment Specialist",
        "investing_style": "Gauges market sentiment and investor behavior to predict market movements and identify opportunities through behavioral analysis.",
        "agent_func": sentiment_analyst_agent,
        "type": "analyst",
        "order": 15,
    },
    "valuation_analyst": {
        "display_name": "Valuation Analyst",
        "description": "Company Valuation Specialist",
        "investing_style": "Specializes in determining the fair value of companies, using various valuation models and financial metrics for investment decisions.",
        "agent_func": valuation_analyst_agent,
        "type": "analyst",
        "order": 16,
    },
    "value_pack": {
        "display_name": "Value Pack Analyst",
        "description": "Comprehensive Value Investing Specialist",
        "investing_style": "Analyzes stocks using a comprehensive value investing framework. Focuses on intrinsic value metrics: FCF yield, PE, PB, EV/EBITDA, ROIC, debt-to-equity, margin of safety, and normalized FCF. Designed as a pure value-oriented philosophy agent.",
        "agent_func": value_pack_agent,
        "type": "analyst",
        "order": 17,
    },
    "quality_pack": {
        "display_name": "Quality Pack Analyst",
        "description": "Comprehensive Quality Investing Specialist",
        "investing_style": "Analyzes stocks using a comprehensive quality investing framework. Focuses on the durability and consistency of financial performance: ROCE, ROE, operating margins, revenue and EPS CAGR stability, FCF conversion efficiency, and operating cash flow consistency. Designed to identify high-quality businesses with durable competitive advantages.",
        "agent_func": quality_pack_agent,
        "type": "analyst",
        "order": 18,
    },
    "momentum": {
        "display_name": "Momentum Analyst",
        "description": "Comprehensive Momentum Investing Specialist",
        "investing_style": "Analyzes stocks using a comprehensive price momentum framework. Focuses on multi-period returns (1D, 1W, 1M, 3M, 6M, 12M), rolling CAGR, maximum drawdown, 12M return excluding last 1M (classic momentum factor), RSI, and volatility-adjusted return (Sharpe-like ratio). A stock with strong, consistent, risk-adjusted momentum across multiple timeframes is a high-conviction candidate.",
        "agent_func": momentum_agent,
        "type": "analyst",
        "order": 19,
    },
    "liquidity": {
        "display_name": "Liquidity Analyst",
        "description": "Comprehensive Liquidity Analysis Specialist",
        "investing_style": "Analyzes stocks using a comprehensive liquidity framework. Focuses on volume metrics (daily volume, 30D avg volume), traded value, Amihud illiquidity ratio, and impact cost proxy via high-low spread. High liquidity reduces execution risk and signals institutional confidence. Low liquidity stocks carry hidden transaction costs that erode returns.",
        "agent_func": liquidity_agent,
        "type": "analyst",
        "order": 20,
    },
    "macro_exposure": {
        "display_name": "Macro Exposure Analyst",
        "description": "Comprehensive Macro Exposure Analysis Specialist",
        "investing_style": "Analyzes stocks using a comprehensive macro exposure framework. Assesses a company's sensitivity to macroeconomic forces via financial statement proxies: interest rate sensitivity (debt load, interest coverage, leverage), inflation sensitivity (gross margin stability, capex intensity), and FX dependency (international revenue exposure, geographic concentration). Companies with low debt, stable margins, and domestic revenue are more resilient across macro regimes.",
        "agent_func": macro_exposure_agent,
        "type": "analyst",
        "order": 21,
    },
    "governance": {
        "display_name": "Governance Analyst",
        "description": "Comprehensive Corporate Governance Analysis Specialist",
        "investing_style": "Analyzes stocks using a comprehensive corporate governance framework. Since promoter holdings, pledge data, auditor history, and related party transactions are not available in the current data model, governance quality is assessed via four available proxies: insider ownership trend (net insider buying/selling as skin-in-the-game proxy), share dilution trend (outstanding shares growth as shareholder alignment proxy), earnings integrity (FCF conversion quality as accounting quality proxy), and news-based governance flags (negative press around governance red flags). Good governance compounds returns. Poor governance destroys them silently.",
        "agent_func": governance_agent,
        "type": "analyst",
        "order": 22,
    },
}

# Derive ANALYST_ORDER from ANALYST_CONFIG for backwards compatibility
ANALYST_ORDER = [(config["display_name"], key) for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])]


def get_analyst_nodes():
    """Get the mapping of analyst keys to their (node_name, agent_func) tuples."""
    return {key: (f"{key}_agent", config["agent_func"]) for key, config in ANALYST_CONFIG.items()}


def get_agents_list():
    """Get the list of agents for API responses."""
    return [
        {
            "key": key,
            "display_name": config["display_name"],
            "description": config["description"],
            "investing_style": config["investing_style"],
            "order": config["order"]
        }
        for key, config in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]
