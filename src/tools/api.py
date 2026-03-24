import datetime
import os
from matplotlib import ticker
import pandas as pd
import requests
import time

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)
# Global cache instance
_cache = get_cache()

# ---------------------------------------------------------------------------
# Twelve Data HTTP helpers (moved from twelve_data_api.py)
# ---------------------------------------------------------------------------

_TWELVE_BASE_URL = "https://api.twelvedata.com"

def _get_twelve_api_key() -> str | None:
    return os.environ.get("TWELVE_DATA_API_KEY")

def _twelve_get(path: str, params: dict) -> dict:
    api_key = _get_twelve_api_key()
    if not api_key:
        return {"error": "Missing TWELVE_DATA_API_KEY"}
    merged = {**params, "apikey": api_key}
    try:
        resp = requests.get(f"{_TWELVE_BASE_URL}{path}", params=merged, timeout=30)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def _fetch_income_statement(symbol: str):
    data = _twelve_get("/income_statement", params={"symbol": symbol})
    if not isinstance(data, dict) or "income_statement" not in data:
        print("TwelveData ERROR:", data)
        return None
    return data

def _fetch_balance_sheet(symbol: str):
    data = _twelve_get("/balance_sheet", params={"symbol": symbol})
    if not isinstance(data, dict) or "balance_sheet" not in data:
        print("TwelveData ERROR:", data)
        return None
    return data

def _fetch_cash_flow(symbol: str):
    data = _twelve_get("/cash_flow", params={"symbol": symbol})
    if not isinstance(data, dict) or "cash_flow" not in data:
        print("TwelveData ERROR:", data)
        return None
    return data

def _fetch_statistics(symbol: str):
    data = _twelve_get("/statistics", params={"symbol": symbol})
    if not isinstance(data, dict) or "statistics" not in data:
        print("TwelveData ERROR:", data)
        return None
    return data

def _fetch_prices(symbol: str, start_date=None, end_date=None, interval="1day", outputsize=5000):
    params = {"symbol": symbol, "interval": interval, "format": "JSON", "outputsize": outputsize}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    data = _twelve_get("/time_series", params=params)
    if not isinstance(data, dict) or "values" not in data:
        print("TwelveData ERROR:", data)
        return []
    prices = []
    for candle in reversed(data["values"]):
        prices.append(Price(
            time=candle["datetime"],
            open=float(candle["open"]),
            high=float(candle["high"]),
            low=float(candle["low"]),
            close=float(candle["close"]),
            volume=int(candle.get("volume", 0) or 0),
        ))
    return prices

def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)

        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from Twelve Data API."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    prices = _fetch_prices(ticker, start_date, end_date)

    if not prices:
        return []

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """
    Fetch financial metrics from Twelve Data API.
    Returns multiple periods by pulling income statement history for
    earnings_per_share growth calculations that agents need.
    """
    cache_key = f"metrics_{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    try:
        # Fetch all sources
        stats_data = _fetch_statistics(ticker)
        income_data = _fetch_income_statement(ticker)
        balance_data = _fetch_balance_sheet(ticker)

        metrics: list[FinancialMetrics] = []

        # Build period-level metrics from income + balance sheet history
        # so agents that need limit=5 actually get 5 historical periods
        income_periods = income_data.get("income_statement", []) if income_data else []
        balance_periods = balance_data.get("balance_sheet", []) if balance_data else []
        currency = (income_data or {}).get("meta", {}).get("currency", "USD")

        # Index balance sheet by fiscal_date for O(1) lookup
        balance_by_date: dict[str, dict] = {}
        for b in balance_periods:
            fd = b.get("fiscal_date")
            if fd:
                balance_by_date[fd] = b

        # Build one FinancialMetrics per income statement period
        for inc in income_periods[:limit]:
            fiscal_date = inc.get("fiscal_date", end_date)
            bal = balance_by_date.get(fiscal_date, {})

            # ----------------------------------------------------------------
            # Safe float helper
            # ----------------------------------------------------------------
            def _f(d: dict, *keys):
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None

            # Revenue and earnings
            revenue = _f(inc, "revenue", "total_revenue", "net_revenue")
            net_income = _f(inc, "net_income")
            gross_profit = _f(inc, "gross_profit")
            operating_income = _f(inc, "operating_income", "ebit")
            interest_expense = _f(inc, "interest_expense")
            ebitda = _f(inc, "ebitda")
            eps = _f(inc, "eps", "diluted_eps", "basic_eps", "earnings_per_share")

            # Balance sheet
            total_assets = _f(bal, "total_assets")
            total_equity = _f(bal, "total_shareholders_equity", "shareholders_equity", "total_equity")
            total_debt = _f(bal, "total_debt", "long_term_debt")
            cash = _f(bal, "cash_and_equivalents", "cash_and_cash_equivalents")

            # Derived ratios
            gross_margin = (gross_profit / revenue) if (gross_profit and revenue and revenue != 0) else None
            operating_margin = (operating_income / revenue) if (operating_income and revenue and revenue != 0) else None
            net_margin = (net_income / revenue) if (net_income and revenue and revenue != 0) else None
            roe = (net_income / total_equity) if (net_income and total_equity and total_equity != 0) else None
            roa = (net_income / total_assets) if (net_income and total_assets and total_assets != 0) else None
            de_ratio = (total_debt / total_equity) if (total_debt is not None and total_equity and total_equity != 0) else None
            interest_coverage = (operating_income / abs(interest_expense)) if (operating_income and interest_expense and interest_expense != 0) else None

            # Use statistics snapshot for valuation metrics (current only)
            stats = (stats_data or {}).get("statistics", {}) if fiscal_date == income_periods[0].get("fiscal_date") else {}
            val = stats.get("valuations_metrics", {})
            fin = stats.get("financials", {})

            metric = FinancialMetrics(
                ticker=ticker,
                report_period=fiscal_date,
                period="annual",
                currency=currency,
                # Valuation (only meaningful for latest period)
                market_cap=_f(val, "market_capitalization") if val else None,
                enterprise_value=_f(val, "enterprise_value") if val else None,
                price_to_earnings_ratio=_f(val, "trailing_pe") if val else None,
                price_to_book_ratio=_f(val, "price_to_book_mrq") if val else None,
                price_to_sales_ratio=_f(val, "price_to_sales_ttm") if val else None,
                enterprise_value_to_ebitda_ratio=_f(val, "enterprise_to_ebitda") if val else None,
                enterprise_value_to_revenue_ratio=_f(val, "enterprise_to_revenue") if val else None,
                peg_ratio=_f(val, "peg_ratio") if val else None,
                free_cash_flow_yield=None,
                # Margins (computed per period)
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                # Returns
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=None,  # requires invested capital calc
                # Efficiency — not computable from income + balance alone
                asset_turnover=(revenue / total_assets) if (revenue and total_assets and total_assets != 0) else None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                # Liquidity
                current_ratio=_f(bal, "current_ratio"),
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                # Leverage
                debt_to_equity=de_ratio,
                debt_to_assets=(total_debt / total_assets) if (total_debt is not None and total_assets and total_assets != 0) else None,
                interest_coverage=interest_coverage,
                # Growth (filled in post-loop)
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                # Per share
                earnings_per_share=eps,
                book_value_per_share=_f(bal, "book_value_per_share"),
                free_cash_flow_per_share=None,
                payout_ratio=None,
            )
            metrics.append(metric)

        # ----------------------------------------------------------------
        # Fill YoY growth rates now that we have the full list
        # metrics[0] = most recent, metrics[1] = prior year
        # ----------------------------------------------------------------
        for i in range(len(metrics) - 1):
            curr = metrics[i]
            prev = metrics[i + 1]

            def _growth(c, p):
                if c is not None and p is not None and p != 0:
                    return (c - p) / abs(p)
                return None

            curr.revenue_growth = _growth(
                _get_revenue_from_income(income_periods[i]),
                _get_revenue_from_income(income_periods[i + 1])
            )
            curr.earnings_growth = _growth(curr.return_on_equity, prev.return_on_equity)
            curr.earnings_per_share_growth = _growth(curr.earnings_per_share, prev.earnings_per_share)
            curr.operating_income_growth = _growth(curr.operating_margin, prev.operating_margin)

        # If no income periods available, fall back to statistics-only (1 period)
        if not metrics and stats_data:
            stats = stats_data.get("statistics", {})
            val = stats.get("valuations_metrics", {})
            fin = stats.get("financials", {})
            inc_stmt = fin.get("income_statement", {})
            bs = fin.get("balance_sheet", {})

            def _f(d, *keys):
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None

            metric = FinancialMetrics(
                ticker=ticker,
                report_period=end_date,
                period="ttm",
                currency=stats_data.get("meta", {}).get("currency", "USD"),
                market_cap=_f(val, "market_capitalization"),
                enterprise_value=_f(val, "enterprise_value"),
                price_to_earnings_ratio=_f(val, "trailing_pe"),
                price_to_book_ratio=_f(val, "price_to_book_mrq"),
                price_to_sales_ratio=_f(val, "price_to_sales_ttm"),
                enterprise_value_to_ebitda_ratio=_f(val, "enterprise_to_ebitda"),
                enterprise_value_to_revenue_ratio=_f(val, "enterprise_to_revenue"),
                peg_ratio=_f(val, "peg_ratio"),
                free_cash_flow_yield=None,
                gross_margin=_f(fin, "gross_margin"),
                operating_margin=_f(fin, "operating_margin"),
                net_margin=_f(fin, "profit_margin"),
                return_on_equity=_f(fin, "return_on_equity_ttm"),
                return_on_assets=_f(fin, "return_on_assets_ttm"),
                return_on_invested_capital=None,
                asset_turnover=None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=None,
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=_f(bs, "total_debt_to_equity_mrq"),
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=_f(inc_stmt, "quarterly_revenue_growth"),
                earnings_growth=_f(inc_stmt, "quarterly_earnings_growth_yoy"),
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=_f(inc_stmt, "diluted_eps_ttm"),
                book_value_per_share=_f(bs, "book_value_per_share_mrq"),
                free_cash_flow_per_share=None,
            )
            metrics.append(metric)

        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    except Exception as e:
        print(f"Error fetching financial metrics for {ticker}: {str(e)}")
        return []


def _get_revenue_from_income(inc: dict) -> float | None:
    """Extract revenue from an income statement dict."""
    for k in ("revenue", "total_revenue", "net_revenue"):
        v = inc.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


###############################################################################
# Field name mapping: Twelve Data → agent field names
# Agents use snake_case names that match what LineItem's extra fields store.
# Twelve Data returns slightly different names for some fields.
# This mapping ensures _safe_get() calls in agents find the right values.
###############################################################################

# Income statement field mapping: twelve_data_key -> agent_key
_INCOME_FIELD_MAP = {
    "fiscal_date":                          "fiscal_date",
    "revenue":                              "revenue",
    "total_revenue":                        "revenue",          # alias
    "gross_profit":                         "gross_profit",
    "gross_margin":                         "gross_margin",
    "operating_income":                     "operating_income",
    "ebit":                                 "ebit",
    "operating_expense":                    "operating_expense",
    "interest_expense":                     "interest_expense",
    "net_income":                           "net_income",
    "net_income_common_stockholders":       "net_income",       # alias
    "ebitda":                               "ebitda",
    "eps":                                  "earnings_per_share",
    "diluted_eps":                          "earnings_per_share",  # alias
    "basic_eps":                            "earnings_per_share",  # alias
    "shares_outstanding":                   "outstanding_shares",
    "weighted_average_shares_outstanding":  "outstanding_shares",  # alias
    "research_and_development":             "research_and_development",
    "selling_general_administrative":       "selling_general_administrative",
    "depreciation_amortization":            "depreciation_and_amortization",
}

# Balance sheet field mapping
_BALANCE_FIELD_MAP = {
    "fiscal_date":                          "fiscal_date",
    "total_assets":                         "total_assets",
    "total_liabilities":                    "total_liabilities",
    "total_shareholders_equity":            "shareholders_equity",
    "shareholders_equity":                  "shareholders_equity",  # alias
    "total_equity":                         "shareholders_equity",  # alias
    "cash_and_equivalents":                 "cash_and_equivalents",
    "cash_and_cash_equivalents":            "cash_and_equivalents",  # alias
    "short_term_investments":               "short_term_investments",
    "accounts_receivable":                  "accounts_receivable",
    "inventory":                            "inventory",
    "current_assets":                       "current_assets",
    "current_liabilities":                  "current_liabilities",
    "long_term_debt":                       "total_debt",
    "total_debt":                           "total_debt",           # alias
    "short_term_debt":                      "short_term_debt",
    "book_value_per_share":                 "book_value_per_share",
    "goodwill":                             "goodwill",
    "intangible_assets":                    "intangible_assets",
    "property_plant_equipment":             "property_plant_equipment",
}

# Cash flow field mapping
_CASHFLOW_FIELD_MAP = {
    "fiscal_date":                                  "fiscal_date",
    "operating_cash_flow":                          "operating_cash_flow",
    "capital_expenditures":                         "capital_expenditure",
    "capital_expenditure":                          "capital_expenditure",  # alias
    "free_cash_flow":                               "free_cash_flow",
    "investing_cash_flow":                          "investing_cash_flow",
    "financing_cash_flow":                          "financing_cash_flow",
    "dividends_paid":                               "dividends_and_other_cash_distributions",
    "dividends_and_other_cash_distributions":       "dividends_and_other_cash_distributions",  # alias
    "issuance_of_stock":                            "issuance_or_purchase_of_equity_shares",
    "repurchase_of_stock":                          "repurchase_of_stock",
    "net_change_in_cash":                           "net_change_in_cash",
    "depreciation_amortization":                    "depreciation_and_amortization",
    "stock_based_compensation":                     "stock_based_compensation",
    "change_in_working_capital":                    "change_in_working_capital",
}


def _map_statement(raw: dict, field_map: dict) -> dict:
    """
    Map raw Twelve Data statement fields to agent-expected field names.
    Preserves fiscal_date and only writes the first match per agent key
    (so aliases don't overwrite already-set values).
    """
    mapped: dict = {}
    for td_key, agent_key in field_map.items():
        if td_key in raw and agent_key not in mapped:
            val = raw[td_key]
            if val is not None:
                # Try to convert numeric strings to float
                try:
                    mapped[agent_key] = float(val)
                except (TypeError, ValueError):
                    mapped[agent_key] = val
    return mapped


def _merge_period_data(
    fiscal_date: str,
    income_map: dict,
    balance_map: dict,
    cashflow_map: dict,
) -> dict:
    """
    Merge all three statement maps into one consolidated dict for a fiscal period.
    Income statement wins on conflicts (revenue, operating_income etc. are more
    reliably reported there).
    """
    merged = {}
    # Apply in reverse priority order so higher priority overwrites
    merged.update(cashflow_map)
    merged.update(balance_map)
    merged.update(income_map)
    # Always force fiscal_date
    merged["fiscal_date"] = fiscal_date

    # Compute gross_margin if not present but revenue + gross_profit are
    if "gross_margin" not in merged:
        rev = merged.get("revenue")
        gp = merged.get("gross_profit")
        if rev and gp and float(rev) != 0:
            merged["gross_margin"] = float(gp) / float(rev)

    # Compute ebit from operating_income if missing
    if "ebit" not in merged and "operating_income" in merged:
        merged["ebit"] = merged["operating_income"]

    # Compute issuance_or_purchase_of_equity_shares from components
    # Negative = net buyback, Positive = net issuance
    if "issuance_or_purchase_of_equity_shares" not in merged:
        issued = merged.get("issuance_of_stock", 0) or 0
        repurchased = merged.get("repurchase_of_stock", 0) or 0
        if issued != 0 or repurchased != 0:
            # repurchased is typically negative in Twelve Data
            merged["issuance_or_purchase_of_equity_shares"] = float(issued) + float(repurchased)

    return merged


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """
    Fetch and consolidate financial line items from Twelve Data.

    Returns one LineItem per fiscal period (not one per statement type).
    Each LineItem contains merged fields from income statement + balance sheet
    + cash flow for that period, mapped to agent-expected field names.

    Agents use _safe_get(item, "free_cash_flow"), _safe_get(item, "revenue") etc.
    All of those field names are correctly mapped here.
    """
    cache_key = f"line_items_{ticker}_{period}_{end_date}_{limit}_{'_'.join(sorted(line_items))}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [LineItem(**item) for item in cached_data]

    try:
        # Fetch all three statement types
        income_data = _fetch_income_statement(ticker)
        balance_data = _fetch_balance_sheet(ticker)
        cashflow_data = _fetch_cash_flow(ticker)

        income_periods: list[dict] = (income_data or {}).get("income_statement", [])
        balance_periods: list[dict] = (balance_data or {}).get("balance_sheet", [])
        cashflow_periods: list[dict] = (cashflow_data or {}).get("cash_flow", [])

        currency = (income_data or balance_data or cashflow_data or {}).get(
            "meta", {}
        ).get("currency", "USD")

        # Index balance sheet and cash flow by fiscal_date for O(1) lookup
        balance_by_date: dict[str, dict] = {}
        for b in balance_periods:
            fd = b.get("fiscal_date")
            if fd:
                balance_by_date[fd] = b

        cashflow_by_date: dict[str, dict] = {}
        for c in cashflow_periods:
            fd = c.get("fiscal_date")
            if fd:
                cashflow_by_date[fd] = c

        result: list[LineItem] = []

        # Build one consolidated LineItem per income statement period
        # (income statement is the primary source for fiscal dates)
        for inc in income_periods[:limit]:
            fiscal_date = inc.get("fiscal_date")
            if not fiscal_date:
                continue

            bal = balance_by_date.get(fiscal_date, {})
            cf = cashflow_by_date.get(fiscal_date, {})

            # Map each statement to agent field names
            inc_mapped = _map_statement(inc, _INCOME_FIELD_MAP)
            bal_mapped = _map_statement(bal, _BALANCE_FIELD_MAP)
            cf_mapped = _map_statement(cf, _CASHFLOW_FIELD_MAP)

            # Merge into one consolidated dict
            merged = _merge_period_data(fiscal_date, inc_mapped, bal_mapped, cf_mapped)

            # Only include requested line item fields (plus required base fields)
            # This filters the merged dict to only what agents asked for,
            # but we include everything since LineItem allows extra fields
            item = LineItem(
                ticker=ticker,
                report_period=fiscal_date,
                period="annual",
                currency=currency,
                **{k: v for k, v in merged.items() if k != "fiscal_date"},
            )
            result.append(item)

        # If income statement returned nothing, try using balance sheet dates
        if not result:
            for bal in balance_periods[:limit]:
                fiscal_date = bal.get("fiscal_date")
                if not fiscal_date:
                    continue

                cf = cashflow_by_date.get(fiscal_date, {})
                bal_mapped = _map_statement(bal, _BALANCE_FIELD_MAP)
                cf_mapped = _map_statement(cf, _CASHFLOW_FIELD_MAP)
                merged = _merge_period_data(fiscal_date, {}, bal_mapped, cf_mapped)

                item = LineItem(
                    ticker=ticker,
                    report_period=fiscal_date,
                    period="annual",
                    currency=currency,
                    **{k: v for k, v in merged.items() if k != "fiscal_date"},
                )
                result.append(item)

        # Cache and return
        _cache.set_financial_metrics(cache_key, [item.model_dump() for item in result])
        return result

    except Exception as e:
        print(f"Error fetching line items for {ticker}: {str(e)}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """
    Insider trade data is not available from Twelve Data API.
    Returns empty list — governance_agent will use its neutral default.
    """
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from NewsData.io"""
    newsdata_key = os.environ.get("NEWSDATA_API_KEY")
    if not newsdata_key:
        return []
    
    # Strip exchange suffix for Indian stocks
    clean_ticker = ticker.split(":")[0]
    
    resp = requests.get(
        "https://newsdata.io/api/1/latest",
        params={
            "apikey": newsdata_key,
            "q": clean_ticker,
            "language": "en",
            "size": min(limit, 10),  # free tier limit
        },
        timeout=15,
    )
    
    if not resp.ok:
        return []
    
    articles = resp.json().get("results", [])
    news = []
    for a in articles:
        news.append(CompanyNews(
            ticker=ticker,
            title=a.get("title", ""),
            author=a.get("creator", [""])[0] if a.get("creator") else "",
            source=a.get("source_id", ""),
            date=a.get("pubDate", end_date),
            url=a.get("link", ""),
            sentiment=a.get("sentiment", None),
        ))
    return news

def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from Twelve Data statistics endpoint."""
    try:
        stats_data = _fetch_statistics(ticker)
        if stats_data and "statistics" in stats_data:
            mc = stats_data["statistics"].get("valuations_metrics", {}).get("market_capitalization")
            if mc is not None:
                return float(mc)
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {str(e)}")

    # Fallback: derive from financial metrics
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if financial_metrics and financial_metrics[0].market_cap:
        return financial_metrics[0].market_cap

    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)