import datetime
import os
import threading
import pandas as pd
import requests
import time

from src.data.cache import get_cache
import src.data.cache as _cache_module
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

_cache = get_cache()

# Thread-safe lock for statement cache
_statement_lock = threading.Lock()

_TWELVE_BASE_URL = "https://api.twelvedata.com"

# ── yfinance fallback switch ──────────────────────────────────────────────────
# Set USE_YFINANCE=true in .env to use yfinance instead of Twelve Data.
# Useful for demos when Twelve Data credits are exhausted.
_USE_YFINANCE = os.environ.get("USE_YFINANCE", "false").lower() == "true"

def _is_yfinance_mode() -> bool:
    """Return True if yfinance mode is active OR Twelve Data key is missing."""
    if _USE_YFINANCE:
        return True
    return not bool(os.environ.get("TWELVE_DATA_API_KEY"))

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

# ── yfinance bridge functions ─────────────────────────────────────────────────

def _yf_get_ticker(symbol: str):
    import yfinance as yf
    clean = symbol.split(":")[0]
    # Map NSE/BSE tickers to yfinance format
    if symbol.endswith(":NSE"):
        clean = clean + ".NS"
    elif symbol.endswith(":BSE"):
        clean = clean + ".BO"
    return yf.Ticker(clean)

def _yf_income_statement(symbol: str):
    try:
        t = _yf_get_ticker(symbol)
        fin = t.financials  # columns = dates, rows = line items
        if fin is None or fin.empty:
            return None
        periods = []
        for col in fin.columns:
            row = {"fiscal_date": str(col.date())}
            def _g(keys):
                for k in keys:
                    if k in fin.index:
                        v = fin.loc[k, col]
                        if v is not None and str(v) != "nan":
                            try: return float(v)
                            except: pass
                return None
            row["sales"]        = _g(["Total Revenue"])
            row["gross_profit"] = _g(["Gross Profit"])
            row["operating_income"] = _g(["Operating Income", "EBIT"])
            row["ebit"]         = _g(["EBIT", "Operating Income"])
            row["ebitda"]       = _g(["EBITDA"])
            row["net_income"]   = _g(["Net Income"])
            row["eps_diluted"]  = _g(["Diluted EPS"])
            row["eps_basic"]    = _g(["Basic EPS"])
            row["diluted_shares_outstanding"] = _g(["Diluted Average Shares"])
            row["interest_expense"] = _g(["Interest Expense"])
            # operating_expense as flat value
            row["operating_expense"] = _g(["Total Operating Expenses", "Operating Expense"])
            row["research_and_development"] = _g(["Research And Development"])
            row["selling_general_administrative"] = _g(["Selling General Administrative"])
            periods.append(row)
        return {"income_statement": periods, "meta": {"currency": "USD"}}
    except Exception as e:
        print(f"[yfinance] income_statement failed for {symbol}: {e}")
        return None

def _yf_balance_sheet(symbol: str):
    try:
        t = _yf_get_ticker(symbol)
        bs = t.balance_sheet
        if bs is None or bs.empty:
            return None
        periods = []
        for col in bs.columns:
            row = {"fiscal_date": str(col.date())}
            def _g(keys):
                for k in keys:
                    if k in bs.index:
                        v = bs.loc[k, col]
                        if v is not None and str(v) != "nan":
                            try: return float(v)
                            except: pass
                return None
            # Build nested structure matching Twelve Data format
            row["assets"] = {
                "total_assets": _g(["Total Assets"]),
                "current_assets": {
                    "total_current_assets": _g(["Current Assets"]),
                    "cash_and_cash_equivalents": _g(["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
                    "accounts_receivable": _g(["Accounts Receivable", "Net Receivables"]),
                    "inventory": _g(["Inventory"]),
                },
                "non_current_assets": {
                    "goodwill": _g(["Goodwill"]),
                    "intangible_assets": _g(["Intangible Assets", "Other Intangible Assets"]),
                },
            }
            row["liabilities"] = {
                "total_liabilities": _g(["Total Liabilities Net Minority Interest", "Total Liab"]),
                "current_liabilities": {
                    "total_current_liabilities": _g(["Current Liabilities"]),
                    "short_term_debt": _g(["Short Term Debt", "Current Debt"]),
                },
                "non_current_liabilities": {
                    "long_term_debt": _g(["Long Term Debt"]),
                },
            }
            row["shareholders_equity"] = {
                "total_shareholders_equity": _g(["Stockholders Equity", "Common Stock Equity"]),
            }
            periods.append(row)
        return {"balance_sheet": periods, "meta": {"currency": "USD"}}
    except Exception as e:
        print(f"[yfinance] balance_sheet failed for {symbol}: {e}")
        return None

def _yf_cash_flow(symbol: str):
    try:
        t = _yf_get_ticker(symbol)
        cf = t.cashflow
        if cf is None or cf.empty:
            return None
        periods = []
        for col in cf.columns:
            row = {"fiscal_date": str(col.date())}
            def _g(keys):
                for k in keys:
                    if k in cf.index:
                        v = cf.loc[k, col]
                        if v is not None and str(v) != "nan":
                            try: return float(v)
                            except: pass
                return None
            row["free_cash_flow"] = _g(["Free Cash Flow"])
            row["operating_activities"] = {
                "operating_cash_flow": _g(["Operating Cash Flow", "Cash From Operations"]),
                "depreciation": _g(["Depreciation And Amortization", "Depreciation"]),
                "stock_based_compensation": _g(["Stock Based Compensation"]),
            }
            row["investing_activities"] = {
                "capital_expenditures": _g(["Capital Expenditure"]),
                "investing_cash_flow": _g(["Investing Cash Flow"]),
            }
            row["financing_activities"] = {
                "financing_cash_flow": _g(["Financing Cash Flow"]),
                "common_dividends": _g(["Common Stock Dividend Paid", "Cash Dividends Paid"]),
                "common_stock_repurchase": _g(["Repurchase Of Capital Stock", "Common Stock Repurchase"]),
                "common_stock_issuance": _g(["Common Stock Issuance"]),
            }
            periods.append(row)
        return {"cash_flow": periods, "meta": {"currency": "USD"}}
    except Exception as e:
        print(f"[yfinance] cash_flow failed for {symbol}: {e}")
        return None

def _yf_statistics(symbol: str):
    try:
        t = _yf_get_ticker(symbol)
        info = t.info or {}
        stats = {
            "valuations_metrics": {
                "market_capitalization": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "trailing_pe": info.get("trailingPE"),
                "price_to_book_mrq": info.get("priceToBook"),
                "price_to_sales_ttm": info.get("priceToSalesTrailing12Months"),
                "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
                "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                "peg_ratio": info.get("pegRatio"),
            },
            "financials": {
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "return_on_equity_ttm": info.get("returnOnEquity"),
                "return_on_assets_ttm": info.get("returnOnAssets"),
            },
        }
        return {"statistics": stats, "meta": {"currency": info.get("currency", "USD")}}
    except Exception as e:
        print(f"[yfinance] statistics failed for {symbol}: {e}")
        return None

def _yf_prices(symbol: str, start_date=None, end_date=None) -> list:
    try:
        import yfinance as yf
        clean = symbol.split(":")[0]
        if symbol.endswith(":NSE"):
            clean += ".NS"
        elif symbol.endswith(":BSE"):
            clean += ".BO"
        hist = yf.Ticker(clean).history(start=start_date, end=end_date)
        if hist is None or hist.empty:
            return []
        prices = []
        for date, row in hist.iterrows():
            prices.append(Price(
                time=str(date.date()),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row.get("Volume", 0) or 0),
            ))
        return prices
    except Exception as e:
        print(f"[yfinance] prices failed for {symbol}: {e}")
        return []

def _fetch_income_statement(symbol: str):
    if _is_yfinance_mode():
        return _yf_income_statement(symbol)
    key = f"income_{symbol}"
    if key in _cache_module._statement_cache:
        return _cache_module._statement_cache[key]
    with _statement_lock:
        if key in _cache_module._statement_cache:
            return _cache_module._statement_cache[key]
        data = _twelve_get("/income_statement", params={"symbol": symbol})
        if not isinstance(data, dict) or "income_statement" not in data:
            print("TwelveData ERROR:", data)
            return None
        _cache_module._statement_cache[key] = data
        return data


def _fetch_balance_sheet(symbol: str):
    if _is_yfinance_mode():
        return _yf_balance_sheet(symbol)
    key = f"balance_{symbol}"
    if key in _cache_module._statement_cache:
        return _cache_module._statement_cache[key]
    with _statement_lock:
        if key in _cache_module._statement_cache:
            return _cache_module._statement_cache[key]
        data = _twelve_get("/balance_sheet", params={"symbol": symbol})
        if not isinstance(data, dict) or "balance_sheet" not in data:
            print("TwelveData ERROR:", data)
            return None
        _cache_module._statement_cache[key] = data
        return data


def _fetch_cash_flow(symbol: str):
    if _is_yfinance_mode():
        return _yf_cash_flow(symbol)
    key = f"cashflow_{symbol}"
    if key in _cache_module._statement_cache:
        return _cache_module._statement_cache[key]
    with _statement_lock:
        if key in _cache_module._statement_cache:
            return _cache_module._statement_cache[key]
        data = _twelve_get("/cash_flow", params={"symbol": symbol})
        if not isinstance(data, dict) or "cash_flow" not in data:
            print("TwelveData ERROR:", data)
            return None
        _cache_module._statement_cache[key] = data
        return data


def _fetch_statistics(symbol: str):
    if _is_yfinance_mode():
        return _yf_statistics(symbol)
    key = f"statistics_{symbol}"
    if key in _cache_module._statement_cache:
        return _cache_module._statement_cache[key]
    with _statement_lock:
        if key in _cache_module._statement_cache:
            return _cache_module._statement_cache[key]
        data = _twelve_get("/statistics", params={"symbol": symbol})
        if not isinstance(data, dict) or "statistics" not in data:
            print("TwelveData ERROR:", data)
            return None
        _cache_module._statement_cache[key] = data
        return data


def _fetch_prices(symbol: str, start_date=None, end_date=None, interval="1day", outputsize=5000):
    if _is_yfinance_mode():
        return _yf_prices(symbol, start_date=start_date, end_date=end_date)
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


def _to_float(v):
    """Safely convert a value to float, returning None if not possible."""
    if v is None or isinstance(v, dict):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Nested structure flatteners
# Twelve Data returns deeply nested JSON. These functions extract and normalize
# all fields into flat dicts keyed by agent-expected field names.
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_income(inc: dict) -> dict:
    """
    Flatten Twelve Data income statement into agent-ready field dict.

    Twelve Data income statement structure (relevant fields):
      sales                          → revenue
      gross_profit                   → gross_profit
      operating_income               → operating_income / ebit
      ebit                           → ebit
      ebitda                         → ebitda
      net_income                     → net_income
      eps_diluted / eps_basic        → earnings_per_share
      diluted_shares_outstanding     → outstanding_shares
      operating_expense (nested dict):
        research_and_development     → research_and_development
        selling_general_and_administrative → selling_general_administrative
      non_operating_interest (nested dict):
        expense                      → interest_expense
    """
    out = {"fiscal_date": inc.get("fiscal_date", "")}

    # Revenue
    out["revenue"] = (
        _to_float(inc.get("sales")) or
        _to_float(inc.get("revenue")) or
        _to_float(inc.get("total_revenue"))
    )

    # P&L
    out["gross_profit"]      = _to_float(inc.get("gross_profit"))
    out["operating_income"]  = _to_float(inc.get("operating_income"))
    out["ebit"]              = _to_float(inc.get("ebit")) or _to_float(inc.get("operating_income"))
    out["ebitda"]            = _to_float(inc.get("ebitda"))
    out["net_income"]        = (_to_float(inc.get("net_income")) or
                                _to_float(inc.get("net_income_continuous_operations")))

    # EPS
    out["earnings_per_share"] = (
        _to_float(inc.get("eps_diluted")) or
        _to_float(inc.get("eps_basic")) or
        _to_float(inc.get("diluted_eps")) or
        _to_float(inc.get("eps"))
    )

    # Shares
    out["outstanding_shares"] = (
        _to_float(inc.get("diluted_shares_outstanding")) or
        _to_float(inc.get("basic_shares_outstanding")) or
        _to_float(inc.get("shares_outstanding")) or
        _to_float(inc.get("weighted_average_shares_outstanding"))
    )

    # Operating expense (nested dict in Twelve Data)
    op_exp = inc.get("operating_expense")
    if isinstance(op_exp, dict):
        out["research_and_development"]      = _to_float(op_exp.get("research_and_development"))
        out["selling_general_administrative"] = _to_float(
            op_exp.get("selling_general_and_administrative") or
            op_exp.get("selling_general_administrative")
        )
    elif op_exp is not None:
        out["operating_expense"] = _to_float(op_exp)

    # Interest expense (nested under non_operating_interest.expense)
    ni = inc.get("non_operating_interest")
    if isinstance(ni, dict):
        out["interest_expense"] = _to_float(ni.get("expense"))
    else:
        out["interest_expense"] = _to_float(inc.get("interest_expense"))

    # Derived: gross_margin
    if out.get("revenue") and out.get("gross_profit") and out["revenue"] != 0:
        out["gross_margin"] = out["gross_profit"] / out["revenue"]

    return {k: v for k, v in out.items() if v is not None or k == "fiscal_date"}


def _flatten_balance(bal: dict) -> dict:
    """
    Flatten Twelve Data balance sheet into agent-ready field dict.

    Twelve Data balance sheet structure:
      assets (nested):
        total_assets                               → total_assets
        current_assets (nested):
          total_current_assets                     → current_assets
          cash_and_cash_equivalents                → cash_and_equivalents
          accounts_receivable                      → accounts_receivable
          inventory                                → inventory
          other_short_term_investments             → short_term_investments
        non_current_assets (nested):
          goodwill                                 → goodwill
          intangible_assets                        → intangible_assets
          land_and_improvements                    → property_plant_equipment
      liabilities (nested):
        total_liabilities                          → total_liabilities
        current_liabilities (nested):
          total_current_liabilities                → current_liabilities
          short_term_debt                          → short_term_debt
        non_current_liabilities (nested):
          long_term_debt                           → total_debt
      shareholders_equity (nested):
        total_shareholders_equity                  → shareholders_equity
    """
    out = {"fiscal_date": bal.get("fiscal_date", "")}

    # ── Assets ──────────────────────────────────────────────────────────────
    assets = bal.get("assets")
    if isinstance(assets, dict):
        out["total_assets"] = _to_float(assets.get("total_assets"))

        curr = assets.get("current_assets")
        if isinstance(curr, dict):
            out["current_assets"]       = _to_float(curr.get("total_current_assets"))
            out["cash_and_equivalents"] = _to_float(curr.get("cash_and_cash_equivalents"))
            out["accounts_receivable"]  = _to_float(curr.get("accounts_receivable"))
            out["inventory"]            = _to_float(curr.get("inventory"))
            out["short_term_investments"] = _to_float(curr.get("other_short_term_investments"))

        non_curr = assets.get("non_current_assets")
        if isinstance(non_curr, dict):
            out["goodwill"]                  = _to_float(non_curr.get("goodwill"))
            out["intangible_assets"]         = _to_float(non_curr.get("intangible_assets"))
            out["property_plant_equipment"]  = _to_float(non_curr.get("land_and_improvements"))
    else:
        # Fallback: some endpoints return flat balance sheet
        out["total_assets"] = _to_float(bal.get("total_assets"))
        out["current_assets"] = _to_float(bal.get("total_current_assets") or bal.get("current_assets"))
        out["cash_and_equivalents"] = _to_float(bal.get("cash_and_cash_equivalents") or bal.get("cash_and_equivalents"))

    # ── Liabilities ──────────────────────────────────────────────────────────
    liab = bal.get("liabilities")
    if isinstance(liab, dict):
        out["total_liabilities"] = _to_float(liab.get("total_liabilities"))

        curr_l = liab.get("current_liabilities")
        if isinstance(curr_l, dict):
            out["current_liabilities"] = _to_float(curr_l.get("total_current_liabilities"))
            out["short_term_debt"]     = _to_float(curr_l.get("short_term_debt"))

        non_curr_l = liab.get("non_current_liabilities")
        if isinstance(non_curr_l, dict):
            out["total_debt"] = _to_float(non_curr_l.get("long_term_debt"))
    else:
        out["total_liabilities"] = _to_float(bal.get("total_liabilities"))
        out["current_liabilities"] = _to_float(bal.get("total_current_liabilities") or bal.get("current_liabilities"))
        out["total_debt"] = _to_float(bal.get("long_term_debt") or bal.get("total_debt"))

    # ── Shareholders equity ───────────────────────────────────────────────────
    eq = bal.get("shareholders_equity")
    if isinstance(eq, dict):
        out["shareholders_equity"]  = _to_float(eq.get("total_shareholders_equity"))
        out["book_value_per_share"] = _to_float(eq.get("book_value_per_share"))
    else:
        out["shareholders_equity"] = (
            _to_float(bal.get("total_shareholders_equity")) or
            _to_float(bal.get("shareholders_equity")) or
            _to_float(bal.get("total_equity"))
        )
        out["book_value_per_share"] = _to_float(bal.get("book_value_per_share"))

    # ── Derived ───────────────────────────────────────────────────────────────
    # current_ratio
    ca = out.get("current_assets")
    cl = out.get("current_liabilities")
    if ca and cl and cl != 0:
        out["current_ratio"] = ca / cl

    return {k: v for k, v in out.items() if v is not None or k == "fiscal_date"}


def _flatten_cashflow(cf: dict) -> dict:
    """
    Flatten Twelve Data cash flow into agent-ready field dict.

    Twelve Data cash flow structure:
      free_cash_flow                               → free_cash_flow (top level)
      operating_activities (nested):
        operating_cash_flow                        → operating_cash_flow
        depreciation                               → depreciation_and_amortization
        stock_based_compensation                   → stock_based_compensation
        other_assets_liabilities                   → change_in_working_capital
      investing_activities (nested):
        capital_expenditures                       → capital_expenditure
        investing_cash_flow                        → investing_cash_flow
      financing_activities (nested):
        financing_cash_flow                        → financing_cash_flow
        common_dividends                           → dividends_and_other_cash_distributions
        common_stock_repurchase                    → repurchase_of_stock
        common_stock_issuance + repurchase         → issuance_or_purchase_of_equity_shares
    """
    out = {"fiscal_date": cf.get("fiscal_date", "")}

    # Top-level
    out["free_cash_flow"] = _to_float(cf.get("free_cash_flow"))

    # Operating activities
    op = cf.get("operating_activities")
    if isinstance(op, dict):
        out["operating_cash_flow"]           = _to_float(op.get("operating_cash_flow"))
        out["depreciation_and_amortization"] = _to_float(op.get("depreciation"))
        out["stock_based_compensation"]      = _to_float(op.get("stock_based_compensation"))
        out["change_in_working_capital"]     = _to_float(op.get("other_assets_liabilities"))
    else:
        out["operating_cash_flow"] = _to_float(cf.get("operating_cash_flow"))
        out["depreciation_and_amortization"] = _to_float(cf.get("depreciation") or cf.get("depreciation_amortization"))

    # Investing activities
    inv = cf.get("investing_activities")
    if isinstance(inv, dict):
        out["capital_expenditure"] = _to_float(inv.get("capital_expenditures") or inv.get("capital_expenditure"))
        out["investing_cash_flow"] = _to_float(inv.get("investing_cash_flow"))
    else:
        out["capital_expenditure"] = _to_float(cf.get("capital_expenditures") or cf.get("capital_expenditure"))

    # Financing activities
    fin = cf.get("financing_activities")
    if isinstance(fin, dict):
        out["financing_cash_flow"]                     = _to_float(fin.get("financing_cash_flow"))
        out["dividends_and_other_cash_distributions"]  = _to_float(fin.get("common_dividends"))
        out["repurchase_of_stock"]                     = _to_float(fin.get("common_stock_repurchase"))

        # Net equity issuance/repurchase
        issued     = _to_float(fin.get("common_stock_issuance")) or 0.0
        repurchased = _to_float(fin.get("common_stock_repurchase")) or 0.0
        net = issued + repurchased
        if net != 0:
            out["issuance_or_purchase_of_equity_shares"] = net
    else:
        out["dividends_and_other_cash_distributions"] = _to_float(cf.get("dividends_paid") or cf.get("common_dividends"))

    # Derived: net_change_in_cash
    out["net_change_in_cash"] = _to_float(cf.get("net_change_in_cash") or cf.get("end_cash_position"))

    return {k: v for k, v in out.items() if v is not None or k == "fiscal_date"}


def _make_api_request(url, headers, method="GET", json_data=None, max_retries=3):
    for attempt in range(max_retries + 1):
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        if response.status_code == 429 and attempt < max_retries:
            delay = 60 + (30 * attempt)
            print(f"Rate limited. Waiting {delay}s...")
            time.sleep(delay)
            continue
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Public API functions
# ─────────────────────────────────────────────────────────────────────────────

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
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
    cache_key = f"metrics_{ticker}_{period}_{end_date}_{limit}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    try:
        stats_data  = _fetch_statistics(ticker)
        income_data = _fetch_income_statement(ticker)
        balance_data = _fetch_balance_sheet(ticker)

        income_periods  = (income_data  or {}).get("income_statement", [])
        balance_periods = (balance_data or {}).get("balance_sheet",    [])
        currency = (income_data or {}).get("meta", {}).get("currency", "USD")

        balance_by_date = {}
        for b in balance_periods:
            fd = b.get("fiscal_date")
            if fd:
                balance_by_date[fd] = b

        metrics = []

        for inc in income_periods[:limit]:
            fiscal_date = inc.get("fiscal_date", end_date)
            bal = balance_by_date.get(fiscal_date, {})

            # Use new flatten functions
            i = _flatten_income(inc)
            b = _flatten_balance(bal)

            revenue          = i.get("revenue")
            net_income       = i.get("net_income")
            gross_profit     = i.get("gross_profit")
            operating_income = i.get("operating_income")
            interest_expense = i.get("interest_expense")
            eps              = i.get("earnings_per_share")

            total_assets  = b.get("total_assets")
            total_equity  = b.get("shareholders_equity")
            total_debt    = b.get("total_debt")
            cash          = b.get("cash_and_equivalents")
            current_ratio = b.get("current_ratio")

            gross_margin    = i.get("gross_margin")
            operating_margin = (operating_income / revenue) if (operating_income and revenue and revenue != 0) else None
            net_margin      = (net_income / revenue) if (net_income and revenue and revenue != 0) else None
            roe = (net_income / total_equity) if (net_income and total_equity and total_equity != 0) else None
            roa = (net_income / total_assets) if (net_income and total_assets and total_assets != 0) else None
            de_ratio = (total_debt / total_equity) if (total_debt is not None and total_equity and total_equity != 0) else None
            interest_coverage = (operating_income / abs(interest_expense)) if (operating_income and interest_expense and interest_expense != 0) else None

            stats = (stats_data or {}).get("statistics", {}) if fiscal_date == (income_periods[0].get("fiscal_date") if income_periods else None) else {}
            val = stats.get("valuations_metrics", {})

            metric = FinancialMetrics(
                ticker=ticker,
                report_period=fiscal_date,
                period="annual",
                currency=currency,
                market_cap=_to_float((val or {}).get("market_capitalization")) if val else None,
                enterprise_value=_to_float((val or {}).get("enterprise_value")) if val else None,
                price_to_earnings_ratio=_to_float((val or {}).get("trailing_pe")) if val else None,
                price_to_book_ratio=_to_float((val or {}).get("price_to_book_mrq")) if val else None,
                price_to_sales_ratio=_to_float((val or {}).get("price_to_sales_ttm")) if val else None,
                enterprise_value_to_ebitda_ratio=_to_float((val or {}).get("enterprise_to_ebitda")) if val else None,
                enterprise_value_to_revenue_ratio=_to_float((val or {}).get("enterprise_to_revenue")) if val else None,
                peg_ratio=_to_float((val or {}).get("peg_ratio")) if val else None,
                free_cash_flow_yield=None,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=None,
                asset_turnover=(revenue / total_assets) if (revenue and total_assets and total_assets != 0) else None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=current_ratio,
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=de_ratio,
                debt_to_assets=(total_debt / total_assets) if (total_debt is not None and total_assets and total_assets != 0) else None,
                interest_coverage=interest_coverage,
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                earnings_per_share=eps,
                book_value_per_share=b.get("book_value_per_share"),
                free_cash_flow_per_share=None,
                payout_ratio=None,
            )
            metrics.append(metric)

        # Fill YoY growth rates
        for idx in range(len(metrics) - 1):
            curr = metrics[idx]
            prev = metrics[idx + 1]

            def _growth(c, p):
                if c is not None and p is not None and p != 0:
                    return (c - p) / abs(p)
                return None

            curr.revenue_growth = _growth(
                _get_revenue_from_income(income_periods[idx]),
                _get_revenue_from_income(income_periods[idx + 1])
            )
            curr.earnings_per_share_growth = _growth(curr.earnings_per_share, prev.earnings_per_share)

        # Fallback: statistics-only
        if not metrics and stats_data:
            stats = stats_data.get("statistics", {})
            val = stats.get("valuations_metrics", {})
            fin = stats.get("financials", {})
            inc_stmt = fin.get("income_statement", {})
            bs = fin.get("balance_sheet", {})

            metric = FinancialMetrics(
                ticker=ticker,
                report_period=end_date,
                period="ttm",
                currency=stats_data.get("meta", {}).get("currency", "USD"),
                market_cap=_to_float((val or {}).get("market_capitalization")),
                enterprise_value=_to_float((val or {}).get("enterprise_value")),
                price_to_earnings_ratio=_to_float((val or {}).get("trailing_pe")),
                price_to_book_ratio=_to_float((val or {}).get("price_to_book_mrq")),
                price_to_sales_ratio=_to_float((val or {}).get("price_to_sales_ttm")),
                enterprise_value_to_ebitda_ratio=_to_float((val or {}).get("enterprise_to_ebitda")),
                enterprise_value_to_revenue_ratio=_to_float((val or {}).get("enterprise_to_revenue")),
                peg_ratio=_to_float((val or {}).get("peg_ratio")),
                free_cash_flow_yield=None,
                gross_margin=_to_float((fin or {}).get("gross_margin")),
                operating_margin=_to_float((fin or {}).get("operating_margin")),
                net_margin=_to_float((fin or {}).get("profit_margin")),
                return_on_equity=_to_float((fin or {}).get("return_on_equity_ttm")),
                return_on_assets=_to_float((fin or {}).get("return_on_assets_ttm")),
                return_on_invested_capital=None,
                asset_turnover=None, inventory_turnover=None, receivables_turnover=None,
                days_sales_outstanding=None, operating_cycle=None, working_capital_turnover=None,
                current_ratio=None, quick_ratio=None, cash_ratio=None, operating_cash_flow_ratio=None,
                debt_to_equity=_to_float((bs or {}).get("total_debt_to_equity_mrq")),
                debt_to_assets=None, interest_coverage=None,
                revenue_growth=_to_float((inc_stmt or {}).get("quarterly_revenue_growth")),
                earnings_growth=_to_float((inc_stmt or {}).get("quarterly_earnings_growth_yoy")),
                book_value_growth=None, earnings_per_share_growth=None,
                free_cash_flow_growth=None, operating_income_growth=None, ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=_to_float((inc_stmt or {}).get("diluted_eps_ttm")),
                book_value_per_share=_to_float((bs or {}).get("book_value_per_share_mrq")),
                free_cash_flow_per_share=None,
            )
            metrics.append(metric)

        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    except Exception as e:
        print(f"Error fetching financial metrics for {ticker}: {str(e)}")
        return []


def _get_revenue_from_income(inc: dict) -> float | None:
    """Extract revenue from a raw income statement dict."""
    return (
        _to_float(inc.get("sales")) or
        _to_float(inc.get("revenue")) or
        _to_float(inc.get("total_revenue"))
    )


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
    Uses _flatten_* functions to correctly handle Twelve Data's nested structure.
    """
    cache_key = f"line_items_{ticker}_{period}_{end_date}_{limit}_{'_'.join(sorted(line_items))}"
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [LineItem(**item) for item in cached_data]

    try:
        income_data  = _fetch_income_statement(ticker)
        balance_data = _fetch_balance_sheet(ticker)
        cashflow_data = _fetch_cash_flow(ticker)

        income_periods   = (income_data   or {}).get("income_statement", [])
        balance_periods  = (balance_data  or {}).get("balance_sheet",    [])
        cashflow_periods = (cashflow_data or {}).get("cash_flow",        [])

        currency = (income_data or balance_data or cashflow_data or {}).get(
            "meta", {}
        ).get("currency", "USD")

        balance_by_date  = {}
        cashflow_by_date = {}

        for b in balance_periods:
            fd = b.get("fiscal_date")
            if fd:
                balance_by_date[fd] = b

        for c in cashflow_periods:
            fd = c.get("fiscal_date")
            if fd:
                cashflow_by_date[fd] = c

        result = []

        for inc in income_periods[:limit]:
            fiscal_date = inc.get("fiscal_date")
            if not fiscal_date:
                continue

            bal = balance_by_date.get(fiscal_date, {})
            cf  = cashflow_by_date.get(fiscal_date, {})

            # Flatten each statement using the new nested-aware functions
            i_flat = _flatten_income(inc)
            b_flat = _flatten_balance(bal)
            c_flat = _flatten_cashflow(cf)

            # Merge: income wins on conflicts (most reliable source for P&L fields)
            merged = {}
            merged.update(c_flat)
            merged.update(b_flat)
            merged.update(i_flat)
            merged["fiscal_date"] = fiscal_date

            item = LineItem(
                ticker=ticker,
                report_period=fiscal_date,
                period="annual",
                currency=currency,
                **{k: v for k, v in merged.items() if k != "fiscal_date"},
            )
            result.append(item)

        # Fallback: use balance sheet dates if no income statement
        if not result:
            for bal in balance_periods[:limit]:
                fiscal_date = bal.get("fiscal_date")
                if not fiscal_date:
                    continue
                cf = cashflow_by_date.get(fiscal_date, {})
                b_flat = _flatten_balance(bal)
                c_flat = _flatten_cashflow(cf)
                merged = {}
                merged.update(c_flat)
                merged.update(b_flat)
                merged["fiscal_date"] = fiscal_date

                item = LineItem(
                    ticker=ticker,
                    report_period=fiscal_date,
                    period="annual",
                    currency=currency,
                    **{k: v for k, v in merged.items() if k != "fiscal_date"},
                )
                result.append(item)

        _cache.set_financial_metrics(cache_key, [item.model_dump() for item in result])
        return result

    except Exception as e:
        print(f"Error fetching line items for {ticker}: {str(e)}")
        return []


def get_insider_trades(ticker, end_date, start_date=None, limit=1000, api_key=None):
    return []


def get_company_news(ticker, end_date, start_date=None, limit=1000, api_key=None):
    newsdata_key = os.environ.get("NEWSDATA_API_KEY")
    if not newsdata_key:
        return []
    clean_ticker = ticker.split(":")[0]
    resp = requests.get(
        "https://newsdata.io/api/1/latest",
        params={"apikey": newsdata_key, "q": clean_ticker, "language": "en", "size": min(limit, 10)},
        timeout=30,
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


def get_market_cap(ticker, end_date, api_key=None):
    try:
        stats_data = _fetch_statistics(ticker)
        if stats_data and "statistics" in stats_data:
            mc = stats_data["statistics"].get("valuations_metrics", {}).get("market_capitalization")
            if mc is not None:
                return _to_float(mc)
    except Exception as e:
        print(f"Error fetching market cap for {ticker}: {str(e)}")
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if financial_metrics and financial_metrics[0].market_cap:
        return financial_metrics[0].market_cap
    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
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


def prefetch_ticker_data(ticker: str, start_date: str = None, end_date: str = None) -> None:
    """Pre-fetch and cache all financial data for a ticker before parallel agents run."""
    for fetch_fn, label in [
        (_fetch_income_statement, "income"),
        (_fetch_balance_sheet,    "balance"),
        (_fetch_cash_flow,        "cashflow"),
        (_fetch_statistics,       "statistics"),
    ]:
        try:
            fetch_fn(ticker)
        except Exception as e:
            print(f"[prefetch] {label} failed for {ticker}: {e}")

    if start_date and end_date:
        try:
            get_prices(ticker, start_date, end_date)
        except Exception as e:
            print(f"[prefetch] prices failed for {ticker}: {e}")