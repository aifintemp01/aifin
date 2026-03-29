"""
src/data/cache.py

Simple key-value cache aligned with how api.py uses it.
api.py passes compound keys like "metrics_AAPL_ttm_2024-01-01_10"
so we store and retrieve by exact key — no merging logic needed.

Also includes _statement_cache for raw Twelve Data API responses
to prevent duplicate API calls within the same server session.
"""


class Cache:
    """In-memory cache for API responses."""

    def __init__(self):
        # All caches use compound string keys passed from api.py
        # e.g. "AAPL_2024-01-01_2024-12-31" for prices
        # e.g. "metrics_AAPL_ttm_2024-01-01_10" for financial metrics
        # e.g. "line_items_AAPL_ttm_2024-01-01_10_..." for line items
        self._prices_cache: dict[str, list[dict]] = {}
        self._financial_metrics_cache: dict[str, list[dict]] = {}
        self._company_news_cache: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(self, key: str) -> list[dict] | None:
        return self._prices_cache.get(key)

    def set_prices(self, key: str, data: list[dict]):
        self._prices_cache[key] = data

    # ------------------------------------------------------------------
    # Financial Metrics + Line Items
    # Both use the same underlying dict since api.py uses
    # get_financial_metrics / set_financial_metrics for both.
    # ------------------------------------------------------------------

    def get_financial_metrics(self, key: str) -> list[dict] | None:
        return self._financial_metrics_cache.get(key)

    def set_financial_metrics(self, key: str, data: list[dict]):
        self._financial_metrics_cache[key] = data

    # ------------------------------------------------------------------
    # Company News
    # ------------------------------------------------------------------

    def get_company_news(self, key: str) -> list[dict] | None:
        return self._company_news_cache.get(key)

    def set_company_news(self, key: str, data: list[dict]):
        self._company_news_cache[key] = data

    # ------------------------------------------------------------------
    # Cache stats (useful for debugging)
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "prices_entries": len(self._prices_cache),
            "metrics_entries": len(self._financial_metrics_cache),
            "news_entries": len(self._company_news_cache),
        }

    def clear(self):
        """Clear all caches — useful for testing."""
        self._prices_cache.clear()
        self._financial_metrics_cache.clear()
        self._company_news_cache.clear()


# Global cache instance — lives for the duration of the server process
_cache = Cache()


def get_cache() -> Cache:
    """Get the global cache instance."""
    return _cache


# ---------------------------------------------------------------------------
# Statement-level cache — prevents duplicate Twelve Data API calls
# within the same request. Keyed by "income_AAPL", "balance_AAPL" etc.
# Lives at module level so it persists across all function calls.
# ---------------------------------------------------------------------------
_statement_cache: dict[str, dict] = {}