from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from app.backend.database.connection import get_db
from app.backend.repositories.market_data_repository import MarketDataRepository
from app.backend.models.schemas import (
    MarketDataCandleResponse,
    MarketDataRawResponse,
    MarketDataCandleRequest,
    MarketDataRawRequest,
    MarketDataQueryParams
)

router = APIRouter()


def get_market_data_repo(db: Session = Depends(get_db)) -> MarketDataRepository:
    """Dependency for getting market data repository"""
    return MarketDataRepository(db)


# Market Data Raw API Routes
@router.post("/raw", response_model=MarketDataRawResponse)
def create_raw_market_data(
    request: MarketDataRawRequest,
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Create or update raw market data entry"""
    try:
        raw_data = repo.create_raw_data(
            provider=request.provider,
            endpoint=request.endpoint,
            ticker=request.ticker,
            raw_response=request.raw_response,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.interval,
            status_code=request.status_code,
            error_message=request.error_message,
            request_id=request.request_id
        )
        return raw_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create raw market data: {str(e)}")


@router.get("/raw/{raw_data_id}", response_model=MarketDataRawResponse)
def get_raw_market_data_by_id(
    raw_data_id: int,
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get raw market data by ID"""
    raw_data = repo.get_raw_data_by_id(raw_data_id)
    if not raw_data:
        raise HTTPException(status_code=404, detail="Raw market data not found")
    return raw_data


@router.get("/raw", response_model=List[MarketDataRawResponse])
def get_raw_market_data_by_ticker(
    ticker: str = Query(..., description="Ticker symbol to filter by"),
    provider: Optional[str] = Query(None, description="Data provider to filter by"),
    endpoint: Optional[str] = Query(None, description="API endpoint to filter by"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get raw market data by ticker symbol"""
    raw_data = repo.get_raw_data_by_ticker(
        ticker=ticker,
        provider=provider,
        endpoint=endpoint,
        limit=limit,
        offset=offset
    )
    return raw_data


# Market Data Candle API Routes
@router.post("/candles", response_model=MarketDataCandleResponse)
def create_candle(
    request: MarketDataCandleRequest,
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Create or update a market data candle"""
    try:
        candle = repo.create_candle(
            ticker=request.ticker,
            provider=request.provider,
            time=request.time,
            interval=request.interval,
            open=request.open,
            high=request.high,
            low=request.low,
            close=request.close,
            volume=request.volume,
            adj_close=request.adj_close,
            source_url=request.source_url,
            raw_data_id=request.raw_data_id
        )
        return candle
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create candle: {str(e)}")


@router.get("/candles", response_model=List[MarketDataCandleResponse])
def get_candles(
    ticker: str = Query(..., description="Ticker symbol to filter by"),
    interval: str = Query("1day", description="Interval to filter by (e.g., 1day, 1hour)"),
    start_date: Optional[str] = Query(None, description="Start date to filter by (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date to filter by (YYYY-MM-DD)"),
    limit: int = Query(5000, ge=1, le=10000, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get market data candles for a specific ticker and interval"""
    candles = repo.get_candles_by_ticker(
        ticker=ticker,
        interval=interval,
        start_time=start_date,
        end_time=end_date,
        limit=limit,
        offset=offset
    )
    return candles


@router.get("/candles/provider", response_model=List[MarketDataCandleResponse])
def get_candles_by_provider(
    provider: str = Query(..., description="Data provider to filter by"),
    ticker: Optional[str] = Query(None, description="Ticker symbol to filter by"),
    interval: Optional[str] = Query(None, description="Interval to filter by"),
    limit: int = Query(1000, ge=1, le=10000, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get market data candles by data provider"""
    candles = repo.get_candles_by_provider(
        provider=provider,
        ticker=ticker,
        interval=interval,
        limit=limit,
        offset=offset
    )
    return candles


# Utility API Routes
@router.get("/tickers", response_model=List[str])
def get_available_tickers(
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get all unique tickers with available market data"""
    return repo.get_tickers_with_data()


@router.get("/providers", response_model=List[str])
def get_available_providers(
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get all unique data providers with available market data"""
    return repo.get_providers_with_data()


@router.get("/intervals", response_model=List[str])
def get_available_intervals(
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Get all unique intervals with available market data"""
    return repo.get_intervals_with_data()


# Delete API Routes
@router.delete("/raw/{ticker}", response_model=dict)
def delete_raw_data_by_ticker(
    ticker: str,
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Delete all raw market data for a specific ticker"""
    deleted_count = repo.delete_raw_data_by_ticker(ticker)
    return {"message": f"Deleted {deleted_count} raw market data records for {ticker}"}


@router.delete("/candles/{ticker}", response_model=dict)
def delete_candles_by_ticker(
    ticker: str,
    repo: MarketDataRepository = Depends(get_market_data_repo)
):
    """Delete all candles for a specific ticker"""
    deleted_count = repo.delete_candles_by_ticker(ticker)
    return {"message": f"Deleted {deleted_count} candles for {ticker}"}