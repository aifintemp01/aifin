from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.backend.database.models import MarketDataRaw, MarketDataCandle


class MarketDataRepository:
    """Repository for market data CRUD operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # Raw Market Data Methods
    
    def create_raw_data(
        self,
        provider: str,
        endpoint: str,
        ticker: str,
        raw_response: Dict[str, Any],
        start_date: str = None,
        end_date: str = None,
        interval: str = None,
        status_code: int = None,
        error_message: str = None,
        request_id: str = None
    ) -> MarketDataRaw:
        """Create or update raw market data entry"""
        # Check if entry already exists
        existing = self.get_raw_data(
            provider=provider,
            endpoint=endpoint,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if existing:
            # Update existing entry
            existing.raw_response = raw_response
            existing.status_code = status_code
            existing.error_message = error_message
            existing.request_id = request_id
        else:
            # Create new entry
            existing = MarketDataRaw(
                provider=provider,
                endpoint=endpoint,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                raw_response=raw_response,
                status_code=status_code,
                error_message=error_message,
                request_id=request_id
            )
            self.db.add(existing)
        
        self.db.commit()
        self.db.refresh(existing)
        return existing
    
    def get_raw_data(
        self,
        provider: str,
        endpoint: str,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        interval: str = None
    ) -> Optional[MarketDataRaw]:
        """Get raw market data entry"""
        query = self.db.query(MarketDataRaw).filter(
            MarketDataRaw.provider == provider,
            MarketDataRaw.endpoint == endpoint,
            MarketDataRaw.ticker == ticker
        )
        
        if start_date:
            query = query.filter(MarketDataRaw.start_date == start_date)
        if end_date:
            query = query.filter(MarketDataRaw.end_date == end_date)
        if interval:
            query = query.filter(MarketDataRaw.interval == interval)
        
        return query.first()
    
    def get_raw_data_by_id(self, raw_data_id: int) -> Optional[MarketDataRaw]:
        """Get raw market data by ID"""
        return self.db.query(MarketDataRaw).filter(MarketDataRaw.id == raw_data_id).first()
    
    def get_raw_data_by_ticker(
        self,
        ticker: str,
        provider: str = None,
        endpoint: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MarketDataRaw]:
        """Get raw market data by ticker"""
        query = self.db.query(MarketDataRaw).filter(MarketDataRaw.ticker == ticker)
        
        if provider:
            query = query.filter(MarketDataRaw.provider == provider)
        if endpoint:
            query = query.filter(MarketDataRaw.endpoint == endpoint)
        
        return query.order_by(desc(MarketDataRaw.created_at)).limit(limit).offset(offset).all()
    
    # Candle Data Methods
    
    def create_candle(
        self,
        ticker: str,
        provider: str,
        time: str,
        interval: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int = None,
        adj_close: float = None,
        source_url: str = None,
        raw_data_id: int = None
    ) -> MarketDataCandle:
        """Create or update a candle entry"""
        # Check if candle already exists
        existing = self.get_candle(ticker, time, interval)
        
        if existing:
            # Update existing candle
            existing.provider = provider
            existing.open = open
            existing.high = high
            existing.low = low
            existing.close = close
            existing.volume = volume
            existing.adj_close = adj_close
            existing.source_url = source_url
            existing.raw_data_id = raw_data_id
        else:
            # Create new candle
            existing = MarketDataCandle(
                ticker=ticker,
                provider=provider,
                time=time,
                interval=interval,
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                adj_close=adj_close,
                source_url=source_url,
                raw_data_id=raw_data_id
            )
            self.db.add(existing)
        
        self.db.commit()
        self.db.refresh(existing)
        return existing
    
    def get_candle(self, ticker: str, time: str, interval: str) -> Optional[MarketDataCandle]:
        """Get a specific candle"""
        return self.db.query(MarketDataCandle).filter(
            MarketDataCandle.ticker == ticker,
            MarketDataCandle.time == time,
            MarketDataCandle.interval == interval
        ).first()
    
    def get_candles_by_ticker(
        self,
        ticker: str,
        interval: str = "1day",
        start_time: str = None,
        end_time: str = None,
        limit: int = 5000,
        offset: int = 0
    ) -> List[MarketDataCandle]:
        """Get candles for a specific ticker and interval"""
        query = self.db.query(MarketDataCandle).filter(
            MarketDataCandle.ticker == ticker,
            MarketDataCandle.interval == interval
        )
        
        if start_time:
            query = query.filter(MarketDataCandle.time >= start_time)
        if end_time:
            query = query.filter(MarketDataCandle.time <= end_time)
        
        return query.order_by(MarketDataCandle.time.asc()).limit(limit).offset(offset).all()
    
    def get_candles_by_provider(
        self,
        provider: str,
        ticker: str = None,
        interval: str = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[MarketDataCandle]:
        """Get candles from a specific provider"""
        query = self.db.query(MarketDataCandle).filter(MarketDataCandle.provider == provider)
        
        if ticker:
            query = query.filter(MarketDataCandle.ticker == ticker)
        if interval:
            query = query.filter(MarketDataCandle.interval == interval)
        
        return query.order_by(desc(MarketDataCandle.time)).limit(limit).offset(offset).all()
    
    def create_candles_batch(
        self,
        candles: List[Dict[str, Any]],
        provider: str,
        source_url: str = None,
        raw_data_id: int = None
    ) -> List[MarketDataCandle]:
        """Create or update multiple candles in batch"""
        created_candles = []
        for candle_data in candles:
            candle = self.create_candle(
                ticker=candle_data["ticker"],
                provider=provider,
                time=candle_data["time"],
                interval=candle_data["interval"],
                open=candle_data["open"],
                high=candle_data["high"],
                low=candle_data["low"],
                close=candle_data["close"],
                volume=candle_data.get("volume"),
                adj_close=candle_data.get("adj_close"),
                source_url=source_url,
                raw_data_id=raw_data_id
            )
            created_candles.append(candle)
        
        return created_candles
    
    # Utility Methods
    
    def get_tickers_with_data(self) -> List[str]:
        """Get all unique tickers with available market data"""
        return [
            row[0] for row in self.db.query(MarketDataCandle.ticker)
            .distinct()
            .order_by(MarketDataCandle.ticker.asc())
            .all()
        ]
    
    def get_providers_with_data(self) -> List[str]:
        """Get all unique data providers with available market data"""
        return [
            row[0] for row in self.db.query(MarketDataCandle.provider)
            .distinct()
            .order_by(MarketDataCandle.provider.asc())
            .all()
        ]
    
    def get_intervals_with_data(self) -> List[str]:
        """Get all unique intervals with available market data"""
        return [
            row[0] for row in self.db.query(MarketDataCandle.interval)
            .distinct()
            .order_by(MarketDataCandle.interval.asc())
            .all()
        ]
    
    def delete_raw_data_by_ticker(self, ticker: str) -> int:
        """Delete all raw market data for a specific ticker"""
        deleted_count = self.db.query(MarketDataRaw).filter(MarketDataRaw.ticker == ticker).delete()
        self.db.commit()
        return deleted_count
    
    def delete_candles_by_ticker(self, ticker: str) -> int:
        """Delete all candles for a specific ticker"""
        deleted_count = self.db.query(MarketDataCandle).filter(MarketDataCandle.ticker == ticker).delete()
        self.db.commit()
        return deleted_count
