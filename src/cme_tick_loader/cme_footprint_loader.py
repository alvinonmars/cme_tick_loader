"""Main CME Footprint Loader API"""

import pandas as pd
from pathlib import Path
from .tick_loader import TickLoader
from .footprint_aggregator import FootprintAggregator
from .footprint_cache import FootprintCache


class CMEFootprintLoader:
    def __init__(self, base_path=None, ticksizes=None):
        """
        Initialize CME Footprint Loader

        Args:
            base_path: Base path for CME futures data (auto-detects if None)
            ticksizes: Dictionary of symbol -> ticksize mappings
        """
        if base_path is None:
            from .config import get_default_base_path
            base_path = get_default_base_path()
        self.base_path = Path(base_path)
        self.tick_loader = TickLoader(base_path, ticksizes=ticksizes)
        self.aggregator = FootprintAggregator()
        self.footprint_cache = FootprintCache(base_path)

    def load_footprint_bars(self, symbol, date, interval='5min',
                           use_cache=True, refresh_cache=False):
        """
        Load footprint bars with caching

        Args:
            symbol: Trading symbol (e.g., 'GC')
            date: Date string in YYYYMMDD format
            interval: Time interval ('1min', '5min', '15min', '30min', '1H')
            use_cache: Use cached footprint bars if available
            refresh_cache: Force regenerate bars

        Returns:
            MultiIndex DataFrame with footprint data and OHLC flags
        """
        # Check footprint cache
        if use_cache and not refresh_cache:
            cached = self.footprint_cache.load_bars(symbol, date, interval)
            if cached is not None:
                return cached

        # Load tick data
        ticks = self.tick_loader.load_ticks(symbol, date, use_cache=use_cache)

        # Get ticksize for symbol
        ticksize = self.tick_loader.get_ticksize(symbol)

        # Aggregate to footprint bars
        footprint = self.aggregator.aggregate_time_bars(ticks, interval, ticksize=ticksize)

        # Cache the result
        if use_cache and not footprint.empty:
            self.footprint_cache.save_bars(footprint, symbol, date, interval)

        return footprint

    def load_date_range(self, symbol, start_date, end_date, interval='5min'):
        """
        Load multiple days and concatenate

        Args:
            symbol: Trading symbol
            start_date: Start date string (YYYYMMDD)
            end_date: End date string (YYYYMMDD)
            interval: Time interval

        Returns:
            Combined MultiIndex DataFrame
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        all_bars = []

        for date in dates:
            date_str = date.strftime('%Y%m%d')
            try:
                bars = self.load_footprint_bars(symbol, date_str, interval)
                if not bars.empty:
                    all_bars.append(bars)
            except FileNotFoundError:
                continue  # Skip missing dates (weekends, holidays)

        return pd.concat(all_bars) if all_bars else pd.DataFrame()

    def get_ohlc(self, footprint, bar_timestamp):
        """Extract OHLC from footprint"""
        return self.aggregator.get_ohlc_from_footprint(footprint, bar_timestamp)

    def analyze_bar(self, footprint, bar_timestamp):
        """Analyze footprint bar"""
        return self.aggregator.analyze_footprint_bar(footprint, bar_timestamp)

    def calculate_value_area(self, footprint, bar_timestamp, percentage=0.70):
        """Calculate value area for bar"""
        return self.aggregator.calculate_value_area(footprint, bar_timestamp, percentage)

    def clear_all_cache(self):
        """Clear both tick and footprint caches"""
        self.tick_loader.clear_cache()
        self.footprint_cache.clear_cache()

    def get_cache_info(self):
        """Get cache information"""
        tick_info = self.tick_loader.get_cache_info()
        footprint_info = self.footprint_cache.get_cache_info()

        return {
            'tick_cache': tick_info,
            'footprint_cache': footprint_info,
            'total_size_mb': tick_info['total_size_mb'] + footprint_info['total_size_mb']
        }