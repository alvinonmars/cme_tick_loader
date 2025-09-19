"""Footprint aggregation cache manager"""

import pandas as pd
from pathlib import Path


class FootprintCache:
    def __init__(self, base_path='/mnt/disk1/cme_futures', cache_dir=None):
        """
        Initialize footprint cache

        Args:
            base_path: Base path for CME futures data
            cache_dir: Cache directory (defaults to base_path/.cache/footprints)
        """
        self.base_path = Path(base_path)
        # Cache directory under base path
        if cache_dir is None:
            self.cache_dir = self.base_path / '.cache' / 'footprints'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get_cache_key(self, symbol, date, interval):
        """Generate cache filename"""
        return f"{symbol}_{date}_{interval}.pkl"

    def save_bars(self, bars, symbol, date, interval):
        """
        Save aggregated footprint bars

        Args:
            bars: DataFrame with footprint data
            symbol: Trading symbol
            date: Date string
            interval: Time interval
        """
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        try:
            bars.to_pickle(cache_path)
        except Exception as e:
            print(f"Warning: Could not save footprint cache: {e}")

    def load_bars(self, symbol, date, interval):
        """
        Load cached footprint bars

        Args:
            symbol: Trading symbol
            date: Date string
            interval: Time interval

        Returns:
            DataFrame with footprint data or None if not cached
        """
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        if cache_path.exists():
            try:
                return pd.read_pickle(cache_path)
            except Exception:
                # If cache read fails, return None
                return None
        return None

    def exists(self, symbol, date, interval):
        """Check if cache exists"""
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        return cache_path.exists()

    def clear_cache(self, symbol=None, date=None, interval=None):
        """
        Clear cache files

        Args:
            symbol: Clear cache for specific symbol (optional)
            date: Clear cache for specific date (optional)
            interval: Clear cache for specific interval (optional)
        """
        if symbol and date and interval:
            # Clear specific file
            cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
            if cache_path.exists():
                cache_path.unlink()
        elif symbol:
            # Clear all files for symbol
            pattern = f"{symbol}_*.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

    def get_cache_info(self):
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            'cache_dir': str(self.cache_dir),
            'file_count': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }