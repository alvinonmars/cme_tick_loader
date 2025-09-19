"""Core tick data loader with cache support"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime
try:
    from .base_cache import BaseCache
except ImportError:
    from base_cache import BaseCache


class TickLoader(BaseCache):
    def __init__(self, base_path='/mnt/disk1/cme_futures', cache_dir=None, ticksizes=None):
        """
        Initialize tick loader

        Args:
            base_path: Base path for CME futures data
            cache_dir: Cache directory (defaults to base_path/.cache/ticks)
            ticksizes: Dictionary of symbol -> ticksize mappings
        """
        super().__init__(base_path, 'ticks', cache_dir)
        self.base_path = Path(base_path)

        # Default ticksizes for common instruments
        self.ticksizes = ticksizes or {
            'GC': 0.1,    # Gold
            'ES': 0.25,   # E-mini S&P 500
            'NQ': 0.25,   # E-mini NASDAQ
            'CL': 0.01,   # Crude Oil
            'ZN': 1/64,   # 10-Year Treasury Note
            'ZB': 1/32,   # 30-Year Treasury Bond
        }

    def load_ticks(self, symbol, date, use_cache=True, refresh_cache=False):
        """
        Load tick data with cache support

        Args:
            symbol: Trading symbol (e.g., 'GC')
            date: Date string in YYYYMMDD format
            use_cache: Whether to use cached pickle file
            refresh_cache: Force refresh cache

        Returns:
            DataFrame with tick data
        """
        # File paths
        csv_path = self.base_path / f"{symbol}_1" / f"{symbol}_1_footprint_{date}.csv"
        cache_key = self.get_cache_key(symbol, date)

        # Check cache first
        if use_cache and not refresh_cache:
            cached_data = self.load_data(cache_key)
            if cached_data is not None:
                # Verify cache is newer than source
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                if csv_path.exists():
                    cache_mtime = cache_path.stat().st_mtime
                    source_mtime = csv_path.stat().st_mtime
                    if cache_mtime > source_mtime:
                        return cached_data

        # Load from CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Add timestamp column with error detection and auto-repair
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        except ValueError as e:
            if 'timestamp_ms' in str(e):
                print(f"\n⚠️  CSV DATA ERROR DETECTED")
                print(f"📄 File: {csv_path}")
                print(f"🚨 Error: {e}")
                print(f"🔍 Likely cause: Duplicate headers in CSV file")

                if self._repair_duplicate_headers(csv_path):
                    print(f"🔄 Retrying data load after repair...")
                    # Clear any cached data for this file
                    self.clear_cache(symbol, date)
                    # Repair successful, reload
                    df = pd.read_csv(csv_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
                    print(f"✅ Successfully loaded repaired file: {len(df)} rows")
                else:
                    print(f"❌ Auto-repair failed, manual intervention required")
                    raise
            else:
                raise

        # Normalize prices to ticksize if available
        if symbol in self.ticksizes:
            ticksize = self.ticksizes[symbol]
            df['price'] = (df['price'] / ticksize).round() * ticksize

        # Save to cache
        if use_cache:
            self.save_data(df, cache_key)

        return df

    def get_ticksize(self, symbol):
        """Get ticksize for symbol"""
        return self.ticksizes.get(symbol, 0.01)  # Default to 0.01 if not found

    def get_cache_key(self, symbol, date):
        """Generate cache key for tick data"""
        return f"{symbol}_1_footprint_{date}"

    def _repair_duplicate_headers(self, csv_path):
        """
        Repair CSV files with duplicate headers

        Args:
            csv_path: Path to the CSV file to repair

        Returns:
            bool: True if repair was successful, False otherwise
        """
        print(f"🔧 Starting repair process for: {csv_path}")

        # 1. Create backup with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = csv_path.with_suffix(f'.backup_{timestamp}')
        try:
            shutil.copy2(csv_path, backup_path)
            print(f"📁 Backup created: {backup_path}")
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return False

        # 2. Detect header positions
        expected_header = "symbol,timeframe,timestamp_ms,price,bid_qty,ask_qty"
        header_positions = []

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip() == expected_header:
                        header_positions.append(line_num)
        except Exception as e:
            print(f"❌ Failed to read CSV file: {e}")
            return False

        print(f"📊 Found {len(header_positions)} headers at lines: {header_positions}")

        if len(header_positions) <= 1:
            print("✅ No duplicate headers found, no repair needed")
            return False

        # 3. Repair: keep data after last header
        last_header_line = max(header_positions)
        temp_path = csv_path.with_suffix('.tmp')

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as infile:
                with open(temp_path, 'w', encoding='utf-8', newline='') as outfile:
                    # Write single header
                    outfile.write("symbol,timeframe,timestamp_ms,price,bid_qty,ask_qty\n")

                    # Keep data after last header
                    lines_written = 0
                    for line_num, line in enumerate(infile, 1):
                        if line_num > last_header_line:
                            outfile.write(line)
                            lines_written += 1

            # Atomic replacement
            temp_path.replace(csv_path)
            print(f"✅ Repair completed: kept {lines_written} data lines from line {last_header_line + 1}")
            return True

        except Exception as e:
            print(f"❌ Repair failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

    def clear_cache(self, symbol=None, date=None):
        """
        Clear cache files

        Args:
            symbol: Clear cache for specific symbol (optional)
            date: Clear cache for specific date (optional)
        """
        if symbol and date:
            cache_key = self.get_cache_key(symbol, date)
            super().clear_cache(cache_key)
        elif symbol:
            pattern = f"{symbol}_1_footprint_*"
            super().clear_cache(pattern)
        else:
            super().clear_cache()