# CME Tick Loader Development Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
   - [Development Installation](#development-installation)
   - [Data Path Configuration](#data-path-configuration)
3. [Quick Start](#quick-start)
4. [Core Data Structure](#core-data-structure)
5. [Implementation Details (API Reference)](#implementation-details-api-reference)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)
8. [Visualization API](#visualization-api)
9. [Ticksize Examples](#ticksize-examples)
10. [Dependencies](#dependencies)

---

## Overview

Lightweight footprint data loader with time-based aggregation and efficient caching using pandas MultiIndex.

### Key Features
- **Auto-detected Data Path**: Automatically detects data location based on OS, or use `CME_DATA_PATH` environment variable
- **Time-based Aggregation**: Aggregate tick data into footprint bars (1min, 5min, 15min, 30min, 1H)
- **Efficient Caching**: Two-level cache system (ticks + footprints) for fast repeated access
- **Ticksize Support**: Handles different price increments for various instruments (GC: 0.1, ES: 0.25, etc.)
- **OHLC Flags**: Built-in open/high/low/close markers for each price level

---

## Installation & Setup

### Development Installation

**Important**: Always switch to conda cs environment before running any Python code:

```bash
# Activate conda cs environment (Ubuntu/Linux)
source ~/anaconda3/bin/activate cs

# Activate conda cs environment (macOS)
source /opt/homebrew/Caskroom/miniconda/base/bin/activate cs

# Install in development mode (local changes take effect immediately)
pip install -e .

# Verify installation
python -c "from cme_tick_loader import CMEFootprintLoader; print('Installation successful')"
```

**Benefits of Development Mode:**
- Local code changes take effect immediately without reinstallation
- Other projects can import directly
- Supports both relative and absolute imports

### Data Path Configuration

#### Overview

The `base_path` is the root directory where CME futures data files are stored. All data files follow the structure:
```
base_path/
├── GC_1/
│   ├── GC_1_footprint_20210104.csv
│   ├── GC_1_footprint_20210105.csv
│   └── ...
├── ES_1/
│   ├── ES_1_footprint_20210104.csv
│   └── ...
└── .cache/
    ├── ticks/        # Cached tick data
    └── footprints/   # Cached aggregated footprints
```

#### Auto-detection

The library auto-detects the data path based on your operating system:
- **macOS**: `/Users/alvinma/Desktop/work/data/cme_futures` (development environment default)
- **Ubuntu/Linux**: `/mnt/disk1/cme_futures`
- **Windows**: Not supported (recommend using environment variable)

**Note**: The macOS default path is configured for the specific development environment.
Other users should set the `CME_DATA_PATH` environment variable to their data location.

#### Configuration Methods

You can override the default path using one of three methods (in priority order):

**Method 1: Environment Variable (Recommended)**
```bash
# Set environment variable before running Python
export CME_DATA_PATH=/your/custom/path

# Or set it in Python before importing
import os
os.environ['CME_DATA_PATH'] = '/your/custom/path'
from cme_tick_loader import CMEFootprintLoader
```

**Method 2: Pass as Parameter**
```python
# Explicitly specify path when initializing
loader = CMEFootprintLoader(base_path='/your/custom/path')
```

**Method 3: Auto-detection (Default)**
```python
# Uses OS-specific default path automatically
loader = CMEFootprintLoader()
```

#### Accessing base_path in Custom Code

```python
# Method 1: Get from loader instance (recommended)
loader = CMEFootprintLoader()
base_path = loader.base_path  # Returns pathlib.Path object

# Method 2: Call get_default_base_path() directly
from cme_tick_loader.config import get_default_base_path
base_path = get_default_base_path()  # Returns string

# Example: Use in custom functions
def list_available_symbols(loader):
    """List all available symbol directories"""
    return [d.name.replace('_1', '') for d in loader.base_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')]
```

---

## Quick Start

### Basic Example

```python
from cme_tick_loader import CMEFootprintLoader

# 1. Initialize loader (auto-detects data path)
loader = CMEFootprintLoader()

# 2. Load 5-minute footprint bars for Gold
footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

# 3. Explore the data
print(f"Loaded {len(footprint)} price levels")
print(f"Time range: {footprint.index.get_level_values(0).min()} to {footprint.index.get_level_values(0).max()}")

# 4. Access a specific bar
bar_time = footprint.index.get_level_values(0).unique()[0]
first_bar = footprint.loc[bar_time]
print(f"\nFirst bar at {bar_time}:")
print(first_bar)

# 5. Get OHLC
ohlc = loader.get_ohlc(footprint, bar_time)
print(f"OHLC: O={ohlc['open']}, H={ohlc['high']}, L={ohlc['low']}, C={ohlc['close']}")
```

### Loading Multiple Days

```python
# Load date range
footprint = loader.load_date_range('GC', '20210104', '20210108', interval='5min')

# Check what data was loaded
print(f"Total bars: {len(footprint.index.get_level_values(0).unique())}")
```

---

## Core Data Structure

### Raw Tick Data
```csv
symbol,timeframe,timestamp_ms,price,bid_qty,ask_qty
GC,1,1609718400003,1917.1,1,0
```

### Ticksize Configuration
Different instruments have different minimum price increments (ticksizes):
- **GC (Gold)**: 0.1
- **ES (E-mini S&P 500)**: 0.25
- **NQ (E-mini NASDAQ)**: 0.25
- **CL (Crude Oil)**: 0.01

### Footprint Bar Structure (MultiIndex DataFrame)
```python
# MultiIndex: (bar_timestamp, price_level)
# Columns: bid_vol, ask_vol, total_vol, delta, is_open, is_high, is_low, is_close

                            bid_vol  ask_vol  total_vol  delta  is_open  is_high  is_low  is_close
bar_timestamp     price
2021-01-04 09:30  1917.0        50       30         80     20    False    False    True     False
                  1917.1       100       80        180     20     True    False   False      True
                  1917.2        75      120        195    -45    False     True   False     False
2021-01-04 09:35  1917.1        60       45        105     15     True     True    True      True  # All flags true when only one price
                  1917.3        90      110        200    -20    False     True   False      True  # Can be both high and close
```

Key points:
- **is_open**: True if price matches the first tick price in the bar
- **is_high**: True if price matches the highest price in the bar
- **is_low**: True if price matches the lowest price in the bar
- **is_close**: True if price matches the last tick price in the bar
- A single price can have multiple flags as True (e.g., open=high=low=close in low volatility)

## Implementation Details (API Reference)

### 1. Core Loader with Cache
```python
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os

class TickLoader:
    def __init__(self, base_path=None, cache_dir=None, ticksizes=None):
        """
        Args:
            base_path: Base directory for CME futures data (auto-detects if None)
        """
        if base_path is None:
            from .config import get_default_base_path
            base_path = get_default_base_path()
        self.base_path = Path(base_path)
        # Cache directory under base path
        if cache_dir is None:
            self.cache_dir = self.base_path / '.cache' / 'ticks'
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

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
            symbol: e.g., 'GC'
            date: YYYYMMDD format string
            use_cache: Whether to use cached pickle file
            refresh_cache: Force refresh cache
        """
        # File paths
        csv_path = self.base_path / f"{symbol}_1" / f"{symbol}_1_footprint_{date}.csv"
        cache_path = self.cache_dir / f"{symbol}_1_footprint_{date}.pkl"

        # Check cache
        if use_cache and not refresh_cache and cache_path.exists():
            return pd.read_pickle(cache_path)

        # Load CSV
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')

        # Normalize prices to ticksize if available
        if symbol in self.ticksizes:
            ticksize = self.ticksizes[symbol]
            df['price'] = (df['price'] / ticksize).round() * ticksize

        # Save cache
        if use_cache:
            df.to_pickle(cache_path)

        return df

    def get_ticksize(self, symbol):
        """Get ticksize for symbol"""
        return self.ticksizes.get(symbol, 0.01)  # Default to 0.01 if not found
```

### 2. Footprint Aggregation with OHLC Flags
```python
class FootprintAggregator:
    @staticmethod
    def aggregate_time_bars(df, interval='5min', ticksize=None):
        """
        Aggregate ticks into footprint bars with OHLC flags

        Args:
            df: DataFrame with columns [timestamp, price, bid_qty, ask_qty]
            interval: Pandas frequency string ('1min', '5min', '15min', '30min', '1H')
            ticksize: Minimum price increment (e.g., 0.1 for GC, 0.25 for ES)

        Returns:
            MultiIndex DataFrame with footprint data and OHLC flags
        """
        # Create time bars
        df = df.copy()
        df['bar_timestamp'] = df['timestamp'].dt.floor(interval)

        # Normalize prices to ticksize if provided
        if ticksize:
            df['price'] = (df['price'] / ticksize).round() * ticksize

        # First, calculate OHLC for each bar
        ohlc_per_bar = df.groupby('bar_timestamp')['price'].agg([
            ('open', 'first'),
            ('high', 'max'),
            ('low', 'min'),
            ('close', 'last')
        ])

        # Aggregate by bar and price level
        footprint = df.groupby(['bar_timestamp', 'price']).agg({
            'bid_qty': 'sum',
            'ask_qty': 'sum'
        }).rename(columns={
            'bid_qty': 'bid_vol',
            'ask_qty': 'ask_vol'
        })

        # Add calculated fields
        footprint['total_vol'] = footprint['bid_vol'] + footprint['ask_vol']
        footprint['delta'] = footprint['bid_vol'] - footprint['ask_vol']

        # Add OHLC flags
        footprint['is_open'] = False
        footprint['is_high'] = False
        footprint['is_low'] = False
        footprint['is_close'] = False

        # Set flags for each bar
        for bar_timestamp in ohlc_per_bar.index:
            ohlc = ohlc_per_bar.loc[bar_timestamp]

            # Get all prices in this bar
            if bar_timestamp in footprint.index.get_level_values(0):
                # Set open flag
                if (bar_timestamp, ohlc['open']) in footprint.index:
                    footprint.loc[(bar_timestamp, ohlc['open']), 'is_open'] = True

                # Set high flag
                if (bar_timestamp, ohlc['high']) in footprint.index:
                    footprint.loc[(bar_timestamp, ohlc['high']), 'is_high'] = True

                # Set low flag
                if (bar_timestamp, ohlc['low']) in footprint.index:
                    footprint.loc[(bar_timestamp, ohlc['low']), 'is_low'] = True

                # Set close flag
                if (bar_timestamp, ohlc['close']) in footprint.index:
                    footprint.loc[(bar_timestamp, ohlc['close']), 'is_close'] = True

        return footprint
```

### 3. Cache Manager
```python
class FootprintCache:
    def __init__(self, base_path=None, cache_dir=None):
        """
        Args:
            base_path: Base directory for CME futures data (auto-detects if None)
        """
        if base_path is None:
            from .config import get_default_base_path
            base_path = get_default_base_path()
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
        """Save aggregated footprint bars"""
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        bars.to_pickle(cache_path)

    def load_bars(self, symbol, date, interval):
        """Load cached footprint bars"""
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        if cache_path.exists():
            return pd.read_pickle(cache_path)
        return None

    def exists(self, symbol, date, interval):
        """Check if cache exists"""
        cache_path = self.cache_dir / self.get_cache_key(symbol, date, interval)
        return cache_path.exists()
```

### 4. Main API
```python
class CMEFootprintLoader:
    def __init__(self, base_path=None, ticksizes=None):
        """
        Args:
            base_path: Base directory for CME futures data (auto-detects if None)
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
        if use_cache:
            self.footprint_cache.save_bars(footprint, symbol, date, interval)

        return footprint

    def load_date_range(self, symbol, start_date, end_date, interval='5min'):
        """
        Load multiple days and concatenate

        Returns:
            Combined MultiIndex DataFrame
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        all_bars = []

        for date in dates:
            date_str = date.strftime('%Y%m%d')
            try:
                bars = self.load_footprint_bars(symbol, date_str, interval)
                all_bars.append(bars)
            except FileNotFoundError:
                continue  # Skip missing dates (weekends, holidays)

        return pd.concat(all_bars) if all_bars else pd.DataFrame()
```

## Usage Examples

### Basic Operations
```python
# Initialize loader (auto-detects data path based on OS)
loader = CMEFootprintLoader()

# Or initialize with custom ticksizes
custom_ticksizes = {
    'GC': 0.1,    # Gold
    'ES': 0.25,   # E-mini S&P 500
    'CL': 0.01,   # Crude Oil
}
loader = CMEFootprintLoader(ticksizes=custom_ticksizes)

# Load 5-minute footprint bars for Gold
footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

# Access specific bar's footprint
bar_time = pd.Timestamp('2021-01-04 09:30:00')
bar_footprint = footprint.loc[bar_time]
print(bar_footprint)
#          bid_vol  ask_vol  total_vol  delta  is_open  is_high  is_low  is_close
# price
# 1917.0       50       30         80     20    False    False    True     False
# 1917.1      100       80        180     20     True    False   False      True
# 1917.2       75      120        195    -45    False     True   False     False

# Get ticksize for symbol
gc_ticksize = loader.tick_loader.get_ticksize('GC')  # Returns 0.1
print(f"GC ticksize: {gc_ticksize}")
```

### Ticksize-Aware Price Formatting
```python
def format_price_by_ticksize(price, ticksize):
    """Format price according to instrument ticksize"""
    if ticksize >= 1:
        return f"{price:.0f}"
    elif ticksize >= 0.1:
        return f"{price:.1f}"
    elif ticksize >= 0.01:
        return f"{price:.2f}"
    elif ticksize == 1/32:  # Treasury bonds
        return f"{int(price)}-{int((price % 1) * 32):02d}"
    elif ticksize == 1/64:  # Treasury notes
        return f"{int(price)}-{int((price % 1) * 64):02d}"
    else:
        return f"{price:.3f}"

# Example usage
gc_price = 1917.1
formatted = format_price_by_ticksize(gc_price, 0.1)  # "1917.1"
```

### Extract OHLC from Flags
```python
def get_ohlc_from_footprint(footprint, bar_timestamp):
    """Extract OHLC prices from footprint flags"""
    bar = footprint.loc[bar_timestamp]

    # Get prices with flags
    open_price = bar[bar['is_open']].index[0] if bar['is_open'].any() else None
    high_price = bar[bar['is_high']].index[0] if bar['is_high'].any() else None
    low_price = bar[bar['is_low']].index[0] if bar['is_low'].any() else None
    close_price = bar[bar['is_close']].index[0] if bar['is_close'].any() else None

    return {
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price
    }

# Example usage
ohlc = get_ohlc_from_footprint(footprint, pd.Timestamp('2021-01-04 09:30:00'))
print(f"OHLC: O={ohlc['open']}, H={ohlc['high']}, L={ohlc['low']}, C={ohlc['close']}")
```

### Analyze Footprint with OHLC Context
```python
def analyze_footprint_bar(footprint, bar_timestamp):
    """Comprehensive analysis of a footprint bar"""
    bar = footprint.loc[bar_timestamp]

    # Find POC (Point of Control)
    poc = bar['total_vol'].idxmax()

    # Get OHLC prices
    open_prices = bar[bar['is_open']].index.tolist()
    close_prices = bar[bar['is_close']].index.tolist()

    # Volume at open vs close
    open_volume = bar[bar['is_open']]['total_vol'].sum() if bar['is_open'].any() else 0
    close_volume = bar[bar['is_close']]['total_vol'].sum() if bar['is_close'].any() else 0

    # Calculate total delta
    total_delta = bar['delta'].sum()

    # Price range
    price_range = bar.index.max() - bar.index.min()

    return {
        'poc': poc,
        'open_prices': open_prices,
        'close_prices': close_prices,
        'open_volume': open_volume,
        'close_volume': close_volume,
        'total_delta': total_delta,
        'price_levels': len(bar),
        'price_range': price_range,
        'strong_close': close_volume > open_volume
    }

# Example usage
analysis = analyze_footprint_bar(footprint, pd.Timestamp('2021-01-04 09:30:00'))
print(f"POC: {analysis['poc']}, Delta: {analysis['total_delta']}")
```

### Volume Profile with OHLC Levels
```python
def create_volume_profile_with_levels(footprint, start_time, end_time):
    """Create volume profile marking OHLC levels"""
    # Filter to time range
    mask = (footprint.index.get_level_values(0) >= start_time) & \
           (footprint.index.get_level_values(0) <= end_time)
    period_data = footprint[mask]

    # Aggregate by price
    profile = period_data.groupby(level='price').agg({
        'bid_vol': 'sum',
        'ask_vol': 'sum',
        'total_vol': 'sum',
        'delta': 'sum',
        'is_open': 'any',
        'is_high': 'any',
        'is_low': 'any',
        'is_close': 'any'
    })

    # Find significant levels
    significant_levels = profile[
        profile['is_open'] | profile['is_high'] |
        profile['is_low'] | profile['is_close']
    ]

    return {
        'profile': profile,
        'significant_levels': significant_levels,
        'poc': profile['total_vol'].idxmax(),
        'total_volume': profile['total_vol'].sum()
    }
```

### Identify Special Bar Patterns
```python
def identify_bar_patterns(footprint, bar_timestamp):
    """Identify special patterns in footprint bars"""
    bar = footprint.loc[bar_timestamp]

    # Check for single price bar (doji-like)
    is_single_price = bar['is_open'].sum() == 1 and \
                     bar['is_open'].equals(bar['is_high']) and \
                     bar['is_open'].equals(bar['is_low']) and \
                     bar['is_open'].equals(bar['is_close'])

    # Check for inside bar (open and close at same price)
    open_close_same = False
    if bar['is_open'].any() and bar['is_close'].any():
        open_price = bar[bar['is_open']].index[0]
        close_price = bar[bar['is_close']].index[0]
        open_close_same = (open_price == close_price)

    # Check for trend bar (close at high or low)
    close_at_high = (bar['is_close'] & bar['is_high']).any()
    close_at_low = (bar['is_close'] & bar['is_low']).any()

    return {
        'is_single_price': is_single_price,
        'is_doji': open_close_same,
        'is_bullish_trend': close_at_high,
        'is_bearish_trend': close_at_low
    }
```

### Batch Processing
```python
def batch_analyze_footprints(loader, symbol, dates, interval='5min'):
    """Batch process and analyze multiple days"""
    results = []

    for date in dates:
        try:
            footprint = loader.load_footprint_bars(symbol, date, interval)

            # Analyze each bar
            for bar_timestamp in footprint.index.get_level_values(0).unique():
                analysis = analyze_footprint_bar(footprint, bar_timestamp)
                patterns = identify_bar_patterns(footprint, bar_timestamp)

                results.append({
                    'timestamp': bar_timestamp,
                    'date': date,
                    **analysis,
                    **patterns
                })
        except FileNotFoundError:
            continue

    return pd.DataFrame(results)
```

## Performance Considerations

### Memory Optimization
```python
# Use appropriate data types to reduce memory
def optimize_footprint_dtypes(footprint):
    """Optimize data types for memory efficiency"""
    footprint['bid_vol'] = footprint['bid_vol'].astype('int32')
    footprint['ask_vol'] = footprint['ask_vol'].astype('int32')
    footprint['total_vol'] = footprint['total_vol'].astype('int32')
    footprint['delta'] = footprint['delta'].astype('int32')
    # Boolean flags are already memory efficient
    return footprint
```

### Cache Management
```python
class CacheManager:
    @staticmethod
    def clear_old_cache(cache_dir, days=30):
        """Remove cache files older than specified days"""
        import time

        cache_path = Path(cache_dir)
        current_time = time.time()

        for cache_file in cache_path.glob('*.pkl'):
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > days * 86400:  # Convert days to seconds
                cache_file.unlink()

    @staticmethod
    def get_cache_size(cache_dir):
        """Get total cache size in MB"""
        cache_path = Path(cache_dir)
        total_size = sum(f.stat().st_size for f in cache_path.glob('*.pkl'))
        return total_size / (1024 * 1024)
```

## Visualization API

### 1. Candlestick Chart
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FootprintVisualizer:
    @staticmethod
    def plot_candlestick(footprint_data, title="CME Footprint Candlestick"):
        """
        Plot candlestick chart from footprint data

        Args:
            footprint_data: MultiIndex DataFrame with OHLC flags
            title: Chart title
        """
        # Extract OHLC for each bar
        ohlc_data = []
        for timestamp in footprint_data.index.get_level_values(0).unique():
            bar = footprint_data.loc[timestamp]

            # Extract prices from flags
            open_price = bar[bar['is_open']].index[0] if bar['is_open'].any() else bar.index[0]
            high_price = bar[bar['is_high']].index[0] if bar['is_high'].any() else bar.index.max()
            low_price = bar[bar['is_low']].index[0] if bar['is_low'].any() else bar.index.min()
            close_price = bar[bar['is_close']].index[0] if bar['is_close'].any() else bar.index[-1]

            volume = bar['total_vol'].sum()

            ohlc_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df_ohlc = pd.DataFrame(ohlc_data)

        # Create subplots with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Volume')
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_ohlc['timestamp'],
                open=df_ohlc['open'],
                high=df_ohlc['high'],
                low=df_ohlc['low'],
                close=df_ohlc['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add volume bars
        colors = ['red' if close < open else 'green'
                  for close, open in zip(df_ohlc['close'], df_ohlc['open'])]

        fig.add_trace(
            go.Bar(
                x=df_ohlc['timestamp'],
                y=df_ohlc['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=False
        )

        return fig
```

### 2. Footprint Chart
```python
    @staticmethod
    def plot_footprint(footprint_data, bar_timestamp, width=400, height=600, ticksize=None):
        """
        Plot detailed footprint chart for a single bar

        Args:
            footprint_data: MultiIndex DataFrame
            bar_timestamp: Timestamp of the bar to plot
            width: Chart width
            height: Chart height
            ticksize: Price increment for formatting (e.g., 0.1 for GC)
        """
        bar = footprint_data.loc[bar_timestamp]

        # Prepare data for heatmap
        prices = sorted(bar.index.tolist())

        # Format prices based on ticksize
        if ticksize and ticksize >= 1:
            price_strings = [f"{p:.0f}" for p in prices]
        elif ticksize and ticksize >= 0.1:
            price_strings = [f"{p:.1f}" for p in prices]
        elif ticksize and ticksize >= 0.01:
            price_strings = [f"{p:.2f}" for p in prices]
        else:
            # Handle fractions like 1/32, 1/64 for bonds
            if ticksize and (ticksize == 1/32 or ticksize == 1/64):
                denominator = int(1/ticksize)
                price_strings = [f"{int(p)}-{int((p % 1) * denominator):02d}" for p in prices]
            else:
                price_strings = [f"{p:.3f}" for p in prices]

        # Create bid/ask matrices
        bid_volumes = bar['bid_vol'].values
        ask_volumes = bar['ask_vol'].values

        # Create custom text for each cell
        text_matrix = []
        for i, price in enumerate(prices):
            row_data = bar.loc[price]
            text = f"Bid: {int(row_data['bid_vol'])}<br>Ask: {int(row_data['ask_vol'])}<br>Δ: {int(row_data['delta'])}"
            text_matrix.append(text)

        # Create figure
        fig = go.Figure()

        # Add bid volume (left side)
        fig.add_trace(go.Bar(
            y=prices,
            x=-bid_volumes,
            orientation='h',
            name='Bid',
            marker_color='green',
            text=bid_volumes,
            textposition='inside',
            hovertemplate='Price: %{y:.1f}<br>Bid Vol: %{text}<extra></extra>'
        ))

        # Add ask volume (right side)
        fig.add_trace(go.Bar(
            y=prices,
            x=ask_volumes,
            orientation='h',
            name='Ask',
            marker_color='red',
            text=ask_volumes,
            textposition='inside',
            hovertemplate='Price: %{y:.1f}<br>Ask Vol: %{text}<extra></extra>'
        ))

        # Mark OHLC levels
        for price in prices:
            if bar.loc[price, 'is_open']:
                fig.add_hline(y=price, line_dash="dot", line_color="blue",
                             annotation_text="Open", annotation_position="left")
            if bar.loc[price, 'is_close']:
                fig.add_hline(y=price, line_dash="dot", line_color="orange",
                             annotation_text="Close", annotation_position="left")
            if bar.loc[price, 'is_high']:
                fig.add_hline(y=price, line_dash="dash", line_color="green",
                             annotation_text="High", annotation_position="right")
            if bar.loc[price, 'is_low']:
                fig.add_hline(y=price, line_dash="dash", line_color="red",
                             annotation_text="Low", annotation_position="right")

        # Find and mark POC
        poc_price = bar['total_vol'].idxmax()
        fig.add_hline(y=poc_price, line_width=2, line_color="purple",
                     annotation_text=f"POC: {poc_price:.1f}", annotation_position="right")

        # Update layout
        fig.update_layout(
            title=f"Footprint Chart - {bar_timestamp}",
            xaxis_title="Volume",
            yaxis_title="Price",
            barmode='relative',
            height=height,
            width=width,
            showlegend=True,
            hovermode='y unified'
        )

        return fig
```

### 3. Advanced Footprint Heatmap
```python
    @staticmethod
    def plot_footprint_heatmap(footprint_data, num_bars=20, show_delta=True):
        """
        Plot multiple footprint bars as a heatmap

        Args:
            footprint_data: MultiIndex DataFrame
            num_bars: Number of bars to display
            show_delta: Show delta instead of total volume
        """
        timestamps = footprint_data.index.get_level_values(0).unique()[-num_bars:]

        # Get all unique prices across selected bars
        all_prices = set()
        for ts in timestamps:
            all_prices.update(footprint_data.loc[ts].index.tolist())
        all_prices = sorted(all_prices)

        # Create matrix for heatmap
        if show_delta:
            matrix = np.zeros((len(all_prices), len(timestamps)))
            for j, ts in enumerate(timestamps):
                bar = footprint_data.loc[ts]
                for i, price in enumerate(all_prices):
                    if price in bar.index:
                        matrix[i, j] = bar.loc[price, 'delta']
        else:
            matrix = np.zeros((len(all_prices), len(timestamps)))
            for j, ts in enumerate(timestamps):
                bar = footprint_data.loc[ts]
                for i, price in enumerate(all_prices):
                    if price in bar.index:
                        matrix[i, j] = bar.loc[price, 'total_vol']

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[ts.strftime('%H:%M') for ts in timestamps],
            y=[f"{p:.1f}" for p in all_prices],
            colorscale='RdBu' if show_delta else 'Viridis',
            zmid=0 if show_delta else None,
            text=matrix,
            texttemplate='%{text:.0f}',
            textfont={"size": 8},
            colorbar=dict(title='Delta' if show_delta else 'Volume')
        ))

        # Add OHLC lines
        for j, ts in enumerate(timestamps):
            bar = footprint_data.loc[ts]

            # Find OHLC indices
            open_idx = None
            close_idx = None
            high_idx = None
            low_idx = None

            for i, price in enumerate(all_prices):
                if price in bar.index:
                    if bar.loc[price, 'is_open']:
                        open_idx = i
                    if bar.loc[price, 'is_close']:
                        close_idx = i
                    if bar.loc[price, 'is_high']:
                        high_idx = i
                    if bar.loc[price, 'is_low']:
                        low_idx = i

            # Draw OHLC box
            if all([idx is not None for idx in [open_idx, close_idx, high_idx, low_idx]]):
                # Draw vertical line from low to high
                fig.add_shape(
                    type="line",
                    x0=j, x1=j,
                    y0=low_idx, y1=high_idx,
                    line=dict(color="white", width=1)
                )

        # Update layout
        fig.update_layout(
            title="Footprint Heatmap" + (" (Delta)" if show_delta else " (Volume)"),
            xaxis_title="Time",
            yaxis_title="Price",
            height=800,
            width=1200
        )

        return fig
```

### 4. Volume Profile
```python
    @staticmethod
    def plot_volume_profile(footprint_data, start_time=None, end_time=None):
        """
        Plot horizontal volume profile

        Args:
            footprint_data: MultiIndex DataFrame
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
        """
        # Filter data if time range specified
        if start_time or end_time:
            mask = pd.Series(True, index=footprint_data.index)
            if start_time:
                mask &= footprint_data.index.get_level_values(0) >= start_time
            if end_time:
                mask &= footprint_data.index.get_level_values(0) <= end_time
            data = footprint_data[mask]
        else:
            data = footprint_data

        # Aggregate by price
        profile = data.groupby(level='price').agg({
            'bid_vol': 'sum',
            'ask_vol': 'sum',
            'total_vol': 'sum',
            'delta': 'sum'
        })

        # Find POC and value area
        poc_price = profile['total_vol'].idxmax()
        total_volume = profile['total_vol'].sum()

        # Calculate value area (70% of volume)
        target_volume = total_volume * 0.7
        profile_sorted = profile.sort_values('total_vol', ascending=False)
        cumsum = 0
        value_area_prices = []

        for price, row in profile_sorted.iterrows():
            cumsum += row['total_vol']
            value_area_prices.append(price)
            if cumsum >= target_volume:
                break

        vah = max(value_area_prices)  # Value Area High
        val = min(value_area_prices)  # Value Area Low

        # Create figure
        fig = go.Figure()

        # Add volume profile bars
        fig.add_trace(go.Bar(
            x=profile['total_vol'],
            y=profile.index,
            orientation='h',
            name='Total Volume',
            marker_color='blue',
            opacity=0.7
        ))

        # Add bid/ask split
        fig.add_trace(go.Bar(
            x=profile['bid_vol'],
            y=profile.index,
            orientation='h',
            name='Bid Volume',
            marker_color='green',
            opacity=0.5
        ))

        # Mark POC
        fig.add_hline(y=poc_price, line_color="red", line_width=2,
                     annotation_text=f"POC: {poc_price:.1f}")

        # Mark value area
        fig.add_hrect(y0=val, y1=vah, fillcolor="yellow", opacity=0.1,
                     annotation_text="Value Area (70%)")

        # Update layout
        fig.update_layout(
            title="Volume Profile",
            xaxis_title="Volume",
            yaxis_title="Price",
            height=800,
            width=600,
            barmode='overlay',
            showlegend=True
        )

        return fig
```

### Complete Visualization Example
```python
# Load data with ticksize awareness
loader = CMEFootprintLoader()
footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

# Get ticksize for proper formatting
gc_ticksize = loader.tick_loader.get_ticksize('GC')  # 0.1

# Initialize visualizer
viz = FootprintVisualizer()

# 1. Plot candlestick chart with volume
fig_candle = viz.plot_candlestick(footprint, title="Gold Futures - 5min")
fig_candle.show()

# 2. Plot detailed footprint for specific bar with ticksize formatting
bar_time = pd.Timestamp('2021-01-04 09:30:00')
fig_footprint = viz.plot_footprint(footprint, bar_time, ticksize=gc_ticksize)
fig_footprint.show()

# 3. Plot footprint heatmap (last 20 bars)
fig_heatmap = viz.plot_footprint_heatmap(footprint, num_bars=20, show_delta=True)
fig_heatmap.show()

# 4. Plot volume profile for the day
fig_profile = viz.plot_volume_profile(footprint)
fig_profile.show()

# Save charts to HTML
fig_candle.write_html("candlestick.html")
fig_footprint.write_html("footprint_bar.html")
fig_heatmap.write_html("footprint_heatmap.html")
fig_profile.write_html("volume_profile.html")
```

### Interactive Dashboard
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(footprint_data, symbol="GC"):
    """Create interactive dashboard with multiple views"""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"{symbol} Price Chart",
            "Volume Profile",
            "Recent Footprint Heatmap",
            "Delta Analysis"
        ),
        specs=[[{"type": "candlestick"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )

    # Add components...
    # (Implementation details omitted for brevity)

    return fig
```

---

## Ticksize Examples

### Multiple Instruments
```python
# Load different instruments with their ticksizes
instruments = {
    'GC': 0.1,    # Gold
    'ES': 0.25,   # E-mini S&P 500
    'CL': 0.01,   # Crude Oil
}

for symbol, expected_ticksize in instruments.items():
    loader = CMEFootprintLoader()
    footprint = loader.load_footprint_bars(symbol, '20210104')

    # Get actual ticksize used
    ticksize = loader.tick_loader.get_ticksize(symbol)
    print(f"{symbol}: ticksize={ticksize}")

    # Use in visualization
    if not footprint.empty:
        first_bar = footprint.index.get_level_values(0).unique()[0]
        fig = viz.plot_footprint(footprint, first_bar, ticksize=ticksize)
```

## Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.0.0
```