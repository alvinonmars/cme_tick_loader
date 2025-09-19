"""Footprint aggregation with OHLC flags"""

import pandas as pd


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
        if df.empty:
            return pd.DataFrame()

        # Create time bars
        df = df.copy()
        df['bar_timestamp'] = df['timestamp'].dt.floor(interval)

        # Normalize prices to ticksize if provided
        if ticksize:
            df['price'] = (df['price'] / ticksize).round() * ticksize

        # Aggregate by bar and price level with OHLC info
        agg_funcs = {
            'bid_qty': 'sum',
            'ask_qty': 'sum',
            'timestamp': ['first', 'last']  # For OHLC detection
        }

        grouped = df.groupby(['bar_timestamp', 'price']).agg(agg_funcs)

        # Flatten column names
        footprint = grouped.copy()
        footprint.columns = ['bid_vol', 'ask_vol', 'first_time', 'last_time']

        # Add calculated fields
        footprint['total_vol'] = footprint['bid_vol'] + footprint['ask_vol']
        footprint['delta'] = footprint['bid_vol'] - footprint['ask_vol']

        # Calculate OHLC for each bar efficiently
        bar_ohlc = df.groupby('bar_timestamp').agg({
            'price': ['first', 'max', 'min', 'last'],
            'timestamp': ['first', 'last']
        })
        bar_ohlc.columns = ['open_price', 'high_price', 'low_price', 'close_price', 'bar_start', 'bar_end']

        # Add OHLC flags efficiently
        footprint['is_open'] = False
        footprint['is_high'] = False
        footprint['is_low'] = False
        footprint['is_close'] = False

        # Set flags using vectorized operations
        for bar_time, bar_info in bar_ohlc.iterrows():
            bar_mask = footprint.index.get_level_values(0) == bar_time
            price_mask_open = footprint.index.get_level_values(1) == bar_info['open_price']
            price_mask_high = footprint.index.get_level_values(1) == bar_info['high_price']
            price_mask_low = footprint.index.get_level_values(1) == bar_info['low_price']
            price_mask_close = footprint.index.get_level_values(1) == bar_info['close_price']

            footprint.loc[bar_mask & price_mask_open, 'is_open'] = True
            footprint.loc[bar_mask & price_mask_high, 'is_high'] = True
            footprint.loc[bar_mask & price_mask_low, 'is_low'] = True
            footprint.loc[bar_mask & price_mask_close, 'is_close'] = True

        # Remove temporary columns
        footprint = footprint.drop(['first_time', 'last_time'], axis=1)

        return footprint

    @staticmethod
    def get_ohlc_from_footprint(footprint, bar_timestamp):
        """
        Extract OHLC prices from footprint flags

        Args:
            footprint: MultiIndex DataFrame with OHLC flags
            bar_timestamp: Timestamp of the bar

        Returns:
            Dictionary with OHLC prices
        """
        if bar_timestamp not in footprint.index.get_level_values(0):
            return None

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

    @staticmethod
    def analyze_footprint_bar(footprint, bar_timestamp):
        """
        Comprehensive analysis of a footprint bar

        Args:
            footprint: MultiIndex DataFrame
            bar_timestamp: Timestamp of the bar

        Returns:
            Dictionary with analysis results
        """
        if bar_timestamp not in footprint.index.get_level_values(0):
            return None

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
            'strong_close': close_volume > open_volume,
            'total_volume': bar['total_vol'].sum()
        }

    @staticmethod
    def calculate_value_area(footprint, bar_timestamp, percentage=0.70):
        """
        Calculate value area for a footprint bar

        Args:
            footprint: MultiIndex DataFrame
            bar_timestamp: Timestamp of the bar
            percentage: Percentage of volume for value area (default 70%)

        Returns:
            Dictionary with value area information
        """
        if bar_timestamp not in footprint.index.get_level_values(0):
            return None

        bar = footprint.loc[bar_timestamp]

        # Find POC
        poc_price = bar['total_vol'].idxmax()
        total_volume = bar['total_vol'].sum()
        target_volume = total_volume * percentage

        # Sort prices by volume
        sorted_prices = bar.sort_values('total_vol', ascending=False)

        # Expand from POC until target volume reached
        accumulated_volume = 0
        value_area_prices = []

        for price, row in sorted_prices.iterrows():
            accumulated_volume += row['total_vol']
            value_area_prices.append(price)
            if accumulated_volume >= target_volume:
                break

        return {
            'poc': poc_price,
            'vah': max(value_area_prices),  # Value Area High
            'val': min(value_area_prices),  # Value Area Low
            'value_area_prices': sorted(value_area_prices),
            'value_area_volume': accumulated_volume,
            'percentage': accumulated_volume / total_volume if total_volume > 0 else 0
        }