"""
Core detectors for Key Zone Strategy

Implements:
- KeyZoneDetector: Detects key price levels and zones
- StructuralSignalDetector: Detects V-reversals and breakouts
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from collections import defaultdict
from scipy.signal import find_peaks

from .models import (
    KeyPriceBook, KeyZone, Signal, ZoneType, SignalType
)
from .config import StrategyConfig


class KeyZoneDetector:
    """
    Detects key price zones from historical data.

    Combines two types of key prices:
    1. Big Delta Prices: Price levels with highest cumulative |delta|
    2. Peak Prices: Local peaks/troughs in price action
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def detect(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame],
        current_close: float
    ) -> Tuple[KeyPriceBook, List[KeyZone]]:
        """
        Detect key zones from historical data.

        Args:
            bars: Recent bars DataFrame with columns: open, high, low, close, volume
            footprints: List of footprint DataFrames (one per bar)
            current_close: Current bar's close price

        Returns:
            (KeyPriceBook, List[KeyZone])
        """
        # Calculate ATR
        atr = self._calculate_atr(bars)

        # Detect big delta prices
        big_delta_resistance, big_delta_support = self._detect_big_delta_prices(
            footprints, current_close
        )

        # Detect peak prices
        peak_resistance, peak_support = self._detect_peak_prices(
            bars, current_close, atr
        )

        # Organize into KeyPriceBook
        book = KeyPriceBook(
            big_delta_ask=big_delta_resistance,
            peak_ask=peak_resistance,
            big_delta_bid=big_delta_support,
            peak_bid=peak_support,
            current_close=current_close
        )

        # Convert to KeyZones
        zones = self._create_zones(book, atr)

        return book, zones

    def _detect_big_delta_prices(
        self,
        footprints: List[pd.DataFrame],
        current_close: float
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Detect key prices based on cumulative delta.

        Returns:
            (resistance_prices, support_prices) - each list has 5 elements (or None)
        """
        # Aggregate delta across all price levels
        price_delta_map = defaultdict(float)

        for footprint in footprints[-self.config.big_delta_lookback:]:
            if footprint is None or footprint.empty:
                continue

            for _, row in footprint.iterrows():
                price = row.name[1] if isinstance(row.name, tuple) else row.name
                delta = row.get('delta', 0)
                price_delta_map[price] += delta  # Sum aggregation

        if not price_delta_map:
            return [None] * 5, [None] * 5

        # Sort by |delta| and take top 10
        sorted_prices = sorted(
            price_delta_map.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

        # Separate into resistance and support
        resistance_candidates = [
            p for p, d in sorted_prices if p > current_close
        ]
        support_candidates = [
            p for p, d in sorted_prices if p < current_close
        ]

        # Sort and take closest 5
        resistance_prices = sorted(resistance_candidates)[:self.config.n_keep_prices]
        support_prices = sorted(support_candidates, reverse=True)[:self.config.n_keep_prices]

        # Pad to 5 elements
        resistance_prices += [None] * (self.config.n_keep_prices - len(resistance_prices))
        support_prices += [None] * (self.config.n_keep_prices - len(support_prices))

        return resistance_prices, support_prices

    def _detect_peak_prices(
        self,
        bars: pd.DataFrame,
        current_close: float,
        atr: float
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Detect key prices from local peaks/troughs.

        Returns:
            (resistance_prices, support_prices)
        """
        recent_bars = bars.tail(self.config.peak_lookback)

        if len(recent_bars) < 10:  # Need minimum data
            return [None] * 5, [None] * 5

        # Extract price series
        high_series = recent_bars['high'].values
        low_series = recent_bars['low'].values

        # Find peaks with prominence filter
        min_prominence = self.config.min_peak_prominence_atr * atr

        # Resistance peaks (local maxima in highs)
        resistance_indices, _ = find_peaks(
            high_series,
            prominence=min_prominence
        )

        # Support peaks (local minima in lows)
        support_indices, _ = find_peaks(
            -low_series,  # Invert to find valleys
            prominence=min_prominence
        )

        # Extract prices
        resistance_candidates = [
            high_series[i] for i in resistance_indices
            if high_series[i] > current_close
        ]

        support_candidates = [
            low_series[i] for i in support_indices
            if low_series[i] < current_close
        ]

        # Sort and take closest 5
        resistance_prices = sorted(resistance_candidates)[:self.config.n_keep_prices]
        support_prices = sorted(support_candidates, reverse=True)[:self.config.n_keep_prices]

        # Pad
        resistance_prices += [None] * (self.config.n_keep_prices - len(resistance_prices))
        support_prices += [None] * (self.config.n_keep_prices - len(support_prices))

        return resistance_prices, support_prices

    def _create_zones(
        self,
        book: KeyPriceBook,
        atr: float
    ) -> List[KeyZone]:
        """
        Convert KeyPriceBook to KeyZones with boundaries.

        Zone width = key_price ± min(0.5*ATR, zone_ticks*ticksize)
        """
        # Calculate zone half-width
        atr_width = 0.5 * atr
        tick_width = self.config.zone_ticks * self.config.ticksize
        half_width = min(atr_width, tick_width)

        # Group prices by value to combine sources
        price_sources_map = defaultdict(list)

        for price, zone_type, label in book.get_all_key_prices():
            price_sources_map[(price, zone_type)].append(label)

        # Create zones
        zones = []
        for (price, zone_type), sources in price_sources_map.items():
            zone = KeyZone(
                center_price=price,
                lower_bound=price - half_width,
                upper_bound=price + half_width,
                zone_type=zone_type,
                sources=sources
            )
            zones.append(zone)

        return zones

    def _calculate_atr(self, bars: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        recent_bars = bars.tail(self.config.atr_period + 1)

        if len(recent_bars) < 2:
            return 1.0  # Default

        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        high = recent_bars['high'].values[1:]
        low = recent_bars['low'].values[1:]
        prev_close = recent_bars['close'].values[:-1]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)

        return atr


class StructuralSignalDetector:
    """
    Detects structural signals: V-reversals and breakouts.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def detect_v_reversal(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame]
    ) -> Optional[Signal]:
        """
        Detect V-reversal pattern.

        Bullish V-reversal:
        1. Prior downtrend (3-5 bars with declining lows)
        2. Recent 2-3 bars form V-shape (low reached, then rebound)
        3. Rebound size > 0.3 * ATR
        4. Positive delta confirmation

        Returns:
            Signal or None
        """
        if len(bars) < self.config.v_reversal_lookback:
            return None

        recent_bars = bars.tail(self.config.v_reversal_lookback)
        atr = self._calculate_atr(bars)

        # Try bullish reversal
        bullish_signal = self._detect_bullish_v_reversal(
            recent_bars, footprints, atr
        )
        if bullish_signal:
            return bullish_signal

        # Try bearish reversal
        bearish_signal = self._detect_bearish_v_reversal(
            recent_bars, footprints, atr
        )
        if bearish_signal:
            return bearish_signal

        return None

    def _detect_bullish_v_reversal(
        self,
        recent_bars: pd.DataFrame,
        footprints: List[pd.DataFrame],
        atr: float
    ) -> Optional[Signal]:
        """Detect bullish V-reversal"""
        bars_list = recent_bars.to_dict('records')

        if len(bars_list) < 5:
            return None

        # Check prior downtrend (first 3 bars)
        prior_lows = [bars_list[i]['low'] for i in range(3)]
        is_downtrend = prior_lows[0] > prior_lows[2]  # Simplified check

        if not is_downtrend:
            return None

        # Check V-shape (last 2-3 bars)
        bar_before = bars_list[-2]
        bar_current = bars_list[-1]

        # Find V bottom
        v_low = min(bar_before['low'], bar_current['low'])

        # Check rebound
        is_rebound = bar_current['close'] > bar_before['close']
        rebound_size = bar_current['close'] - v_low

        if not is_rebound:
            return None

        # Check rebound size
        min_size = self.config.min_reversal_size_atr * atr
        if rebound_size < min_size:
            return None

        # Delta confirmation
        if footprints and len(footprints) > 0:
            current_footprint = footprints[-1]
            if current_footprint is not None and not current_footprint.empty:
                try:
                    total_delta = current_footprint['delta'].sum()
                    if total_delta <= 0:
                        return None
                except:
                    pass

        # Generate signal
        strength = min(1.0, rebound_size / atr)

        return Signal(
            signal_type=SignalType.V_REVERSAL_BULLISH,
            strength=strength,
            bar_index=len(bars_list) - 1,
            metadata={'rebound_size': rebound_size, 'v_low': v_low}
        )

    def _detect_bearish_v_reversal(
        self,
        recent_bars: pd.DataFrame,
        footprints: List[pd.DataFrame],
        atr: float
    ) -> Optional[Signal]:
        """Detect bearish V-reversal (inverted logic)"""
        bars_list = recent_bars.to_dict('records')

        if len(bars_list) < 5:
            return None

        # Check prior uptrend
        prior_highs = [bars_list[i]['high'] for i in range(3)]
        is_uptrend = prior_highs[0] < prior_highs[2]

        if not is_uptrend:
            return None

        # Check inverted V
        bar_before = bars_list[-2]
        bar_current = bars_list[-1]

        v_high = max(bar_before['high'], bar_current['high'])

        # Check decline
        is_decline = bar_current['close'] < bar_before['close']
        decline_size = v_high - bar_current['close']

        if not is_decline:
            return None

        # Check size
        min_size = self.config.min_reversal_size_atr * atr
        if decline_size < min_size:
            return None

        # Delta confirmation (negative)
        if footprints and len(footprints) > 0:
            current_footprint = footprints[-1]
            if current_footprint is not None and not current_footprint.empty:
                try:
                    total_delta = current_footprint['delta'].sum()
                    if total_delta >= 0:
                        return None
                except:
                    pass

        strength = min(1.0, decline_size / atr)

        return Signal(
            signal_type=SignalType.V_REVERSAL_BEARISH,
            strength=strength,
            bar_index=len(bars_list) - 1,
            metadata={'decline_size': decline_size, 'v_high': v_high}
        )

    def detect_breakout(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame]
    ) -> Optional[Signal]:
        """
        Detect breakout pattern.

        Bullish breakout:
        1. Close > max(high[-lookback:-1])
        2. Continuous 2 bars up: close[-1] > close[-2] > close[-3]
        3. Continuous 2 bars positive delta

        Returns:
            Signal or None
        """
        if len(bars) < self.config.breakout_lookback + 3:
            return None

        # Try bullish breakout
        bullish_signal = self._detect_bullish_breakout(bars, footprints)
        if bullish_signal:
            return bullish_signal

        # Try bearish breakout
        bearish_signal = self._detect_bearish_breakout(bars, footprints)
        if bearish_signal:
            return bearish_signal

        return None

    def _detect_bullish_breakout(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame]
    ) -> Optional[Signal]:
        """Detect bullish breakout"""
        lookback = self.config.breakout_lookback

        # Get recent bars
        bars_for_range = bars.iloc[-(lookback+3):-1]  # Exclude current
        current_bar = bars.iloc[-1]
        prev_bars = bars.iloc[-3:]

        # Check breakout
        max_high = bars_for_range['high'].max()
        current_close = current_bar['close']

        is_breakout = current_close > max_high

        if not is_breakout:
            return None

        # Check continuous 2 bars up
        closes = prev_bars['close'].values
        is_continuous_up = (closes[0] < closes[1] < closes[2])

        if not is_continuous_up:
            return None

        # Check continuous delta
        if footprints and len(footprints) >= 2:
            try:
                delta_1 = footprints[-2]['delta'].sum()
                delta_2 = footprints[-1]['delta'].sum()

                if delta_1 <= 0 or delta_2 <= 0:
                    return None
            except:
                pass

        # Calculate strength
        atr = self._calculate_atr(bars)
        breakout_size = current_close - max_high
        strength = min(1.0, breakout_size / atr)

        return Signal(
            signal_type=SignalType.BREAKOUT_BULLISH,
            strength=strength,
            bar_index=len(bars) - 1,
            metadata={'breakout_size': breakout_size, 'max_high': max_high}
        )

    def _detect_bearish_breakout(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame]
    ) -> Optional[Signal]:
        """Detect bearish breakout"""
        lookback = self.config.breakout_lookback

        bars_for_range = bars.iloc[-(lookback+3):-1]
        current_bar = bars.iloc[-1]
        prev_bars = bars.iloc[-3:]

        # Check breakdown
        min_low = bars_for_range['low'].min()
        current_close = current_bar['close']

        is_breakdown = current_close < min_low

        if not is_breakdown:
            return None

        # Check continuous down
        closes = prev_bars['close'].values
        is_continuous_down = (closes[0] > closes[1] > closes[2])

        if not is_continuous_down:
            return None

        # Check negative delta
        if footprints and len(footprints) >= 2:
            try:
                delta_1 = footprints[-2]['delta'].sum()
                delta_2 = footprints[-1]['delta'].sum()

                if delta_1 >= 0 or delta_2 >= 0:
                    return None
            except:
                pass

        atr = self._calculate_atr(bars)
        breakdown_size = min_low - current_close
        strength = min(1.0, breakdown_size / atr)

        return Signal(
            signal_type=SignalType.BREAKOUT_BEARISH,
            strength=strength,
            bar_index=len(bars) - 1,
            metadata={'breakdown_size': breakdown_size, 'min_low': min_low}
        )

    def _calculate_atr(self, bars: pd.DataFrame) -> float:
        """Calculate ATR (same as KeyZoneDetector)"""
        recent_bars = bars.tail(self.config.atr_period + 1)

        if len(recent_bars) < 2:
            return 1.0

        high = recent_bars['high'].values[1:]
        low = recent_bars['low'].values[1:]
        prev_close = recent_bars['close'].values[:-1]

        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)

        return atr
