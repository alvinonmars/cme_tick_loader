"""
Data models for Key Zone Strategy

Defines core data structures for key prices, zones, signals, and events.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ZoneType(Enum):
    """Zone type enum"""
    SUPPORT = 'support'
    RESISTANCE = 'resistance'


class SignalType(Enum):
    """Signal type enum"""
    V_REVERSAL_BULLISH = 'v_reversal_bullish'
    V_REVERSAL_BEARISH = 'v_reversal_bearish'
    BREAKOUT_BULLISH = 'breakout_bullish'
    BREAKOUT_BEARISH = 'breakout_bearish'


class TradeAction(Enum):
    """Trade action enum"""
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


@dataclass
class KeyPriceBook:
    """
    Organized key prices similar to bid/ask book structure.

    Each list contains up to 5 prices, ordered by proximity to current_close:
    - ask0 = closest resistance above close
    - bid0 = closest support below close
    """
    # Resistance (above close)
    big_delta_ask: List[Optional[float]] = field(default_factory=lambda: [None] * 5)
    peak_ask: List[Optional[float]] = field(default_factory=lambda: [None] * 5)

    # Support (below close)
    big_delta_bid: List[Optional[float]] = field(default_factory=lambda: [None] * 5)
    peak_bid: List[Optional[float]] = field(default_factory=lambda: [None] * 5)

    current_close: float = 0.0

    def get_all_key_prices(self) -> List[Tuple[float, ZoneType, str]]:
        """
        Get all key prices with metadata.

        Returns:
            List of (price, zone_type, label) tuples
        """
        results = []

        # Ask side (resistance)
        for i, price in enumerate(self.big_delta_ask):
            if price is not None:
                results.append((price, ZoneType.RESISTANCE, f'big_delta_ask{i}'))

        for i, price in enumerate(self.peak_ask):
            if price is not None:
                results.append((price, ZoneType.RESISTANCE, f'peak_ask{i}'))

        # Bid side (support)
        for i, price in enumerate(self.big_delta_bid):
            if price is not None:
                results.append((price, ZoneType.SUPPORT, f'big_delta_bid{i}'))

        for i, price in enumerate(self.peak_bid):
            if price is not None:
                results.append((price, ZoneType.SUPPORT, f'peak_bid{i}'))

        return results


@dataclass
class KeyZone:
    """
    A key price zone with boundaries.

    Zone width = key_price ± min(0.5*ATR, zone_ticks*ticksize)
    """
    center_price: float
    lower_bound: float
    upper_bound: float
    zone_type: ZoneType
    sources: List[str]  # e.g., ['big_delta_bid0', 'peak_bid1']
    strength: int = 1  # Number of sources

    def __post_init__(self):
        self.strength = len(self.sources)

    def contains(self, price: float) -> bool:
        """Check if price is within zone"""
        return self.lower_bound <= price <= self.upper_bound


@dataclass
class Signal:
    """
    Structural signal (V-reversal or breakout).
    """
    signal_type: SignalType
    strength: float  # 0-1, normalized
    bar_index: int
    metadata: dict = field(default_factory=dict)  # Additional info

    def is_bullish(self) -> bool:
        """Check if signal is bullish"""
        return 'bullish' in self.signal_type.value

    def is_bearish(self) -> bool:
        """Check if signal is bearish"""
        return 'bearish' in self.signal_type.value


@dataclass
class TouchEvent:
    """
    Event when price touches a key zone.
    """
    zone: KeyZone
    bar_index: int
    touch_type: str  # 'high_touch' or 'low_touch'
    touch_price: float  # Actual price that touched the zone


@dataclass
class TradeSignal:
    """
    Final trade signal combining zone touch and structural signal.
    """
    action: TradeAction
    strength: float  # Combined strength
    zone: KeyZone
    structural_signal: Signal
    confidence: float = 0.0  # 0-1

    def __post_init__(self):
        # Calculate confidence based on strength and zone strength
        self.confidence = self.strength * (self.zone.strength / 2.0)  # Normalize
        self.confidence = min(1.0, self.confidence)
