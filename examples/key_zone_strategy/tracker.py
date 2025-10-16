"""
Touch tracking and signal combination logic
"""

from typing import List, Optional
from collections import deque

from .models import (
    KeyZone, TouchEvent, Signal, TradeSignal,
    TradeAction, ZoneType
)
from .config import StrategyConfig


class TouchTracker:
    """
    Tracks zone touch events within a sliding window.

    Maintains a history of recent touch events to enable
    signal combination with structural signals.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.touch_history: deque = deque(maxlen=100)  # Keep last 100 touches

    def check_touches(
        self,
        bar_index: int,
        bar_high: float,
        bar_low: float,
        zones: List[KeyZone]
    ) -> List[TouchEvent]:
        """
        Check if current bar touches any zones.

        Args:
            bar_index: Current bar index
            bar_high: Bar's high price
            bar_low: Bar's low price
            zones: List of KeyZones to check

        Returns:
            List of TouchEvent
        """
        touches = []

        for zone in zones:
            # Check high touch
            if zone.contains(bar_high):
                touch = TouchEvent(
                    zone=zone,
                    bar_index=bar_index,
                    touch_type='high_touch',
                    touch_price=bar_high
                )
                touches.append(touch)

            # Check low touch (avoid duplicate if high==low)
            elif zone.contains(bar_low):
                touch = TouchEvent(
                    zone=zone,
                    bar_index=bar_index,
                    touch_type='low_touch',
                    touch_price=bar_low
                )
                touches.append(touch)

        return touches

    def update(self, touches: List[TouchEvent]):
        """Add new touches to history"""
        for touch in touches:
            self.touch_history.append(touch)

    def get_recent_touches(
        self,
        current_bar_index: int,
        window: Optional[int] = None
    ) -> List[TouchEvent]:
        """
        Get touches within recent window.

        Args:
            current_bar_index: Current bar index
            window: Look back window (bars). If None, use config.touch_window

        Returns:
            List of TouchEvent within window
        """
        if window is None:
            window = self.config.touch_window

        min_bar_index = current_bar_index - window

        recent = [
            touch for touch in self.touch_history
            if touch.bar_index >= min_bar_index
        ]

        return recent

    def clear(self):
        """Clear touch history"""
        self.touch_history.clear()


class SignalCombiner:
    """
    Combines zone touches with structural signals to generate trade signals.

    Rules:
    - Touch support + Bullish signal → BUY
    - Touch resistance + Bearish signal → SELL
    - Touch resistance + Bullish breakout → BUY (breakout)
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def combine(
        self,
        recent_touches: List[TouchEvent],
        structural_signals: List[Signal]
    ) -> Optional[TradeSignal]:
        """
        Combine touches and signals to generate trade signal.

        Args:
            recent_touches: Touches within recent window
            structural_signals: Structural signals detected

        Returns:
            TradeSignal or None
        """
        if not recent_touches or not structural_signals:
            return None

        # Try to match each touch with each signal
        for touch in recent_touches:
            for signal in structural_signals:
                trade_signal = self._try_match(touch, signal)
                if trade_signal:
                    return trade_signal

        return None

    def _try_match(
        self,
        touch: TouchEvent,
        signal: Signal
    ) -> Optional[TradeSignal]:
        """
        Try to match a touch with a signal.

        Returns:
            TradeSignal if match found, else None
        """
        zone_type = touch.zone.zone_type
        is_bullish = signal.is_bullish()
        is_bearish = signal.is_bearish()

        # Rule 1: Support + Bullish → BUY
        if zone_type == ZoneType.SUPPORT and is_bullish:
            combined_strength = signal.strength * (touch.zone.strength / 2.0)
            combined_strength = min(1.0, combined_strength)

            return TradeSignal(
                action=TradeAction.BUY,
                strength=combined_strength,
                zone=touch.zone,
                structural_signal=signal
            )

        # Rule 2: Resistance + Bearish → SELL
        elif zone_type == ZoneType.RESISTANCE and is_bearish:
            combined_strength = signal.strength * (touch.zone.strength / 2.0)
            combined_strength = min(1.0, combined_strength)

            return TradeSignal(
                action=TradeAction.SELL,
                strength=combined_strength,
                zone=touch.zone,
                structural_signal=signal
            )

        # Rule 3: Resistance + Bullish Breakout → BUY
        # (Simplified: accept all bullish signals at resistance as potential breakout)
        elif zone_type == ZoneType.RESISTANCE and is_bullish:
            # Reduce strength slightly for resistance breakout
            combined_strength = signal.strength * (touch.zone.strength / 2.0) * 0.8
            combined_strength = min(1.0, combined_strength)

            return TradeSignal(
                action=TradeAction.BUY,
                strength=combined_strength,
                zone=touch.zone,
                structural_signal=signal
            )

        # No match
        return None
