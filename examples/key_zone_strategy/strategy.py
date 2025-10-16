"""
PyBroker integration for Key Zone Strategy
"""

import pandas as pd
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import StrategyConfig, get_config
from .detectors import KeyZoneDetector, StructuralSignalDetector
from .tracker import TouchTracker, SignalCombiner
from .models import TradeAction


class KeyZoneStrategy:
    """
    Main strategy class integrating all components.

    This class maintains state across bars and can be used
    with PyBroker's execution function.
    """

    def __init__(self, symbol: str = 'GC', config: Optional[StrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            symbol: Trading symbol (determines default config)
            config: Custom configuration (optional)
        """
        self.symbol = symbol
        self.config = config or get_config(symbol)

        # Initialize components
        self.zone_detector = KeyZoneDetector(self.config)
        self.structural_detector = StructuralSignalDetector(self.config)
        self.touch_tracker = TouchTracker(self.config)
        self.signal_combiner = SignalCombiner(self.config)

        # State
        self.current_bar_index = 0
        self.last_zones = []
        self.last_trade_signal = None

    def update(
        self,
        bars: pd.DataFrame,
        footprints: list,
        current_bar_index: int
    ) -> Optional[dict]:
        """
        Update strategy with new bar.

        Args:
            bars: Historical bars DataFrame
            footprints: List of footprint DataFrames
            current_bar_index: Current bar index

        Returns:
            Dict with 'action', 'zones', 'signals', etc. or None
        """
        self.current_bar_index = current_bar_index

        if len(bars) < self.config.big_delta_lookback:
            return None

        current_bar = bars.iloc[-1]
        current_close = current_bar['close']

        # Step 1: Detect key zones
        try:
            book, zones = self.zone_detector.detect(
                bars=bars,
                footprints=footprints,
                current_close=current_close
            )
            self.last_zones = zones
        except Exception as e:
            print(f"Zone detection error: {e}")
            return None

        # Step 2: Check zone touches
        try:
            touches = self.touch_tracker.check_touches(
                bar_index=current_bar_index,
                bar_high=current_bar['high'],
                bar_low=current_bar['low'],
                zones=zones
            )
            self.touch_tracker.update(touches)
        except Exception as e:
            print(f"Touch detection error: {e}")
            touches = []

        # Step 3: Detect structural signals
        structural_signals = []

        try:
            v_signal = self.structural_detector.detect_v_reversal(bars, footprints)
            if v_signal:
                structural_signals.append(v_signal)
        except Exception as e:
            print(f"V-reversal detection error: {e}")

        try:
            breakout_signal = self.structural_detector.detect_breakout(bars, footprints)
            if breakout_signal:
                structural_signals.append(breakout_signal)
        except Exception as e:
            print(f"Breakout detection error: {e}")

        # Step 4: Combine signals
        recent_touches = self.touch_tracker.get_recent_touches(current_bar_index)

        try:
            trade_signal = self.signal_combiner.combine(
                recent_touches=recent_touches,
                structural_signals=structural_signals
            )
            self.last_trade_signal = trade_signal
        except Exception as e:
            print(f"Signal combination error: {e}")
            trade_signal = None

        # Return result dict
        result = {
            'bar_index': current_bar_index,
            'zones': zones,
            'touches': touches,
            'recent_touches': recent_touches,
            'structural_signals': structural_signals,
            'trade_signal': trade_signal,
            'action': trade_signal.action if trade_signal else TradeAction.HOLD,
            'confidence': trade_signal.confidence if trade_signal else 0.0,
        }

        return result

    def get_last_trade_signal(self):
        """Get last generated trade signal"""
        return self.last_trade_signal

    def get_last_zones(self):
        """Get last detected zones"""
        return self.last_zones

    def reset(self):
        """Reset strategy state"""
        self.touch_tracker.clear()
        self.current_bar_index = 0
        self.last_zones = []
        self.last_trade_signal = None


# ============================================================================
# PyBroker Integration Example
# ============================================================================

def create_key_zone_exec_fn(symbol: str = 'GC', config: Optional[StrategyConfig] = None):
    """
    Factory function to create a PyBroker execution function.

    Usage:
        from key_zone_strategy import create_key_zone_exec_fn
        exec_fn = create_key_zone_exec_fn('GC')

        strategy = Strategy(data_source, '2021-01-04', '2021-01-06')
        strategy.add_execution(exec_fn, ['GC'])
        result = strategy.backtest()

    Args:
        symbol: Trading symbol
        config: Custom configuration

    Returns:
        Execution function compatible with PyBroker
    """
    # Create strategy instance (stateful)
    key_zone_strategy = KeyZoneStrategy(symbol=symbol, config=config)

    def exec_fn(ctx):
        """
        PyBroker execution function.

        Note: This requires access to historical bars and footprints.
        See example usage in tests for how to properly integrate.
        """
        # This is a placeholder - actual implementation depends on
        # how footprint data is passed to the execution context
        pass

    return exec_fn


def example_standalone_usage():
    """
    Example of using KeyZoneStrategy in standalone mode (without PyBroker).
    """
    from pybroker_integration import CMEDataSource

    # Initialize
    data_source = CMEDataSource(resolution='MIN', num_units=5)
    strategy = KeyZoneStrategy(symbol='GC')

    # Load data
    df = data_source.query(['GC'], '2021-01-04', '2021-01-04')

    # Convert to bars format
    bars = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Get footprints (simplified - need actual footprint data)
    footprints = []  # Would need to retrieve from data_source

    # Run strategy
    for i in range(len(bars)):
        window_bars = bars.iloc[:i+1]
        window_footprints = footprints[:i+1]

        result = strategy.update(
            bars=window_bars,
            footprints=window_footprints,
            current_bar_index=i
        )

        if result and result['trade_signal']:
            print(f"Bar {i}: {result['action']} - Confidence: {result['confidence']:.2f}")
            print(f"  Zone: {result['trade_signal'].zone.center_price}")
            print(f"  Signal: {result['trade_signal'].structural_signal.signal_type.value}")


if __name__ == '__main__':
    print("Key Zone Strategy - PyBroker Integration")
    print("See tests for usage examples")
