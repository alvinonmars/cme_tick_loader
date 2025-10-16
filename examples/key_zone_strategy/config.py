"""
Configuration for Key Zone Strategy
"""

from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuration parameters for Key Zone Strategy"""

    # KeyZoneDetector parameters
    big_delta_lookback: int = 100      # Big delta lookback bar count
    peak_lookback: int = 100           # Peak detection lookback bar count
    n_keep_prices: int = 5             # Number of bid/ask prices to keep
    zone_ticks: int = 20               # Zone width in ticks
    min_peak_prominence_atr: float = 0.5  # Min peak prominence (multiple of ATR)

    # StructuralSignalDetector parameters
    v_reversal_lookback: int = 5       # V-reversal observation window
    breakout_lookback: int = 10        # Breakout lookback period
    min_reversal_size_atr: float = 0.3  # Min reversal size (multiple of ATR)

    # TouchTracker parameters
    touch_window: int = 3              # Touch event window (bars)

    # ATR parameters
    atr_period: int = 14               # ATR calculation period

    # Symbol-specific
    ticksize: float = 0.1              # Tick size (default for GC)


# Default configurations for common symbols
SYMBOL_CONFIGS = {
    'GC': StrategyConfig(ticksize=0.1),     # Gold
    'ES': StrategyConfig(ticksize=0.25),    # E-mini S&P
    'NQ': StrategyConfig(ticksize=0.25),    # E-mini NASDAQ
    'CL': StrategyConfig(ticksize=0.01),    # Crude Oil
    'ZN': StrategyConfig(ticksize=1/64),    # 10-Year Note
}


def get_config(symbol: str = 'GC') -> StrategyConfig:
    """Get configuration for symbol"""
    return SYMBOL_CONFIGS.get(symbol, StrategyConfig())
