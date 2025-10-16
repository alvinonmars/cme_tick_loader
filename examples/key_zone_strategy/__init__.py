"""
Key Zone Strategy Package

A systematic trading strategy based on key price zones and structural signals.
"""

from .models import (
    KeyPriceBook, KeyZone, Signal, TouchEvent, TradeSignal,
    ZoneType, SignalType, TradeAction
)
from .config import StrategyConfig, get_config, SYMBOL_CONFIGS
from .detectors import KeyZoneDetector, StructuralSignalDetector
from .tracker import TouchTracker, SignalCombiner
from .strategy import KeyZoneStrategy

__version__ = "1.0.0"

__all__ = [
    # Models
    'KeyPriceBook',
    'KeyZone',
    'Signal',
    'TouchEvent',
    'TradeSignal',
    'ZoneType',
    'SignalType',
    'TradeAction',
    # Config
    'StrategyConfig',
    'get_config',
    'SYMBOL_CONFIGS',
    # Detectors
    'KeyZoneDetector',
    'StructuralSignalDetector',
    # Tracker
    'TouchTracker',
    'SignalCombiner',
    # Strategy
    'KeyZoneStrategy',
]
