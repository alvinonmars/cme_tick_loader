"""CME Tick Loader - Footprint data loading and aggregation"""

from .tick_loader import TickLoader
from .cme_bars_loader import CMEBarsLoader
from .visualizer import FootprintVisualizer, FootprintConfig
from .chart_api import ChartAPI
from .footprint_bar_cache import FootprintBarDataCache
from .footprint_bar_data import FootprintBarData, get_detection_data, find_nearest_timestamp

__version__ = "3.0.0"
__all__ = [
    "TickLoader",
    "CMEBarsLoader",
    "FootprintVisualizer",
    "FootprintConfig",
    "ChartAPI",
    "FootprintBarDataCache",
    "FootprintBarData",
    "get_detection_data",
    "find_nearest_timestamp"
]