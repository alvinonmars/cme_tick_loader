#!/usr/bin/env python3
"""Test rendering logic consistency with reference code"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer, FootprintConfig


def test_rendering_logic():
    """Test that rendering logic matches reference code patterns"""

    print("Testing rendering logic consistency...")

    # Initialize
    loader = CMEFootprintLoader()
    viz = FootprintVisualizer()

    # Load minimal test data
    footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

    if footprint.empty:
        print("No data loaded")
        return False

    # Get just 2 bars for testing
    timestamps = footprint.index.get_level_values(0).unique()[:2]
    test_data = footprint.loc[timestamps]

    ticksize = loader.tick_loader.get_ticksize('GC')

    print(f"Testing with {len(timestamps)} bars, {len(test_data)} price levels")

    # Test with custom config to verify rendering parameters
    config = FootprintConfig()

    # Verify column ratios (should be 2:12:10 like reference code)
    assert config.column_ratios == (2, 12, 10), f"Column ratios wrong: {config.column_ratios}"
    print("✓ Column ratios match reference code (2:12:10)")

    # Test local scaling
    fig_local = viz.plot_footprint(
        test_data,
        ticksize=ticksize,
        scaling_mode='local',
        config=config
    )

    # Check figure has shapes (candles, delta bars, volume bars)
    if fig_local.layout.shapes:
        print(f"✓ Local scaling created {len(fig_local.layout.shapes)} shapes")
    else:
        print("✗ No shapes created")
        return False

    # Test global scaling
    fig_global = viz.plot_footprint(
        test_data,
        ticksize=ticksize,
        scaling_mode='global',
        config=config
    )

    if fig_global.layout.shapes:
        print(f"✓ Global scaling created {len(fig_global.layout.shapes)} shapes")
    else:
        print("✗ No shapes created")
        return False

    # Verify OHLC extraction
    first_timestamp = timestamps[0]
    bar_data = footprint.loc[first_timestamp]

    # Find OHLC from flags
    open_price = bar_data[bar_data['is_open']].index[0] if bar_data['is_open'].any() else None
    high_price = bar_data[bar_data['is_high']].index[0] if bar_data['is_high'].any() else None
    low_price = bar_data[bar_data['is_low']].index[0] if bar_data['is_low'].any() else None
    close_price = bar_data[bar_data['is_close']].index[0] if bar_data['is_close'].any() else None

    print(f"✓ OHLC extracted: O={open_price}, H={high_price}, L={low_price}, C={close_price}")

    # Verify color logic
    is_bullish = close_price >= open_price if (close_price and open_price) else False
    expected_color = config.colors['body_bullish'] if is_bullish else config.colors['body_bearish']
    print(f"✓ Body color logic: {'Bullish' if is_bullish else 'Bearish'} -> {expected_color}")

    # Save test outputs
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    fig_local.write_html(output_dir / "test_render_local.html")
    fig_global.write_html(output_dir / "test_render_global.html")

    print("✓ Test outputs saved to output/")
    print("✓ All rendering logic tests passed!")

    return True


if __name__ == "__main__":
    success = test_rendering_logic()
    sys.exit(0 if success else 1)