#!/usr/bin/env python3
"""Quick test of new FootprintVisualizer implementation"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer

def test_new_visualizer():
    """Test the new simplified FootprintVisualizer"""
    print("Testing new FootprintVisualizer implementation...")

    # Initialize loader
    loader = CMEFootprintLoader()

    try:
        # Load a small amount of data
        print("Loading GC footprint data...")
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

        if footprint.empty:
            print("No data loaded. Data file may not exist.")
            return False

        print(f"Loaded {len(footprint)} price levels")

        # Get ticksize
        gc_ticksize = loader.tick_loader.get_ticksize('GC')
        print(f"GC ticksize: {gc_ticksize}")

        # Test new visualizer
        viz = FootprintVisualizer()

        # Test with limited data (first few bars only)
        timestamps = footprint.index.get_level_values(0).unique()
        if len(timestamps) > 3:
            # Use only first 3 bars to avoid timeout
            limited_footprint = footprint.loc[timestamps[:3]]
            print(f"Testing with first 3 bars: {len(limited_footprint)} price levels")
        else:
            limited_footprint = footprint
            print(f"Using all {len(timestamps)} bars")

        # Test local scaling
        print("Creating footprint chart with local scaling...")
        fig_local = viz.plot_footprint(
            limited_footprint,
            ticksize=gc_ticksize,
            scaling_mode='local'
        )
        print("✓ Local scaling chart created successfully")

        # Test global scaling
        print("Creating footprint chart with global scaling...")
        fig_global = viz.plot_footprint(
            limited_footprint,
            ticksize=gc_ticksize,
            scaling_mode='global'
        )
        print("✓ Global scaling chart created successfully")

        # Save test outputs
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        fig_local.write_html(output_dir / "test_footprint_local.html")
        fig_global.write_html(output_dir / "test_footprint_global.html")

        print(f"✓ Test charts saved to {output_dir}/")
        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_visualizer()
    sys.exit(0 if success else 1)