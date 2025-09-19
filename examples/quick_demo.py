"""Quick demo of CME Footprint Loader with limited data"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer


def main():
    """Quick demonstration with limited data"""

    # Initialize loader
    print("Initializing CME Footprint Loader...")
    loader = CMEFootprintLoader()

    try:
        # Load footprint bars for Gold futures
        print("Loading GC footprint data for 2021-01-04...")
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

        if footprint.empty:
            print("No data loaded. Please check if data file exists.")
            return

        # Show basic info
        bar_timestamps = footprint.index.get_level_values(0).unique()
        print(f"Loaded {len(footprint)} price levels across {len(bar_timestamps)} bars")

        # Get ticksize
        gc_ticksize = loader.tick_loader.get_ticksize('GC')
        print(f"GC ticksize: {gc_ticksize}")

        # Use only first 5 bars for quick demo
        first_5_bars = bar_timestamps[:5]
        demo_footprint = footprint.loc[first_5_bars]
        print(f"Demo using first 5 bars: {len(demo_footprint)} price levels")

        # Show first few entries
        print("\nFirst few footprint entries:")
        print(demo_footprint.head(10))

        # Create visualizations
        print("\nCreating visualizations...")
        viz = FootprintVisualizer()

        # Candlestick chart (all data - this is fast)
        fig_candle = viz.plot_candlestick(footprint, title="Gold Futures - 5min Bars")
        print("- Candlestick chart created")

        # Professional footprint chart - demo data with local scaling
        fig_footprint_local = viz.plot_footprint(
            demo_footprint, ticksize=gc_ticksize, scaling_mode='local'
        )
        print("- Professional footprint chart (local scaling) created")

        # Professional footprint chart - demo data with global scaling
        fig_footprint_global = viz.plot_footprint(
            demo_footprint, ticksize=gc_ticksize, scaling_mode='global'
        )
        print("- Professional footprint chart (global scaling) created")

        # Save charts
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        fig_candle.write_html(output_dir / "demo_candlestick.html")
        fig_footprint_local.write_html(output_dir / "demo_footprint_local.html")
        fig_footprint_global.write_html(output_dir / "demo_footprint_global.html")

        print(f"Charts saved to {output_dir}/")
        print("✓ Quick demo completed successfully!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists in the correct location.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()