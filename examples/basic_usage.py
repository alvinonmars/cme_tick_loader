"""Basic usage example for CME Footprint Loader"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer


def main():
    """Demonstrate basic usage of CME Footprint Loader"""

    # Initialize loader with ticksize support
    print("Initializing CME Footprint Loader...")
    loader = CMEFootprintLoader()

    try:
        # Load footprint bars for Gold futures
        print("Loading GC footprint data for 2021-01-04...")
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

        if footprint.empty:
            print("No data loaded. Please check if data file exists at:")
            print("/mnt/disk1/cme_futures/GC_1/GC_1_footprint_20210104.csv")
            return

        print(f"Loaded {len(footprint)} price levels across {len(footprint.index.get_level_values(0).unique())} bars")

        # Show ticksize info
        gc_ticksize = loader.tick_loader.get_ticksize('GC')
        print(f"GC ticksize: {gc_ticksize}")

        # Show first few bars
        print("\nFirst few footprint entries:")
        print(footprint.head(10))

        # Analyze first bar
        bar_timestamps = footprint.index.get_level_values(0).unique()
        if len(bar_timestamps) > 0:
            first_bar = bar_timestamps[0]
            print(f"\nAnalyzing first bar: {first_bar}")

            # Get OHLC
            ohlc = loader.get_ohlc(footprint, first_bar)
            if ohlc:
                print(f"OHLC: O={ohlc['open']:.1f} H={ohlc['high']:.1f} L={ohlc['low']:.1f} C={ohlc['close']:.1f}")

            # Analyze bar
            analysis = loader.analyze_bar(footprint, first_bar)
            if analysis:
                print(f"POC: {analysis['poc']:.1f}")
                print(f"Total Volume: {analysis['total_volume']}")
                print(f"Total Delta: {analysis['total_delta']}")
                print(f"Price Levels: {analysis['price_levels']}")

            # Calculate value area
            value_area = loader.calculate_value_area(footprint, first_bar)
            if value_area:
                print(f"Value Area: {value_area['val']:.1f} - {value_area['vah']:.1f}")
                print(f"POC: {value_area['poc']:.1f}")

        # Show cache info
        cache_info = loader.get_cache_info()
        print(f"\nCache Info:")
        print(f"Tick cache: {cache_info['tick_cache']['file_count']} files, {cache_info['tick_cache']['total_size_mb']:.2f} MB")
        print(f"Footprint cache: {cache_info['footprint_cache']['file_count']} files, {cache_info['footprint_cache']['total_size_mb']:.2f} MB")

        # Create visualizations (if plotly is available)
        try:
            print("\nCreating visualizations...")
            viz = FootprintVisualizer()

            # Candlestick chart
            fig_candle = viz.plot_candlestick(footprint, title="Gold Futures - 5min Bars")
            print("- Candlestick chart created")

            # Professional footprint chart - balanced for verification and performance
            bar_timestamps_list = list(bar_timestamps)
            if len(bar_timestamps_list) > 15:
                recent_start = bar_timestamps_list[-15]  # Last 15 bars for verification
                recent_footprint = footprint.loc[footprint.index.get_level_values(0) >= recent_start]
                print(f"Using last 15 bars for footprint chart: {len(recent_footprint)} price levels")
            else:
                recent_footprint = footprint
                print(f"Using all {len(bar_timestamps_list)} bars: {len(recent_footprint)} price levels")

            fig_footprint_local = viz.plot_footprint(
                recent_footprint, ticksize=gc_ticksize, scaling_mode='local'
            )
            print("- Professional footprint chart (local scaling) created")

            # Professional footprint chart - same recent data with global scaling
            fig_footprint_global = viz.plot_footprint(
                recent_footprint, ticksize=gc_ticksize, scaling_mode='global'
            )
            print("- Professional footprint chart (global scaling) created")

            # Time-filtered footprint chart (recent data only)
            bar_timestamps_list = list(bar_timestamps)
            if len(bar_timestamps_list) > 8:
                recent_start = bar_timestamps_list[-8]  # Last 8 bars
                fig_footprint_recent = viz.plot_footprint(
                    footprint, ticksize=gc_ticksize, scaling_mode='local',
                    start_time=recent_start
                )
                print("- Recent footprint chart (last 8 bars) created")
            else:
                fig_footprint_recent = None

            # Save charts (optional)
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            fig_candle.write_html(output_dir / "candlestick.html")
            fig_footprint_local.write_html(output_dir / "footprint_local.html")
            fig_footprint_global.write_html(output_dir / "footprint_global.html")

            if fig_footprint_recent:
                fig_footprint_recent.write_html(output_dir / "footprint_recent.html")

            print(f"Charts saved to {output_dir}/")

        except ImportError:
            print("Plotly not available, skipping visualizations")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file exists in the correct location.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()