"""Basic usage example for CME Bars Loader"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEBarsLoader


def main():
    """Demonstrate basic usage of CME Bars Loader"""

    # Initialize loader
    print("Initializing CME Bars Loader...")
    loader = CMEBarsLoader()

    try:
        # Example 1: Load 5-minute TIME bars with footprint
        print("\n=== Example 1: 5-Minute TIME Bars ===")
        result = loader.load_bars(
            symbol='GC',
            date='20210104',
            resolution='MIN',
            num_units=5,
            enable_footprint=True,
            verbose=False
        )

        bars = result['bars']
        footprint = result['footprint']

        print(f"Loaded {len(bars)} bars")
        print(f"Footprint: {len(footprint)} price levels")

        # Show first few bars
        print("\nFirst 3 bars:")
        print(bars.head(3)[['date_time', 'open', 'high', 'low', 'close', 'volume']])

        # Show footprint for first bar
        if len(bars) > 0:
            first_bar_time = bars['date_time'].iloc[0]
            print(f"\nFootprint for bar at {first_bar_time}:")
            bar_footprint = footprint.loc[first_bar_time]
            print(bar_footprint.head(5))

            # Find POC (Point of Control)
            poc = bar_footprint['total_vol'].idxmax()
            print(f"POC: {poc}")

        # Example 2: Load VOLUME bars
        print("\n=== Example 2: VOLUME Bars (threshold=10000) ===")
        result_volume = loader.load_bars(
            symbol='GC',
            date='20210104',
            resolution='VOLUME',
            num_units=10000,
            enable_footprint=True,
            verbose=False
        )

        print(f"Loaded {len(result_volume['bars'])} volume bars")

        # Example 3: Load date range
        print("\n=== Example 3: Date Range (3 days) ===")
        result_range = loader.load_date_range(
            symbol='GC',
            start_date='20210104',
            end_date='20210106',
            resolution='MIN',
            num_units=5,
            enable_footprint=True,
            verbose=False
        )

        print(f"Loaded {len(result_range['bars'])} bars across multiple days")

        # Show cache info
        print("\n=== Cache Information ===")
        cache_info = loader.get_cache_info()
        print(f"Tick cache: {cache_info['tick_cache']['count']} files, "
              f"{cache_info['tick_cache']['size_mb']:.2f} MB")
        print(f"Result cache: {cache_info['result_cache']['count']} files, "
              f"{cache_info['result_cache']['size_mb']:.2f} MB")
        print(f"Total cache size: {cache_info['total_size_mb']:.2f} MB")

        # Example 4: Different resolutions
        print("\n=== Example 4: Different Resolutions ===")
        resolutions = [
            ('H', 1, '1-Hour'),
            ('MIN', 15, '15-Minute'),
            ('TICK', 1000, 'Tick bars (1000 ticks)'),
        ]

        for resolution, num_units, description in resolutions:
            result_test = loader.load_bars(
                symbol='GC',
                date='20210104',
                resolution=resolution,
                num_units=num_units,
                enable_footprint=False,  # Only bars, no footprint
                verbose=False
            )
            print(f"{description}: {len(result_test)} bars")

        print("\n✅ All examples completed successfully!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the data file exists:")
        print(f"Expected: {loader.base_path}/GC_1/GC_1_footprint_20210104.csv")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
