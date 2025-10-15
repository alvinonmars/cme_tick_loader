"""Quick demonstration of CME Bars Loader"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEBarsLoader


def main():
    # Initialize
    loader = CMEBarsLoader()

    # Load 5-minute bars with footprint
    result = loader.load_bars(
        symbol='GC',
        date='20210104',
        resolution='MIN',
        num_units=5
    )

    # Access data
    bars = result['bars']
    footprint = result['footprint']

    # Print summary
    print(f"Loaded {len(bars)} bars")
    print(f"Footprint: {len(footprint)} price levels")
    print("\nFirst 3 bars:")
    print(bars.head(3))


if __name__ == "__main__":
    main()
