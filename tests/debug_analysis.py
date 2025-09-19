#!/usr/bin/env python3
"""Deep analysis and debugging of CME Tick Loader implementation"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Import the package
from cme_tick_loader import CMEFootprintLoader, TickLoader, FootprintAggregator

class DeepAnalyzer:
    def __init__(self):
        self.test_dir = None

    def create_minimal_test_data(self):
        """Create minimal test data for detailed analysis"""
        print("Creating minimal test data for deep analysis...")

        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        test_path = Path(self.test_dir)
        (test_path / 'GC_1').mkdir(parents=True)

        # Create very simple test data - just 2 bars with clear OHLC
        data = [
            # Bar 1: 09:30:00 - 09:35:00
            ['GC', 1, 1609718400000, 1917.0, 2, 1],  # First tick (Open)
            ['GC', 1, 1609718410000, 1917.2, 1, 2],  # High
            ['GC', 1, 1609718420000, 1916.8, 3, 1],  # Low
            ['GC', 1, 1609718430000, 1917.1, 1, 1],  # Close

            # Bar 2: 09:35:00 - 09:40:00
            ['GC', 1, 1609718700000, 1917.3, 1, 2],  # First tick (Open)
            ['GC', 1, 1609718710000, 1917.5, 2, 1],  # High
            ['GC', 1, 1609718720000, 1917.2, 1, 3],  # Low
            ['GC', 1, 1609718730000, 1917.4, 2, 1],  # Close
        ]

        df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
        csv_file = test_path / 'GC_1' / 'GC_1_footprint_20210104.csv'
        df.to_csv(csv_file, index=False)

        print(f"✓ Created test data: {len(data)} ticks")
        print(f"✓ Test directory: {self.test_dir}")
        return test_path

    def analyze_raw_data_loading(self, test_path):
        """Step 1: Analyze raw data loading"""
        print("\n" + "="*60)
        print("STEP 1: ANALYZING RAW DATA LOADING")
        print("="*60)

        # Test basic TickLoader
        loader = TickLoader(str(test_path))

        print(f"GC Ticksize: {loader.get_ticksize('GC')}")

        # Load raw ticks
        ticks = loader.load_ticks('GC', '20210104')
        print(f"\nRaw ticks loaded: {len(ticks)}")
        print("\nRaw tick data:")
        print(ticks[['timestamp', 'price', 'bid_qty', 'ask_qty']])

        # Check price normalization
        print(f"\nPrice normalization check:")
        print(f"Unique prices: {sorted(ticks['price'].unique())}")

        ticksize = loader.get_ticksize('GC')
        for price in ticks['price'].unique():
            remainder = (price / ticksize) % 1
            is_aligned = abs(remainder) < 1e-10 or abs(remainder - 1) < 1e-10
            print(f"Price {price}: aligned to {ticksize} = {is_aligned}")

        return ticks

    def analyze_aggregation_step_by_step(self, ticks):
        """Step 2: Analyze aggregation step by step"""
        print("\n" + "="*60)
        print("STEP 2: ANALYZING FOOTPRINT AGGREGATION")
        print("="*60)

        # Manual step-by-step aggregation
        df = ticks.copy()
        interval = '5min'
        ticksize = 0.1

        print("Before aggregation:")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Step 1: Create bar timestamps
        df['bar_timestamp'] = df['timestamp'].dt.floor(interval)
        print(f"\nUnique bar timestamps: {df['bar_timestamp'].unique()}")

        # Step 2: Apply ticksize normalization
        if ticksize:
            original_prices = df['price'].copy()
            df['price'] = (df['price'] / ticksize).round() * ticksize
            print(f"\nPrice normalization:")
            for i, (orig, norm) in enumerate(zip(original_prices, df['price'])):
                if orig != norm:
                    print(f"  Tick {i}: {orig} -> {norm}")

        # Step 3: Calculate OHLC for each bar
        print(f"\nCalculating OHLC for each bar:")
        ohlc_per_bar = df.groupby('bar_timestamp')['price'].agg([
            ('open', 'first'),
            ('high', 'max'),
            ('low', 'min'),
            ('close', 'last')
        ])
        print("OHLC per bar:")
        print(ohlc_per_bar)

        # Step 4: Group by bar and price
        print(f"\nGrouping by bar_timestamp and price:")
        grouped = df.groupby(['bar_timestamp', 'price']).agg({
            'bid_qty': 'sum',
            'ask_qty': 'sum',
            'timestamp': ['first', 'last']
        })

        print(f"Grouped data shape: {grouped.shape}")
        print(f"Grouped index type: {type(grouped.index)}")
        print(f"Grouped index names: {grouped.index.names}")
        print(f"Grouped index levels: {grouped.index.nlevels}")

        # Step 5: Create final footprint structure
        footprint = grouped.copy()
        footprint.columns = ['bid_vol', 'ask_vol', 'first_time', 'last_time']

        # Add calculated fields
        footprint['total_vol'] = footprint['bid_vol'] + footprint['ask_vol']
        footprint['delta'] = footprint['bid_vol'] - footprint['ask_vol']

        # Initialize OHLC flags
        footprint['is_open'] = False
        footprint['is_high'] = False
        footprint['is_low'] = False
        footprint['is_close'] = False

        print(f"\nFootprint before OHLC flags:")
        print(f"Shape: {footprint.shape}")
        print(f"Index: {footprint.index}")
        print(f"Columns: {list(footprint.columns)}")

        # Step 6: Set OHLC flags
        print(f"\nSetting OHLC flags:")
        for bar_time, bar_info in ohlc_per_bar.iterrows():
            print(f"\nBar {bar_time}:")
            print(f"  OHLC: O={bar_info['open']}, H={bar_info['high']}, L={bar_info['low']}, C={bar_info['close']}")

            # Check if these prices exist in footprint
            bar_mask = footprint.index.get_level_values(0) == bar_time
            available_prices = footprint[bar_mask].index.get_level_values(1)
            print(f"  Available prices: {list(available_prices)}")

            # Set flags
            for flag_name, price in [('is_open', bar_info['open']),
                                   ('is_high', bar_info['high']),
                                   ('is_low', bar_info['low']),
                                   ('is_close', bar_info['close'])]:
                price_mask = footprint.index.get_level_values(1) == price
                combined_mask = bar_mask & price_mask
                matches = footprint[combined_mask]
                if len(matches) > 0:
                    footprint.loc[combined_mask, flag_name] = True
                    print(f"  Set {flag_name} flag for price {price}")
                else:
                    print(f"  WARNING: Price {price} not found for {flag_name}")

        # Remove temporary columns
        footprint = footprint.drop(['first_time', 'last_time'], axis=1)

        print(f"\nFinal footprint:")
        print(f"Shape: {footprint.shape}")
        print(f"Index type: {type(footprint.index)}")
        print(f"Index names: {footprint.index.names}")
        print(footprint)

        return footprint

    def compare_with_actual_implementation(self, test_path):
        """Step 3: Compare with actual implementation"""
        print("\n" + "="*60)
        print("STEP 3: COMPARING WITH ACTUAL IMPLEMENTATION")
        print("="*60)

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

        print(f"Actual implementation result:")
        print(f"Shape: {footprint.shape}")
        print(f"Index type: {type(footprint.index)}")
        print(f"Index names: {footprint.index.names}")
        print(f"Columns: {list(footprint.columns)}")
        print("\nActual footprint data:")
        print(footprint)

        # Check OHLC flags
        print(f"\nOHLC flags verification:")
        for bar_time in footprint.index.get_level_values(0).unique():
            bar_data = footprint.loc[bar_time]
            print(f"\nBar {bar_time}:")

            for flag in ['is_open', 'is_high', 'is_low', 'is_close']:
                flagged_prices = bar_data[bar_data[flag]].index.tolist()
                print(f"  {flag}: {flagged_prices}")

        return footprint

    def analyze_cache_behavior(self, test_path):
        """Step 4: Analyze cache behavior"""
        print("\n" + "="*60)
        print("STEP 4: ANALYZING CACHE BEHAVIOR")
        print("="*60)

        loader = CMEFootprintLoader(str(test_path))

        # Clear cache first
        loader.clear_all_cache()

        # First load
        print("First load (should create cache):")
        import time
        start = time.time()
        footprint1 = loader.load_footprint_bars('GC', '20210104', interval='5min')
        time1 = time.time() - start
        print(f"Time: {time1:.4f}s")

        # Check cache
        cache_info = loader.get_cache_info()
        print(f"Cache info after first load:")
        print(f"  Tick cache files: {cache_info['tick_cache']['file_count']}")
        print(f"  Footprint cache files: {cache_info['footprint_cache']['file_count']}")

        # Second load
        print("\nSecond load (should use cache):")
        start = time.time()
        footprint2 = loader.load_footprint_bars('GC', '20210104', interval='5min')
        time2 = time.time() - start
        print(f"Time: {time2:.4f}s")

        # Compare results
        try:
            pd.testing.assert_frame_equal(footprint1, footprint2)
            print("✓ Cache consistency verified - both loads identical")
        except Exception as e:
            print(f"✗ Cache inconsistency: {e}")

        print(f"Speed improvement: {time1/time2:.2f}x faster" if time2 > 0 else "No timing difference")

    def analyze_visualization_ticksize(self, test_path):
        """Step 5: Analyze visualization with ticksize"""
        print("\n" + "="*60)
        print("STEP 5: ANALYZING VISUALIZATION TICKSIZE")
        print("="*60)

        try:
            from cme_tick_loader import FootprintVisualizer

            loader = CMEFootprintLoader(str(test_path))
            footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
            viz = FootprintVisualizer()

            if not footprint.empty:
                bar_timestamp = footprint.index.get_level_values(0).unique()[0]
                ticksize = loader.tick_loader.get_ticksize('GC')

                print(f"Testing visualization with ticksize {ticksize}")

                # Get bar data
                bar_data = footprint.loc[bar_timestamp]
                prices = sorted(bar_data.index.tolist())

                print(f"Prices in bar: {prices}")

                # Test price formatting
                if ticksize and ticksize >= 1:
                    price_labels = [f"{p:.0f}" for p in prices]
                elif ticksize and ticksize >= 0.1:
                    price_labels = [f"{p:.1f}" for p in prices]
                elif ticksize and ticksize >= 0.01:
                    price_labels = [f"{p:.2f}" for p in prices]
                else:
                    price_labels = [f"{p:.3f}" for p in prices]

                print(f"Formatted price labels: {price_labels}")

                # Create visualization
                fig = viz.plot_footprint(footprint, bar_timestamp, ticksize=ticksize)
                print("✓ Visualization created successfully")

        except ImportError:
            print("⚠ Plotly not available for visualization testing")
        except Exception as e:
            print(f"✗ Visualization error: {e}")

    def cleanup(self):
        """Clean up test directory"""
        if self.test_dir:
            shutil.rmtree(self.test_dir)
            print(f"\n✓ Cleaned up test directory")

    def run_full_analysis(self):
        """Run complete deep analysis"""
        print("CME Tick Loader - Deep Analysis")
        print("="*50)

        try:
            test_path = self.create_minimal_test_data()

            # Step-by-step analysis
            ticks = self.analyze_raw_data_loading(test_path)
            manual_footprint = self.analyze_aggregation_step_by_step(ticks)
            actual_footprint = self.compare_with_actual_implementation(test_path)

            # Advanced analysis
            self.analyze_cache_behavior(test_path)
            self.analyze_visualization_ticksize(test_path)

            # Final comparison
            print("\n" + "="*60)
            print("FINAL ANALYSIS SUMMARY")
            print("="*60)

            print(f"Manual aggregation shape: {manual_footprint.shape}")
            print(f"Actual implementation shape: {actual_footprint.shape}")

            if manual_footprint.shape == actual_footprint.shape:
                print("✓ Shape consistency verified")
            else:
                print("✗ Shape mismatch detected")

            # Check index consistency
            print(f"Manual index names: {manual_footprint.index.names}")
            print(f"Actual index names: {actual_footprint.index.names}")

            # Deep data comparison
            try:
                # Compare without index names
                manual_reset = manual_footprint.reset_index()
                actual_reset = actual_footprint.reset_index()

                if manual_reset.shape == actual_reset.shape:
                    print("✓ Data structure consistency verified")
                else:
                    print("✗ Data structure inconsistency detected")

            except Exception as e:
                print(f"⚠ Could not compare data structures: {e}")

        except Exception as e:
            print(f"\n✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

if __name__ == "__main__":
    analyzer = DeepAnalyzer()
    analyzer.run_full_analysis()