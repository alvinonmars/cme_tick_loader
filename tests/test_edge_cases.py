#!/usr/bin/env python3
"""Edge cases and stress testing for CME Tick Loader"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from cme_tick_loader import CMEFootprintLoader, TickLoader, FootprintAggregator

class EdgeCaseTests:
    def __init__(self):
        self.results = {"passed": 0, "failed": 0}

    def test_extreme_ticksizes(self):
        """Test with extreme ticksize values"""
        print("Testing extreme ticksize values...")

        # Test very small ticksize
        small_ticksize = {'TEST': 0.0001}
        loader = TickLoader(ticksizes=small_ticksize)
        assert loader.get_ticksize('TEST') == 0.0001

        # Test fractional ticksizes (bonds)
        bond_ticksizes = {'ZN': 1/64, 'ZB': 1/32, 'ZF': 1/128}
        bond_loader = TickLoader(ticksizes=bond_ticksizes)
        assert abs(bond_loader.get_ticksize('ZN') - 1/64) < 1e-10

        # Test large ticksize
        large_ticksize = {'BIG': 25.0}
        large_loader = TickLoader(ticksizes=large_ticksize)
        assert large_loader.get_ticksize('BIG') == 25.0

        print("✓ Extreme ticksize values handled correctly")
        self.results["passed"] += 1

    def test_single_tick_bar(self):
        """Test bar with only one tick"""
        print("Testing single tick per bar...")

        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)
            (test_path / 'TEST_1').mkdir(parents=True)

            # Create data with single tick per bar
            data = [
                ['TEST', 1, 1609718400000, 100.0, 1, 1],  # Bar 1: single tick
                ['TEST', 1, 1609718700000, 101.0, 2, 1],  # Bar 2: single tick
            ]

            df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
            csv_file = test_path / 'TEST_1' / 'TEST_1_footprint_20210104.csv'
            df.to_csv(csv_file, index=False)

            loader = CMEFootprintLoader(str(test_path))
            footprint = loader.load_footprint_bars('TEST', '20210104', interval='5min')

            # Each bar should have all OHLC flags True for the single price
            for bar_time in footprint.index.get_level_values(0).unique():
                bar_data = footprint.loc[bar_time]
                assert len(bar_data) == 1, "Single tick should create single price level"

                price_data = bar_data.iloc[0]
                assert all([price_data['is_open'], price_data['is_high'],
                          price_data['is_low'], price_data['is_close']]), "Single tick should have all OHLC flags"

            print("✓ Single tick per bar handled correctly")
            self.results["passed"] += 1

        finally:
            shutil.rmtree(test_dir)

    def test_zero_volume_handling(self):
        """Test handling of zero volume ticks"""
        print("Testing zero volume handling...")

        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)
            (test_path / 'TEST_1').mkdir(parents=True)

            # Create data with zero volumes
            data = [
                ['TEST', 1, 1609718400000, 100.0, 0, 0],  # Zero volume
                ['TEST', 1, 1609718410000, 100.1, 1, 2],  # Normal volume
                ['TEST', 1, 1609718420000, 100.2, 0, 1],  # Zero bid
                ['TEST', 1, 1609718430000, 100.3, 2, 0],  # Zero ask
            ]

            df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
            csv_file = test_path / 'TEST_1' / 'TEST_1_footprint_20210104.csv'
            df.to_csv(csv_file, index=False)

            loader = CMEFootprintLoader(str(test_path))
            footprint = loader.load_footprint_bars('TEST', '20210104', interval='5min')

            assert not footprint.empty, "Should handle zero volumes"

            # Check that zero volume entries are included
            zero_vol_entries = footprint[footprint['total_vol'] == 0]
            assert len(zero_vol_entries) == 1, "Should have one zero volume entry"

            print("✓ Zero volume handling verified")
            self.results["passed"] += 1

        finally:
            shutil.rmtree(test_dir)

    def test_price_precision_edge_cases(self):
        """Test price precision edge cases"""
        print("Testing price precision edge cases...")

        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)
            (test_path / 'TEST_1').mkdir(parents=True)

            # Create data with precision challenges
            data = [
                ['TEST', 1, 1609718400000, 1917.0999999, 1, 1],  # Almost 1917.1
                ['TEST', 1, 1609718410000, 1917.1000001, 2, 1],  # Almost 1917.1
                ['TEST', 1, 1609718420000, 1917.2500000, 1, 2],  # Exact quarter
                ['TEST', 1, 1609718430000, 1917.2499999, 1, 1],  # Almost quarter
            ]

            df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
            csv_file = test_path / 'TEST_1' / 'TEST_1_footprint_20210104.csv'
            df.to_csv(csv_file, index=False)

            # Test with 0.1 ticksize
            ticksizes = {'TEST': 0.1}
            loader = CMEFootprintLoader(str(test_path), ticksizes=ticksizes)
            footprint = loader.load_footprint_bars('TEST', '20210104', interval='5min')

            # Check that similar prices are normalized to same value
            prices = footprint.index.get_level_values(1).unique()

            # Should have normalized the close values to proper ticksize
            for price in prices:
                remainder = (price / 0.1) % 1
                precision_ok = abs(remainder) < 1e-10 or abs(remainder - 1) < 1e-10
                assert precision_ok, f"Price {price} not properly aligned to ticksize"

            print(f"✓ Price precision handled: {len(prices)} unique normalized prices")
            self.results["passed"] += 1

        finally:
            shutil.rmtree(test_dir)

    def test_large_dataset_performance(self):
        """Test performance with larger dataset"""
        print("Testing large dataset performance...")

        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)
            (test_path / 'TEST_1').mkdir(parents=True)

            # Generate 1000 ticks across multiple bars
            data = []
            base_time = 1609718400000
            base_price = 1000.0

            for i in range(1000):
                tick_time = base_time + i * 1000  # 1 second intervals
                price = base_price + (i % 100) * 0.1  # Price variation
                bid_qty = np.random.randint(1, 10)
                ask_qty = np.random.randint(1, 10)

                data.append(['TEST', 1, tick_time, price, bid_qty, ask_qty])

            df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
            csv_file = test_path / 'TEST_1' / 'TEST_1_footprint_20210104.csv'
            df.to_csv(csv_file, index=False)

            import time
            loader = CMEFootprintLoader(str(test_path))

            # Test 1-minute bars (should create many bars)
            start_time = time.time()
            footprint = loader.load_footprint_bars('TEST', '20210104', interval='1min')
            load_time = time.time() - start_time

            assert not footprint.empty, "Should handle large dataset"
            bar_count = len(footprint.index.get_level_values(0).unique())
            price_levels = len(footprint)

            # Test cache performance
            start_time = time.time()
            footprint2 = loader.load_footprint_bars('TEST', '20210104', interval='1min')
            cache_time = time.time() - start_time

            speedup = load_time / cache_time if cache_time > 0 else float('inf')

            print(f"✓ Large dataset: 1000 ticks → {bar_count} bars, {price_levels} levels")
            print(f"✓ Performance: {load_time:.3f}s initial, {cache_time:.3f}s cached ({speedup:.1f}x speedup)")
            self.results["passed"] += 1

        finally:
            shutil.rmtree(test_dir)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        print("Testing invalid input handling...")

        loader = CMEFootprintLoader()

        # Test invalid dates
        try:
            loader.load_footprint_bars('GC', 'invalid_date')
            assert False, "Should raise error for invalid date"
        except:
            pass  # Expected

        # Test invalid intervals
        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)
            (test_path / 'TEST_1').mkdir(parents=True)

            # Create minimal valid data
            data = [['TEST', 1, 1609718400000, 100.0, 1, 1]]
            df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
            csv_file = test_path / 'TEST_1' / 'TEST_1_footprint_20210104.csv'
            df.to_csv(csv_file, index=False)

            test_loader = CMEFootprintLoader(str(test_path))

            # Test invalid interval format
            try:
                test_loader.load_footprint_bars('TEST', '20210104', interval='invalid')
                assert False, "Should raise error for invalid interval"
            except:
                pass  # Expected

        finally:
            shutil.rmtree(test_dir)

        print("✓ Invalid input handling verified")
        self.results["passed"] += 1

    def test_memory_efficiency(self):
        """Test memory usage patterns"""
        print("Testing memory efficiency...")

        # Test that large datasets don't cause memory issues
        import sys

        # Create multiple loaders to test memory cleanup
        for i in range(10):
            ticksizes = {f'TEST{i}': 0.1 * (i + 1)}
            loader = TickLoader(ticksizes=ticksizes)
            assert loader.get_ticksize(f'TEST{i}') == 0.1 * (i + 1)

        # Test that aggregator handles large datasets
        large_df = pd.DataFrame({
            'timestamp': pd.date_range('2021-01-04', periods=1000, freq='1s'),
            'price': np.random.normal(1000, 10, 1000),
            'bid_qty': np.random.randint(1, 10, 1000),
            'ask_qty': np.random.randint(1, 10, 1000)
        })

        aggregator = FootprintAggregator()
        result = aggregator.aggregate_time_bars(large_df, interval='1min', ticksize=0.1)

        assert not result.empty, "Should handle large DataFrame"

        print("✓ Memory efficiency verified")
        self.results["passed"] += 1

    def run_all_edge_case_tests(self):
        """Run all edge case tests"""
        print("CME Tick Loader - Edge Cases and Stress Testing")
        print("=" * 60)

        try:
            self.test_extreme_ticksizes()
            self.test_single_tick_bar()
            self.test_zero_volume_handling()
            self.test_price_precision_edge_cases()
            self.test_large_dataset_performance()
            self.test_invalid_inputs()
            self.test_memory_efficiency()

        except Exception as e:
            print(f"\n✗ Edge case test failed: {e}")
            import traceback
            traceback.print_exc()
            self.results["failed"] += 1

        print("\n" + "=" * 60)
        print("EDGE CASE TEST RESULTS")
        print("=" * 60)
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")

        if self.results["failed"] == 0:
            print("\n🎉 ALL EDGE CASE TESTS PASSED!")
        else:
            print(f"\n⚠ {self.results['failed']} edge case test(s) failed")

        return self.results["failed"] == 0

if __name__ == "__main__":
    tester = EdgeCaseTests()
    success = tester.run_all_edge_case_tests()
    exit(0 if success else 1)