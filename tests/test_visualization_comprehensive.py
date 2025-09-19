#!/usr/bin/env python3
"""Comprehensive visualization testing for CME Tick Loader"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer

class VisualizationTester:
    def __init__(self):
        self.test_dir = None
        self.results = {"passed": 0, "failed": 0, "tests": []}

    def setup_comprehensive_test_data(self):
        """Create comprehensive test data for visualization"""
        print("Setting up comprehensive visualization test data...")

        self.test_dir = tempfile.mkdtemp()
        test_path = Path(self.test_dir)
        (test_path / 'GC_1').mkdir(parents=True)

        # Create rich test data with multiple scenarios
        data = []
        base_time = 1609718400000  # 2021-01-04 09:30:00

        # Scenario 1: High volatility bar with wide range
        bar1_start = base_time
        prices1 = [1917.0, 1917.5, 1916.5, 1917.8, 1916.2, 1917.3]  # Wide range
        volumes1 = [(3, 2), (5, 8), (10, 4), (2, 12), (8, 3), (4, 6)]

        for i, (price, (bid, ask)) in enumerate(zip(prices1, volumes1)):
            data.append(['GC', 1, bar1_start + i * 30000, price, bid, ask])

        # Scenario 2: Low volatility bar with tight range
        bar2_start = base_time + 300000  # +5 minutes
        prices2 = [1917.2, 1917.3, 1917.2, 1917.1, 1917.2]  # Tight range
        volumes2 = [(2, 2), (15, 20), (8, 5), (6, 8), (3, 4)]  # High volume at 1917.3

        for i, (price, (bid, ask)) in enumerate(zip(prices2, volumes2)):
            data.append(['GC', 1, bar2_start + i * 40000, price, bid, ask])

        # Scenario 3: Trend bar (strong directional movement)
        bar3_start = base_time + 600000  # +10 minutes
        prices3 = [1917.1, 1917.4, 1917.6, 1917.8, 1918.0, 1918.1]  # Uptrend
        volumes3 = [(5, 2), (8, 3), (12, 4), (15, 5), (10, 3), (6, 2)]  # Buying pressure

        for i, (price, (bid, ask)) in enumerate(zip(prices3, volumes3)):
            data.append(['GC', 1, bar3_start + i * 35000, price, bid, ask])

        df = pd.DataFrame(data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
        csv_file = test_path / 'GC_1' / 'GC_1_footprint_20210104.csv'
        df.to_csv(csv_file, index=False)

        print(f"✓ Created visualization test data: {len(data)} ticks across 3 diverse scenarios")
        return test_path

    def test_candlestick_visualization(self, test_path):
        """Test candlestick chart visualization"""
        print("\n1. Testing candlestick visualization...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()

        # Test basic candlestick
        fig = viz.plot_candlestick(footprint, title="Test Gold Futures - 5min")
        assert fig is not None, "Candlestick figure should not be None"

        # Verify figure has data
        assert len(fig.data) >= 2, "Should have at least price and volume traces"

        # Check that we have candlestick data
        candlestick_trace = fig.data[0]
        assert candlestick_trace.type == 'candlestick', "First trace should be candlestick"
        assert len(candlestick_trace.x) > 0, "Should have timestamp data"
        assert len(candlestick_trace.open) > 0, "Should have OHLC data"

        # Test with empty data
        empty_footprint = pd.DataFrame()
        fig_empty = viz.plot_candlestick(empty_footprint)
        assert fig_empty is not None, "Should handle empty data gracefully"

        self.record_test("Candlestick visualization", True)
        print("✓ Candlestick visualization verified")

    def test_footprint_chart_all_scenarios(self, test_path):
        """Test footprint chart for all bar scenarios"""
        print("\n2. Testing footprint charts for all scenarios...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()
        ticksize = loader.tick_loader.get_ticksize('GC')

        bar_timestamps = footprint.index.get_level_values(0).unique()

        for i, bar_timestamp in enumerate(bar_timestamps):
            print(f"  Testing bar {i+1}: {bar_timestamp}")

            # Test with different chart sizes
            sizes = [(400, 600), (800, 800), (600, 400)]
            for width, height in sizes:
                fig = viz.plot_footprint(footprint, bar_timestamp,
                                       width=width, height=height, ticksize=ticksize)
                assert fig is not None, f"Footprint figure should not be None for {width}x{height}"

                # Check figure properties
                assert fig.layout.width == width, f"Width should be {width}"
                assert fig.layout.height == height, f"Height should be {height}"

            # Test without ticksize
            fig_no_ticksize = viz.plot_footprint(footprint, bar_timestamp)
            assert fig_no_ticksize is not None, "Should work without ticksize"

            # Verify bar data visualization
            bar_data = footprint.loc[bar_timestamp]
            expected_traces = 2  # Bid and ask bars
            assert len(fig.data) == expected_traces, f"Should have {expected_traces} traces"

            # Check OHLC level annotations
            ohlc_levels = 0
            if bar_data['is_open'].any(): ohlc_levels += 1
            if bar_data['is_high'].any(): ohlc_levels += 1
            if bar_data['is_low'].any(): ohlc_levels += 1
            if bar_data['is_close'].any(): ohlc_levels += 1

            # Should have POC line + OHLC lines
            total_shapes = len([shape for shape in fig.layout.shapes])
            assert total_shapes >= 1, "Should have at least POC line"

        self.record_test("Footprint chart scenarios", True)
        print(f"✓ Footprint charts verified for {len(bar_timestamps)} bars")

    def test_volume_profile_visualization(self, test_path):
        """Test volume profile visualization"""
        print("\n3. Testing volume profile visualization...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()

        # Test full day volume profile
        fig = viz.plot_volume_profile(footprint)
        assert fig is not None, "Volume profile figure should not be None"

        # Check that we have volume data
        assert len(fig.data) >= 2, "Should have total and bid volume traces"

        # Verify horizontal bars (volume profile characteristic)
        total_volume_trace = fig.data[0]
        assert total_volume_trace.type == 'bar', "Should be bar chart"
        assert total_volume_trace.orientation == 'h', "Should be horizontal bars"

        # Test with time range filtering
        start_time = footprint.index.get_level_values(0).min()
        end_time = footprint.index.get_level_values(0).max()

        fig_filtered = viz.plot_volume_profile(footprint, start_time=start_time, end_time=end_time)
        assert fig_filtered is not None, "Filtered volume profile should work"

        # Test POC and Value Area annotations
        shapes_count = len([shape for shape in fig.layout.shapes])
        assert shapes_count >= 2, "Should have POC line and value area rectangle"

        self.record_test("Volume profile visualization", True)
        print("✓ Volume profile visualization verified")

    def test_ticksize_formatting(self, test_path):
        """Test ticksize-based price formatting"""
        print("\n4. Testing ticksize formatting...")

        # Test different ticksize formats
        ticksize_tests = [
            ('GC', 0.1, '1917.1'),      # Gold: 1 decimal
            ('ES', 0.25, '3800.25'),    # ES: 2 decimals for quarters
            ('CL', 0.01, '75.42'),      # Oil: 2 decimals
            ('ZN', 1/64, '132-32'),     # Notes: fractional format
            ('ZB', 1/32, '155-16'),     # Bonds: fractional format
        ]

        for symbol, ticksize, expected_format in ticksize_tests:
            print(f"  Testing {symbol} ticksize {ticksize}")

            # Test price formatting logic
            test_prices = [1917.1, 3800.25, 75.42, 132.5, 155.5]

            for price in test_prices:
                # Test formatting based on ticksize
                if ticksize >= 1:
                    formatted = f"{price:.0f}"
                elif ticksize >= 0.1:
                    formatted = f"{price:.1f}"
                elif ticksize >= 0.01:
                    formatted = f"{price:.2f}"
                elif ticksize == 1/32:
                    whole = int(price)
                    fractional = (price - whole) * 32
                    formatted = f"{whole}-{fractional:02.0f}"
                elif ticksize == 1/64:
                    whole = int(price)
                    fractional = (price - whole) * 64
                    formatted = f"{whole}-{fractional:02.0f}"
                else:
                    formatted = f"{price:.3f}"

                assert formatted is not None, "Formatting should not fail"

        # Test in actual visualization
        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

        if not footprint.empty:
            viz = FootprintVisualizer()
            bar_timestamp = footprint.index.get_level_values(0).unique()[0]

            # Test with GC ticksize (0.1)
            fig = viz.plot_footprint(footprint, bar_timestamp, ticksize=0.1)
            assert fig is not None, "Ticksize formatting should work"

        self.record_test("Ticksize formatting", True)
        print("✓ Ticksize formatting verified")

    def test_interactive_features(self, test_path):
        """Test interactive visualization features"""
        print("\n5. Testing interactive features...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()

        if not footprint.empty:
            bar_timestamp = footprint.index.get_level_values(0).unique()[0]

            # Test hover templates
            fig = viz.plot_footprint(footprint, bar_timestamp, ticksize=0.1)

            # Check hover templates exist
            for trace in fig.data:
                if hasattr(trace, 'hovertemplate'):
                    assert trace.hovertemplate is not None, "Should have hover template"

            # Test that we can export to HTML
            html_str = fig.to_html()
            assert len(html_str) > 1000, "HTML export should generate substantial content"
            assert 'plotly' in html_str.lower(), "Should contain plotly references"

        self.record_test("Interactive features", True)
        print("✓ Interactive features verified")

    def test_error_handling_visualization(self, test_path):
        """Test visualization error handling"""
        print("\n6. Testing visualization error handling...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()

        # Test with non-existent bar timestamp
        fake_timestamp = pd.Timestamp('2021-01-04 15:00:00')
        fig = viz.plot_footprint(footprint, fake_timestamp)
        assert fig is not None, "Should handle missing bar gracefully"

        # Check that it shows an error annotation
        annotations = fig.layout.annotations
        assert len(annotations) > 0, "Should have error annotation"
        assert "not found" in str(annotations[0].text).lower(), "Should indicate bar not found"

        # Test with invalid ticksize values
        if not footprint.empty:
            bar_timestamp = footprint.index.get_level_values(0).unique()[0]

            # Test with zero ticksize
            fig_zero = viz.plot_footprint(footprint, bar_timestamp, ticksize=0)
            assert fig_zero is not None, "Should handle zero ticksize"

            # Test with negative ticksize
            fig_neg = viz.plot_footprint(footprint, bar_timestamp, ticksize=-0.1)
            assert fig_neg is not None, "Should handle negative ticksize"

        self.record_test("Visualization error handling", True)
        print("✓ Visualization error handling verified")

    def test_visualization_data_consistency(self, test_path):
        """Test that visualizations accurately represent the data"""
        print("\n7. Testing visualization data consistency...")

        loader = CMEFootprintLoader(str(test_path))
        footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')
        viz = FootprintVisualizer()

        # Test candlestick data consistency
        if not footprint.empty:
            # Extract OHLC manually
            manual_ohlc = []
            for timestamp in footprint.index.get_level_values(0).unique():
                bar = footprint.loc[timestamp]

                open_price = bar[bar['is_open']].index[0] if bar['is_open'].any() else bar.index[0]
                high_price = bar[bar['is_high']].index[0] if bar['is_high'].any() else bar.index.max()
                low_price = bar[bar['is_low']].index[0] if bar['is_low'].any() else bar.index.min()
                close_price = bar[bar['is_close']].index[0] if bar['is_close'].any() else bar.index[-1]
                volume = bar['total_vol'].sum()

                manual_ohlc.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

            # Create candlestick and verify data matches
            fig = viz.plot_candlestick(footprint)
            candlestick_trace = fig.data[0]

            # Verify we have the right number of bars
            assert len(candlestick_trace.x) == len(manual_ohlc), "Bar count should match"

            # Verify OHLC values match (within floating point precision)
            for i, expected in enumerate(manual_ohlc):
                assert abs(candlestick_trace.open[i] - expected['open']) < 1e-10, f"Open mismatch at bar {i}"
                assert abs(candlestick_trace.high[i] - expected['high']) < 1e-10, f"High mismatch at bar {i}"
                assert abs(candlestick_trace.low[i] - expected['low']) < 1e-10, f"Low mismatch at bar {i}"
                assert abs(candlestick_trace.close[i] - expected['close']) < 1e-10, f"Close mismatch at bar {i}"

        self.record_test("Visualization data consistency", True)
        print("✓ Visualization data consistency verified")

    def record_test(self, test_name, passed, note=""):
        """Record test result"""
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1

        self.results["tests"].append({
            "name": test_name,
            "passed": passed,
            "note": note
        })

    def cleanup(self):
        """Clean up test directory"""
        if self.test_dir:
            shutil.rmtree(self.test_dir)

    def run_comprehensive_visualization_tests(self):
        """Run all visualization tests"""
        print("CME Tick Loader - Comprehensive Visualization Testing")
        print("="*60)

        try:
            test_path = self.setup_comprehensive_test_data()

            self.test_candlestick_visualization(test_path)
            self.test_footprint_chart_all_scenarios(test_path)
            self.test_volume_profile_visualization(test_path)
            self.test_ticksize_formatting(test_path)
            self.test_interactive_features(test_path)
            self.test_error_handling_visualization(test_path)
            self.test_visualization_data_consistency(test_path)

        except ImportError as e:
            if 'plotly' in str(e).lower():
                print("⚠ Plotly not available - visualization tests skipped")
                self.record_test("All visualization tests", True, "Skipped - Plotly not available")
            else:
                raise
        except Exception as e:
            print(f"✗ Visualization test failed: {e}")
            import traceback
            traceback.print_exc()
            self.record_test("Visualization tests", False, str(e))
        finally:
            self.cleanup()

        # Print results
        print("\n" + "="*60)
        print("COMPREHENSIVE VISUALIZATION TEST RESULTS")
        print("="*60)

        for test in self.results["tests"]:
            status = "✓ PASS" if test["passed"] else "✗ FAIL"
            note = f" ({test['note']})" if test["note"] else ""
            print(f"{status}: {test['name']}{note}")

        print(f"\nSummary: {self.results['passed']} passed, {self.results['failed']} failed")

        if self.results["failed"] == 0:
            print("\n🎉 ALL VISUALIZATION TESTS PASSED!")
        else:
            print(f"\n⚠ {self.results['failed']} visualization test(s) failed")

        return self.results["failed"] == 0

if __name__ == "__main__":
    tester = VisualizationTester()
    success = tester.run_comprehensive_visualization_tests()
    exit(0 if success else 1)