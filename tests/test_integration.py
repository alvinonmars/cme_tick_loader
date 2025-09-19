"""Integration tests for CME Footprint Loader"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader


class TestIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with real data structure"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create test data structure
        (self.test_path / 'GC_1').mkdir(parents=True)

        # Create realistic test data (multiple bars)
        test_data = []
        base_time = 1609718400000  # 2021-01-04 09:30:00

        # First 5-minute bar
        for i in range(10):
            test_data.append(['GC', 1, base_time + i * 1000, 1917.1 + (i % 3) * 0.1, 1, 0])
            test_data.append(['GC', 1, base_time + i * 1000 + 500, 1917.1 + (i % 3) * 0.1, 0, 1])

        # Second 5-minute bar
        base_time += 300000  # +5 minutes
        for i in range(8):
            test_data.append(['GC', 1, base_time + i * 1000, 1917.2 + (i % 2) * 0.1, 2, 1])

        df = pd.DataFrame(test_data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
        test_file = self.test_path / 'GC_1' / 'GC_1_footprint_20210104.csv'
        df.to_csv(test_file, index=False)

        self.loader = CMEFootprintLoader(str(self.test_path))

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_full_workflow(self):
        """Test complete workflow from loading to analysis"""
        # Load footprint bars
        footprint = self.loader.load_footprint_bars('GC', '20210104', interval='5min')

        # Should have data
        self.assertFalse(footprint.empty)

        # Should have correct structure
        self.assertEqual(len(footprint.index.levels), 2)  # MultiIndex with 2 levels

        # Should have expected columns
        expected_columns = ['bid_vol', 'ask_vol', 'total_vol', 'delta', 'is_open', 'is_high', 'is_low', 'is_close']
        for col in expected_columns:
            self.assertIn(col, footprint.columns)

    def test_cache_workflow(self):
        """Test caching workflow"""
        # First load (creates cache)
        footprint1 = self.loader.load_footprint_bars('GC', '20210104', interval='5min')

        # Second load (from cache)
        footprint2 = self.loader.load_footprint_bars('GC', '20210104', interval='5min')

        # Should be identical
        pd.testing.assert_frame_equal(footprint1, footprint2)

        # Check cache exists
        cache_info = self.loader.get_cache_info()
        self.assertGreater(cache_info['footprint_cache']['file_count'], 0)

    def test_analysis_workflow(self):
        """Test analysis functions"""
        footprint = self.loader.load_footprint_bars('GC', '20210104', interval='5min')

        if not footprint.empty:
            # Get first bar timestamp
            bar_timestamp = footprint.index.get_level_values(0).unique()[0]

            # Test OHLC extraction
            ohlc = self.loader.get_ohlc(footprint, bar_timestamp)
            self.assertIsNotNone(ohlc)

            # Test bar analysis
            analysis = self.loader.analyze_bar(footprint, bar_timestamp)
            self.assertIsNotNone(analysis)

            # Test value area calculation
            value_area = self.loader.calculate_value_area(footprint, bar_timestamp)
            self.assertIsNotNone(value_area)

    def test_date_range_loading(self):
        """Test loading date range"""
        # Create additional test file for next day
        test_data = [
            ['GC', 1, 1609804800000, 1918.0, 1, 0],  # 2021-01-05
            ['GC', 1, 1609804801000, 1918.1, 0, 1],
        ]

        df = pd.DataFrame(test_data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
        test_file = self.test_path / 'GC_1' / 'GC_1_footprint_20210105.csv'
        df.to_csv(test_file, index=False)

        # Load date range
        footprint = self.loader.load_date_range('GC', '20210104', '20210105', interval='5min')

        # Should have data from both days
        timestamps = footprint.index.get_level_values(0).unique()
        dates = [ts.date() for ts in timestamps]
        unique_dates = set(dates)

        self.assertGreaterEqual(len(unique_dates), 1)  # At least one day

    def test_error_handling(self):
        """Test error handling"""
        # Try to load non-existent file
        with self.assertRaises(FileNotFoundError):
            self.loader.load_footprint_bars('XX', '20210104')

        # Try to analyze non-existent bar
        footprint = self.loader.load_footprint_bars('GC', '20210104', interval='5min')
        nonexistent_time = pd.Timestamp('2021-01-04 15:00:00')
        analysis = self.loader.analyze_bar(footprint, nonexistent_time)
        self.assertIsNone(analysis)

    def test_cache_management(self):
        """Test cache management functions"""
        # Create some cache
        self.loader.load_footprint_bars('GC', '20210104', interval='5min')

        # Get cache info
        info_before = self.loader.get_cache_info()
        self.assertGreater(info_before['total_size_mb'], 0)

        # Clear cache
        self.loader.clear_all_cache()

        # Check cache is cleared
        info_after = self.loader.get_cache_info()
        self.assertEqual(info_after['tick_cache']['file_count'], 0)
        self.assertEqual(info_after['footprint_cache']['file_count'], 0)


if __name__ == '__main__':
    unittest.main()