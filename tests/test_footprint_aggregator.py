"""Test cases for FootprintAggregator"""

import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import FootprintAggregator


class TestFootprintAggregator(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create test tick data
        test_data = [
            ['2021-01-04 09:30:00.003', 1917.1, 1, 0],
            ['2021-01-04 09:30:00.288', 1917.1, 1, 0],
            ['2021-01-04 09:30:00.319', 1917.2, 0, 1],
            ['2021-01-04 09:30:00.384', 1917.0, 1, 0],
            ['2021-01-04 09:30:00.454', 1917.2, 0, 1],
            ['2021-01-04 09:31:00.100', 1917.3, 0, 2],
            ['2021-01-04 09:31:00.200', 1917.1, 2, 0],
            ['2021-01-04 09:31:00.300', 1917.2, 1, 1],
        ]

        self.df = pd.DataFrame(test_data, columns=['timestamp', 'price', 'bid_qty', 'ask_qty'])
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.aggregator = FootprintAggregator()

    def test_basic_aggregation(self):
        """Test basic time bar aggregation"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        # Should have 2 bars (09:30 and 09:31)
        bar_count = len(footprint.index.get_level_values(0).unique())
        self.assertEqual(bar_count, 2)

        # Check columns
        expected_columns = ['bid_vol', 'ask_vol', 'total_vol', 'delta', 'is_open', 'is_high', 'is_low', 'is_close']
        for col in expected_columns:
            self.assertIn(col, footprint.columns)

    def test_ohlc_flags(self):
        """Test OHLC flag setting"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        # Check first bar (09:30)
        bar1_time = pd.Timestamp('2021-01-04 09:30:00')
        if bar1_time in footprint.index.get_level_values(0):
            bar1 = footprint.loc[bar1_time]

            # Should have open, high, low, close flags
            self.assertTrue(bar1['is_open'].any())
            self.assertTrue(bar1['is_high'].any())
            self.assertTrue(bar1['is_low'].any())
            self.assertTrue(bar1['is_close'].any())

    def test_volume_aggregation(self):
        """Test volume aggregation by price level"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        bar1_time = pd.Timestamp('2021-01-04 09:30:00')
        if bar1_time in footprint.index.get_level_values(0):
            bar1 = footprint.loc[bar1_time]

            # Check if volumes are aggregated correctly
            if 1917.1 in bar1.index:
                # Price 1917.1 appears twice in first bar with bid_qty=1 each
                self.assertEqual(bar1.loc[1917.1, 'bid_vol'], 2)

    def test_get_ohlc_from_footprint(self):
        """Test OHLC extraction"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        bar1_time = pd.Timestamp('2021-01-04 09:30:00')
        ohlc = self.aggregator.get_ohlc_from_footprint(footprint, bar1_time)

        if ohlc:
            self.assertIn('open', ohlc)
            self.assertIn('high', ohlc)
            self.assertIn('low', ohlc)
            self.assertIn('close', ohlc)

    def test_analyze_footprint_bar(self):
        """Test bar analysis"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        bar1_time = pd.Timestamp('2021-01-04 09:30:00')
        analysis = self.aggregator.analyze_footprint_bar(footprint, bar1_time)

        if analysis:
            self.assertIn('poc', analysis)
            self.assertIn('total_delta', analysis)
            self.assertIn('price_levels', analysis)
            self.assertIn('total_volume', analysis)

    def test_calculate_value_area(self):
        """Test value area calculation"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        bar1_time = pd.Timestamp('2021-01-04 09:30:00')
        value_area = self.aggregator.calculate_value_area(footprint, bar1_time)

        if value_area:
            self.assertIn('poc', value_area)
            self.assertIn('vah', value_area)
            self.assertIn('val', value_area)
            self.assertIn('percentage', value_area)

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame(columns=['timestamp', 'price', 'bid_qty', 'ask_qty'])
        footprint = self.aggregator.aggregate_time_bars(empty_df, '1min')

        self.assertTrue(footprint.empty)

    def test_nonexistent_bar(self):
        """Test handling of nonexistent bar"""
        footprint = self.aggregator.aggregate_time_bars(self.df, '1min')

        # Try to analyze nonexistent bar
        nonexistent_time = pd.Timestamp('2021-01-04 10:00:00')
        analysis = self.aggregator.analyze_footprint_bar(footprint, nonexistent_time)

        self.assertIsNone(analysis)


if __name__ == '__main__':
    unittest.main()