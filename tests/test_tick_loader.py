"""Test cases for TickLoader"""

import unittest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import TickLoader


class TestTickLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create test data structure
        (self.test_path / 'GC_1').mkdir(parents=True)

        # Create test CSV file
        test_data = [
            ['GC', 1, 1609718400003, 1917.1, 1, 0],
            ['GC', 1, 1609718400288, 1917.1, 1, 0],
            ['GC', 1, 1609718400319, 1917.2, 0, 1],
            ['GC', 1, 1609718400384, 1917.0, 1, 0],
            ['GC', 1, 1609718400454, 1917.2, 0, 1]
        ]

        df = pd.DataFrame(test_data, columns=['symbol', 'timeframe', 'timestamp_ms', 'price', 'bid_qty', 'ask_qty'])
        test_file = self.test_path / 'GC_1' / 'GC_1_footprint_20210104.csv'
        df.to_csv(test_file, index=False)

        self.loader = TickLoader(str(self.test_path))

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_load_ticks_basic(self):
        """Test basic tick loading"""
        df = self.loader.load_ticks('GC', '20210104', use_cache=False)

        self.assertEqual(len(df), 5)
        self.assertIn('timestamp', df.columns)
        self.assertIn('price', df.columns)
        self.assertEqual(df['symbol'].iloc[0], 'GC')

    def test_cache_functionality(self):
        """Test cache save and load"""
        # First load (creates cache)
        df1 = self.loader.load_ticks('GC', '20210104', use_cache=True)

        # Check cache file exists
        cache_file = self.test_path / '.cache' / 'ticks' / 'GC_1_footprint_20210104.pkl'
        self.assertTrue(cache_file.exists())

        # Second load (from cache)
        df2 = self.loader.load_ticks('GC', '20210104', use_cache=True)

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_refresh_cache(self):
        """Test cache refresh"""
        # Load with cache
        df1 = self.loader.load_ticks('GC', '20210104', use_cache=True)

        # Load with refresh
        df2 = self.loader.load_ticks('GC', '20210104', use_cache=True, refresh_cache=True)

        # Should be identical (but cache was refreshed)
        pd.testing.assert_frame_equal(df1, df2)

    def test_file_not_found(self):
        """Test handling of missing files"""
        with self.assertRaises(FileNotFoundError):
            self.loader.load_ticks('GC', '20210199')  # Invalid date

    def test_clear_cache(self):
        """Test cache clearing"""
        # Create cache
        self.loader.load_ticks('GC', '20210104', use_cache=True)

        # Clear specific cache
        self.loader.clear_cache('GC', '20210104')

        cache_file = self.test_path / '.tick_cache' / 'GC_1_footprint_20210104.pkl'
        self.assertFalse(cache_file.exists())

    def test_get_cache_info(self):
        """Test cache info"""
        # Create cache
        self.loader.load_ticks('GC', '20210104', use_cache=True)

        info = self.loader.get_cache_info()

        self.assertIn('cache_dir', info)
        self.assertIn('file_count', info)
        self.assertIn('total_size_mb', info)
        self.assertEqual(info['file_count'], 1)


if __name__ == '__main__':
    unittest.main()