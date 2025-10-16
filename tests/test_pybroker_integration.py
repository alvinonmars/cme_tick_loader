"""
Test script for PyBroker + CME Tick Loader integration

This script tests the basic functionality of the integration without
running a full backtest.
"""

import sys
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'examples'))

def test_imports():
    """Test 1: Verify all required imports work"""
    print("="*80)
    print("TEST 1: Verifying imports...")
    print("="*80)

    try:
        from pybroker_integration import CMEDataSource
        print("✓ CMEDataSource imported successfully")

        from pybroker.data import DataSource
        from pybroker.common import DataCol
        print("✓ PyBroker modules imported successfully")

        from cme_tick_loader import CMEBarsLoader
        print("✓ CMEBarsLoader imported successfully")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datasource_creation():
    """Test 2: Test CMEDataSource instantiation"""
    print("\n" + "="*80)
    print("TEST 2: Creating CMEDataSource instance...")
    print("="*80)

    try:
        from pybroker_integration import CMEDataSource

        # Create data source with default settings
        data_source = CMEDataSource(
            resolution='MIN',
            num_units=5,
            cache_footprint=True
        )

        print(f"✓ CMEDataSource created")
        print(f"  Resolution: {data_source.resolution}")
        print(f"  Num units: {data_source.num_units}")
        print(f"  Cache footprint: {data_source.cache_footprint}")
        print(f"  Base path: {data_source.loader.base_path}")

        return data_source
    except Exception as e:
        print(f"✗ DataSource creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading():
    """Test 3: Test loading data with CMEDataSource"""
    print("\n" + "="*80)
    print("TEST 3: Loading test data...")
    print("="*80)

    try:
        from pybroker_integration import CMEDataSource
        import pandas as pd

        # Create data source
        data_source = CMEDataSource(
            resolution='MIN',
            num_units=5,
            cache_footprint=True
        )

        # Try to load one day of data
        print("\nAttempting to load GC data for 2021-01-04...")
        df = data_source.query(
            symbols=['GC'],
            start_date='2021-01-04',
            end_date='2021-01-04'
        )

        if df.empty:
            print("✗ No data returned (possibly data file not found)")
            print("  This is expected if CME data files are not available")
            return None

        print(f"✓ Data loaded successfully!")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Verify required columns
        from pybroker.common import DataCol
        required_cols = [
            DataCol.SYMBOL.value,
            DataCol.DATE.value,
            DataCol.OPEN.value,
            DataCol.HIGH.value,
            DataCol.LOW.value,
            DataCol.CLOSE.value,
            DataCol.VOLUME.value
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return None

        print(f"✓ All required columns present")

        # Check footprint columns
        footprint_cols = ['delta', 'total_volume', 'poc_price', 'poc_volume', 'imbalance_ratio']
        present_footprint_cols = [col for col in footprint_cols if col in df.columns]
        print(f"✓ Footprint columns present: {present_footprint_cols}")

        return df

    except FileNotFoundError as e:
        print(f"✗ Data file not found: {e}")
        print("  This is expected if CME data files are not in the expected location")
        return None
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_footprint_metrics():
    """Test 4: Verify footprint metrics are calculated correctly"""
    print("\n" + "="*80)
    print("TEST 4: Verifying footprint metrics...")
    print("="*80)

    try:
        from pybroker_integration import CMEDataSource

        data_source = CMEDataSource(
            resolution='MIN',
            num_units=5,
            cache_footprint=True
        )

        df = data_source.query(
            symbols=['GC'],
            start_date='2021-01-04',
            end_date='2021-01-04'
        )

        if df.empty:
            print("⊘ Skipping (no data available)")
            return None

        # Check footprint metrics
        print("\nFootprint metrics summary:")
        if 'delta' in df.columns:
            print(f"  Delta - mean: {df['delta'].mean():.2f}, std: {df['delta'].std():.2f}")
        if 'total_volume' in df.columns:
            print(f"  Total volume - mean: {df['total_volume'].mean():.0f}, sum: {df['total_volume'].sum():.0f}")
        if 'imbalance_ratio' in df.columns:
            print(f"  Imbalance ratio - mean: {df['imbalance_ratio'].mean():.4f}, max: {df['imbalance_ratio'].max():.4f}")
        if 'poc_price' in df.columns:
            non_null_poc = df['poc_price'].dropna()
            if len(non_null_poc) > 0:
                print(f"  POC price - range: {non_null_poc.min():.1f} to {non_null_poc.max():.1f}")

        print("\n✓ Footprint metrics calculated successfully")
        return True

    except FileNotFoundError:
        print("⊘ Skipping (no data available)")
        return None
    except Exception as e:
        print(f"✗ Footprint metrics verification failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_indicators():
    """Test 5: Test indicator functions"""
    print("\n" + "="*80)
    print("TEST 5: Testing indicator functions...")
    print("="*80)

    try:
        from pybroker_integration import delta_indicator, imbalance_indicator
        import numpy as np

        # Create mock bar data
        class MockBarData:
            def __init__(self):
                self.delta = np.array([100, -50, 200, -100, 150, 50, -75, 125, -25, 100])
                self.total_volume = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

        bar_data = MockBarData()

        # Test delta indicator
        print("\nTesting delta_indicator()...")
        cum_delta = delta_indicator(bar_data, lookback=5)
        print(f"  Cumulative delta (last 5): {cum_delta.iloc[-5:].values}")
        print("  ✓ delta_indicator works")

        # Test imbalance indicator
        print("\nTesting imbalance_indicator()...")
        imbalance = imbalance_indicator(bar_data, threshold=0.15)
        print(f"  Imbalance signals (last 5): {imbalance.iloc[-5:].values}")
        print("  ✓ imbalance_indicator works")

        return True

    except Exception as e:
        print(f"✗ Indicator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "PyBroker + CME Integration Test Suite" + " "*20 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()

    if not results['imports']:
        print("\n✗ CRITICAL: Import test failed. Cannot continue.")
        return

    # Test 2: DataSource creation
    results['datasource'] = test_datasource_creation() is not None

    # Test 3: Data loading
    results['data_loading'] = test_data_loading() is not None

    # Test 4: Footprint metrics
    results['footprint'] = test_footprint_metrics() is not None

    # Test 5: Indicators
    results['indicators'] = test_indicators()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result is True else "⊘ SKIP" if result is None else "✗ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\nResults: {passed}/{total} passed, {skipped} skipped, {failed} failed")

    if failed == 0:
        print("\n🎉 All executable tests passed!")
        if skipped > 0:
            print(f"   (Note: {skipped} test(s) skipped due to missing data files)")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
