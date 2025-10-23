"""Test GenericTimeSeriesCache with real CME data"""

import pandas as pd
from cme_tick_loader import CMEBarsLoader, GenericTimeSeriesCache


def create_loader_func():
    """Create loader function wrapping CMEBarsLoader"""
    loader = CMEBarsLoader()

    def load_func(symbol, resolution, num_units, start_time, end_time):
        start_date = start_time.strftime('%Y%m%d')
        end_date = end_time.strftime('%Y%m%d')

        result = loader.load_date_range(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            num_units=num_units,
            timezone_naive=True,
            set_index=True
        )

        return result['bars'], result['footprint']

    return load_func


def test_basic_get():
    """Test basic get functionality"""
    print("\n=== Test 1: Basic Get ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 23:59:59')

    bars, footprint = cache.get('GC', 'MIN', 5, start, end)

    print(f"Loaded {len(bars)} bars")
    print(f"Time range: {bars.index.min()} - {bars.index.max()}")
    print(f"Footprint rows: {len(footprint)}")
    print(f"\nFirst 3 bars:\n{bars.head(3)}")

    assert not bars.empty
    assert len(bars) > 0


def test_cache_hit():
    """Test cache hit on second query"""
    print("\n=== Test 2: Cache Hit ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 23:59:59')

    # First query (miss)
    print("First query (should load from CME)...")
    bars1, _ = cache.get('GC', 'MIN', 5, start, end)
    print(f"  Loaded {len(bars1)} bars")

    # Second query (hit)
    print("Second query (should hit cache)...")
    bars2, _ = cache.get('GC', 'MIN', 5, start, end)
    print(f"  Retrieved {len(bars2)} bars")

    # Should be identical
    pd.testing.assert_frame_equal(bars1, bars2)
    print("  ✓ Cache hit successful")


def test_append_realtime():
    """Test append for real-time bar"""
    print("\n=== Test 3: Append Real-time Bar ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    # Load historical data
    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 01:00:00')

    bars, _ = cache.get('GC', 'MIN', 5, start, end)
    original_count = len(bars)
    print(f"Original bars: {original_count}")

    # Simulate new bar
    new_time = pd.Timestamp('2021-01-04 01:05:00')
    new_bar = pd.DataFrame([{
        'open': 1850.0,
        'high': 1855.0,
        'low': 1845.0,
        'close': 1850.0,
        'volume': 1200
    }], index=[new_time])

    empty_fp = pd.DataFrame(
        columns=['bid_vol', 'ask_vol'],
        index=pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])
    )

    cache.append('GC', 'MIN', 5, new_bar, empty_fp)
    print(f"Appended bar at {new_time}")

    # Query extended range
    bars_new, _ = cache.get('GC', 'MIN', 5, start, new_time)
    print(f"New total bars: {len(bars_new)}")
    print(f"Last bar:\n{bars_new.tail(1)}")

    assert len(bars_new) > original_count
    assert new_time in bars_new.index


def test_multi_day_range():
    """Test multi-day query"""
    print("\n=== Test 4: Multi-day Range ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-06 23:59:59')

    bars, _ = cache.get('GC', 'MIN', 5, start, end)

    print(f"Loaded {len(bars)} bars across 3 days")
    print(f"Time range: {bars.index.min()} - {bars.index.max()}")

    # Group by date
    bars['date'] = bars.index.date
    daily_counts = bars.groupby('date').size()
    print(f"\nBars per day:\n{daily_counts}")

    assert len(bars) > 0


def test_different_resolutions():
    """Test multiple resolutions cached separately"""
    print("\n=== Test 5: Different Resolutions ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 23:59:59')

    # 5-minute bars
    bars_5m, _ = cache.get('GC', 'MIN', 5, start, end)
    print(f"5-minute bars: {len(bars_5m)}")

    # 15-minute bars
    bars_15m, _ = cache.get('GC', 'MIN', 15, start, end)
    print(f"15-minute bars: {len(bars_15m)}")

    # 1-hour bars
    bars_1h, _ = cache.get('GC', 'H', 1, start, end)
    print(f"1-hour bars: {len(bars_1h)}")

    # Check stats
    stats = cache.get_stats()
    print(f"\nCache entries: {stats['total_keys']}")
    for key, info in stats['entries'].items():
        print(f"  {key}: {info['bars_count']} bars, {info['memory_usage_mb']:.2f} MB")

    assert stats['total_keys'] == 3


def test_has_data():
    """Test has_data check"""
    print("\n=== Test 6: Has Data Check ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 23:59:59')

    # Before loading
    has_before = cache.has_data('GC', 'MIN', 5, start, end)
    print(f"Has data before load: {has_before}")

    # Load
    bars, _ = cache.get('GC', 'MIN', 5, start, end)
    actual_start = bars.index.min()
    actual_end = bars.index.max()
    print(f"Actual data range: {actual_start} - {actual_end}")

    # After loading - check actual data range
    has_after = cache.has_data('GC', 'MIN', 5, actual_start, actual_end)
    print(f"Has data after load (actual range): {has_after}")

    # Extended range (not covered)
    extended_end = pd.Timestamp('2021-01-05 23:59:59')
    has_extended = cache.has_data('GC', 'MIN', 5, actual_start, extended_end)
    print(f"Has extended range: {has_extended}")

    assert not has_before
    assert has_after
    assert not has_extended


def test_clear():
    """Test cache clearing"""
    print("\n=== Test 7: Clear Cache ===")

    cache = GenericTimeSeriesCache(create_loader_func())

    start = pd.Timestamp('2021-01-04 00:00:00')
    end = pd.Timestamp('2021-01-04 23:59:59')

    # Load multiple keys
    cache.get('GC', 'MIN', 5, start, end)
    cache.get('GC', 'MIN', 15, start, end)

    stats_before = cache.get_stats()
    print(f"Entries before clear: {stats_before['total_keys']}")

    # Clear GC_MIN_5
    cache.clear('GC_MIN_5')

    stats_after = cache.get_stats()
    print(f"Entries after clear GC_MIN_5: {stats_after['total_keys']}")

    # Clear all
    cache.clear()

    stats_final = cache.get_stats()
    print(f"Entries after clear all: {stats_final['total_keys']}")

    assert stats_before['total_keys'] == 2
    assert stats_after['total_keys'] == 1
    assert stats_final['total_keys'] == 0


if __name__ == '__main__':
    print("Testing GenericTimeSeriesCache with real CME data")
    print("=" * 60)

    try:
        test_basic_get()
        test_cache_hit()
        test_append_realtime()
        test_multi_day_range()
        test_different_resolutions()
        test_has_data()
        test_clear()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
