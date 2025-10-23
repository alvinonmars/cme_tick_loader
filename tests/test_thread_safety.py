"""Test thread safety of GenericTimeSeriesCache"""

import threading
import time
import pandas as pd
from cme_tick_loader import GenericTimeSeriesCache


def mock_loader(symbol, resolution, num_units, start_time, end_time):
    """Mock loader for testing"""
    return pd.DataFrame(), pd.DataFrame(
        columns=['bid_vol', 'ask_vol'],
        index=pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])
    )


def test_concurrent_append():
    """Test multiple threads appending data concurrently"""
    print("\n=== Test 1: Concurrent Append ===")

    cache = GenericTimeSeriesCache(mock_loader, thread_safe=True)

    # Prepare data for two threads
    thread_a_data = []
    thread_b_data = []

    for i in range(100):
        # Thread A: timestamps 00:00, 00:02, 00:04, ... (even)
        ts_a = pd.Timestamp('2021-01-04') + pd.Timedelta(minutes=i*2)
        bar_a = pd.DataFrame([{
            'open': 1800.0 + i,
            'high': 1805.0,
            'low': 1795.0,
            'close': 1800.0,
            'volume': 1000
        }], index=[ts_a])
        thread_a_data.append(bar_a)

        # Thread B: timestamps 00:01, 00:03, 00:05, ... (odd)
        ts_b = pd.Timestamp('2021-01-04') + pd.Timedelta(minutes=i*2+1)
        bar_b = pd.DataFrame([{
            'open': 1900.0 + i,
            'high': 1905.0,
            'low': 1895.0,
            'close': 1900.0,
            'volume': 2000
        }], index=[ts_b])
        thread_b_data.append(bar_b)

    empty_fp = pd.DataFrame(
        columns=['bid_vol', 'ask_vol'],
        index=pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])
    )

    def thread_a_work():
        for bar in thread_a_data:
            cache.append('GC', 'MIN', 5, bar, empty_fp)

    def thread_b_work():
        for bar in thread_b_data:
            cache.append('GC', 'MIN', 5, bar, empty_fp)

    # Start threads
    thread_a = threading.Thread(target=thread_a_work)
    thread_b = threading.Thread(target=thread_b_work)

    start_time = time.time()
    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()
    elapsed = time.time() - start_time

    # Verify results
    stats = cache.get_stats()
    bars_count = stats['entries']['GC_MIN_5']['bars_count']

    print(f"Expected: 200 bars (100 from each thread)")
    print(f"Actual: {bars_count} bars")
    print(f"Time elapsed: {elapsed:.3f}s")

    # Get actual data (access cache directly)
    key = ('GC', 'MIN', 5)
    bars = cache._cache[key].bars

    # Check sorted
    is_sorted = bars.index.is_monotonic_increasing
    print(f"Data sorted: {is_sorted}")

    assert bars_count == 200, f"Expected 200 bars, got {bars_count}"
    assert is_sorted, "Data should be sorted by timestamp"
    print("✓ Test passed: No data loss, correctly sorted")


def test_out_of_order_timestamps():
    """Test threads writing out-of-order timestamps"""
    print("\n=== Test 2: Out-of-Order Timestamps ===")

    cache = GenericTimeSeriesCache(mock_loader, thread_safe=True)

    empty_fp = pd.DataFrame(
        columns=['bid_vol', 'ask_vol'],
        index=pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])
    )

    def realtime_thread():
        """Simulate realtime data: 10:00, 10:05, 10:10, ..."""
        for i in range(10):
            ts = pd.Timestamp('2021-01-04 10:00:00') + pd.Timedelta(minutes=i*5)
            bar = pd.DataFrame([{
                'open': 1800.0, 'high': 1805.0,
                'low': 1795.0, 'close': 1800.0, 'volume': 1000
            }], index=[ts])
            cache.append('GC', 'MIN', 5, bar, empty_fp)
            time.sleep(0.001)  # Simulate realtime interval

    def backfill_thread():
        """Simulate backfill: 09:00, 09:05, 09:10, ... (earlier data)"""
        time.sleep(0.005)  # Start slightly later
        for i in range(10):
            ts = pd.Timestamp('2021-01-04 09:00:00') + pd.Timedelta(minutes=i*5)
            bar = pd.DataFrame([{
                'open': 1700.0, 'high': 1705.0,
                'low': 1695.0, 'close': 1700.0, 'volume': 500
            }], index=[ts])
            cache.append('GC', 'MIN', 5, bar, empty_fp)
            time.sleep(0.001)

    # Start threads
    t1 = threading.Thread(target=realtime_thread)
    t2 = threading.Thread(target=backfill_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Verify results (access cache directly to avoid loader)
    stats = cache.get_stats()
    bars_count = stats['entries']['GC_MIN_5']['bars_count']

    # Access cache directly
    key = ('GC', 'MIN', 5)
    bars = cache._cache[key].bars

    print(f"Total bars: {len(bars)}")
    if len(bars) > 0:
        print(f"First timestamp: {bars.index[0]}")
        print(f"Last timestamp: {bars.index[-1]}")
        print(f"Data sorted: {bars.index.is_monotonic_increasing}")

    # Check data integrity
    expected_count = 20
    assert len(bars) == expected_count, f"Expected {expected_count} bars, got {len(bars)}"
    assert bars.index.is_monotonic_increasing, "Data should be sorted"

    # Check backfill data is present
    backfill_bars = bars[bars.index < pd.Timestamp('2021-01-04 10:00:00')]
    realtime_bars = bars[bars.index >= pd.Timestamp('2021-01-04 10:00:00')]

    print(f"Backfill bars: {len(backfill_bars)}")
    print(f"Realtime bars: {len(realtime_bars)}")

    assert len(backfill_bars) == 10, "Backfill data missing"
    assert len(realtime_bars) == 10, "Realtime data missing"
    print("✓ Test passed: Out-of-order data correctly sorted")


def test_without_thread_safety():
    """Test that without thread_safe=True, data may be lost"""
    print("\n=== Test 3: Without Thread Safety (Expected Race Condition) ===")

    cache = GenericTimeSeriesCache(mock_loader, thread_safe=False)

    empty_fp = pd.DataFrame(
        columns=['bid_vol', 'ask_vol'],
        index=pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])
    )

    def worker(start_idx):
        for i in range(start_idx, start_idx + 50):
            ts = pd.Timestamp('2021-01-04') + pd.Timedelta(minutes=i)
            bar = pd.DataFrame([{
                'open': 1800.0, 'high': 1805.0,
                'low': 1795.0, 'close': 1800.0, 'volume': 1000
            }], index=[ts])
            cache.append('GC', 'MIN', 5, bar, empty_fp)

    # Start multiple threads
    threads = [threading.Thread(target=worker, args=(i*50,)) for i in range(4)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Count result
    stats = cache.get_stats()
    bars_count = stats['entries']['GC_MIN_5']['bars_count']

    print(f"Expected: 200 bars")
    print(f"Actual: {bars_count} bars")

    if bars_count < 200:
        print(f"⚠️  Data loss detected: {200 - bars_count} bars lost")
        print("   This is expected without thread_safe=True")
    else:
        print("   (No data loss detected in this run, but race condition still exists)")


if __name__ == '__main__':
    print("Testing GenericTimeSeriesCache Thread Safety")
    print("=" * 60)

    try:
        test_concurrent_append()
        test_out_of_order_timestamps()
        test_without_thread_safety()

        print("\n" + "=" * 60)
        print("✓ All thread safety tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
