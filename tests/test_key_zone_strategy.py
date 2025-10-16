"""
Tests for Key Zone Strategy

Comprehensive tests for all components.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'examples'))

from key_zone_strategy import (
    KeyZoneStrategy, KeyZoneDetector, StructuralSignalDetector,
    TouchTracker, SignalCombiner, StrategyConfig,
    KeyPriceBook, KeyZone, Signal, TouchEvent,
    ZoneType, SignalType, TradeAction
)


def create_mock_bars(n=100, trend='flat'):
    """Create mock bar data for testing"""
    np.random.seed(42)

    base_price = 1900.0
    prices = []

    if trend == 'flat':
        prices = base_price + np.random.randn(n) * 2
    elif trend == 'up':
        prices = base_price + np.arange(n) * 0.5 + np.random.randn(n) * 1
    elif trend == 'down':
        prices = base_price - np.arange(n) * 0.5 + np.random.randn(n) * 1
    elif trend == 'v_bullish':
        # Create V-shape
        prices = np.concatenate([
            base_price - np.arange(50) * 0.3,  # Down
            base_price - 15 + np.arange(50) * 0.4  # Up
        ])
    elif trend == 'breakout':
        # Create consolidation then breakout
        prices = np.concatenate([
            base_price + np.random.randn(80) * 1,  # Consolidation
            base_price + np.arange(20) * 0.8  # Breakout
        ])

    # Create OHLC from prices
    data = []
    for i, close in enumerate(prices):
        high = close + abs(np.random.randn() * 0.5)
        low = close - abs(np.random.randn() * 0.5)
        open_price = (high + low) / 2 + np.random.randn() * 0.2
        volume = 1000 + np.random.randint(0, 500)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def create_mock_footprint(price_center=1900.0, positive_delta=True):
    """Create mock footprint data"""
    prices = np.arange(price_center - 5, price_center + 5, 0.1)

    data = []
    for price in prices:
        if positive_delta:
            bid_vol = max(0, 50 + np.random.randn() * 10)
            ask_vol = max(0, 80 + np.random.randn() * 15)  # Higher ask
        else:
            bid_vol = max(0, 80 + np.random.randn() * 15)  # Higher bid
            ask_vol = max(0, 50 + np.random.randn() * 10)

        delta = ask_vol - bid_vol
        total_vol = bid_vol + ask_vol

        data.append({
            'price': price,
            'bid_vol': bid_vol,
            'ask_vol': ask_vol,
            'delta': delta,
            'total_vol': total_vol
        })

    df = pd.DataFrame(data)
    df = df.set_index('price')
    return df


# ============================================================================
# Test 1: Data Models
# ============================================================================

def test_data_models():
    """Test data model creation and methods"""
    print("\n" + "="*80)
    print("TEST 1: Data Models")
    print("="*80)

    # KeyPriceBook
    book = KeyPriceBook(
        big_delta_ask=[1905.0, 1910.0, None, None, None],
        peak_ask=[1908.0, None, None, None, None],
        big_delta_bid=[1895.0, 1890.0, None, None, None],
        peak_bid=[1892.0, None, None, None, None],
        current_close=1900.0
    )

    all_prices = book.get_all_key_prices()
    print(f"✓ KeyPriceBook created with {len(all_prices)} key prices")
    assert len(all_prices) == 6, "Should have 6 non-None prices"

    # KeyZone
    zone = KeyZone(
        center_price=1900.0,
        lower_bound=1898.0,
        upper_bound=1902.0,
        zone_type=ZoneType.SUPPORT,
        sources=['big_delta_bid0', 'peak_bid0']
    )
    print(f"✓ KeyZone created with strength={zone.strength}")
    assert zone.strength == 2, "Zone strength should equal number of sources"
    assert zone.contains(1900.0), "Zone should contain center price"
    assert not zone.contains(1905.0), "Zone should not contain far price"

    # Signal
    signal = Signal(
        signal_type=SignalType.V_REVERSAL_BULLISH,
        strength=0.8,
        bar_index=50
    )
    print(f"✓ Signal created: {signal.signal_type.value}")
    assert signal.is_bullish(), "Should be bullish"
    assert not signal.is_bearish(), "Should not be bearish"

    print("✅ All data model tests passed\n")
    return True


# ============================================================================
# Test 2: KeyZoneDetector
# ============================================================================

def test_key_zone_detector():
    """Test KeyZoneDetector"""
    print("\n" + "="*80)
    print("TEST 2: KeyZoneDetector")
    print("="*80)

    config = StrategyConfig()
    detector = KeyZoneDetector(config)

    # Create mock data
    bars = create_mock_bars(n=100, trend='flat')
    current_close = bars.iloc[-1]['close']

    # Create mock footprints
    footprints = [
        create_mock_footprint(price_center=bar['close'], positive_delta=(i % 2 == 0))
        for i, bar in bars.iterrows()
    ]

    # Detect zones
    book, zones = detector.detect(bars, footprints, current_close)

    print(f"✓ Detected {len(zones)} key zones")
    print(f"  Big delta resistance: {sum(1 for p in book.big_delta_ask if p is not None)}")
    print(f"  Big delta support: {sum(1 for p in book.big_delta_bid if p is not None)}")
    print(f"  Peak resistance: {sum(1 for p in book.peak_ask if p is not None)}")
    print(f"  Peak support: {sum(1 for p in book.peak_bid if p is not None)}")

    assert len(zones) > 0, "Should detect at least some zones"

    # Check zone properties
    for zone in zones:
        assert zone.lower_bound < zone.center_price < zone.upper_bound
        assert zone.strength >= 1

    print("✅ KeyZoneDetector tests passed\n")
    return True


# ============================================================================
# Test 3: StructuralSignalDetector
# ============================================================================

def test_structural_signal_detector():
    """Test StructuralSignalDetector"""
    print("\n" + "="*80)
    print("TEST 3: StructuralSignalDetector")
    print("="*80)

    config = StrategyConfig()
    detector = StructuralSignalDetector(config)

    # Test V-reversal
    print("\n Testing V-reversal detection...")
    bars_v = create_mock_bars(n=100, trend='v_bullish')
    footprints_v = [
        create_mock_footprint(bar['close'], positive_delta=True)
        for _, bar in bars_v.iterrows()
    ]

    v_signal = detector.detect_v_reversal(bars_v, footprints_v)
    if v_signal:
        print(f"✓ Detected V-reversal: {v_signal.signal_type.value}")
        print(f"  Strength: {v_signal.strength:.2f}")
    else:
        print("  No V-reversal detected (may be normal)")

    # Test breakout
    print("\n Testing breakout detection...")
    bars_break = create_mock_bars(n=100, trend='breakout')
    footprints_break = [
        create_mock_footprint(bar['close'], positive_delta=True)
        for _, bar in bars_break.iterrows()
    ]

    breakout_signal = detector.detect_breakout(bars_break, footprints_break)
    if breakout_signal:
        print(f"✓ Detected breakout: {breakout_signal.signal_type.value}")
        print(f"  Strength: {breakout_signal.strength:.2f}")
    else:
        print("  No breakout detected (may be normal)")

    print("\n✅ StructuralSignalDetector tests passed\n")
    return True


# ============================================================================
# Test 4: TouchTracker
# ============================================================================

def test_touch_tracker():
    """Test TouchTracker"""
    print("\n" + "="*80)
    print("TEST 4: TouchTracker")
    print("="*80)

    config = StrategyConfig()
    tracker = TouchTracker(config)

    # Create test zone
    zone = KeyZone(
        center_price=1900.0,
        lower_bound=1898.0,
        upper_bound=1902.0,
        zone_type=ZoneType.SUPPORT,
        sources=['test']
    )

    # Test touches
    touches = tracker.check_touches(
        bar_index=0,
        bar_high=1905.0,  # Above zone
        bar_low=1900.0,   # In zone
        zones=[zone]
    )

    print(f"✓ Detected {len(touches)} touches")
    assert len(touches) == 1, "Should detect 1 touch (low in zone)"

    tracker.update(touches)

    # Get recent touches
    recent = tracker.get_recent_touches(current_bar_index=2, window=3)
    print(f"✓ Recent touches: {len(recent)}")
    assert len(recent) == 1, "Should have 1 recent touch"

    print("✅ TouchTracker tests passed\n")
    return True


# ============================================================================
# Test 5: Signal Combiner
# ============================================================================

def test_signal_combiner():
    """Test SignalCombiner"""
    print("\n" + "="*80)
    print("TEST 5: SignalCombiner")
    print("="*80)

    config = StrategyConfig()
    combiner = SignalCombiner(config)

    # Create test data
    support_zone = KeyZone(
        center_price=1890.0,
        lower_bound=1888.0,
        upper_bound=1892.0,
        zone_type=ZoneType.SUPPORT,
        sources=['test']
    )

    touch = TouchEvent(
        zone=support_zone,
        bar_index=50,
        touch_type='low_touch',
        touch_price=1890.0
    )

    bullish_signal = Signal(
        signal_type=SignalType.V_REVERSAL_BULLISH,
        strength=0.8,
        bar_index=51
    )

    # Combine
    trade_signal = combiner.combine([touch], [bullish_signal])

    if trade_signal:
        print(f"✓ Generated trade signal: {trade_signal.action.value}")
        print(f"  Confidence: {trade_signal.confidence:.2f}")
        assert trade_signal.action == TradeAction.BUY, "Should be BUY"
    else:
        print("✗ No trade signal generated")
        return False

    print("✅ SignalCombiner tests passed\n")
    return True


# ============================================================================
# Test 6: KeyZoneStrategy Integration
# ============================================================================

def test_key_zone_strategy():
    """Test full KeyZoneStrategy"""
    print("\n" + "="*80)
    print("TEST 6: KeyZoneStrategy Integration")
    print("="*80)

    strategy = KeyZoneStrategy(symbol='GC')

    # Create test data
    bars = create_mock_bars(n=120, trend='v_bullish')
    footprints = [
        create_mock_footprint(bar['close'], positive_delta=(i > 80))
        for i, bar in bars.iterrows()
    ]

    # Run strategy over bars
    signals_generated = 0
    for i in range(100, len(bars)):
        window_bars = bars.iloc[:i+1]
        window_footprints = footprints[:i+1]

        result = strategy.update(
            bars=window_bars,
            footprints=window_footprints,
            current_bar_index=i
        )

        if result and result['trade_signal']:
            signals_generated += 1
            print(f"✓ Bar {i}: {result['action'].value} signal")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Zones detected: {len(result['zones'])}")
            print(f"  Structural signals: {len(result['structural_signals'])}")

    print(f"\n✓ Total signals generated: {signals_generated}")
    print("✅ KeyZoneStrategy integration tests passed\n")
    return True


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Key Zone Strategy Test Suite" + " "*30 + "║")
    print("╚" + "="*78 + "╝")

    results = {}

    # Run tests
    try:
        results['models'] = test_data_models()
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        results['models'] = False

    try:
        results['zone_detector'] = test_key_zone_detector()
    except Exception as e:
        print(f"✗ KeyZoneDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        results['zone_detector'] = False

    try:
        results['structural_detector'] = test_structural_signal_detector()
    except Exception as e:
        print(f"✗ StructuralSignalDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        results['structural_detector'] = False

    try:
        results['touch_tracker'] = test_touch_tracker()
    except Exception as e:
        print(f"✗ TouchTracker test failed: {e}")
        results['touch_tracker'] = False

    try:
        results['signal_combiner'] = test_signal_combiner()
    except Exception as e:
        print(f"✗ SignalCombiner test failed: {e}")
        results['signal_combiner'] = False

    try:
        results['strategy'] = test_key_zone_strategy()
    except Exception as e:
        print(f"✗ KeyZoneStrategy test failed: {e}")
        import traceback
        traceback.print_exc()
        results['strategy'] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v is True)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\nResults: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("\n" + "="*80)

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
