"""Test CMEBarsLoader functionality"""

import pandas as pd
from cme_tick_loader import CMEBarsLoader


def test_basic_time_bars():
    """测试基本的时间 bars 功能"""
    loader = CMEBarsLoader()

    # 加载 5 分钟 time bars
    result = loader.load_bars(
        symbol='GC',
        date='20210104',
        resolution='MIN',
        num_units=5,
        use_cache=False,  # 第一次不使用缓存
        verbose=False
    )

    # 验证返回格式
    assert isinstance(result, dict)
    assert 'bars' in result
    assert 'footprint' in result

    # 验证 bars DataFrame
    bars = result['bars']
    assert isinstance(bars, pd.DataFrame)
    assert 'date_time' in bars.columns
    assert 'open' in bars.columns
    assert 'high' in bars.columns
    assert 'low' in bars.columns
    assert 'close' in bars.columns

    # 验证 date_time 是 datetime 对象（TIME bars 已修复）
    assert isinstance(bars['date_time'].iloc[0], pd.Timestamp)
    assert bars['date_time'].dt.tz is None  # 默认 timezone_naive=True，无 timezone

    # 验证 footprint DataFrame
    footprint = result['footprint']
    assert isinstance(footprint, pd.DataFrame)
    assert footprint.index.names == ['bar_timestamp', 'price']
    assert 'bid_vol' in footprint.columns
    assert 'ask_vol' in footprint.columns
    assert 'is_open' in footprint.columns
    assert 'is_high' in footprint.columns
    assert 'is_low' in footprint.columns
    assert 'is_close' in footprint.columns

    # 验证 footprint 的 timestamp 是 datetime
    assert isinstance(footprint.index.levels[0][0], pd.Timestamp)
    assert footprint.index.levels[0].tz is None  # 默认 timezone_naive=True，无 timezone

    print("✅ TIME bars 测试通过")


def test_cache_functionality():
    """测试缓存功能"""
    loader = CMEBarsLoader()

    # 第一次加载（无缓存）
    result1 = loader.load_bars(
        symbol='GC',
        date='20210104',
        resolution='MIN',
        num_units=5,
        use_cache=True,
        refresh_cache=False,
        verbose=False
    )

    # 第二次加载（命中缓存）
    result2 = loader.load_bars(
        symbol='GC',
        date='20210104',
        resolution='MIN',
        num_units=5,
        use_cache=True,
        refresh_cache=False,
        verbose=False
    )

    # 验证缓存结果一致
    pd.testing.assert_frame_equal(result1['bars'], result2['bars'])
    pd.testing.assert_frame_equal(result1['footprint'], result2['footprint'])

    # 验证缓存信息
    cache_info = loader.get_cache_info()
    assert cache_info['result_cache']['count'] > 0
    assert cache_info['total_size_mb'] > 0

    print("✅ 缓存功能测试通过")


def test_volume_bars():
    """测试 VOLUME bars"""
    loader = CMEBarsLoader()

    result = loader.load_bars(
        symbol='GC',
        date='20210104',
        resolution='VOLUME',
        num_units=10000,  # 每 10000 成交量采样
        use_cache=False,
        verbose=False
    )

    assert isinstance(result, dict)
    assert 'bars' in result
    assert 'footprint' in result

    # VOLUME bars 的 timestamp 应该已经是 datetime（不需要 fix）
    bars = result['bars']
    assert isinstance(bars['date_time'].iloc[0], pd.Timestamp)

    print("✅ VOLUME bars 测试通过")


def test_different_resolutions():
    """测试不同的 resolution"""
    loader = CMEBarsLoader()

    test_cases = [
        ('MIN', 5),      # 5 分钟
        ('H', 1),        # 1 小时
        ('VOLUME', 10000),  # Volume bars
        ('TICK', 1000),     # Tick bars
    ]

    for resolution, num_units in test_cases:
        result = loader.load_bars(
            symbol='GC',
            date='20210104',
            resolution=resolution,
            num_units=num_units,
            use_cache=False,
            verbose=False
        )

        assert isinstance(result, dict)
        assert 'bars' in result
        assert 'footprint' in result

        print(f"✅ {resolution}:{num_units} 测试通过")


def test_date_range():
    """测试日期范围加载"""
    loader = CMEBarsLoader()

    result = loader.load_date_range(
        symbol='GC',
        start_date='20210104',
        end_date='20210106',
        resolution='MIN',
        num_units=5,
        use_cache=False,
        verbose=False
    )

    assert isinstance(result, dict)
    assert 'bars' in result
    assert 'footprint' in result

    # 应该包含多天数据
    bars = result['bars']
    assert len(bars) > 0

    print("✅ 日期范围加载测试通过")


def test_cache_key_format():
    """测试缓存键格式（可读性）"""
    loader = CMEBarsLoader()

    # 生成缓存键
    key = loader._get_result_cache_key('GC', '20210104', 'MIN', 5)

    # 验证格式
    assert key == 'GC_20210104_MIN_5'

    # 验证可读性（展开，不哈希）
    assert '_' in key
    assert '20210104' in key
    assert 'MIN' in key

    print("✅ 缓存键格式测试通过")


def test_clear_cache():
    """测试清除缓存"""
    loader = CMEBarsLoader()

    # 先生成一些缓存
    loader.load_bars('GC', '20210104', resolution='MIN', num_units=5, use_cache=True)

    # 清除 result cache
    loader.clear_cache(level='result')

    cache_info = loader.get_cache_info()
    assert cache_info['result_cache']['count'] == 0

    print("✅ 清除缓存测试通过")


if __name__ == '__main__':
    print("开始测试 CMEBarsLoader...\n")

    test_basic_time_bars()
    test_cache_functionality()
    test_volume_bars()
    test_different_resolutions()
    test_date_range()
    test_cache_key_format()
    test_clear_cache()

    print("\n✅ 所有测试通过！")
