"""CME Bars Loader - 基于 mlfinlab API 的按天缓存封装"""

import pandas as pd
from pathlib import Path


class CMEBarsLoader:
    """
    CME Bars Loader - 基于 mlfinlab API 的按天缓存封装

    核心特性：
    1. 支持多种 bar 类型（time, volume, tick, dollar）
    2. 两级缓存（tick + result）
    3. 按天管理数据
    4. 预聚合避免重复索引
    5. 正确处理时间戳转换

    设计原则：
    - 简洁：不考虑并发
    - 可读：cache 文件名展开
    - 正确：确保 mlfinlab 一次性处理
    """

    def __init__(self, base_path=None):
        """
        初始化

        Args:
            base_path: CME 数据根目录（自动检测）
        """
        if base_path is None:
            from .config import get_default_base_path
            base_path = get_default_base_path()

        self.base_path = Path(base_path)

        # Tick cache 目录
        self.tick_cache_dir = self.base_path / '.cache' / 'ticks'
        self.tick_cache_dir.mkdir(parents=True, exist_ok=True)

        # Result cache 目录
        self.result_cache_dir = self.base_path / '.cache' / 'results'
        self.result_cache_dir.mkdir(parents=True, exist_ok=True)

        # TickLoader（负责加载原始 tick 数据）
        from .tick_loader import TickLoader
        self.tick_loader = TickLoader(base_path)

    # ========== 核心方法 ==========

    def load_bars(self, symbol, date, resolution='MIN', num_units=5,
                  use_cache=True, refresh_cache=False, verbose=False,
                  timezone_naive=True, set_index=True):
        """
        加载单日 bars 和 footprint

        Args:
            symbol: 交易符号 (e.g., 'GC', 'ES')
            date: 日期字符串 YYYYMMDD (e.g., '20210104')
            resolution: bar 类型和时间单位
                TIME bars: 'D', 'H', 'MIN', 'S'
                其他: 'VOLUME', 'TICK', 'DOLLAR'
            num_units:
                TIME bars: 时间单位数量 (e.g., 5 for 5分钟)
                其他: threshold (e.g., 10000 for volume bars)
            use_cache: 是否使用缓存
            refresh_cache: 是否刷新 result cache
                True: 跳过 result cache，重新聚合
                False: 优先使用 result cache
            verbose: 是否打印 mlfinlab 日志
            timezone_naive: 是否移除timezone信息 (default: True)
                True: datetime为naive（无timezone）
                False: datetime为aware（UTC timezone）
            set_index: 是否将date_time设为index (default: True)
                True: bars.index = bars.date_time.values，保留date_time列
                False: 使用默认整数索引

        Returns:
            dict: 始终返回完整的 bars 和 footprint
                {
                    'bars': pd.DataFrame,
                    'footprint': pd.DataFrame
                }

        示例:
            # 5分钟 time bars（默认：timezone-naive + datetime index）
            result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
            bars = result['bars']
            footprint = result['footprint']

            # 保留UTC timezone
            result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                                     timezone_naive=False)

            # 使用整数索引
            result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                                     set_index=False)
        """
        # Step 1: 生成 result cache key
        result_cache_key = self._get_result_cache_key(symbol, date, resolution, num_units)

        # Step 2: 检查 result cache
        if use_cache and not refresh_cache:
            cached_result = self._load_result_cache(result_cache_key)
            if cached_result is not None:
                # 从缓存加载后仍然应用datetime格式转换
                cached_result = self._process_datetime_format(cached_result, timezone_naive, set_index)
                return cached_result

        # Step 3: 加载 tick 数据（TickLoader 负责 tick cache）
        ticks = self.tick_loader.load_ticks(
            symbol=symbol,
            date=date,
            use_cache=use_cache,
            refresh_cache=refresh_cache
        )

        # Step 4: 转换为 mlfinlab 格式
        ticks_mlfinlab = self._convert_to_mlfinlab_format(ticks)

        # Step 5: 预聚合（按 date_time, price 累加）
        ticks_agg = self._aggregate_duplicates(ticks_mlfinlab)

        # Step 6: 调用 mlfinlab API (总是启用 footprint)
        result = self._call_mlfinlab_api(
            df=ticks_agg,
            resolution=resolution,
            num_units=num_units,
            verbose=verbose
        )

        # Step 7: 修复 time bar 的 timestamp
        bar_type = self._get_bar_type(resolution)
        if bar_type == 'time':
            result = self._fix_time_bar_timestamp(result)

        # Step 8: 保存 result cache
        if use_cache:
            self._save_result_cache(result, result_cache_key)

        # Step 9: 处理datetime格式（timezone和index）
        result = self._process_datetime_format(result, timezone_naive, set_index)

        return result

    def load_date_range(self, symbol, start_date, end_date, **kwargs):
        """
        加载时间范围内的 bars 和 footprint

        Args:
            symbol: 交易符号
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            **kwargs: 传递给 load_bars 的参数
                resolution, num_units, use_cache, verbose,
                timezone_naive, set_index

        Returns:
            dict: 合并后的结果（格式与 load_bars 一致）
                {
                    'bars': pd.DataFrame,
                    'footprint': pd.DataFrame
                }

        注意:
            - timezone_naive和set_index参数会透传给每日load_bars
            - 合并时bars使用ignore_index=True（多日数据重建整数索引）
            - 如果需要datetime index，会在合并后重新设置

        示例:
            # 默认：timezone-naive + datetime index
            result = loader.load_date_range(
                'GC', '20210104', '20210108',
                resolution='MIN', num_units=5
            )

            # 保留UTC timezone
            result = loader.load_date_range(
                'GC', '20210104', '20210108',
                resolution='MIN', num_units=5,
                timezone_naive=False
            )
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        all_bars = []
        all_footprints = []

        for date in dates:
            date_str = date.strftime('%Y%m%d')
            try:
                result = self.load_bars(symbol, date_str, **kwargs)
                all_bars.append(result['bars'])
                all_footprints.append(result['footprint'])

            except FileNotFoundError:
                # 跳过缺失日期（周末、节假日）
                continue

        # 合并结果
        if not all_bars:
            return {'bars': pd.DataFrame(), 'footprint': pd.DataFrame()}

        # 合并bars（ignore_index=True，重建整数索引）
        merged_bars = pd.concat(all_bars, ignore_index=True)

        # 如果set_index=True，需要在合并后重新设置datetime index
        set_index = kwargs.get('set_index', True)
        if set_index and 'date_time' in merged_bars.columns:
            merged_bars.index = merged_bars.date_time.values

        return {
            'bars': merged_bars,
            'footprint': pd.concat(all_footprints)
        }

    # ========== 核心内部方法 ==========

    @staticmethod
    def _convert_to_mlfinlab_format(cme_ticks):
        """
        将 CME 格式转换为 mlfinlab 格式

        关键：timestamp_ms → date_time (datetime对象, UTC)

        Args:
            cme_ticks: CME 格式 DataFrame
                列：timestamp_ms, price, bid_qty, ask_qty

        Returns:
            mlfinlab 格式 DataFrame
                列：date_time, price, bid_qty, ask_qty
        """
        # 转换 timestamp_ms 为 datetime 对象（UTC）
        date_time = pd.to_datetime(cme_ticks['timestamp_ms'], unit='ms', utc=True)

        # 创建 mlfinlab 格式（date_time 必须是第一列）
        return pd.DataFrame({
            'date_time': date_time,
            'price': cme_ticks['price'],
            'bid_qty': cme_ticks['bid_qty'],
            'ask_qty': cme_ticks['ask_qty']
        })

    @staticmethod
    def _aggregate_duplicates(df):
        """
        预聚合：按 (date_time, price) 分组，累加 bid_qty 和 ask_qty

        Args:
            df: mlfinlab 格式 DataFrame
                列：date_time, price, bid_qty, ask_qty

        Returns:
            聚合后的 DataFrame（同样格式）
        """
        # 按 (date_time, price) 分组聚合
        df_agg = df.groupby(['date_time', 'price'], as_index=False).agg({
            'bid_qty': 'sum',
            'ask_qty': 'sum'
        })

        # 确保列顺序
        df_agg = df_agg[['date_time', 'price', 'bid_qty', 'ask_qty']]

        return df_agg

    def _get_bar_type(self, resolution):
        """
        从 resolution 判断 bar type

        Args:
            resolution: 'D', 'H', 'MIN', 'S', 'VOLUME', 'TICK', 'DOLLAR'

        Returns:
            'time' | 'volume' | 'tick' | 'dollar'
        """
        if resolution in ['D', 'H', 'MIN', 'S']:
            return 'time'
        elif resolution == 'VOLUME':
            return 'volume'
        elif resolution == 'TICK':
            return 'tick'
        elif resolution == 'DOLLAR':
            return 'dollar'
        else:
            raise ValueError(f"Unknown resolution: {resolution}. "
                           f"Expected: 'D', 'H', 'MIN', 'S', 'VOLUME', 'TICK', 'DOLLAR'")

    def _call_mlfinlab_api(self, df, resolution, num_units, verbose):
        """
        调用对应的 mlfinlab API

        Args:
            df: 预聚合后的 DataFrame
            resolution: bar 类型
            num_units: 参数值
            verbose: 是否打印日志

        Returns:
            mlfinlab 返回的结果 (总是包含 footprint)
        """
        from mlfinlab.data_structures import (
            get_time_bars,
            get_volume_bars,
            get_tick_bars,
            get_dollar_bars
        )

        bar_type = self._get_bar_type(resolution)

        # 关键：batch_size 要大于单天数据量，确保一次性处理
        batch_size = 50000000

        # 总是启用 footprint
        if bar_type == 'time':
            result = get_time_bars(
                file_path_or_df=df,
                resolution=resolution,
                num_units=num_units,
                batch_size=batch_size,
                verbose=verbose,
                enable_footprint=True
            )
        elif bar_type == 'volume':
            result = get_volume_bars(
                file_path_or_df=df,
                threshold=num_units,
                batch_size=batch_size,
                verbose=verbose,
                enable_footprint=True
            )
        elif bar_type == 'tick':
            result = get_tick_bars(
                file_path_or_df=df,
                threshold=num_units,
                batch_size=batch_size,
                verbose=verbose,
                enable_footprint=True
            )
        elif bar_type == 'dollar':
            result = get_dollar_bars(
                file_path_or_df=df,
                threshold=num_units,
                batch_size=batch_size,
                verbose=verbose,
                enable_footprint=True
            )

        return result

    @staticmethod
    def _fix_time_bar_timestamp(result):
        """
        修复 time bars 的 Unix timestamp 为 datetime

        直接修改 result，in-place 操作

        Args:
            result: get_time_bars 返回的 dict {'bars': df, 'footprint': df}

        Returns:
            修改后的 result（同一对象）
        """
        # 修复 bars 的 date_time 列
        bars = result['bars']
        bars['date_time'] = pd.to_datetime(bars['date_time'], unit='s', utc=True)

        # 修复 footprint 的 MultiIndex 第一级（bar_timestamp）
        footprint = result['footprint']
        footprint.index = footprint.index.set_levels(
            pd.to_datetime(footprint.index.levels[0], unit='s', utc=True),
            level=0
        )

        result['footprint'] = footprint
        result['bars'] = bars

        return result

    @staticmethod
    def _process_datetime_format(result, timezone_naive, set_index):
        """
        处理datetime格式：timezone转换和index设置

        Args:
            result: {'bars': DataFrame, 'footprint': DataFrame}
            timezone_naive: 是否移除timezone
            set_index: 是否设置datetime为index

        Returns:
            处理后的result
        """
        bars = result['bars']
        footprint = result['footprint']

        # 1. 处理timezone
        if timezone_naive and bars['date_time'].dt.tz is not None:
            # 移除timezone（转为naive datetime）
            bars['date_time'] = bars['date_time'].dt.tz_localize(None)

            # footprint的MultiIndex第一级也要处理
            if footprint.index.levels[0].tz is not None:
                footprint.index = footprint.index.set_levels(
                    footprint.index.levels[0].tz_localize(None),
                    level=0
                )

        # 2. 设置index（关键：使用.values）
        if set_index:
            # 使用.values避免pandas自动对齐索引
            bars.index = bars.date_time.values
            # 保留date_time列（不删除）

        result['bars'] = bars
        result['footprint'] = footprint

        return result

    # ========== 缓存管理 ==========

    def _get_result_cache_key(self, symbol, date, resolution, num_units):
        """
        生成 result cache key（展开，不哈希）

        Args:
            symbol: 'GC'
            date: '20210104'
            resolution: 'MIN'
            num_units: 5

        Returns:
            'GC_20210104_MIN_5'
        """
        return f"{symbol}_{date}_{resolution}_{num_units}"

    def _load_result_cache(self, cache_key):
        """
        加载 result cache

        Args:
            cache_key: 'GC_20210104_MIN_5'

        Returns:
            cached result or None
        """
        cache_path = self.result_cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            return pd.read_pickle(cache_path)
        return None

    def _save_result_cache(self, result, cache_key):
        """
        保存 result cache（简单写入，不考虑并发）

        Args:
            result: {'bars': df, 'footprint': df} or DataFrame
            cache_key: 'GC_20210104_MIN_5'
        """
        cache_path = self.result_cache_dir / f"{cache_key}.pkl"
        pd.to_pickle(result, cache_path)

    def clear_cache(self, level='all'):
        """
        清除缓存

        Args:
            level: 'tick' | 'result' | 'all'
        """
        import shutil

        if level in ['tick', 'all']:
            if self.tick_cache_dir.exists():
                shutil.rmtree(self.tick_cache_dir)
                self.tick_cache_dir.mkdir(parents=True, exist_ok=True)

        if level in ['result', 'all']:
            if self.result_cache_dir.exists():
                shutil.rmtree(self.result_cache_dir)
                self.result_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_info(self):
        """
        获取缓存信息

        Returns:
            {
                'tick_cache': {'count': int, 'size_mb': float},
                'result_cache': {'count': int, 'size_mb': float},
                'total_size_mb': float
            }
        """
        def _get_dir_info(dir_path):
            if not dir_path.exists():
                return {'count': 0, 'size_mb': 0.0}

            files = list(dir_path.glob('*.pkl'))
            total_size = sum(f.stat().st_size for f in files)
            return {
                'count': len(files),
                'size_mb': total_size / (1024 * 1024)
            }

        tick_info = _get_dir_info(self.tick_cache_dir)
        result_info = _get_dir_info(self.result_cache_dir)

        return {
            'tick_cache': tick_info,
            'result_cache': result_info,
            'total_size_mb': tick_info['size_mb'] + result_info['size_mb']
        }

    def get_ticksize(self, symbol):
        """
        获取指定交易品种的ticksize

        Args:
            symbol: 交易符号 (e.g., 'GC', 'ES')

        Returns:
            float: ticksize值

        示例:
            ticksize = loader.get_ticksize('GC')  # 0.1
            ticksize = loader.get_ticksize('ES')  # 0.25
        """
        return self.tick_loader.get_ticksize(symbol)
