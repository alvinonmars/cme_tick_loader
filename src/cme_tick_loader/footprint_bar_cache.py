"""
Footprint Bar 数据缓存

核心设计：
1. 缓存 FootprintBarData（已计算 VWAP/POC/VAH/VAL）
2. append/merge 验证 metadata 一致性
3. 纯内存缓存，数据源可插拔
4. 支持按交易时长加载（支持模糊时间戳匹配）
"""

from contextlib import nullcontext
from typing import Dict, Tuple, Callable, Union
import pandas as pd
import threading
import logging

from .footprint_bar_data import FootprintBarData

logger = logging.getLogger(__name__)

# Type aliases
CacheKey = Tuple[str, str, int]  # (symbol, resolution, num_units)
LoaderFunc = Callable[[str, str, int, pd.Timestamp, pd.Timestamp], Tuple[pd.DataFrame, pd.DataFrame]]


class FootprintBarDataCache:
    """
    Footprint Bar 数据缓存（纯内存）

    特性：
    - 缓存已计算的 FootprintBarData（包含 VWAP/POC/VAH/VAL）
    - append/merge 验证 metadata 一致性
    - 时间范围查询
    - 可选线程安全

    Example:
        >>> def loader(symbol, resolution, num_units, start, end):
        ...     result = cme_loader.load_date_range(symbol, ...)
        ...     return result['bars'], result['footprint']
        >>>
        >>> cache = FootprintBarDataCache(loader)
        >>> bar_data = cache.get('GC', 'MIN', 5, start_time, end_time)
    """

    def __init__(self, loader_func: LoaderFunc, thread_safe: bool = False):
        """
        初始化缓存

        Args:
            loader_func: 数据源函数，签名：
                (symbol, resolution, num_units, start_time, end_time)
                -> (bars_df, footprint_df)
            thread_safe: 是否启用线程安全（默认 False）
        """
        self._cache: Dict[CacheKey, FootprintBarData] = {}
        self._loader = loader_func
        self._lock = threading.RLock() if thread_safe else None

    def get(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        start_time: Union[pd.Timestamp, int],
        end_time: Union[pd.Timestamp, int],
        use_cache: bool = True
    ) -> FootprintBarData:
        """
        获取数据（返回 FootprintBarData）

        Args:
            symbol: 交易符号
            resolution: bar 类型
            num_units: 时间单位或 threshold
            start_time: 开始时间
            end_time: 结束时间
            use_cache: 是否使用缓存

        Returns:
            FootprintBarData 实例

        Raises:
            ValueError: 参数无效
        """
        with self._lock if self._lock else nullcontext():
            # 验证输入
            if num_units <= 0:
                raise ValueError(f"num_units must be positive, got {num_units}")

            key = (symbol, resolution, num_units)
            start = self._normalize_timestamp(start_time)
            end = self._normalize_timestamp(end_time)

            if start > end:
                raise ValueError(f"start_time ({start}) must be <= end_time ({end})")

            # Level 1: 检查缓存
            if use_cache and key in self._cache:
                cached = self._cache[key]
                if self._has_complete_coverage(cached, start, end):
                    extracted = self._extract_range(cached, start, end)
                    # 🔥 纵深防御：即使coverage检查通过，仍需验证提取结果非空
                    # 防止边缘情况下的空数据（如缓存空洞、切片错误）
                    if not extracted.bars.empty:
                        return extracted
                    # 提取为空，fallback到数据源加载

            # Level 2: 从数据源加载
            bars, footprint = self._loader(symbol, resolution, num_units, start, end)

            # 空数据
            if bars.empty:
                return self._empty_bar_data(resolution, num_units)

            # 创建 FootprintBarData
            bar_data = FootprintBarData.from_result(
                {'bars': bars, 'footprint': footprint},
                resolution=resolution,
                num_units=num_units
            )

            # 合并到缓存
            if use_cache:
                self._merge_to_cache(key, bar_data)
                # 从缓存提取范围
                return self._extract_range(self._cache[key], start, end)
            else:
                return bar_data

    def append(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        bar_data: FootprintBarData
    ) -> None:
        """
        追加数据（实时更新）

        验证 metadata 一致性，去重（keep='last'）

        Args:
            symbol: 交易符号
            resolution: bar 类型
            num_units: 时间单位
            bar_data: 要追加的数据

        Raises:
            ValueError: metadata 不一致
        """
        with self._lock if self._lock else nullcontext():
            self._validate_metadata(bar_data, resolution, num_units)
            self._merge_internal(symbol, resolution, num_units, bar_data, strategy='keep_last')

    def merge(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        bar_data: FootprintBarData,
        strategy: str = 'keep_last'
    ) -> None:
        """
        合并数据（批量更新）

        验证 metadata 一致性

        Args:
            symbol: 交易符号
            resolution: bar 类型
            num_units: 时间单位
            bar_data: 要合并的数据
            strategy: 'keep_last' 或 'keep_first'

        Raises:
            ValueError: metadata 不一致或 strategy 无效
        """
        with self._lock if self._lock else nullcontext():
            self._validate_metadata(bar_data, resolution, num_units)
            if strategy not in ('keep_last', 'keep_first'):
                raise ValueError(f"strategy must be 'keep_last' or 'keep_first', got '{strategy}'")
            self._merge_internal(symbol, resolution, num_units, bar_data, strategy)

    def delete(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        start_time: pd.Timestamp = None,
        end_time: pd.Timestamp = None
    ) -> None:
        """删除数据（整个 key 或时间范围）"""
        with self._lock if self._lock else nullcontext():
            key = (symbol, resolution, num_units)

            if key not in self._cache:
                return

            # 删除整个 key
            if start_time is None or end_time is None:
                del self._cache[key]
                return

            # 删除时间范围
            cached = self._cache[key]
            start = self._normalize_timestamp(start_time)
            end = self._normalize_timestamp(end_time)

            # 过滤掉范围内的数据
            bars = cached.bars[~((cached.bars.index >= start) & (cached.bars.index <= end))]
            footprint = cached.footprint_df
            if not footprint.empty:
                idx_level0 = footprint.index.get_level_values(0)
                mask = ~((idx_level0 >= start) & (idx_level0 <= end))
                footprint = footprint[mask]

            # 重新创建 FootprintBarData
            if not bars.empty:
                self._cache[key] = FootprintBarData(
                    bars=bars,
                    footprint_df=footprint,
                    resolution=cached.resolution,
                    num_units=cached.num_units
                )
            else:
                del self._cache[key]

    def clear_key(
        self,
        symbol: str,
        resolution: str = None,
        num_units: int = None
    ) -> None:
        """清除缓存（结构化清除）"""
        with self._lock if self._lock else nullcontext():
            if resolution is None:
                keys_to_delete = [k for k in self._cache.keys() if k[0] == symbol]
            elif num_units is None:
                keys_to_delete = [
                    k for k in self._cache.keys()
                    if k[0] == symbol and k[1] == resolution
                ]
            else:
                keys_to_delete = [(symbol, resolution, num_units)]

            for key in keys_to_delete:
                if key in self._cache:
                    del self._cache[key]

    def get_stats(self) -> dict:
        """获取缓存统计"""
        stats = {
            'total_keys': len(self._cache),
            'entries': {}
        }

        for (symbol, resolution, num_units), bar_data in self._cache.items():
            key_str = f"{symbol}_{resolution}_{num_units}"

            entry_stats = {
                'bars_count': len(bar_data.bars),
                'footprint_count': len(bar_data.footprint_df),
                'start_time': str(bar_data.bars.index.min()) if not bar_data.bars.empty else None,
                'end_time': str(bar_data.bars.index.max()) if not bar_data.bars.empty else None,
            }

            # 内存使用
            bars_mem = bar_data.bars.memory_usage(deep=True).sum()
            footprint_mem = bar_data.footprint_df.memory_usage(deep=True).sum() if not bar_data.footprint_df.empty else 0
            entry_stats['memory_usage_mb'] = (bars_mem + footprint_mem) / (1024 * 1024)

            stats['entries'][key_str] = entry_stats

        return stats

    def has_data(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> bool:
        """检查缓存是否有完整覆盖"""
        key = (symbol, resolution, num_units)
        if key not in self._cache:
            return False

        start = self._normalize_timestamp(start_time)
        end = self._normalize_timestamp(end_time)

        return self._has_complete_coverage(self._cache[key], start, end)

    def get_by_trading_hours(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        end_time: Union[pd.Timestamp, int],
        trading_hours: float,
        tolerance_seconds: float = 0.0,
        normal_gap_threshold_hours: float = 5.0,
        trading_time_tolerance: float = 0.8,
        max_load_attempts: int = 5,
        use_cache: bool = True
    ) -> FootprintBarData:
        """
        按交易时长加载数据（支持模糊时间戳匹配）

        核心功能：
        1. 将交易时长需求转换为日历时间窗口
        2. 迭代调整窗口大小，确保加载到足够的交易时长
        3. 支持模糊时间戳匹配（容差 > 0 时）

        Args:
            symbol: 交易品种
            resolution: 时间分辨率
            num_units: 单位数量
            end_time: 数据终点时间（可以是 pd.Timestamp 或 Unix 时间戳 ms）
            trading_hours: 需要的交易时长（小时）
            tolerance_seconds: 时间戳匹配容差（秒），0 表示精确匹配
            normal_gap_threshold_hours: 正常交易间隔阈值（小时）
            trading_time_tolerance: 交易时长容差（0.8 = 80%即可）
            max_load_attempts: 最大加载尝试次数
            use_cache: 是否使用缓存

        Returns:
            FootprintBarData: 包含bars和footprint的数据对象

        算法：
            1. 如果 tolerance_seconds > 0，进行模糊匹配（获取最新时间戳）
            2. 初始估算：calendar_lookback = trading_hours（乐观估计）
            3. 迭代加载：
               - 调用 self.get 加载数据
               - 计算实际交易时长
               - 判断是否满足要求
            4. 不足则扩大窗口：+1天
            5. 如果数据过多，切片到精确的交易时长

        Example:
            >>> # 精确匹配（与原实现完全一致）
            >>> data = cache.get_by_trading_hours(
            ...     'GC', 'MIN', 5, end_time, trading_hours=48
            ... )
            >>>
            >>> # 模糊匹配（容差30秒）
            >>> data = cache.get_by_trading_hours(
            ...     'GC', 'MIN', 5, end_time, trading_hours=48,
            ...     tolerance_seconds=30
            ... )
        """
        with self._lock if self._lock else nullcontext():
            # Step 1: 标准化 end_time
            end_time = self._normalize_timestamp(end_time)

            # Step 2: 如果 tolerance > 0，进行模糊匹配
            if tolerance_seconds > 0:
                actual_end_time = self._find_nearest_end_time(
                    symbol, resolution, num_units, end_time,
                    tolerance_seconds, use_cache
                )
                logger.info(
                    f"Fuzzy timestamp match: {end_time} → {actual_end_time} "
                    f"(tolerance: {tolerance_seconds}s)"
                )
            else:
                actual_end_time = end_time

            # Step 3: 迭代加载（完全复制原逻辑）
            return self._load_by_trading_hours_iterative(
                symbol, resolution, num_units, actual_end_time,
                trading_hours, normal_gap_threshold_hours,
                trading_time_tolerance, max_load_attempts, use_cache
            )

    # ===== Private methods =====

    def _find_nearest_end_time(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        target_time: pd.Timestamp,
        tolerance_seconds: float,
        use_cache: bool
    ) -> pd.Timestamp:
        """
        模糊匹配：找到距离target最近的时间戳（在容差内）

        策略：
        1. 加载 target 前后各1小时的数据（足够大的窗口）
        2. 找到距离 target 最近的时间戳（<= target）
        3. 验证距离是否在容差内

        注意：
        - 此方法在 get_by_trading_hours() 的锁内部调用
        - 内部会调用 self.get()，导致锁重入（RLock支持）

        Args:
            target_time: 目标时间
            tolerance_seconds: 容差（秒）
            use_cache: 是否使用缓存

        Returns:
            最接近的时间戳（<= target）

        Raises:
            ValueError: 容差范围内找不到数据
        """
        # 加载较大的窗口（target 前后各1小时）
        window_start = target_time - pd.Timedelta(hours=1)
        window_end = target_time + pd.Timedelta(hours=1)

        logger.debug(
            f"Fuzzy timestamp matching: loading window "
            f"[{window_start}, {window_end}] (tolerance: {tolerance_seconds}s)"
        )

        # 调用 self.get()
        # 注意：这会导致锁重入（RLock支持），性能开销可接受
        recent_data = self.get(
            symbol, resolution, num_units,
            window_start, window_end,
            use_cache=use_cache
        )

        if recent_data.bars.empty:
            raise ValueError(
                f"No data available in window [{window_start}, {window_end}]"
            )

        # 找到 <= target 的最近时间戳
        valid_times = recent_data.bars.index[recent_data.bars.index <= target_time]

        if len(valid_times) == 0:
            raise ValueError(
                f"No bars found before or at target time {target_time}"
            )

        # 获取最接近的时间戳
        nearest_time = valid_times[-1]

        # 验证容差
        time_diff = abs((target_time - nearest_time).total_seconds())
        if time_diff <= tolerance_seconds:
            logger.info(
                f"Fuzzy match: {target_time} → {nearest_time} (diff: {time_diff:.1f}s)"
            )
            return nearest_time
        else:
            raise ValueError(
                f"Nearest bar at {nearest_time} is {time_diff:.1f}s away from target {target_time}, "
                f"exceeds tolerance {tolerance_seconds}s"
            )

    def _load_by_trading_hours_iterative(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        end_time: pd.Timestamp,
        required_trading_hours: float,
        normal_gap_threshold_hours: float,
        trading_time_tolerance: float,
        max_load_attempts: int,
        use_cache: bool
    ) -> FootprintBarData:
        """
        迭代加载直到满足交易时长

        完全复制 indicator_server.load_data_by_trading_hours 的逻辑

        算法：
        1. 初始估算：calendar_lookback = required_trading_hours
        2. 迭代加载：
           - 调用 self.get() 加载数据
           - 计算实际交易时长
           - 判断是否满足要求
        3. 不足则扩大窗口：+1天
        4. 如果数据过多，切片到精确的交易时长
        """
        # 初始估算：乐观假设所有时间都是交易时间
        calendar_lookback = pd.Timedelta(hours=required_trading_hours)
        normal_threshold = pd.Timedelta(hours=normal_gap_threshold_hours)

        logger.info(
            f"Loading data by trading hours: {required_trading_hours:.1f}h "
            f"(tolerance: {trading_time_tolerance})"
        )

        # 迭代加载：不够就+1天
        bar_data = None
        actual_hours = 0.0

        for attempt in range(max_load_attempts):
            load_start = end_time - calendar_lookback

            logger.info(
                f"Attempt {attempt + 1}/{max_load_attempts}: "
                f"Loading {calendar_lookback.days}d {calendar_lookback.seconds//3600}h calendar window "
                f"from [{load_start}] to [{end_time}]"
            )

            # 通过缓存层加载数据（纯日历时间）
            bar_data = self.get(
                symbol=symbol,
                resolution=resolution,
                num_units=num_units,
                start_time=load_start,
                end_time=end_time,
                use_cache=use_cache
            )

            if len(bar_data.bars) == 0:
                logger.warning(f"No data loaded in attempt {attempt + 1}")
                calendar_lookback += pd.Timedelta(days=1)
                continue

            # 计算实际交易时长
            actual_hours = self._calculate_trading_hours(bar_data.bars, normal_threshold)

            logger.info(
                f"  → Loaded {len(bar_data.bars)} bars, "
                f"{actual_hours:.1f}h trading time (required: {required_trading_hours:.1f}h)"
            )

            # 检查是否满足容差
            if actual_hours >= required_trading_hours * trading_time_tolerance:
                logger.info(f"✓ Sufficient trading hours")

                # 关键：第2+次尝试可能加载过多，需要切片到精确的交易时长
                if attempt > 0 and actual_hours > required_trading_hours:
                    logger.info(f"Slicing to required trading hours ({required_trading_hours:.1f}h)...")
                    sliced_bars, sliced_footprint = self._slice_to_trading_hours(
                        bar_data.bars, bar_data.footprint_df,
                        required_trading_hours, normal_threshold
                    )
                    # 创建切片后的 FootprintBarData
                    bar_data = FootprintBarData(
                        bars=sliced_bars,
                        footprint_df=sliced_footprint,
                        resolution=resolution,
                        num_units=num_units
                    )
                    # 打印切片后数据范围和时长
                    actual_hours = self._calculate_trading_hours(bar_data.bars, normal_threshold)
                    logger.info(
                        f"Sliced data: {len(bar_data.bars)} bars, "
                        f"{actual_hours:.1f}h trading time "
                        f"from [{bar_data.bars.index[0]} to {bar_data.bars.index[-1]}]"
                    )

                return bar_data

            # 不够就+1天
            calendar_lookback += pd.Timedelta(days=1)
            logger.info(f"  → Insufficient, expanding window to {calendar_lookback.days}d")

        # 达到最大尝试次数，返回最后一次的数据（即使不够也返回）
        if bar_data is not None and len(bar_data.bars) > 0:
            logger.warning(
                f"Reached max attempts ({max_load_attempts}), "
                f"returning data with {actual_hours:.1f}h (required: {required_trading_hours:.1f}h)"
            )
            return bar_data
        else:
            logger.error(f"Failed to load any data after {max_load_attempts} attempts")
            # 返回空的 FootprintBarData
            return self._empty_bar_data(resolution, num_units)

    @staticmethod
    def _calculate_trading_hours(bars: pd.DataFrame, normal_threshold: pd.Timedelta) -> float:
        """
        计算实际交易时长（排除周末/节假日大间隔）

        策略：只累加相邻bar之间的"正常"间隔（<阈值）

        Args:
            bars: bar数据
            normal_threshold: 正常间隔阈值

        Returns:
            交易时长（小时）
        """
        if len(bars) < 2:
            return 0.0

        # 计算相邻bar的时间间隔
        time_diffs = bars.index.to_series().diff().dropna()

        # 只累加"正常"间隔，过滤周末/节假日的大间隔
        normal_intervals = time_diffs[time_diffs < normal_threshold]

        trading_hours = normal_intervals.sum().total_seconds() / 3600

        return trading_hours

    @staticmethod
    def _slice_to_trading_hours(
        bars: pd.DataFrame,
        footprint: pd.DataFrame,
        required_hours: float,
        normal_threshold: pd.Timedelta
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        切片到指定的交易时长（从最新向前回溯）

        策略：从最新的bar向前累计交易时间，直到达到required_hours

        Args:
            bars: bar数据
            footprint: footprint数据
            required_hours: 需要的交易时长
            normal_threshold: 正常间隔阈值

        Returns:
            (切片后的bars, 切片后的footprint)
        """
        if len(bars) < 2:
            return bars, footprint

        # 从后往前累计交易时间
        accumulated_hours = 0.0
        cutoff_index = 0

        for i in range(len(bars) - 1, 0, -1):
            gap = bars.index[i] - bars.index[i - 1]

            # 只累加正常间隔
            if gap < normal_threshold:
                accumulated_hours += gap.total_seconds() / 3600

            # 达到要求的交易时长
            if accumulated_hours >= required_hours:
                cutoff_index = i - 1
                break

        # 切片 bars
        sliced_bars = bars.iloc[cutoff_index:]

        # 切片 footprint（保持一致）
        sliced_footprint = footprint[
            footprint.index.get_level_values(0).isin(sliced_bars.index)
        ]

        logger.info(
            f"Sliced: {len(bars)} → {len(sliced_bars)} bars "
            f"({accumulated_hours:.1f}h trading time)"
        )

        return sliced_bars, sliced_footprint

    @staticmethod
    def _validate_metadata(bar_data: FootprintBarData, resolution: str, num_units: int) -> None:
        """验证 metadata 一致性"""
        if bar_data.resolution != resolution:
            raise ValueError(
                f"Resolution mismatch: bar_data.resolution='{bar_data.resolution}' != '{resolution}'"
            )
        if bar_data.num_units != num_units:
            raise ValueError(
                f"num_units mismatch: bar_data.num_units={bar_data.num_units} != {num_units}"
            )

    def _merge_internal(
        self,
        symbol: str,
        resolution: str,
        num_units: int,
        bar_data: FootprintBarData,
        strategy: str = 'keep_last'
    ) -> None:
        """内部合并（无锁，假设调用者持有锁）"""
        key = (symbol, resolution, num_units)

        if bar_data.bars.empty:
            return

        if key not in self._cache:
            self._cache[key] = bar_data
        else:
            cached = self._cache[key]
            # 验证兼容性
            cached.validate_compatible(bar_data)

            # 合并 bars 和 footprint
            merged_bars = self._merge_dataframes(cached.bars, bar_data.bars, strategy)
            merged_footprint = self._merge_dataframes(cached.footprint_df, bar_data.footprint_df, strategy)

            # 创建新的 FootprintBarData
            self._cache[key] = FootprintBarData(
                bars=merged_bars,
                footprint_df=merged_footprint,
                resolution=resolution,
                num_units=num_units
            )

    def _merge_to_cache(self, key: CacheKey, bar_data: FootprintBarData) -> None:
        """合并到缓存"""
        symbol, resolution, num_units = key
        self._merge_internal(symbol, resolution, num_units, bar_data, strategy='keep_last')

    @staticmethod
    def _merge_dataframes(
        old: pd.DataFrame,
        new: pd.DataFrame,
        strategy: str = 'keep_last'
    ) -> pd.DataFrame:
        """合并并去重 DataFrames"""
        if old.empty:
            return new.copy()
        if new.empty:
            return old.copy()

        combined = pd.concat([old, new])
        keep = 'last' if strategy == 'keep_last' else 'first'
        deduplicated = combined[~combined.index.duplicated(keep=keep)].copy()

        return deduplicated.sort_index()

    @staticmethod
    def _normalize_timestamp(ts: Union[pd.Timestamp, int]) -> pd.Timestamp:
        """转换为 timezone-naive Timestamp"""
        if isinstance(ts, int):
            return pd.Timestamp(ts, unit='ms').tz_localize(None)
        elif isinstance(ts, pd.Timestamp):
            if ts.tz is None:
                return ts
            else:
                return ts.replace(tzinfo=None)
        else:
            raise ValueError(f"Unsupported timestamp type: {type(ts)}")

    @staticmethod
    def _has_complete_coverage(
        bar_data: FootprintBarData,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> bool:
        """
        检查是否完整覆盖 [start, end]

        关键修复：不仅检查边界，还要检查范围内是否有实际数据
        防止缓存空洞导致误判
        """
        if bar_data.bars.empty:
            return False

        # 提取查询范围内的实际数据
        range_bars = bar_data.bars.loc[start:end]

        # 范围内必须有数据
        if range_bars.empty:
            return False

        # 检查范围内数据的边界是否覆盖查询范围
        range_start = range_bars.index.min()
        range_end = range_bars.index.max()

        return range_start <= start and range_end >= end

    @staticmethod
    def _extract_range(
        bar_data: FootprintBarData,
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> FootprintBarData:
        """从缓存提取 [start, end] 范围"""
        bars = bar_data.bars.loc[start:end].copy()

        if not bar_data.footprint_df.empty:
            footprint = bar_data.footprint_df.loc[start:end, :].copy()
        else:
            footprint = bar_data.footprint_df.copy()

        return FootprintBarData(
            bars=bars,
            footprint_df=footprint,
            resolution=bar_data.resolution,
            num_units=bar_data.num_units
        )

    @staticmethod
    def _empty_bar_data(resolution: str, num_units: int) -> FootprintBarData:
        """返回空的 FootprintBarData"""
        empty_bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        empty_bars.index = pd.DatetimeIndex([], name='timestamp')

        empty_footprint = pd.DataFrame(columns=['bid_vol', 'ask_vol', 'total_vol', 'delta'])
        empty_footprint.index = pd.MultiIndex.from_tuples([], names=['timestamp', 'price'])

        return FootprintBarData(
            bars=empty_bars,
            footprint_df=empty_footprint,
            resolution=resolution,
            num_units=num_units
        )
