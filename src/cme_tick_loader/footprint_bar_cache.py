"""
Footprint Bar 数据缓存

核心设计：
1. 缓存 FootprintBarData（已计算 VWAP/POC/VAH/VAL）
2. append/merge 验证 metadata 一致性
3. 纯内存缓存，数据源可插拔
"""

from contextlib import nullcontext
from typing import Dict, Tuple, Callable, Union
import pandas as pd
import threading

from .footprint_bar_data import FootprintBarData


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

    # ===== Private methods =====

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
