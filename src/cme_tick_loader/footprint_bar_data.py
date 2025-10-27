"""
Footprint Bar 数据格式 - 所有下游系统的统一输入

核心设计：
1. FootprintBarData = bars + footprint + metadata
2. 严格验证：bars.index == footprint.index.levels[0]
3. 避免重复计算：OHLC 由 mlfinlab 聚合，只计算增强指标
4. Metadata 记录：resolution, num_units 用于验证一致性
5. 切片能力：slice(index) 根据时间戳索引切片数据

工具函数：
- get_detection_data(): 计算检测窗口的时间戳索引（纯索引计算）
- find_nearest_timestamp(): 模糊查找最接近的时间戳

架构：
    CMEBarsLoader → FootprintBarData.from_result() → key_zone_detector / trading_system

使用示例：
    # 1. 加载数据
    result = loader.load_bars('GC', '20210104', 'MIN', 5)
    bar_data = FootprintBarData.from_result(result, 'MIN', 5)

    # 2. 计算检测窗口索引
    detection_index = get_detection_data(bar_data.bars.index, target_ts, 50)

    # 3. 切片数据
    if detection_index is not None:
        detection_data = bar_data.slice(detection_index)
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass


@dataclass
class FootprintBarData:
    """
    Footprint Bar 数据 - 所有下游系统的统一输入

    核心保证：
    1. bars.index == footprint.index.levels[0] （严格对齐）
    2. 无重复、已排序、timezone-naive
    3. 包含 metadata（resolution, num_units）用于验证

    列结构：
    - bars.index: DatetimeIndex (timezone-naive)
    - bars.columns: open, high, low, close, volume, [vwap, poc, vah, val]
    - footprint.index: MultiIndex(timestamp, price)
    - footprint.columns: bid_vol, ask_vol, total_vol, delta, is_open, is_high, is_low, is_close

    使用示例：
        >>> result = loader.load_bars('GC', '20210104', 'MIN', 5)
        >>> bar_data = FootprintBarData.from_result(result, resolution='MIN', num_units=5)
        >>> print(len(bar_data))
        >>> print(bar_data.bars[['vwap', 'poc', 'vah', 'val']].head())
    """

    # ===== 列定义（Schema - Single Source of Truth）=====

    # Bars 核心列（OHLCV，必需）
    BARS_CORE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

    # Bars 增强列（Volume Profile 指标，可选）
    BARS_ENHANCED_COLUMNS = ['vwap', 'poc', 'vah', 'val']

    # Bars 标准列（默认使用，包含所有列）
    BARS_STANDARD_COLUMNS = BARS_CORE_COLUMNS + BARS_ENHANCED_COLUMNS

    # Footprint 核心列（买卖量，必需）
    FOOTPRINT_CORE_COLUMNS = ['bid_vol', 'ask_vol']

    # Footprint 计算列（由核心列计算）
    FOOTPRINT_CALC_COLUMNS = ['total_vol', 'delta']

    # Footprint OHLC 标记列（价格层级标记）
    FOOTPRINT_OHLC_MARKERS = ['is_open', 'is_high', 'is_low', 'is_close']

    # Footprint 标准列（默认使用，包含所有列）
    FOOTPRINT_STANDARD_COLUMNS = (
        FOOTPRINT_CORE_COLUMNS +
        FOOTPRINT_CALC_COLUMNS +
        FOOTPRINT_OHLC_MARKERS
    )

    # ===== 数据字段 =====
    bars: pd.DataFrame
    footprint_df: pd.DataFrame
    resolution: str
    num_units: int

    def __post_init__(self):
        """数据验证：确保一致性"""
        if not self.bars.empty and not self.footprint_df.empty:
            # 验证 bars.index 是 DatetimeIndex
            if not isinstance(self.bars.index, pd.DatetimeIndex):
                raise TypeError("bars.index must be DatetimeIndex")

            # 验证 footprint 是 MultiIndex
            if not isinstance(self.footprint_df.index, pd.MultiIndex):
                raise TypeError("footprint_df.index must be MultiIndex")

            # 验证时间戳一致性
            bars_times = set(self.bars.index)
            footprint_times = set(self.footprint_df.index.get_level_values(0).unique())

            if bars_times != footprint_times:
                missing_in_footprint = bars_times - footprint_times
                missing_in_bars = footprint_times - bars_times
                raise ValueError(
                    f"Timestamp mismatch between bars and footprint. "
                    f"Missing in footprint: {len(missing_in_footprint)}, "
                    f"Missing in bars: {len(missing_in_bars)}"
                )

            # 验证无重复
            if self.bars.index.duplicated().any():
                raise ValueError("bars.index contains duplicate timestamps")

            # 验证排序
            if not self.bars.index.is_monotonic_increasing:
                raise ValueError("bars.index must be sorted in ascending order")

        # 验证 metadata
        if not self.resolution:
            raise ValueError("resolution cannot be empty")
        if self.num_units <= 0:
            raise ValueError(f"num_units must be positive, got {self.num_units}")

    def __len__(self) -> int:
        """返回 bar 数量"""
        return len(self.bars)

    def slice(self, index: pd.DatetimeIndex) -> 'FootprintBarData':
        """
        根据时间戳索引切片，返回新的 FootprintBarData

        Args:
            index: 要切片的时间戳索引（DatetimeIndex 或 list/array of timestamps）

        Returns:
            新的 FootprintBarData 实例

        Raises:
            ValueError: 如果 index 中有不存在的时间戳

        Example:
            >>> # 获取检测窗口的索引
            >>> detection_index = get_detection_data(bar_data.bars.index, target_ts, 50)
            >>> # 切片数据
            >>> detection_data = bar_data.slice(detection_index)
        """
        # 转换为 DatetimeIndex（如果不是）
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)

        # 验证所有时间戳都存在
        missing = index.difference(self.bars.index)
        if len(missing) > 0:
            raise ValueError(
                f"Index contains {len(missing)} timestamps not in bars.index. "
                f"First few missing: {missing[:5].tolist()}"
            )

        # 切片 bars（保持 DatetimeIndex）
        sliced_bars = self.bars.loc[index].copy()

        # 切片 footprint（MultiIndex 的第一层）
        sliced_footprint = self.footprint_df.loc[
            self.footprint_df.index.get_level_values(0).isin(index)
        ].copy()

        # 返回新实例（保持 metadata）
        return FootprintBarData(
            bars=sliced_bars,
            footprint_df=sliced_footprint,
            resolution=self.resolution,
            num_units=self.num_units
        )

    @property
    def price_range(self) -> tuple:
        """价格范围 (low_min, high_max)"""
        return (self.bars['low'].min(), self.bars['high'].max())

    @property
    def total_volume(self) -> int:
        """总成交量"""
        return int(self.bars['volume'].sum())

    @property
    def bars_columns(self) -> tuple:
        """bars 的列名（用于验证）"""
        return tuple(self.bars.columns)

    @property
    def footprint_columns(self) -> tuple:
        """footprint 的列名（用于验证）"""
        return tuple(self.footprint_df.columns)

    def validate_compatible(self, other: 'FootprintBarData') -> None:
        """
        验证与另一个 FootprintBarData 是否兼容（可合并）

        检查：
        1. resolution 一致
        2. num_units 一致
        3. 列名一致

        Raises:
            ValueError: 如果不兼容
        """
        if self.resolution != other.resolution:
            raise ValueError(
                f"Resolution mismatch: {self.resolution} != {other.resolution}"
            )

        if self.num_units != other.num_units:
            raise ValueError(
                f"num_units mismatch: {self.num_units} != {other.num_units}"
            )

        if self.bars_columns != other.bars_columns:
            raise ValueError(
                f"bars columns mismatch: {self.bars_columns} != {other.bars_columns}"
            )

        if self.footprint_columns != other.footprint_columns:
            raise ValueError(
                f"footprint columns mismatch: {self.footprint_columns} != {other.footprint_columns}"
            )

    @classmethod
    def from_result(cls,
                    result: dict,
                    resolution: str,
                    num_units: int,
                    tick_size: float = 0.1,
                    add_vwap: bool = True,
                    add_volume_profile: bool = True,
                    validate: bool = True,
                    sort: bool = True,
                    deduplicate: bool = True) -> 'FootprintBarData':
        """
        从 CMEBarsLoader.load_bars() 结果创建 FootprintBarData

        自动完成：
        1. 数据验证（去重、排序、对齐）
        2. 计算 VWAP（可选）
        3. 计算 Volume Profile 指标：POC, VAH, VAL（可选）

        Args:
            result: CMEBarsLoader.load_bars() 的返回值
                   {'bars': DataFrame, 'footprint': DataFrame}
            resolution: bar 类型 ('MIN', 'H', 'VOLUME', 'TICK', 'DOLLAR')
            num_units: 时间单位或 threshold
            tick_size: tick 大小（预留参数）
            add_vwap: 是否添加 VWAP 列
            add_volume_profile: 是否添加 POC/VAH/VAL 列
            validate: 是否验证数据一致性
            sort: 是否排序
            deduplicate: 是否去重

        Returns:
            FootprintBarData 实例

        Example:
            >>> result = loader.load_bars('GC', '20210104', 'MIN', 5)
            >>> bar_data = FootprintBarData.from_result(result, 'MIN', 5)
        """
        bars = result['bars'].copy()
        footprint = result['footprint'].copy()

        # 1. 确保 bars.index 是 DatetimeIndex
        if not isinstance(bars.index, pd.DatetimeIndex):
            if 'date_time' in bars.columns:
                bars.set_index('date_time', inplace=True)
            else:
                raise ValueError("bars must have DatetimeIndex or 'date_time' column")

        # 2. 去重
        if deduplicate:
            if bars.index.duplicated().any():
                bars = bars[~bars.index.duplicated(keep='last')]

            if footprint.index.duplicated().any():
                footprint = footprint[~footprint.index.duplicated(keep='last')]

        # 3. 排序
        if sort:
            bars = bars.sort_index()
            footprint = footprint.sort_index()

        # 4. 添加 VWAP
        if add_vwap and 'vwap' not in bars.columns:
            bars = cls._add_vwap(bars, footprint)

        # 5. 添加 Volume Profile 指标
        if add_volume_profile:
            if 'poc' not in bars.columns:
                bars = cls._add_volume_profile(bars, footprint)

        # 6. 过滤列：只保留标准列（使用类常量确保一致性）
        # 过滤掉 mlfinlab 的 metadata 列（tick_num, open_time_ms, close_time_ms, cum_buy_volume, cum_ticks, cum_dollar_value）
        bars = bars[cls.BARS_STANDARD_COLUMNS]

        # 7. 标准化 footprint：添加计算列和标记列（如果不存在）
        # 这确保所有数据源（CME/protobuf）都返回相同的8列结构
        if not footprint.empty:
            # 7.1 添加计算列（如果不存在）
            if 'total_vol' not in footprint.columns:
                footprint['total_vol'] = footprint['bid_vol'] + footprint['ask_vol']
            if 'delta' not in footprint.columns:
                footprint['delta'] = footprint['ask_vol'] - footprint['bid_vol']

            # 7.2 添加OHLC标记列（如果不存在）
            if 'is_open' not in footprint.columns:
                # 初始化所有标记列为False
                footprint['is_open'] = False
                footprint['is_high'] = False
                footprint['is_low'] = False
                footprint['is_close'] = False

                # 为每个timestamp设置标记
                for ts in bars.index:
                    try:
                        # 获取该timestamp的OHLC值
                        ohlc = {
                            'open': bars.loc[ts, 'open'],
                            'high': bars.loc[ts, 'high'],
                            'low': bars.loc[ts, 'low'],
                            'close': bars.loc[ts, 'close']
                        }

                        # 设置标记（如果价格存在于footprint中）
                        if ts in footprint.index.get_level_values(0):
                            for marker, price in ohlc.items():
                                if (ts, price) in footprint.index:
                                    footprint.loc[(ts, price), f'is_{marker}'] = True
                    except (KeyError, IndexError):
                        # 某些timestamp可能没有footprint数据，跳过
                        pass

        # 8. 创建实例（会自动验证）
        return cls(
            bars=bars,
            footprint_df=footprint,
            resolution=resolution,
            num_units=num_units
        )

    @staticmethod
    def _add_vwap(bars: pd.DataFrame, footprint: pd.DataFrame) -> pd.DataFrame:
        """添加 VWAP 列"""
        bars = bars.copy()
        vwap_list = []

        for ts in bars.index:
            try:
                time_data = footprint.loc[ts]
                if isinstance(time_data.index, pd.MultiIndex):
                    prices = time_data.index.get_level_values(1).values
                else:
                    prices = time_data.index.values

                volumes = time_data['total_vol'].values
                total_volume = volumes.sum()

                if total_volume > 0:
                    vwap = float((prices * volumes).sum() / total_volume)
                else:
                    vwap = bars.loc[ts, 'close']

                vwap_list.append(vwap)
            except (KeyError, IndexError):
                vwap_list.append(bars.loc[ts, 'close'])

        bars['vwap'] = vwap_list
        return bars

    @staticmethod
    def _add_volume_profile(bars: pd.DataFrame, footprint: pd.DataFrame) -> pd.DataFrame:
        """添加 Volume Profile 指标：POC, VAH, VAL"""
        bars = bars.copy()
        poc_list = []
        vah_list = []
        val_list = []

        for ts in bars.index:
            try:
                time_data = footprint.loc[ts]
                vp_metrics = FootprintBarData._calculate_volume_profile_for_bar(time_data)
                poc_list.append(vp_metrics['poc'])
                vah_list.append(vp_metrics['vah'])
                val_list.append(vp_metrics['val'])
            except (KeyError, IndexError):
                poc_list.append(bars.loc[ts, 'close'])
                vah_list.append(bars.loc[ts, 'high'])
                val_list.append(bars.loc[ts, 'low'])

        bars['poc'] = poc_list
        bars['vah'] = vah_list
        bars['val'] = val_list

        return bars

    @staticmethod
    def _calculate_volume_profile_for_bar(time_data: pd.DataFrame) -> dict:
        """计算单个 bar 的 Volume Profile 指标"""
        if isinstance(time_data.index, pd.MultiIndex):
            prices = time_data.index.get_level_values(1).values
        else:
            prices = time_data.index.values

        volumes = time_data['total_vol'].values

        if len(prices) == 0 or volumes.sum() == 0:
            return {
                'poc': float(prices[0]) if len(prices) > 0 else 0.0,
                'vah': float(prices.max()) if len(prices) > 0 else 0.0,
                'val': float(prices.min()) if len(prices) > 0 else 0.0
            }

        # 1. POC
        poc_idx = volumes.argmax()
        poc = float(prices[poc_idx])

        # 2. Value Area (70%)
        total_volume = volumes.sum()
        target_volume = total_volume * 0.7

        sorted_indices = prices.argsort()
        sorted_prices = prices[sorted_indices]
        sorted_volumes = volumes[sorted_indices]

        poc_sorted_idx = np.where(sorted_prices == poc)[0][0]

        accumulated_volume = sorted_volumes[poc_sorted_idx]
        low_idx = poc_sorted_idx
        high_idx = poc_sorted_idx

        while accumulated_volume < target_volume:
            expand_up = high_idx < len(sorted_prices) - 1
            expand_down = low_idx > 0

            if expand_up and expand_down:
                up_vol = sorted_volumes[high_idx + 1] if high_idx < len(sorted_volumes) - 1 else 0
                down_vol = sorted_volumes[low_idx - 1] if low_idx > 0 else 0

                if up_vol >= down_vol:
                    high_idx += 1
                    accumulated_volume += up_vol
                else:
                    low_idx -= 1
                    accumulated_volume += down_vol
            elif expand_up:
                high_idx += 1
                if high_idx < len(sorted_volumes):
                    accumulated_volume += sorted_volumes[high_idx]
            elif expand_down:
                low_idx -= 1
                if low_idx >= 0:
                    accumulated_volume += sorted_volumes[low_idx]
            else:
                break

        val = float(sorted_prices[low_idx])
        vah = float(sorted_prices[high_idx])

        return {'poc': poc, 'vah': vah, 'val': val}


# ===== 工具函数 =====

def find_nearest_timestamp(
    all_times: pd.DatetimeIndex,
    target_time: pd.Timestamp,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=10)
) -> Optional[pd.Timestamp]:
    """
    模糊查找最接近的时间戳（严格 <= target，避免未来数据）

    使用 pandas get_indexer(method='ffill') 高效实现

    Args:
        all_times: 所有可用时间戳（已排序）
        target_time: 目标时间戳
        tolerance: 最大容差（默认10分钟，适配常见bar周期）

    Returns:
        最接近且 <= target_time 的时间戳，超出容差则返回 None

    Example:
        >>> times = pd.DatetimeIndex(['10:00:00.123', '10:05:00.456'])
        >>> find_nearest_timestamp(times, pd.Timestamp('10:05:00'))
        Timestamp('10:00:00.123')  # 严格 <= target
    """
    if len(all_times) == 0:
        return None

    # ffill: 返回 <= target 的最近索引
    idx = all_times.get_indexer([target_time], method='ffill')[0]

    if idx == -1:  # target < 所有时间戳
        return None

    nearest = all_times[idx]

    # 检查容差
    if (target_time - nearest) <= tolerance:
        return nearest

    return None


def get_detection_data(
    bars_index: pd.DatetimeIndex,
    target_bar_time: pd.Timestamp,
    lookback_bars: int,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=10)
) -> Optional[pd.DatetimeIndex]:
    """
    计算检测窗口的时间戳索引 - 支持模糊时间戳匹配

    职责：纯索引计算，不涉及数据切片
    使用：配合 FootprintBarData.slice() 完成数据提取

    核心特性：
    1. 模糊查找时间戳（容差内匹配）
    2. 确保返回 <= target_bar_time（避免未来数据泄露）
    3. 精确控制 bar 数量

    Args:
        bars_index: bars 的时间戳索引（DatetimeIndex，已排序）
        target_bar_time: 目标 bar 时间（可以是近似时间）
        lookback_bars: 回看 bar 数量（包含 target 本身）
        tolerance: 时间戳匹配容差（默认10分钟）

    Returns:
        检测窗口的时间戳索引（长度为 lookback_bars），数据不足或超出容差则返回 None

    Example:
        >>> # 1. 计算检测窗口的索引
        >>> detection_index = get_detection_data(
        ...     bar_data.bars.index,
        ...     target_timestamp,
        ...     lookback_bars=50
        ... )
        >>> # 2. 切片数据
        >>> if detection_index is not None:
        ...     detection_data = bar_data.slice(detection_index)
    """
    # 确保索引已排序
    all_times = bars_index.sort_values()

    # 模糊查找最近的时间戳（<= target）
    actual_target = find_nearest_timestamp(all_times, target_bar_time, tolerance)

    if actual_target is None:
        return None

    # 找到实际时间戳的索引
    target_idx = all_times.get_loc(actual_target)

    # 检查历史数据充足性
    if target_idx < lookback_bars - 1:
        return None

    # 精确取 lookback_bars 个时间点
    start_idx = target_idx - lookback_bars + 1
    selected_times = all_times[start_idx:target_idx + 1]

    # 严格验证数量
    if len(selected_times) != lookback_bars:
        return None

    return selected_times
