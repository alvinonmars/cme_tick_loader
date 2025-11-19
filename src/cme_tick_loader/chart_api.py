#!/usr/bin/env python3
"""
通用图表API - 基于DataFrame的金融图表绘制工具
支持蜡烛图 + 矩形 + 线条 + 标签的组合绘制
专为Jupyter notebook调试设计
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class ChartAPI:
    """
    通用图表API

    设计原则：
    1. DataFrame统一输入层
    2. 精确时间戳匹配
    3. 智能边界处理
    4. 链式API调用
    5. 详细日志反馈
    """

    # zone_type → 完整样式映射
    # 依赖约束：zone_type必须全局唯一（见CLAUDE.md）
    ZONE_STYLES = {
        # Peak zones（时点型：从detected_at延伸、显示符号）
        'PEAK_HIGH': {
            'color': 'red',
            'symbol': '▼',
            'opacity': 0.1,
            'render_mode': 'peak',
        },
        'PEAK_LOW': {
            'color': 'green',
            'symbol': '▲',
            'opacity': 0.1,
            'render_mode': 'peak',
        },

        # VP zones（空间型：横跨全范围、无符号）
        'VP_LARGE_ORDER': {
            'color': 'gold',
            'opacity': 0.2,
            'render_mode': 'spatial',
        },
        'VP_DELTA_BUY': {
            'color': 'green',
            'opacity': 0.2,
            'render_mode': 'spatial',
        },
        'VP_DELTA_SELL': {
            'color': 'red',
            'opacity': 0.2,
            'render_mode': 'spatial',
        },
        'VP_BATTLE': {
            'color': 'blue',
            'opacity': 0.2,
            'render_mode': 'spatial',
        },
    }

    DEFAULT_STYLE = {
        'color': 'gray',
        'opacity': 0.15,
        'render_mode': 'spatial',
    }

    def __init__(self, symbol: str = 'CHART', tick_size: float = 0.1, use_index_mode: bool = False):
        """
        初始化图表API

        Args:
            symbol: 交易品种符号
            tick_size: tick大小，用于价格格式化
            use_index_mode: 是否使用索引模式（默认False）
                          - False: x轴使用真实时间戳（适合均匀时间bar，如5分钟）
                          - True: x轴使用bar索引，等宽显示（适合非均匀时间bar，如VOLUME bar）
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.use_index_mode = use_index_mode
        self.fig = None
        self.ohlcv_df = None
        self.num_rows = 1  # 总行数（包括主图和子图）
        self.main_row = 1  # 主图所在行（永远是第1行）

        # 索引模式的映射表（仅在use_index_mode=True时使用）
        self._timestamp_to_idx = {}  # {pd.Timestamp: int}
        self._idx_to_timestamp = []  # [pd.Timestamp, ...]

    @staticmethod
    def _calculate_atr(ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算Average True Range（ATR）

        Args:
            ohlcv_df: OHLCV DataFrame
            period: ATR周期（默认14）

        Returns:
            ATR序列
        """
        high_low = ohlcv_df['high'] - ohlcv_df['low']
        high_close = np.abs(ohlcv_df['high'] - ohlcv_df['close'].shift())
        low_close = np.abs(ohlcv_df['low'] - ohlcv_df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def _calculate_cum_delta(footprint_df: pd.DataFrame) -> pd.Series:
        """
        计算Cumulative Volume Delta（累计买卖压力）

        Args:
            footprint_df: MultiIndex DataFrame (bar_timestamp, price) with 'delta' column

        Returns:
            Cumulative Delta序列（按bar_timestamp索引）
        """
        # 按bar_timestamp分组，汇总每个bar的delta
        bar_delta = footprint_df.groupby(level=0)['delta'].sum()
        # 累计求和
        cum_delta = bar_delta.cumsum()

        return cum_delta

    def _to_x_coord(self, timestamp: pd.Timestamp):
        """
        将时间戳转换为x坐标（自动适应时间戳/索引模式）

        Args:
            timestamp: pd.Timestamp 时间戳

        Returns:
            x坐标：索引模式返回int，时间戳模式返回pd.Timestamp

        注意：
            - 即使在时间戳模式下，如果timestamp不在bars中，也会返回最近的时间戳
            - 这个fallback机制对于zone.detected_at等可能不在bars.index中的时间戳是必需的
        """
        if not self.use_index_mode:
            # 时间戳模式：直接返回时间戳
            # 但如果不在index中，返回最近的（fallback）
            if self.ohlcv_df is None:
                return timestamp

            if timestamp in self.ohlcv_df.index:
                return timestamp

            # Fallback: 找最接近的时间戳
            timestamps = self.ohlcv_df.index
            closest_idx = min(
                range(len(timestamps)),
                key=lambda i: abs((timestamps[i] - timestamp).total_seconds())
            )
            return timestamps[closest_idx]

        # 索引模式：转换为索引
        if timestamp in self._timestamp_to_idx:
            return self._timestamp_to_idx[timestamp]

        # Fallback: 找最接近的时间戳对应的索引
        if not self._idx_to_timestamp:
            raise RuntimeError("索引模式下timestamp映射表为空，请先调用create_candlestick")

        closest_idx = min(
            range(len(self._idx_to_timestamp)),
            key=lambda i: abs((self._idx_to_timestamp[i] - timestamp).total_seconds())
        )
        return closest_idx

    def _setup_index_mode_ticks(self):
        """
        设置索引模式的x轴刻度（显示真实时间而非索引）

        仅在use_index_mode=True时调用
        在所有子图上设置相同的刻度，但只在底部子图显示刻度文本
        """
        if not self.use_index_mode or not self._idx_to_timestamp:
            return

        n = len(self._idx_to_timestamp)
        if n == 0:
            return

        # 约10个刻度
        tick_step = max(1, n // 10)
        tick_indices = list(range(0, n, tick_step))
        tick_labels = [self._idx_to_timestamp[i].strftime('%m-%d %H:%M')
                      for i in tick_indices]

        # 检测是否为单图模式（无子图）
        is_single_plot = not hasattr(self.fig, '_grid_ref') or self.fig._grid_ref is None

        if is_single_plot:
            # 单图模式：直接更新xaxis
            self.fig.update_xaxes(
                title_text='Time (等间距bar)',
                tickmode='array',
                tickvals=tick_indices,
                ticktext=tick_labels
            )
        else:
            # 多子图模式：更新所有子图的x轴
            for row in range(1, self.num_rows + 1):
                if row == self.num_rows:
                    # 底部子图：显示刻度文本和标题
                    self.fig.update_xaxes(
                        title_text='Time (等间距bar)',
                        tickmode='array',
                        tickvals=tick_indices,
                        ticktext=tick_labels,
                        row=row, col=1
                    )
                else:
                    # 其他子图：只设置刻度位置，不显示文本
                    self.fig.update_xaxes(
                        tickmode='array',
                        tickvals=tick_indices,
                        ticktext=[''] * len(tick_indices),  # 空文本
                        row=row, col=1
                    )

    def create_candlestick(
        self,
        ohlcv_df: pd.DataFrame,
        footprint_df: pd.DataFrame = None,
        show_volume: bool = True,
        show_cum_delta: bool = False,
        show_atr: bool = False,
        atr_period: int = 14
    ) -> 'ChartAPI':
        """
        创建主蜡烛图（支持多个技术指标子图）

        Args:
            ohlcv_df: DataFrame with DatetimeIndex and columns ['open', 'high', 'low', 'close', 'volume']
            footprint_df: MultiIndex DataFrame (bar_timestamp, price) for calculating cumulative delta
            show_volume: 是否显示volume子图
            show_cum_delta: 是否显示cumulative delta子图（需要footprint_df）
            show_atr: 是否显示ATR子图
            atr_period: ATR计算周期（默认14）

        Returns:
            self: 支持链式调用
        """
        if ohlcv_df.empty:
            raise ValueError("OHLCV DataFrame cannot be empty")

        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ohlcv_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 确保时间戳索引有序
        self.ohlcv_df = ohlcv_df.sort_index()

        # 建立索引模式的映射表（如果启用）
        if self.use_index_mode:
            self._timestamp_to_idx = {ts: i for i, ts in enumerate(self.ohlcv_df.index)}
            self._idx_to_timestamp = list(self.ohlcv_df.index)
            x_data = list(range(len(self.ohlcv_df)))  # 使用索引作为x坐标
        else:
            x_data = self.ohlcv_df.index  # 使用时间戳作为x坐标

        # 确定要显示的子图
        subplot_configs = []

        if show_volume and 'volume' in self.ohlcv_df.columns:
            subplot_configs.append('volume')

        if show_cum_delta:
            if footprint_df is None:
                print("⚠️  show_cum_delta=True但未提供footprint_df，跳过cumulative delta子图")
            else:
                subplot_configs.append('cum_delta')

        if show_atr:
            subplot_configs.append('atr')

        # 计算总行数和高度分配
        num_subplots = len(subplot_configs)
        self.num_rows = 1 + num_subplots  # 主图 + 子图数量

        if num_subplots == 0:
            # 无子图，单图表
            row_heights = None
        elif num_subplots == 1:
            row_heights = [0.7, 0.3]
        elif num_subplots == 2:
            row_heights = [0.6, 0.2, 0.2]
        elif num_subplots == 3:
            row_heights = [0.55, 0.15, 0.15, 0.15]
        else:
            # 更多子图的话均分
            main_height = 0.5
            subplot_height = (1.0 - main_height) / num_subplots
            row_heights = [main_height] + [subplot_height] * num_subplots

        # 创建子图标题
        subplot_titles = [f'{self.symbol} Price']
        for config in subplot_configs:
            if config == 'volume':
                subplot_titles.append('Volume')
            elif config == 'cum_delta':
                subplot_titles.append('Cumulative Delta')
            elif config == 'atr':
                subplot_titles.append(f'ATR({atr_period})')

        # 创建figure
        if num_subplots > 0:
            # 创建多行子图
            self.fig = make_subplots(
                rows=self.num_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=row_heights,
                subplot_titles=tuple(subplot_titles)
            )

            # 添加蜡烛图到主图（row=1）
            candlestick_trace = go.Candlestick(
                x=x_data,
                open=self.ohlcv_df['open'],
                high=self.ohlcv_df['high'],
                low=self.ohlcv_df['low'],
                close=self.ohlcv_df['close'],
                name=f'{self.symbol} OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            )

            # 索引模式：添加时间戳到hover信息
            if self.use_index_mode:
                # Candlestick不支持hovertemplate，使用hovertext
                hover_texts = []
                for ts, row in self.ohlcv_df.iterrows():
                    text = (
                        f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>" +
                        f"Open: {row['open']:.2f}<br>" +
                        f"High: {row['high']:.2f}<br>" +
                        f"Low: {row['low']:.2f}<br>" +
                        f"Close: {row['close']:.2f}"
                    )
                    hover_texts.append(text)
                candlestick_trace.text = hover_texts
                candlestick_trace.hoverinfo = 'text'

            self.fig.add_trace(candlestick_trace, row=1, col=1)

            # 添加各个子图
            current_row = 2  # 从第2行开始添加子图
            colors = ['red' if close < open else 'green'
                     for close, open in zip(self.ohlcv_df['close'], self.ohlcv_df['open'])]

            for config in subplot_configs:
                if config == 'volume':
                    # Volume柱状图
                    volume_trace = go.Bar(
                        x=x_data,
                        y=self.ohlcv_df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7,
                        showlegend=False
                    )

                    # 索引模式：添加时间戳到hover信息
                    if self.use_index_mode:
                        timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in self.ohlcv_df.index]
                        volume_trace.customdata = timestamp_strs
                        volume_trace.hovertemplate = '<b>Time: %{customdata}</b><br>Volume: %{y}<extra></extra>'

                    self.fig.add_trace(volume_trace, row=current_row, col=1)

                elif config == 'cum_delta':
                    # Cumulative Delta 线图
                    cum_delta = self._calculate_cum_delta(footprint_df)
                    # 对齐时间戳（footprint可能有不同的时间戳）
                    aligned_cum_delta = cum_delta.reindex(self.ohlcv_df.index, method='ffill')

                    # 根据正负着色
                    delta_colors = ['green' if d >= 0 else 'red' for d in aligned_cum_delta]

                    cum_delta_trace = go.Scatter(
                        x=x_data,
                        y=aligned_cum_delta,
                        name='Cum Delta',
                        line=dict(color='blue', width=2),
                        fill='tozeroy',
                        showlegend=False
                    )

                    # 索引模式：添加时间戳到hover信息
                    if self.use_index_mode:
                        timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in self.ohlcv_df.index]
                        cum_delta_trace.customdata = timestamp_strs
                        cum_delta_trace.hovertemplate = '<b>Time: %{customdata}</b><br>Cum Delta: %{y:.0f}<extra></extra>'

                    self.fig.add_trace(cum_delta_trace, row=current_row, col=1)

                    # 添加零线
                    self.fig.add_hline(
                        y=0,
                        line=dict(color='gray', width=1, dash='dash'),
                        row=current_row, col=1
                    )

                elif config == 'atr':
                    # ATR 线图
                    atr = self._calculate_atr(self.ohlcv_df, period=atr_period)

                    atr_trace = go.Scatter(
                        x=x_data,
                        y=atr,
                        name=f'ATR({atr_period})',
                        line=dict(color='orange', width=2),
                        showlegend=False
                    )

                    # 索引模式：添加时间戳到hover信息
                    if self.use_index_mode:
                        timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in self.ohlcv_df.index]
                        atr_trace.customdata = timestamp_strs
                        atr_trace.hovertemplate = '<b>Time: %{customdata}</b><br>ATR: %{y:.2f}<extra></extra>'

                    self.fig.add_trace(atr_trace, row=current_row, col=1)

                current_row += 1

            # 动态计算图表高度
            chart_height = 400 + num_subplots * 150

            # 更新布局
            self.fig.update_layout(
                title=f'{self.symbol} Chart Analysis',
                template='plotly_white',
                showlegend=True,
                hovermode='x unified',
                height=chart_height
            )

            # 更新x轴设置（只在底部显示）
            for row in range(1, self.num_rows):
                self.fig.update_xaxes(title_text='', row=row, col=1, rangeslider_visible=False)
            self.fig.update_xaxes(title_text='Time', row=self.num_rows, col=1)

            # 更新y轴设置
            self.fig.update_yaxes(title_text='Price', row=1, col=1)

            current_row = 2
            for config in subplot_configs:
                if config == 'volume':
                    self.fig.update_yaxes(title_text='Volume', row=current_row, col=1)
                elif config == 'cum_delta':
                    self.fig.update_yaxes(title_text='Cum Delta', row=current_row, col=1)
                elif config == 'atr':
                    self.fig.update_yaxes(title_text='ATR', row=current_row, col=1)
                current_row += 1

        else:
            # 创建单图表（仅价格）
            self.fig = go.Figure()

            # 添加蜡烛图
            candlestick_trace = go.Candlestick(
                x=x_data,
                open=self.ohlcv_df['open'],
                high=self.ohlcv_df['high'],
                low=self.ohlcv_df['low'],
                close=self.ohlcv_df['close'],
                name=f'{self.symbol} OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            )

            # 索引模式：添加时间戳到hover信息
            if self.use_index_mode:
                # Candlestick不支持hovertemplate，使用hovertext
                hover_texts = []
                for ts, row in self.ohlcv_df.iterrows():
                    text = (
                        f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>" +
                        f"Open: {row['open']:.2f}<br>" +
                        f"High: {row['high']:.2f}<br>" +
                        f"Low: {row['low']:.2f}<br>" +
                        f"Close: {row['close']:.2f}"
                    )
                    hover_texts.append(text)
                candlestick_trace.text = hover_texts
                candlestick_trace.hoverinfo = 'text'

            self.fig.add_trace(candlestick_trace)

            # 基础布局
            self.fig.update_layout(
                title=f'{self.symbol} Chart Analysis',
                xaxis_title='Time',
                yaxis_title='Price',
                template='plotly_white',
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )

        # 设置索引模式的x轴刻度（如果启用）
        self._setup_index_mode_ticks()

        # 日志输出
        subplot_info = f" with {', '.join(subplot_configs)}" if subplot_configs else ""
        print(f"✅ Created candlestick chart with {len(self.ohlcv_df)} bars{subplot_info}")
        return self

    def _get_chart_time_bounds(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """获取图表时间边界"""
        if self.ohlcv_df is None:
            raise RuntimeError("Must create candlestick chart first")
        return self.ohlcv_df.index.min(), self.ohlcv_df.index.max()

    def _is_point_in_bounds(self, timestamp: pd.Timestamp) -> bool:
        """检查时间点是否在图表范围内"""
        chart_start, chart_end = self._get_chart_time_bounds()
        return chart_start <= timestamp <= chart_end

    def _clip_time_range(self, x_start: pd.Timestamp, x_end: pd.Timestamp) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], bool]:
        """
        裁剪时间范围到图表边界

        Returns:
            clipped_start: 裁剪后的开始时间
            clipped_end: 裁剪后的结束时间
            is_valid: 是否有有效重叠
        """
        chart_start, chart_end = self._get_chart_time_bounds()

        # 完全在范围外
        if x_end < chart_start or x_start > chart_end:
            return None, None, False

        # 裁剪到有效范围
        clipped_start = max(x_start, chart_start)
        clipped_end = min(x_end, chart_end)

        return clipped_start, clipped_end, True

    def _align_timestamps(self, timestamps: pd.Series) -> Tuple[List[Optional[int]], int]:
        """
        精确时间戳匹配

        Args:
            timestamps: 待匹配的时间戳序列

        Returns:
            indices: 匹配的蜡烛图索引列表（None表示无匹配）
            filtered_count: 被过滤的元素数量
        """
        candlestick_times = self.ohlcv_df.index
        indices = []
        filtered_count = 0

        for ts in timestamps:
            if pd.isna(ts) or ts not in candlestick_times:
                indices.append(None)
                filtered_count += 1
            else:
                idx = candlestick_times.get_loc(ts)
                indices.append(idx)

        return indices, filtered_count

    def add_rectangles(self, rectangles_df: pd.DataFrame) -> 'ChartAPI':
        """
        添加矩形区域（如显著区zones）

        Args:
            rectangles_df: DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end', 'fill_color', 'line_color', ...]

        Returns:
            self: 支持链式调用
        """
        if rectangles_df.empty:
            print("⚠️  Empty rectangles DataFrame, skipping")
            return self

        required_cols = ['x_start', 'x_end', 'y_start', 'y_end']
        missing_cols = [col for col in required_cols if col not in rectangles_df.columns]
        if missing_cols:
            raise ValueError(f"Rectangles DataFrame missing columns: {missing_cols}")

        valid_rectangles = []
        bounds_filtered = 0
        exact_filtered = 0
        clipped_count = 0

        for _, rect in rectangles_df.iterrows():
            x_start, x_end = rect['x_start'], rect['x_end']

            # 1. 边界裁剪
            clipped_start, clipped_end, is_valid = self._clip_time_range(x_start, x_end)

            if not is_valid:
                bounds_filtered += 1
                continue

            # 2. 精确匹配检查
            start_exists = clipped_start in self.ohlcv_df.index
            end_exists = clipped_end in self.ohlcv_df.index

            if not (start_exists and end_exists):
                exact_filtered += 1
                continue

            # 3. 记录是否发生了裁剪
            if clipped_start != x_start or clipped_end != x_end:
                clipped_count += 1

            # 4. 构建有效矩形（转换为x坐标）
            fill_color = rect.get('fill_color', 'rgba(255,0,0,0.2)')
            line_color = rect.get('line_color', 'red')
            opacity = rect.get('opacity', 0.3)
            name = rect.get('name', 'Rectangle')

            valid_rectangles.append({
                'x_start': self._to_x_coord(clipped_start),
                'x_end': self._to_x_coord(clipped_end),
                'y_start': rect['y_start'],
                'y_end': rect['y_end'],
                'fill_color': fill_color,
                'line_color': line_color,
                'opacity': opacity,
                'name': name
            })

        # 渲染有效矩形
        for i, rect in enumerate(valid_rectangles):
            # 检查是否有子图（volume）
            if hasattr(self.fig, '_grid_ref') and self.fig._grid_ref is not None and len(self.fig._grid_ref) > 1:
                # 有子图，添加到主图（row=1）
                self.fig.add_shape(
                    type="rect",
                    x0=rect['x_start'], x1=rect['x_end'],
                    y0=rect['y_start'], y1=rect['y_end'],
                    fillcolor=rect['fill_color'],
                    line=dict(color=rect['line_color'], width=1),
                    opacity=rect['opacity'],
                    name=f"{rect['name']}_{i}",
                    xref="x", yref="y"  # 明确指定主图坐标系
                )
            else:
                # 单图，正常添加
                self.fig.add_shape(
                    type="rect",
                    x0=rect['x_start'], x1=rect['x_end'],
                    y0=rect['y_start'], y1=rect['y_end'],
                    fillcolor=rect['fill_color'],
                    line=dict(color=rect['line_color'], width=1),
                    opacity=rect['opacity'],
                    name=f"{rect['name']}_{i}"
                )

        # 日志反馈
        if bounds_filtered > 0:
            print(f"⚠️  Filtered {bounds_filtered} rectangle(s) - completely outside chart bounds")
        if exact_filtered > 0:
            print(f"⚠️  Filtered {exact_filtered} rectangle(s) - clipped timestamps not found")
        if clipped_count > 0:
            print(f"📏 Clipped {clipped_count} rectangle(s) to chart boundaries")

        print(f"✅ Added {len(valid_rectangles)} rectangle(s)")
        return self

    def add_lines(self, lines_df: pd.DataFrame) -> 'ChartAPI':
        """
        添加通用线条（水平线、垂直线、斜线）

        Args:
            lines_df: DataFrame with columns ['x_start', 'x_end', 'y_start', 'y_end', 'color', 'width', 'dash', 'name']
                     None值表示延伸到边界

        Returns:
            self: 支持链式调用
        """
        if lines_df.empty:
            print("⚠️  Empty lines DataFrame, skipping")
            return self

        chart_start, chart_end = self._get_chart_time_bounds()
        chart_y_min = self.ohlcv_df['low'].min()
        chart_y_max = self.ohlcv_df['high'].max()

        valid_lines = []
        filtered_count = 0

        for _, line in lines_df.iterrows():
            x_start = line['x_start'] if pd.notna(line['x_start']) else chart_start
            x_end = line['x_end'] if pd.notna(line['x_end']) else chart_end
            y_start = line['y_start'] if pd.notna(line['y_start']) else chart_y_min
            y_end = line['y_end'] if pd.notna(line['y_end']) else chart_y_max

            # 检查线条类型和有效性
            is_horizontal = y_start == y_end
            is_vertical = x_start == x_end

            if is_vertical:
                # 垂直线：检查时间戳精确匹配和边界
                if x_start not in self.ohlcv_df.index or not self._is_point_in_bounds(x_start):
                    filtered_count += 1
                    continue
            elif is_horizontal:
                # 水平线：时间范围使用图表边界
                x_start, x_end = chart_start, chart_end
            else:
                # 斜线：检查两端时间戳
                if (x_start not in self.ohlcv_df.index or
                    x_end not in self.ohlcv_df.index or
                    not self._is_point_in_bounds(x_start) or
                    not self._is_point_in_bounds(x_end)):
                    filtered_count += 1
                    continue

            color = line.get('color', 'blue')
            width = line.get('width', 2)
            dash = line.get('dash', 'solid')
            name = line.get('name', 'Line')

            valid_lines.append({
                'x_start': self._to_x_coord(x_start),
                'x_end': self._to_x_coord(x_end),
                'y_start': y_start, 'y_end': y_end,
                'color': color, 'width': width, 'dash': dash, 'name': name
            })

        # 渲染有效线条
        for i, line in enumerate(valid_lines):
            # 检查是否有子图（volume）
            if hasattr(self.fig, '_grid_ref') and self.fig._grid_ref is not None and len(self.fig._grid_ref) > 1:
                # 有子图，添加到主图（row=1）
                self.fig.add_shape(
                    type="line",
                    x0=line['x_start'], x1=line['x_end'],
                    y0=line['y_start'], y1=line['y_end'],
                    line=dict(
                        color=line['color'],
                        width=line['width'],
                        dash=line['dash']
                    ),
                    name=f"{line['name']}_{i}",
                    xref="x", yref="y"  # 明确指定主图坐标系
                )
            else:
                # 单图，正常添加
                self.fig.add_shape(
                    type="line",
                    x0=line['x_start'], x1=line['x_end'],
                    y0=line['y_start'], y1=line['y_end'],
                    line=dict(
                        color=line['color'],
                        width=line['width'],
                        dash=line['dash']
                    ),
                    name=f"{line['name']}_{i}"
                )

        if filtered_count > 0:
            print(f"⚠️  Filtered {filtered_count} line(s) - invalid timestamps or outside bounds")
        print(f"✅ Added {len(valid_lines)} line(s)")
        return self

    def add_horizontal_lines(self, y_values: List[float], **style_kwargs) -> 'ChartAPI':
        """
        便捷方法：添加水平线（自动跨越整个图表）

        Args:
            y_values: 水平线的y坐标列表
            **style_kwargs: 样式参数 (color, width, dash, name)

        Returns:
            self: 支持链式调用
        """
        if not y_values:
            print("⚠️  Empty y_values list, skipping")
            return self

        hlines_df = pd.DataFrame({
            'x_start': [None] * len(y_values),  # None表示延伸到边界
            'x_end': [None] * len(y_values),
            'y_start': y_values,
            'y_end': y_values,
            'color': [style_kwargs.get('color', 'blue')] * len(y_values),
            'width': [style_kwargs.get('width', 2)] * len(y_values),
            'dash': [style_kwargs.get('dash', 'solid')] * len(y_values),
            'name': [style_kwargs.get('name', 'HLine')] * len(y_values)
        })

        return self.add_lines(hlines_df)

    def add_vertical_lines(self, x_values: List[pd.Timestamp], **style_kwargs) -> 'ChartAPI':
        """
        便捷方法：添加垂直线

        Args:
            x_values: 垂直线的x坐标（时间戳）列表
            **style_kwargs: 样式参数 (color, width, dash, name)

        Returns:
            self: 支持链式调用
        """
        if not x_values:
            print("⚠️  Empty x_values list, skipping")
            return self

        vlines_df = pd.DataFrame({
            'x_start': x_values,
            'x_end': x_values,  # 垂直线起始结束相同
            'y_start': [None] * len(x_values),  # None表示延伸到边界
            'y_end': [None] * len(x_values),
            'color': [style_kwargs.get('color', 'red')] * len(x_values),
            'width': [style_kwargs.get('width', 2)] * len(x_values),
            'dash': [style_kwargs.get('dash', 'solid')] * len(x_values),
            'name': [style_kwargs.get('name', 'VLine')] * len(x_values)
        })

        return self.add_lines(vlines_df)

    def add_scatter(
        self,
        x_values: List[pd.Timestamp],
        y_values: List[float],
        mode: str = 'markers',
        marker: dict = None,
        line: dict = None,
        name: str = 'Scatter',
        showlegend: bool = True
    ) -> 'ChartAPI':
        """
        添加散点图或线图

        Args:
            x_values: x坐标（时间戳）列表
            y_values: y坐标（价格）列表
            mode: 绘制模式 ('markers', 'lines', 'lines+markers')
            marker: marker样式字典，例如 dict(size=8, color='red', symbol='triangle-down')
            line: line样式字典，例如 dict(color='red', width=2, dash='dash')
            name: trace名称（显示在legend中）
            showlegend: 是否显示在legend中

        Returns:
            self: 支持链式调用

        Example:
            # 添加Peak点（markers）
            chart.add_scatter(
                [ts1, ts2], [price1, price2],
                mode='markers',
                marker=dict(size=8, color='red', symbol='triangle-down'),
                name='PEAK_HIGH'
            )

            # 添加趋势线（lines）
            chart.add_scatter(
                [start_ts, end_ts], [start_price, end_price],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='Trend Line'
            )
        """
        if len(x_values) == 0 or len(y_values) == 0:
            print("⚠️  Empty x_values or y_values, skipping")
            return self

        if len(x_values) != len(y_values):
            raise ValueError(f"x_values and y_values must have same length: {len(x_values)} vs {len(y_values)}")

        # 转换为x坐标（支持索引模式）
        x_coords = [self._to_x_coord(ts) for ts in x_values]

        # 构建trace参数
        trace_kwargs = {
            'x': x_coords,
            'y': y_values,
            'mode': mode,
            'name': name,
            'showlegend': showlegend
        }

        # 添加marker样式
        if marker and 'markers' in mode:
            trace_kwargs['marker'] = marker

        # 添加line样式
        if line and 'lines' in mode:
            trace_kwargs['line'] = line

        # 创建Scatter trace
        scatter_trace = go.Scatter(**trace_kwargs)

        # 添加时间戳到hover信息（索引模式）
        if self.use_index_mode:
            timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in x_values]
            scatter_trace.customdata = timestamp_strs
            scatter_trace.hovertemplate = '<b>Time: %{customdata}</b><br>Price: %{y:.2f}<extra></extra>'

        # 添加到主图（如果有子图的话）
        if hasattr(self.fig, '_grid_ref') and self.fig._grid_ref is not None and len(self.fig._grid_ref) > 1:
            # 有子图，添加到主图（row=1）
            self.fig.add_trace(scatter_trace, row=1, col=1)
        else:
            # 单图，正常添加
            self.fig.add_trace(scatter_trace)

        print(f"✅ Added scatter trace: {name} ({len(x_values)} points, mode={mode})")
        return self

    def add_labels(self, labels_df: pd.DataFrame) -> 'ChartAPI':
        """
        添加文本标签

        Args:
            labels_df: DataFrame with columns ['x', 'y', 'text', 'color', 'size', 'angle']

        Returns:
            self: 支持链式调用
        """
        if labels_df.empty:
            print("⚠️  Empty labels DataFrame, skipping")
            return self

        required_cols = ['x', 'y', 'text']
        missing_cols = [col for col in required_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"Labels DataFrame missing columns: {missing_cols}")

        # 时间戳对齐和边界检查
        indices, exact_filtered = self._align_timestamps(labels_df['x'])
        bounds_filtered = 0

        valid_labels = []
        for i, (_, label) in enumerate(labels_df.iterrows()):
            if indices[i] is None:
                continue

            if not self._is_point_in_bounds(label['x']):
                bounds_filtered += 1
                continue

            color = label.get('color', 'black')
            size = label.get('size', 12)
            angle = label.get('angle', 0)

            valid_labels.append({
                'x': self._to_x_coord(label['x']),
                'y': label['y'],
                'text': label['text'],
                'color': color,
                'size': size,
                'angle': angle
            })

        # 渲染有效标签
        for i, label in enumerate(valid_labels):
            # 检查是否有子图（volume）
            if hasattr(self.fig, '_grid_ref') and self.fig._grid_ref is not None and len(self.fig._grid_ref) > 1:
                # 有子图，添加到主图（row=1）
                self.fig.add_annotation(
                    x=label['x'],
                    y=label['y'],
                    text=label['text'],
                    showarrow=False,
                    font=dict(color=label['color'], size=label['size']),
                    textangle=label['angle'],
                    name=f"Label_{i}",
                    xref="x", yref="y"  # 明确指定主图坐标系
                )
            else:
                # 单图，正常添加
                self.fig.add_annotation(
                    x=label['x'],
                    y=label['y'],
                    text=label['text'],
                    showarrow=False,
                    font=dict(color=label['color'], size=label['size']),
                    textangle=label['angle'],
                    name=f"Label_{i}"
                )

        # 日志反馈
        if exact_filtered > 0:
            print(f"⚠️  Filtered {exact_filtered} label(s) - timestamp not found")
        if bounds_filtered > 0:
            print(f"⚠️  Filtered {bounds_filtered} label(s) - outside chart bounds")
        print(f"✅ Added {len(valid_labels)} label(s)")
        return self

    def get_figure(self) -> go.Figure:
        """获取plotly figure对象"""
        if self.fig is None:
            raise RuntimeError("No chart created. Call create_candlestick() first")
        return self.fig

    def show(self) -> None:
        """在notebook中显示图表"""
        if self.fig is None:
            raise RuntimeError("No chart created. Call create_candlestick() first")
        self.fig.show()

    def plot_zones(self, zones: list, target_timestamp, bars: pd.DataFrame,
                   footprint_df: pd.DataFrame = None,
                   show_labels: bool = True,
                   show_center_lines: bool = True,
                   show_peak_symbols: bool = True,
                   show_extension_lines: bool = True,
                   show_volume: bool = False,
                   show_cum_delta: bool = False,
                   show_atr: bool = False,
                   atr_period: int = 14) -> 'ChartAPI':
        """
        一站式绘制zones检测结果（支持多个技术指标子图）

        自动完成：
        1. 创建蜡烛图（使用bars，范围应大于检测窗口以观察未来）
        2. 标记检测点（target_timestamp垂直线）
        3. 根据zone_type自动选择样式和绘制模式
        4. 绘制所有zones（矩形、中心线、标签、符号）

        绘制元素：
        - 所有zones: 矩形 + 中心线(在key_price) + 强度标签
        - Peak zones额外: 符号(▼▲) + 延伸线(到检测点)

        Args:
            zones: List[KeyZone] - detector返回的zones列表
            target_timestamp: pd.Timestamp - 检测的目标时间点
            bars: pd.DataFrame - 用于绘制的bar数据（建议比检测窗口大）
            footprint_df: pd.DataFrame - MultiIndex footprint数据（用于cumulative delta）
            show_labels: bool - 是否显示强度标签
            show_center_lines: bool - 是否显示中心线
            show_peak_symbols: bool - 是否显示Peak的▼▲符号
            show_extension_lines: bool - 是否显示Peak的延伸线
            show_volume: bool - 是否显示volume子图
            show_cum_delta: bool - 是否显示cumulative delta子图（需要footprint_df）
            show_atr: bool - 是否显示ATR子图
            atr_period: int - ATR计算周期（默认14）

        Returns:
            self - 支持链式调用

        Example:
            # 基础用法（仅主图）
            chart = ChartAPI('GC', tick_size=0.1)
            chart.plot_zones(lv_zones, TARGET_TS, viz_bar_data.bars)
            chart.show()

            # 完整用法（主图 + 3个技术指标子图）
            chart = ChartAPI('GC', tick_size=0.1)
            chart.plot_zones(
                zones=all_zones,
                target_timestamp=TARGET_TS,
                bars=viz_bar_data.bars,
                footprint_df=viz_footprint_df,
                show_volume=True,
                show_cum_delta=True,
                show_atr=True
            )
            chart.show()
        """
        # 1. 创建蜡烛图（带可选的技术指标子图）
        self.create_candlestick(
            bars,
            footprint_df=footprint_df,
            show_volume=show_volume,
            show_cum_delta=show_cum_delta,
            show_atr=show_atr,
            atr_period=atr_period
        )

        # 2. 添加检测点垂直线
        self.add_vertical_lines([target_timestamp], color='purple', width=3,
                               dash='solid', name='检测点')

        if not zones:
            print("⚠️  zones列表为空，只显示蜡烛图和检测点")
            return self

        # 3. 收集所有绘制元素
        rectangles = []
        center_lines = []
        labels = []
        symbols = []
        extension_lines = []

        for zone in zones:
            # 获取样式配置
            style = self.ZONE_STYLES.get(zone.zone_type, self.DEFAULT_STYLE)
            color = style['color']
            opacity = style['opacity']
            render_mode = style['render_mode']

            # 根据填充透明度生成rgba
            if color == 'gold':
                fill_color = f'rgba(255,215,0,{opacity})'
            elif color == 'green':
                fill_color = f'rgba(0,200,0,{opacity})'
            elif color == 'red':
                fill_color = f'rgba(255,0,0,{opacity})'
            elif color == 'blue':
                fill_color = f'rgba(0,0,255,{opacity})'
            elif color == 'gray':
                fill_color = f'rgba(128,128,128,{opacity})'
            else:
                fill_color = f'rgba(128,128,128,{opacity})'

            # 确定矩形的时间范围
            if render_mode == 'peak':
                x_start = zone.detected_at
                x_end = bars.index[-1]
            else:  # spatial
                x_start = bars.index[0]
                x_end = bars.index[-1]

            # 添加矩形
            rectangles.append({
                'x_start': x_start,
                'x_end': x_end,
                'y_start': zone.lower_bound,
                'y_end': zone.upper_bound,
                'fill_color': fill_color,
                'line_color': color,
                'name': f"{zone.zone_type}({zone.strength:.1f})"
            })

            # 添加中心线
            if show_center_lines:
                center_lines.append({
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': zone.key_price,
                    'y_end': zone.key_price,
                    'color': color,
                    'width': 2,
                    'dash': 'solid',
                    'name': 'Center_Line'
                })

            # 添加标签和符号
            if render_mode == 'peak':
                # Peak zones: 符号 + 强度标签
                if show_peak_symbols and 'symbol' in style:
                    symbols.append({
                        'x': zone.detected_at,
                        'y': zone.key_price,
                        'text': style['symbol'],
                        'color': color,
                        'size': min(20, max(10, int(zone.strength / 5) + 10))
                    })

                if show_labels:
                    offset = 1.5 if 'HIGH' in zone.zone_type else -1.5
                    labels.append({
                        'x': zone.detected_at,
                        'y': zone.key_price + offset,
                        'text': f"{zone.strength:.0f}",
                        'color': color,
                        'size': 10
                    })

                # 延伸线到检测点
                if show_extension_lines and zone.detected_at <= target_timestamp:
                    extension_lines.append({
                        'x_start': zone.detected_at,
                        'x_end': target_timestamp,
                        'y_start': zone.key_price,
                        'y_end': zone.key_price,
                        'color': color,
                        'width': 1,
                        'dash': 'dot',
                        'name': 'Extension_Line'
                    })
            else:
                # Spatial zones: 强度标签在右侧
                if show_labels:
                    labels.append({
                        'x': bars.index[-1],
                        'y': zone.key_price,
                        'text': f"{zone.strength:.0f}",
                        'color': color,
                        'size': 12
                    })

        # 4. 批量添加所有元素
        if rectangles:
            self.add_rectangles(pd.DataFrame(rectangles))
        if center_lines:
            self.add_lines(pd.DataFrame(center_lines))
        if labels:
            self.add_labels(pd.DataFrame(labels))
        if symbols:
            self.add_labels(pd.DataFrame(symbols))
        if extension_lines:
            self.add_lines(pd.DataFrame(extension_lines))

        print(f"✅ 已绘制 {len(zones)} 个zones")

        return self

    def save_html(self, filename: str) -> str:
        """
        保存为HTML文件

        Args:
            filename: 输出文件名

        Returns:
            保存的文件路径
        """
        if self.fig is None:
            raise RuntimeError("No chart created. Call create_candlestick() first")

        self.fig.write_html(
            filename,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )

        print(f"💾 Chart saved: {filename}")
        return filename

    # ========== Volume/Delta Profile可视化 ==========

    def plot_volume_profile(self, price_stats: Dict[float, Dict],
                           detected_zones: List = None,
                           gmm_params: Dict = None,
                           boundaries: List[float] = None,
                           title: str = "Volume Profile") -> 'ChartAPI':
        """
        绘制Volume Profile（横向柱状图）

        Args:
            price_stats: {price: {'total_volume': ...}} 价格统计字典
            detected_zones: 检测到的zones列表（可选）
            gmm_params: GMM参数 {'means': [...], 'stds': [...], 'weights': [...]}（可选）
            boundaries: 边界价格列表（可选）
            title: 图表标题

        Returns:
            self（链式调用）
        """
        prices = np.array(sorted(price_stats.keys()))
        volumes = np.array([price_stats[p]['total_volume'] for p in prices])

        # 创建横向柱状图
        fig = go.Figure()

        # Volume柱状图（横向）
        fig.add_trace(go.Bar(
            x=volumes,
            y=prices,
            orientation='h',
            name='Volume',
            marker=dict(color='steelblue', opacity=0.6),
            hovertemplate='Price: %{y}<br>Volume: %{x}<extra></extra>'
        ))

        # 叠加GMM拟合曲线
        if gmm_params:
            means = gmm_params['means']
            stds = gmm_params['stds']
            weights = gmm_params['weights']

            price_range = np.linspace(prices.min(), prices.max(), 200)
            total_volume = volumes.sum()

            for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
                # 高斯分布曲线（缩放到volume量纲）
                gaussian = weight * total_volume * np.exp(-0.5 * ((price_range - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

                fig.add_trace(go.Scatter(
                    x=gaussian,
                    y=price_range,
                    mode='lines',
                    name=f'Distribution {i+1}',
                    line=dict(width=2, dash='dash'),
                    hovertemplate='Price: %{y}<br>Density: %{x}<extra></extra>'
                ))

        # 标注边界
        if boundaries:
            max_volume = volumes.max()
            for boundary in boundaries:
                fig.add_hline(
                    y=boundary,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=f"Boundary @ {boundary:.1f}",
                    annotation_position="right"
                )

        # 标注检测的zones
        if detected_zones:
            for zone in detected_zones:
                fig.add_hrect(
                    y0=zone.lower_bound,
                    y1=zone.upper_bound,
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line_width=0,
                    annotation_text=zone.zone_type,
                    annotation_position="right"
                )

        fig.update_layout(
            title=title,
            xaxis_title="Volume",
            yaxis_title="Price",
            height=600,
            showlegend=True,
            hovermode='closest'
        )

        self.fig = fig
        return self

    def plot_delta_profile(self, price_stats: Dict[float, Dict],
                          detected_zones: List = None,
                          title: str = "Delta Profile") -> 'ChartAPI':
        """
        绘制Delta Profile（横向柱状图，正delta向右，负delta向左）

        Args:
            price_stats: {price: {'total_delta': ...}} 价格统计字典
            detected_zones: 检测到的zones列表（可选）
            title: 图表标题

        Returns:
            self（链式调用）
        """
        prices = np.array(sorted(price_stats.keys()))
        deltas = np.array([price_stats[p]['total_delta'] for p in prices])

        # 创建横向柱状图
        fig = go.Figure()

        # 正Delta（买压）
        positive_mask = deltas > 0
        if positive_mask.any():
            fig.add_trace(go.Bar(
                x=deltas[positive_mask],
                y=prices[positive_mask],
                orientation='h',
                name='Buy Pressure',
                marker=dict(color='green', opacity=0.6),
                hovertemplate='Price: %{y}<br>Delta: +%{x}<extra></extra>'
            ))

        # 负Delta（卖压）
        negative_mask = deltas < 0
        if negative_mask.any():
            fig.add_trace(go.Bar(
                x=deltas[negative_mask],
                y=prices[negative_mask],
                orientation='h',
                name='Sell Pressure',
                marker=dict(color='red', opacity=0.6),
                hovertemplate='Price: %{y}<br>Delta: %{x}<extra></extra>'
            ))

        # 零线
        fig.add_vline(x=0, line=dict(color='gray', width=1, dash='dash'))

        # 标注检测的zones
        if detected_zones:
            for zone in detected_zones:
                color = 'rgba(0, 255, 0, 0.2)' if 'BUY' in zone.zone_type else 'rgba(255, 0, 0, 0.2)'
                fig.add_hrect(
                    y0=zone.lower_bound,
                    y1=zone.upper_bound,
                    fillcolor=color,
                    line_width=0,
                    annotation_text=zone.zone_type,
                    annotation_position="right"
                )

        fig.update_layout(
            title=title,
            xaxis_title="Delta (Positive=Buy, Negative=Sell)",
            yaxis_title="Price",
            height=600,
            showlegend=True,
            hovermode='closest'
        )

        self.fig = fig
        return self