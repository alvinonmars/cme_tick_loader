"""Professional Footprint visualization - Direct implementation based on reference code logic"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class FootprintConfig:
    """Professional Footprint configuration - simplified"""

    # Three-column layout ratios (Body:Delta:Volume) - from reference code
    column_ratios: Tuple[int, int, int] = (2, 12, 10)

    # Professional color scheme - from reference code
    colors: Dict[str, str] = field(default_factory=lambda: {
        'background': '#FFFFFF',
        'body_bullish': '#00AA00',      # Green for bullish candles
        'body_bearish': '#FF0000',      # Red for bearish candles
        'body_border': '#000000',       # Black border for candle bodies
        'delta_positive': '#00CC00',    # Bright green for positive delta
        'delta_negative': '#FF4444',    # Bright red for negative delta
        'volume_bar': '#4A90E2',        # Professional blue for volume
        'text_main': '#000000',         # Black main text
        'text_shadow': '#FFFFFF',       # White text shadow/outline
        'grid_line': '#E0E0E0',         # Light gray grid lines
    })

    # Performance settings (reference code approach)
    max_bars: int = 50  # Maximum bars to render (prevents performance issues)

    # Display settings
    price_level_height: float = 0.8    # Height of each price level in plot units
    font_size: int = 8
    show_text_overlay: bool = True


class FootprintVisualizer:
    """Professional Footprint Visualizer - Direct reference code logic implementation"""

    def __init__(self, config: Optional[FootprintConfig] = None):
        """Initialize with optional configuration"""
        self.config = config or FootprintConfig()

    @staticmethod
    def plot_candlestick(footprint_data, title="CME Footprint Candlestick"):
        """
        Plot candlestick chart from footprint data

        Args:
            footprint_data: MultiIndex DataFrame with OHLC flags
            title: Chart title

        Returns:
            Plotly Figure object
        """
        if footprint_data.empty:
            return go.Figure().add_annotation(text="No data available",
                                             xref="paper", yref="paper",
                                             x=0.5, y=0.5, showarrow=False)

        # Extract OHLC for each bar
        ohlc_data = []
        for timestamp in footprint_data.index.get_level_values(0).unique():
            bar = footprint_data.loc[timestamp]
            prices = bar.index.tolist()

            # Extract prices from flags with proper fallbacks
            open_price = bar[bar['is_open']].index[0] if bar['is_open'].any() else prices[0]
            high_price = bar[bar['is_high']].index[0] if bar['is_high'].any() else max(prices)
            low_price = bar[bar['is_low']].index[0] if bar['is_low'].any() else min(prices)
            close_price = bar[bar['is_close']].index[0] if bar['is_close'].any() else prices[-1]

            volume = bar['total_vol'].sum()

            ohlc_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df_ohlc = pd.DataFrame(ohlc_data)

        # Create subplots with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Volume')
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_ohlc['timestamp'],
                open=df_ohlc['open'],
                high=df_ohlc['high'],
                low=df_ohlc['low'],
                close=df_ohlc['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add volume bars
        colors = ['red' if close < open else 'green'
                  for close, open in zip(df_ohlc['close'], df_ohlc['open'])]

        fig.add_trace(
            go.Bar(
                x=df_ohlc['timestamp'],
                y=df_ohlc['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=False
        )

        return fig

    def plot_footprint(self, footprint_data, ticksize=None, scaling_mode='local',
                      start_time=None, end_time=None, config=None) -> go.Figure:
        """
        Professional footprint chart - Direct implementation of reference code logic

        Args:
            footprint_data: MultiIndex DataFrame with footprint data
            ticksize: Price increment for formatting (e.g., 0.1 for GC)
            scaling_mode: Volume bar scaling strategy ('local' or 'global')
            start_time: Optional start timestamp for filtering
            end_time: Optional end timestamp for filtering
            config: Optional configuration override

        Returns:
            go.Figure: Professional footprint chart
        """
        # Use provided config or instance config
        cfg = config or self.config

        if footprint_data.empty:
            return go.Figure().add_annotation(text="No data available",
                                             xref="paper", yref="paper",
                                             x=0.5, y=0.5, showarrow=False)

        # 1. Direct data extraction with performance optimization
        timestamps = footprint_data.index.get_level_values(0).unique()

        # Filter by time range if specified
        if start_time:
            mask = timestamps >= start_time
            timestamps = timestamps[mask]
        if end_time:
            mask = timestamps <= end_time
            timestamps = timestamps[mask]

        # Performance optimization: limit bars (reference code approach)
        if len(timestamps) > cfg.max_bars:
            timestamps = timestamps[-cfg.max_bars:]  # Take last max_bars

        if len(timestamps) == 0:
            return go.Figure().add_annotation(text="No data in time range",
                                             xref="paper", yref="paper",
                                             x=0.5, y=0.5, showarrow=False)

        # 2. Price mapping (reference code algorithm, optimized for our data)
        price_mapping = self._calculate_price_mapping(footprint_data, timestamps, ticksize)

        # 3. Three-column layout rendering (exactly like reference code)
        return self._render_atas_layout(footprint_data, timestamps, price_mapping, scaling_mode, cfg)

    def _calculate_price_mapping(self, footprint_data, timestamps, ticksize):
        """Efficient price mapping algorithm - based on reference code approach"""

        # Find price range across all selected timestamps (more efficient)
        min_price = float('inf')
        max_price = float('-inf')

        for timestamp in timestamps:
            bar_data = footprint_data.loc[timestamp]
            if len(bar_data) > 0:
                bar_min = bar_data.index.min()
                bar_max = bar_data.index.max()
                min_price = min(min_price, bar_min)
                max_price = max(max_price, bar_max)

        if min_price == float('inf'):
            return {'price_levels': [], 'price_to_y': {}, 'num_levels': 0}

        # Use numpy-based approach for efficient price level generation
        if ticksize:
            import numpy as np

            # Align to ticksize boundaries (reference code algorithm)
            max_aligned = np.ceil(max_price / ticksize) * ticksize
            min_aligned = np.floor(min_price / ticksize) * ticksize

            # Calculate number of levels efficiently
            num_levels = int(np.round((max_aligned - min_aligned) / ticksize)) + 1

            # Generate price levels using numpy (much more efficient than loop)
            price_levels = np.linspace(max_aligned, min_aligned, num_levels)
            price_levels = np.round(price_levels / ticksize) * ticksize  # Ensure ticksize alignment
            price_levels = price_levels.tolist()
        else:
            # Fallback: collect actual prices (but only from selected timestamps)
            all_prices = set()
            for timestamp in timestamps:
                bar_data = footprint_data.loc[timestamp]
                all_prices.update(bar_data.index.tolist())
            price_levels = sorted(all_prices, reverse=True)

        # Create price to y-coordinate mapping
        price_to_y = {price: i for i, price in enumerate(price_levels)}

        return {
            'price_levels': price_levels,
            'price_to_y': price_to_y,
            'num_levels': len(price_levels)
        }

    def _render_atas_layout(self, footprint_data, timestamps, price_mapping, scaling_mode, cfg):
        """Reference code three-column layout logic - direct implementation"""

        fig = go.Figure()

        # Calculate scaling reference (reference code algorithm)
        scaling_ref = self._calculate_scaling_reference(footprint_data, timestamps, scaling_mode)

        # Reference code three-column ratios
        total_ratio = sum(cfg.column_ratios)
        col_widths = [ratio / total_ratio for ratio in cfg.column_ratios]

        # Render each bar (reference code structure)
        for bar_idx, timestamp in enumerate(timestamps):
            bar_data = footprint_data.loc[timestamp]

            # Extract OHLC (direct from our data structure)
            ohlc = self._extract_ohlc(bar_data)

            x_center = bar_idx

            # Column positions (reference code layout)
            col1_start = x_center - 0.4
            col1_width = col_widths[0] * 0.8
            col2_start = x_center - 0.4 + col_widths[0] * 0.8
            col2_width = col_widths[1] * 0.8
            col3_start = x_center - 0.4 + (col_widths[0] + col_widths[1]) * 0.8
            col3_width = col_widths[2] * 0.8

            # Reference code three-column rendering
            self._render_column1_body(fig, ohlc, col1_start, col1_width, price_mapping, cfg)
            self._render_column2_delta(fig, bar_data, col2_start, col2_width, price_mapping, scaling_ref, cfg)
            self._render_column3_volume(fig, bar_data, col3_start, col3_width, price_mapping, scaling_ref, cfg)

            # Text overlay (reference code style)
            if cfg.show_text_overlay:
                self._render_text_overlay(fig, bar_data, x_center, price_mapping, cfg)

        # Configure axes and layout
        self._configure_figure(fig, price_mapping, timestamps, cfg)

        return fig

    def _extract_ohlc(self, bar_data):
        """Extract OHLC from our data structure - simple and direct"""
        prices = bar_data.index.tolist()

        open_price = bar_data[bar_data['is_open']].index[0] if bar_data['is_open'].any() else prices[0]
        high_price = bar_data[bar_data['is_high']].index[0] if bar_data['is_high'].any() else max(prices)
        low_price = bar_data[bar_data['is_low']].index[0] if bar_data['is_low'].any() else min(prices)
        close_price = bar_data[bar_data['is_close']].index[0] if bar_data['is_close'].any() else prices[-1]

        return {
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        }

    def _calculate_scaling_reference(self, footprint_data, timestamps, scaling_mode):
        """Reference code scaling calculation"""
        if scaling_mode == 'global':
            # Global scaling: find max across all bars
            max_delta = 0
            max_volume = 0

            for timestamp in timestamps:
                bar_data = footprint_data.loc[timestamp]
                if not bar_data.empty:
                    max_delta = max(max_delta, bar_data['delta'].abs().max())
                    max_volume = max(max_volume, bar_data['total_vol'].max())

            return {'max_delta': max_delta, 'max_volume': max_volume, 'mode': 'global'}
        else:
            # Local scaling will be calculated per bar
            return {'mode': 'local'}

    def _render_column1_body(self, fig, ohlc, col_start, col_width, price_mapping, cfg):
        """Render candlestick body in column 1 - reference code logic"""

        # Get y-coordinates for OHLC prices
        open_y = self._price_to_y(ohlc['open'], price_mapping)
        high_y = self._price_to_y(ohlc['high'], price_mapping)
        low_y = self._price_to_y(ohlc['low'], price_mapping)
        close_y = self._price_to_y(ohlc['close'], price_mapping)

        if any(y is None for y in [open_y, high_y, low_y, close_y]):
            return

        # Determine color (reference code logic)
        is_bullish = ohlc['close'] >= ohlc['open']
        body_color = cfg.colors['body_bullish'] if is_bullish else cfg.colors['body_bearish']

        # Draw candlestick body (corrected for our coordinate system)
        # In our system: smaller y = higher price, larger y = lower price
        body_top = min(open_y, close_y)  # Higher price (smaller y)
        body_bottom = max(open_y, close_y)  # Lower price (larger y)

        if body_top == body_bottom:
            body_bottom += 0.1  # Minimum height

        fig.add_shape(
            type="rect",
            x0=col_start, x1=col_start + col_width,
            y0=body_top, y1=body_bottom,  # Top to bottom
            fillcolor=body_color,
            line=dict(color=cfg.colors['body_border'], width=1),
            opacity=0.8
        )

        # Draw shadows (corrected for our coordinate system)
        center_x = col_start + col_width / 2

        # Upper shadow (high price to body top)
        if high_y < body_top:  # High is above body (smaller y)
            fig.add_shape(
                type="line",
                x0=center_x, x1=center_x,
                y0=high_y, y1=body_top,
                line=dict(color=cfg.colors['body_border'], width=2)
            )

        # Lower shadow (body bottom to low price)
        if low_y > body_bottom:  # Low is below body (larger y)
            fig.add_shape(
                type="line",
                x0=center_x, x1=center_x,
                y0=body_bottom, y1=low_y,
                line=dict(color=cfg.colors['body_border'], width=2)
            )

    def _render_column2_delta(self, fig, bar_data, col_start, col_width, price_mapping, scaling_ref, cfg):
        """Render delta bars in column 2 - reference code logic"""

        # Calculate scaling reference (reference code approach)
        if scaling_ref['mode'] == 'local':
            max_delta = bar_data['delta'].abs().max() if not bar_data.empty else 1
        else:
            max_delta = scaling_ref['max_delta']

        if max_delta == 0:
            max_delta = 1

        col_center = col_start + col_width / 2

        # Render each price level (reference code structure)
        for price in bar_data.index:
            delta = bar_data.loc[price, 'delta']
            if delta == 0:
                continue

            y_coord = self._price_to_y(price, price_mapping)
            if y_coord is None:
                continue

            # Calculate bar width and position (reference code algorithm)
            width_ratio = abs(delta) / max_delta
            bar_width = width_ratio * col_width * 0.45  # Max 45% of column width

            if delta > 0:
                # Positive delta: green bar extending right from center
                bar_start = col_center
                color = cfg.colors['delta_positive']
            else:
                # Negative delta: red bar extending left from center
                bar_start = col_center - bar_width
                color = cfg.colors['delta_negative']

            # Draw delta bar (reference code style)
            fig.add_shape(
                type="rect",
                x0=bar_start, x1=bar_start + bar_width,
                y0=y_coord - cfg.price_level_height/2, y1=y_coord + cfg.price_level_height/2,
                fillcolor=color,
                line=dict(width=0),
                opacity=0.7
            )

    def _render_column3_volume(self, fig, bar_data, col_start, col_width, price_mapping, scaling_ref, cfg):
        """Render volume bars in column 3 - reference code logic"""

        # Calculate scaling reference (reference code approach)
        if scaling_ref['mode'] == 'local':
            max_volume = bar_data['total_vol'].max() if not bar_data.empty else 1
        else:
            max_volume = scaling_ref['max_volume']

        if max_volume == 0:
            max_volume = 1

        # Render each price level (reference code structure)
        for price in bar_data.index:
            volume = bar_data.loc[price, 'total_vol']
            if volume == 0:
                continue

            y_coord = self._price_to_y(price, price_mapping)
            if y_coord is None:
                continue

            # Calculate bar width (reference code algorithm)
            width_ratio = volume / max_volume
            bar_width = width_ratio * col_width * 0.9  # Max 90% of column width

            # Draw volume bar from left (reference code style)
            fig.add_shape(
                type="rect",
                x0=col_start, x1=col_start + bar_width,
                y0=y_coord - cfg.price_level_height/2, y1=y_coord + cfg.price_level_height/2,
                fillcolor=cfg.colors['volume_bar'],
                line=dict(width=0),
                opacity=0.6
            )

    def _render_text_overlay(self, fig, bar_data, x_center, price_mapping, cfg):
        """Render text overlay - reference code style"""

        for price in bar_data.index:
            y_coord = self._price_to_y(price, price_mapping)
            if y_coord is None:
                continue

            bid_vol = int(bar_data.loc[price, 'bid_vol'])
            ask_vol = int(bar_data.loc[price, 'ask_vol'])
            total_vol = int(bar_data.loc[price, 'total_vol'])
            delta = int(bar_data.loc[price, 'delta'])

            if total_vol == 0:
                continue

            # Column 2 text: "ask bid" format (reference code format)
            if bid_vol > 0 or ask_vol > 0:
                bid_ask_text = f"{ask_vol} {bid_vol}"
                fig.add_annotation(
                    x=x_center - 0.1, y=y_coord,
                    text=bid_ask_text,
                    showarrow=False,
                    font=dict(size=cfg.font_size, color=cfg.colors['text_main']),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=0.5
                )

            # Column 3 text: total volume and delta (reference code approach)
            if total_vol > 0:
                # Total volume
                fig.add_annotation(
                    x=x_center + 0.1, y=y_coord,
                    text=str(total_vol),
                    showarrow=False,
                    font=dict(size=cfg.font_size, color=cfg.colors['text_main']),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=0.5
                )

            # Delta with sign (reference code format)
            if delta != 0:
                delta_text = f"{delta:+d}"
                fig.add_annotation(
                    x=x_center + 0.25, y=y_coord,
                    text=delta_text,
                    showarrow=False,
                    font=dict(size=cfg.font_size, color=cfg.colors['text_main']),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=0.5
                )

    def _configure_figure(self, fig, price_mapping, timestamps, cfg):
        """Configure figure layout - reference code style"""

        # Configure axes (reference code approach)
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(len(timestamps))),
            ticktext=[ts.strftime('%H:%M') for ts in timestamps],
            title_text="Time",
            showgrid=True,
            gridcolor=cfg.colors['grid_line'],
            gridwidth=0.5
        )

        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(price_mapping['price_levels']))),
            ticktext=[f"{price:.2f}" for price in price_mapping['price_levels']],
            title_text="Price",
            showgrid=True,
            gridcolor=cfg.colors['grid_line'],
            gridwidth=0.5,
            autorange='reversed'  # Critical: reverse y-axis to match reference code (high prices at top)
        )

        # Update layout (reference code style)
        fig.update_layout(
            plot_bgcolor=cfg.colors['background'],
            paper_bgcolor=cfg.colors['background'],
            font=dict(size=cfg.font_size),
            margin=dict(l=80, r=40, t=40, b=60),
            showlegend=False,
            height=600 + len(price_mapping['price_levels']) * 15,
            width=max(800, len(timestamps) * 100)
        )

    def _price_to_y(self, price, price_mapping):
        """Convert price to y-coordinate - reference code algorithm"""
        if 'price_to_y' not in price_mapping:
            return None

        # Try exact match first
        if price in price_mapping['price_to_y']:
            return price_mapping['price_to_y'][price]

        # Find closest price if no exact match (reference code tolerance)
        closest_price = min(
            price_mapping['price_to_y'].keys(),
            key=lambda x: abs(x - price)
        )

        return price_mapping['price_to_y'][closest_price]


# Convenience functions
def create_footprint_config(**kwargs) -> FootprintConfig:
    """Create FootprintConfig with custom parameters"""
    return FootprintConfig(**kwargs)


def save_footprint_chart(footprint_data, filename, **kwargs):
    """Save footprint chart to file"""
    viz = FootprintVisualizer()
    fig = viz.plot_footprint(footprint_data, **kwargs)

    if filename.endswith('.html'):
        fig.write_html(filename)
    else:
        fig.write_image(filename)

    print(f"Footprint chart saved to: {filename}")