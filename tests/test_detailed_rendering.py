#!/usr/bin/env python3
"""深度验证footprint渲染的正确性"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cme_tick_loader import CMEFootprintLoader, FootprintVisualizer, FootprintConfig


def analyze_footprint_data(footprint_data, timestamps):
    """详细分析footprint数据以便验证渲染"""

    print(f"\n=== 数据分析 ===")
    print(f"时间范围: {timestamps[0]} 到 {timestamps[-1]}")
    print(f"总bars: {len(timestamps)}")

    # 分析每个bar
    for i, timestamp in enumerate(timestamps[:5]):  # 只分析前5个bar
        bar_data = footprint_data.loc[timestamp]

        # OHLC信息
        open_price = bar_data[bar_data['is_open']].index[0] if bar_data['is_open'].any() else None
        high_price = bar_data[bar_data['is_high']].index[0] if bar_data['is_high'].any() else None
        low_price = bar_data[bar_data['is_low']].index[0] if bar_data['is_low'].any() else None
        close_price = bar_data[bar_data['is_close']].index[0] if bar_data['is_close'].any() else None

        # Delta统计
        total_delta = bar_data['delta'].sum()
        max_positive_delta = bar_data[bar_data['delta'] > 0]['delta'].max() if (bar_data['delta'] > 0).any() else 0
        max_negative_delta = bar_data[bar_data['delta'] < 0]['delta'].min() if (bar_data['delta'] < 0).any() else 0

        # Volume统计
        total_volume = bar_data['total_vol'].sum()
        max_volume = bar_data['total_vol'].max()

        # POC (Point of Control)
        poc_price = bar_data['total_vol'].idxmax()

        print(f"\nBar {i+1} ({timestamp.strftime('%H:%M')}):")
        print(f"  OHLC: O={open_price:.1f} H={high_price:.1f} L={low_price:.1f} C={close_price:.1f}")
        print(f"  价格层级: {len(bar_data)} 个")
        print(f"  总Delta: {total_delta:+d}")
        print(f"  Delta范围: {max_negative_delta:+.0f} 到 {max_positive_delta:+.0f}")
        print(f"  总Volume: {total_volume}")
        print(f"  最大Volume: {max_volume} (在价格 {poc_price:.1f})")

        # 验证OHLC逻辑
        is_bullish = close_price >= open_price if (close_price and open_price) else False
        print(f"  蜡烛类型: {'看涨' if is_bullish else '看跌'}")

        # 价格分布
        price_range = high_price - low_price if (high_price and low_price) else 0
        print(f"  价格范围: {price_range:.1f}")


def verify_rendering_details(fig, footprint_data, timestamps, config):
    """验证渲染细节"""

    print(f"\n=== 渲染验证 ===")

    # 1. 检查图形元素数量
    shapes_count = len(fig.layout.shapes) if fig.layout.shapes else 0
    annotations_count = len(fig.layout.annotations) if fig.layout.annotations else 0

    print(f"图形形状数量: {shapes_count}")
    print(f"注释数量: {annotations_count}")

    # 2. 验证列宽比例
    expected_ratios = config.column_ratios
    total_ratio = sum(expected_ratios)
    expected_percentages = [ratio/total_ratio for ratio in expected_ratios]

    print(f"列宽比例配置: {expected_ratios} = {[f'{p:.1%}' for p in expected_percentages]}")

    # 3. 检查价格轴设置
    if fig.layout.yaxis:
        autorange = getattr(fig.layout.yaxis, 'autorange', None)
        print(f"Y轴autorange: {autorange} {'✓' if autorange == 'reversed' else '✗'}")

        if hasattr(fig.layout.yaxis, 'ticktext') and fig.layout.yaxis.ticktext:
            price_labels = fig.layout.yaxis.ticktext
            print(f"价格标签数量: {len(price_labels)}")
            print(f"价格标签范围: {price_labels[0]} 到 {price_labels[-1]}")

    # 4. 分析图形形状类型
    if fig.layout.shapes:
        shape_types = {}
        for shape in fig.layout.shapes:
            shape_type = shape.type
            shape_types[shape_type] = shape_types.get(shape_type, 0) + 1

        print(f"图形类型分布: {shape_types}")

        # 预期: rectangles (蜡烛体 + delta条 + volume条) + lines (影线)
        expected_bars = len(timestamps)
        expected_min_shapes = expected_bars * 3  # 至少每个bar有3个shape (body + delta + volume)

        print(f"预期最少形状数: {expected_min_shapes} (bars={expected_bars} × 3)")
        print(f"实际形状数: {shapes_count} {'✓' if shapes_count >= expected_min_shapes else '✗'}")

    # 5. 检查颜色配置
    print(f"\n颜色配置验证:")
    colors = config.colors
    for key, color in colors.items():
        print(f"  {key}: {color}")


def main():
    """主验证流程"""

    print("=== FootprintVisualizer 深度渲染验证 ===")

    # 初始化
    loader = CMEFootprintLoader()
    viz = FootprintVisualizer()
    config = FootprintConfig()

    # 加载数据
    print("加载数据...")
    footprint = loader.load_footprint_bars('GC', '20210104', interval='5min')

    if footprint.empty:
        print("无数据")
        return False

    # 选择合理数量的bars进行验证
    all_timestamps = footprint.index.get_level_values(0).unique()
    test_timestamps = all_timestamps[-20:]  # 最后20个bars
    test_footprint = footprint.loc[test_timestamps]

    ticksize = loader.tick_loader.get_ticksize('GC')

    print(f"总数据: {len(all_timestamps)} bars")
    print(f"验证数据: {len(test_timestamps)} bars")
    print(f"价格层级: {len(test_footprint)}")
    print(f"Ticksize: {ticksize}")

    # 详细分析数据
    analyze_footprint_data(test_footprint, test_timestamps)

    # 生成图表
    print(f"\n=== 生成图表 ===")

    # Local scaling
    print("创建Local scaling图表...")
    fig_local = viz.plot_footprint(
        test_footprint,
        ticksize=ticksize,
        scaling_mode='local',
        config=config
    )

    # Global scaling
    print("创建Global scaling图表...")
    fig_global = viz.plot_footprint(
        test_footprint,
        ticksize=ticksize,
        scaling_mode='global',
        config=config
    )

    # 验证渲染细节
    verify_rendering_details(fig_local, test_footprint, test_timestamps, config)

    # 保存验证输出
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    fig_local.write_html(output_dir / "detailed_verification_local.html")
    fig_global.write_html(output_dir / "detailed_verification_global.html")

    print(f"\n=== 验证完成 ===")
    print(f"验证图表已保存到 {output_dir}/")
    print("请查看HTML文件来验证:")
    print("1. 高价格是否在上方，低价格在下方")
    print("2. 蜡烛体颜色是否正确（涨绿跌红）")
    print("3. Delta条是否从中心向两边延伸（正绿负红）")
    print("4. Volume条是否按比例显示")
    print("5. 三列布局比例是否为 2:12:10")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)