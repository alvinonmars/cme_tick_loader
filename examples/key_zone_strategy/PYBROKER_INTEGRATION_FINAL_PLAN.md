# PyBroker集成最终方案

**版本**: 1.0 Final
**日期**: 2025-10-16
**状态**: Ready for Implementation

---

## 📋 方案概述

### 核心思想

将复杂的KeyZoneStrategy逻辑**预先计算**为简单的indicator列，然后用PyBroker进行专业回测。

**关键优势**：
- ✅ 完全分离计算层和决策层
- ✅ PyBroker不需要知道footprint数据的存在
- ✅ 保持KeyZoneStrategy核心逻辑不变
- ✅ 止损止盈基于真实市场结构（zone center price）
- ✅ 简洁、可维护

### 架构流程

```
┌─────────────────────────────────────────┐
│ Step 1: 加载原始CME数据                  │
│ - CMEDataSource.query()                 │
│ - bars + footprints                     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Step 2: 预计算Key Zone Indicators       │
│ - KeyZoneIndicatorComputer              │
│ - 输入: bars, footprints                │
│ - 输出: enhanced_bars (9 new columns)  │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Step 3: 缓存（可选）                     │
│ - 保存到parquet                         │
│ - 后续直接加载                           │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Step 4: PyBroker Backtest               │
│ - Strategy + exec_fn                    │
│ - 只使用预计算的indicator列              │
└─────────────────────────────────────────┘
```

---

## 🎯 设计决策总结

所有决策基于深度讨论后确定，不再更改：

| 决策点 | 最终选择 | 说明 |
|--------|----------|------|
| Zone Touch粒度 | 区分类型 | `support_touch`, `resistance_touch` |
| 支撑/阻力数量 | 2个 | `_1` (最近), `_2` (第二近) |
| "最近"定义 | 距离 | `abs(price - close)` 排序 |
| Zone价格 | center_price | **不是**zone边界 |
| Structure表达 | 分离 | `type`, `direction`（**无strength**）|
| Structure强度 | 不使用 | 保持简洁 |
| Touch强度 | 不使用 | 保持简洁 |
| 同时touch | 重置为None | `last_touch = None` |
| 止损Buffer | ATR比例 | `0.2 * ATR` |
| 无阻力 | Trailing stop | `min(ATR, distance)` |
| Entry价格 | Close | `ctx.buy_limit_price = close` |
| ATR | 预计算 | 添加`atr`列 |
| Structure优先级 | 多个→忽略 | 只保留一个signal |
| 时间范围 | 可配置 | `start_date`, `end_date` |
| 仓位 | 固定 | `position_size = 100` |
| 方向 | 多空都支持 | `enable_long`, `enable_short` |

---

## 📊 Indicator Schema（预计算列）

共9列添加到bars DataFrame：

```python
# Zone Touch (2列)
support_touch: bool         # 当前bar是否touch任何support zone
resistance_touch: bool      # 当前bar是否touch任何resistance zone

# Support Prices (2列)
support_1: float           # 最近的support zone的center_price (可NaN)
support_2: float           # 第二近的support zone的center_price (可NaN)

# Resistance Prices (2列)
resistance_1: float        # 最近的resistance zone的center_price (可NaN)
resistance_2: float        # 第二近的resistance zone的center_price (可NaN)

# Structure Signal (2列)
structure_type: str        # 'v_reversal', 'breakout', None
structure_direction: str   # 'bullish', 'bearish', None

# Technical Indicator (1列)
atr: float                 # Average True Range (预计算)
```

### 重要说明

1. **Zone价格是center_price**：
   - Zone定义：`[center_price - half_width, center_price + half_width]`
   - 暴露的是`center_price`，不是边界
   - 止损基于`center_price - buffer`，不是`lower_bound`

2. **无strength相关列**：
   - 不记录`structure_strength`
   - 不记录`touch_strength`
   - 不记录`zone_strength`
   - 保持最简洁

3. **Structure的处理**：
   - 如果同时检测到多个structure（v_reversal + breakout）→ 都设为None（忽略）
   - 只有唯一一个structure时才记录

4. **NaN的语义**：
   - `support_1 = NaN` → 当前没有support zone（下方无关键区）
   - `resistance_1 = NaN` → 当前没有resistance zone（上方无关键区）

---

## ⚙️ 配置类

```python
from dataclasses import dataclass

@dataclass
class KeyZoneBacktestConfig:
    """
    Key Zone Strategy的PyBroker回测配置

    原则：简洁、可配置、无冗余
    """

    # ========== Zone Detection（预计算阶段使用） ==========
    big_delta_lookback: int = 50           # Big delta回溯窗口
    peak_lookback: int = 50                # Peak检测回溯窗口
    n_keep_prices: int = 5                 # 保留的key prices数量（每个方向）
    zone_ticks: int = 20                   # Zone宽度（ticks）
    min_peak_prominence_atr: float = 0.5   # Peak最小prominence（ATR倍数）
    atr_period: int = 14                   # ATR计算周期
    ticksize: float = 0.1                  # 品种tick大小（GC=0.1）

    # ========== Signal Combination（exec_fn使用） ==========
    touch_window: int = 3                  # 检查最近N个bars的touch

    # ========== Risk Management ==========
    stop_buffer_atr_ratio: float = 0.2     # 止损buffer = ratio * ATR
    trailing_stop_atr_cap: float = 1.0     # Trailing止损上限 = cap * ATR

    # ========== Position Management ==========
    position_size: int = 100               # 每次交易shares数量
    enable_long: bool = True               # 是否允许做多
    enable_short: bool = True              # 是否允许做空

    # ========== PyBroker Settings ==========
    initial_cash: float = 100_000          # 初始资金
    max_long_positions: int = 1            # 最多同时持有多头数
    max_short_positions: int = 1           # 最多同时持有空头数

    # ========== Backtest Range（可配置）==========
    start_date: str = '2024-01-02'
    end_date: str = '2024-01-31'
    symbol: str = 'GC'
```

### 配置使用示例

```python
# 默认配置
config = KeyZoneBacktestConfig()

# 自定义配置
config_custom = KeyZoneBacktestConfig(
    touch_window=5,              # 更宽的touch窗口
    stop_buffer_atr_ratio=0.3,   # 更宽的止损
    start_date='2024-01-01',
    end_date='2024-03-31',       # 测试Q1
)

# 只做多配置
config_long_only = KeyZoneBacktestConfig(
    enable_long=True,
    enable_short=False,
)
```

---

## 🔧 核心逻辑实现

### 1. get_last_valid_touch（关键函数）

```python
def get_last_valid_touch(ctx, window=3):
    """
    获取最近N个bars中最后一次有效touch的类型

    规则：
    - 只touch一个方向（support或resistance）→ 有效，记录
    - 同时touch两个方向 → 无效，重置为None
    - 最后返回最近的有效touch类型

    Args:
        ctx: PyBroker ExecContext
        window: 回溯窗口（bars数）

    Returns:
        'support', 'resistance', 或 None
    """
    last_touch = None

    for i in range(-window, 0):  # 从 T-window 到 T-1
        support = ctx.support_touch[i]
        resistance = ctx.resistance_touch[i]

        # 同时touch两个方向 → 重置（市场不确定）
        if support and resistance:
            last_touch = None  # 重要：重置，不是跳过
            continue

        # 只touch一个方向 → 记录
        if support:
            last_touch = 'support'
        elif resistance:
            last_touch = 'resistance'

    return last_touch
```

**关键点**：
- 同时touch时必须`last_touch = None`，不能只`continue`
- 这确保了信号的清晰性（避免模糊信号）

**测试案例**：
```
Bar T-3: support_touch=True  → last_touch = 'support'
Bar T-2: both=True           → last_touch = None（重置）
Bar T-1: nothing             → last_touch = None（不变）
Bar T:   bullish structure   → 不匹配 → 不交易 ✓
```

---

### 2. exec_fn主逻辑

```python
def create_exec_fn(config: KeyZoneBacktestConfig):
    """
    工厂函数：创建PyBroker兼容的exec_fn

    Args:
        config: KeyZoneBacktestConfig实例

    Returns:
        exec_fn函数
    """

    def exec_fn(ctx: ExecContext):
        """
        PyBroker执行函数

        在每个bar调用，决定是否交易
        """

        # ===== Step 1: 获取最近有效touch =====
        last_touch = get_last_valid_touch(ctx, window=config.touch_window)

        # ===== Step 2: 获取当前structure =====
        structure_direction = ctx.structure_direction[-1]  # 'bullish', 'bearish', None

        # ===== Step 3: 信号组合判断 =====
        should_long = (
            config.enable_long and
            last_touch == 'support' and      # 最近touch了support
            structure_direction == 'bullish'  # 当前有bullish结构
        )

        should_short = (
            config.enable_short and
            last_touch == 'resistance' and    # 最近touch了resistance
            structure_direction == 'bearish'  # 当前有bearish结构
        )

        # ===== Step 4: Entry执行 =====
        if should_long and not ctx.long_pos() and not ctx.short_pos():
            enter_long(ctx, config)

        elif should_short and not ctx.long_pos() and not ctx.short_pos():
            enter_short(ctx, config)

        # ===== Step 5: Exit逻辑（反向信号平仓）=====
        if ctx.long_pos() and structure_direction == 'bearish':
            ctx.sell_all_shares()  # 平多

        if ctx.short_pos() and structure_direction == 'bullish':
            ctx.cover_all_shares()  # 平空

    return exec_fn
```

---

### 3. enter_long（做多逻辑）

```python
def enter_long(ctx, config):
    """
    执行做多entry

    Args:
        ctx: PyBroker ExecContext
        config: KeyZoneBacktestConfig
    """

    # ===== 获取价格数据 =====
    estimated_entry = ctx.close[-1]        # 当前close（作为entry估算）
    support_1 = ctx.support_1[-1]          # 最近的support（center_price）
    resistance_1 = ctx.resistance_1[-1]    # 最近的resistance（center_price）
    atr = ctx.atr[-1]                      # 预计算的ATR

    # ===== 前置检查：必须有support =====
    if pd.isna(support_1):
        return  # 无support，不进场（无止损依据）

    # ===== 计算止损价格 =====
    buffer = config.stop_buffer_atr_ratio * atr  # Buffer = 0.2 * ATR
    stop_price = support_1 - buffer              # 止损价 = support中心 - buffer
    stop_distance = abs(estimated_entry - stop_price)  # 止损距离（正数）

    # ===== Entry Order =====
    ctx.buy_limit_price = estimated_entry  # 用close作为limit price
    ctx.buy_shares = config.position_size  # 固定仓位

    # ===== 设置止损止盈 =====
    if pd.isna(resistance_1):
        # 情况1: 无上方阻力 → 使用Trailing Stop（让利润run）
        trailing_distance = min(
            atr * config.trailing_stop_atr_cap,  # 上限：1.0 * ATR
            stop_distance                         # 或者到support的距离
        )
        ctx.stop_trailing = trailing_distance  # PyBroker自动跟踪

    else:
        # 情况2: 有上方阻力 → 使用Fixed Stop Loss + Take Profit
        ctx.stop_loss = stop_distance  # 固定止损（正数，PyBroker自动计算方向）

        # 止盈：略低于resistance
        profit_distance = resistance_1 - estimated_entry
        if profit_distance > 0:
            ctx.stop_profit = profit_distance  # 固定止盈（正数）
```

**关键点**：
1. **必须有support**：无support不进场（风险不可控）
2. **止损基于center_price**：不是zone边界，而是关键价格
3. **PyBroker的stop都是正数**：方向自动处理
4. **无阻力用trailing**：捕获趋势
5. **有阻力用fixed**：锁定利润

---

### 4. enter_short（做空逻辑）

```python
def enter_short(ctx, config):
    """
    执行做空entry（镜像做多逻辑）

    Args:
        ctx: PyBroker ExecContext
        config: KeyZoneBacktestConfig
    """

    # ===== 获取价格数据 =====
    estimated_entry = ctx.close[-1]
    resistance_1 = ctx.resistance_1[-1]    # 最近的resistance
    support_1 = ctx.support_1[-1]          # 最近的support
    atr = ctx.atr[-1]

    # ===== 前置检查：必须有resistance =====
    if pd.isna(resistance_1):
        return  # 无resistance，不进场

    # ===== 计算止损价格 =====
    buffer = config.stop_buffer_atr_ratio * atr
    stop_price = resistance_1 + buffer         # 止损价 = resistance中心 + buffer
    stop_distance = abs(stop_price - estimated_entry)  # 正数

    # ===== Entry Order =====
    ctx.sell_limit_price = estimated_entry
    ctx.sell_shares = config.position_size

    # ===== 设置止损止盈 =====
    if pd.isna(support_1):
        # 无下方支撑 → Trailing Stop
        trailing_distance = min(
            atr * config.trailing_stop_atr_cap,
            stop_distance
        )
        ctx.stop_trailing = trailing_distance

    else:
        # 有下方支撑 → Fixed Stops
        ctx.stop_loss = stop_distance

        profit_distance = estimated_entry - support_1
        if profit_distance > 0:
            ctx.stop_profit = profit_distance
```

---

## 🔬 PyBroker Stop机制说明

从PyBroker源码确认的行为：

### Stop类型

```python
# 1. Fixed Stop Loss（固定止损）
ctx.stop_loss = 5.0  # 距离entry 5个点
# 多头：实际止损 = entry_price - 5.0
# 空头：实际止损 = entry_price + 5.0

# 2. Fixed Take Profit（固定止盈）
ctx.stop_profit = 10.0  # 距离entry 10个点
# 多头：实际止盈 = entry_price + 10.0
# 空头：实际止盈 = entry_price - 10.0

# 3. Trailing Stop（跟踪止损）
ctx.stop_trailing = 3.0  # 距离最高/低价 3个点
# 多头：止损 = max(high) - 3.0（只升不降）
# 空头：止损 = min(low) + 3.0（只降不升）
```

### 关键原则

1. **所有距离都是正数**：方向由PyBroker根据多空自动判断
2. **Trailing自动更新**：PyBroker每个bar自动计算新的trailing价格
3. **不能同时用stop_loss和stop_trailing**：只能选一个

### 我们的使用

```python
# 有明确目标 → Fixed stops
ctx.stop_loss = distance_to_support
ctx.stop_profit = distance_to_resistance

# 无明确目标 → Trailing stop
ctx.stop_trailing = min(ATR, distance_to_support)
```

---

## 🏗️ 预计算Pipeline设计

### KeyZoneIndicatorComputer类

```python
from tqdm import tqdm
import numpy as np
import pandas as pd

class KeyZoneIndicatorComputer:
    """
    预计算Key Zone Strategy的indicators

    职责：
    - 调用KeyZoneStrategy逐bar计算
    - 提取结果转换为9个indicator列
    - 处理特殊情况（同时多个structure等）
    """

    def __init__(self, config: StrategyConfig):
        """
        Args:
            config: KeyZoneStrategy的配置（不是BacktestConfig）
        """
        from key_zone_strategy import KeyZoneStrategy

        self.strategy = KeyZoneStrategy(config=config)
        self.config = config

    def compute_all(
        self,
        bars: pd.DataFrame,
        footprints: List[pd.DataFrame],
        symbol: str
    ) -> pd.DataFrame:
        """
        对所有bars预计算indicators

        Args:
            bars: OHLCV DataFrame
            footprints: List of footprint DataFrames（每个bar一个）
            symbol: 品种代码

        Returns:
            enhanced_bars: 原始bars + 9个新列
        """

        # 复制原始DataFrame
        bars = bars.copy()

        # 初始化9列
        bars['support_touch'] = False
        bars['resistance_touch'] = False
        bars['support_1'] = np.nan
        bars['support_2'] = np.nan
        bars['resistance_1'] = np.nan
        bars['resistance_2'] = np.nan
        bars['structure_type'] = None
        bars['structure_direction'] = None
        bars['atr'] = np.nan

        # 逐bar计算
        print(f"\n预计算 {symbol} indicators...")
        for i in tqdm(range(len(bars)), desc=f"{symbol}"):

            # 跳过warmup期
            if i < self.config.big_delta_lookback:
                continue

            # 调用KeyZoneStrategy
            try:
                result = self.strategy.update(
                    bars=bars.iloc[:i+1],
                    footprints=footprints[:i+1],
                    current_bar_index=i
                )
            except Exception as e:
                print(f"\nError at bar {i}: {e}")
                continue

            if not result:
                continue

            # 提取并填充数据
            self._extract_touch_info(bars, i, result)
            self._extract_zone_prices(bars, i, result)
            self._extract_structure_info(bars, i, result)
            self._extract_atr(bars, i, result)

        print(f"✓ 预计算完成")
        return bars

    def _extract_touch_info(self, bars, idx, result):
        """提取touch信息"""
        current_bar = bars.iloc[idx]
        zones = result['zones']

        for zone in zones:
            # 检查bar的high或low是否在zone内
            if zone.contains(current_bar['high']) or zone.contains(current_bar['low']):
                if zone.zone_type == ZoneType.SUPPORT:
                    bars.at[idx, 'support_touch'] = True
                else:
                    bars.at[idx, 'resistance_touch'] = True

    def _extract_zone_prices(self, bars, idx, result):
        """提取support/resistance价格（center_price）"""
        zones = result['zones']
        current_close = bars.iloc[idx]['close']

        # 分类
        support_zones = [z for z in zones if z.zone_type == ZoneType.SUPPORT]
        resistance_zones = [z for z in zones if z.zone_type == ZoneType.RESISTANCE]

        # 按距离排序
        support_zones_sorted = sorted(
            support_zones,
            key=lambda z: abs(z.center_price - current_close)
        )
        resistance_zones_sorted = sorted(
            resistance_zones,
            key=lambda z: abs(z.center_price - current_close)
        )

        # 取最近的2个
        if len(support_zones_sorted) >= 1:
            bars.at[idx, 'support_1'] = support_zones_sorted[0].center_price
        if len(support_zones_sorted) >= 2:
            bars.at[idx, 'support_2'] = support_zones_sorted[1].center_price

        if len(resistance_zones_sorted) >= 1:
            bars.at[idx, 'resistance_1'] = resistance_zones_sorted[0].center_price
        if len(resistance_zones_sorted) >= 2:
            bars.at[idx, 'resistance_2'] = resistance_zones_sorted[1].center_price

    def _extract_structure_info(self, bars, idx, result):
        """
        提取structure信息

        规则：
        - 如果同时有多个structural signals → 忽略（设为None）
        - 如果只有1个signal → 记录type和direction
        """
        structural_signals = result['structural_signals']

        # 只有唯一一个signal时才记录
        if len(structural_signals) == 1:
            signal = structural_signals[0]

            # 提取type
            signal_type_value = signal.signal_type.value  # 'v_reversal_bullish'
            if 'v_reversal' in signal_type_value:
                bars.at[idx, 'structure_type'] = 'v_reversal'
            elif 'breakout' in signal_type_value:
                bars.at[idx, 'structure_type'] = 'breakout'

            # 提取direction
            if signal.is_bullish():
                bars.at[idx, 'structure_direction'] = 'bullish'
            elif signal.is_bearish():
                bars.at[idx, 'structure_direction'] = 'bearish'

        # len(signals) == 0 或 > 1 时，保持None（忽略）

    def _extract_atr(self, bars, idx, result):
        """提取ATR"""
        # 使用KeyZoneDetector的_calculate_atr方法
        atr = self.strategy.zone_detector._calculate_atr(bars.iloc[:idx+1])
        bars.at[idx, 'atr'] = atr
```

### 使用示例

```python
# 1. 加载原始数据
from cme_tick_loader.examples.pybroker_integration import CMEDataSource

data_source = CMEDataSource(
    resolution='MIN',
    num_units=5,
    cache_footprint=True
)

# 2. 创建computer
from key_zone_strategy import StrategyConfig

strategy_config = StrategyConfig(
    big_delta_lookback=50,
    peak_lookback=50,
    zone_ticks=20,
    ticksize=0.1,
)

computer = KeyZoneIndicatorComputer(config=strategy_config)

# 3. 预计算
# 需要先从data_source获取bars和footprints
# （具体实现见后续）

enhanced_bars = computer.compute_all(bars, footprints, symbol='GC')

# 4. 可选：缓存结果
enhanced_bars.to_parquet('enhanced_bars_2024_01.parquet')
```

---

## 📝 完整使用流程

### 方法1：一体化流程

```python
from pybroker import Strategy, StrategyConfig as PyBrokerStrategyConfig
from key_zone_strategy import StrategyConfig, KeyZoneIndicatorComputer
from cme_tick_loader.examples.pybroker_integration import CMEDataSource

# ===== 配置 =====
backtest_config = KeyZoneBacktestConfig(
    start_date='2024-01-02',
    end_date='2024-01-31',
    symbol='GC',
)

# ===== Step 1: 加载原始数据 =====
data_source = CMEDataSource(
    resolution='MIN',
    num_units=5,
    cache_footprint=True
)

# 这里需要实现一个helper函数来获取bars和footprints
bars, footprints = load_cme_data(
    data_source,
    symbol=backtest_config.symbol,
    start_date=backtest_config.start_date,
    end_date=backtest_config.end_date
)

# ===== Step 2: 预计算indicators =====
strategy_config = StrategyConfig(
    big_delta_lookback=backtest_config.big_delta_lookback,
    peak_lookback=backtest_config.peak_lookback,
    # ... 其他参数
)

computer = KeyZoneIndicatorComputer(config=strategy_config)
enhanced_bars = computer.compute_all(bars, footprints, backtest_config.symbol)

# ===== Step 3: 创建PyBroker DataSource =====
# 需要将enhanced_bars转换为PyBroker格式
# 可以使用pandas_data等PyBroker内置方法

# ===== Step 4: 回测 =====
pybroker_config = PyBrokerStrategyConfig(
    initial_cash=backtest_config.initial_cash,
    max_long_positions=backtest_config.max_long_positions,
    max_short_positions=backtest_config.max_short_positions,
)

exec_fn = create_exec_fn(backtest_config)

strategy = Strategy(
    enhanced_data_source,
    start_date=backtest_config.start_date,
    end_date=backtest_config.end_date,
    config=pybroker_config
)

strategy.add_execution(
    fn=exec_fn,
    symbols=[backtest_config.symbol]
)

result = strategy.backtest(warmup=50)

# ===== Step 5: 分析结果 =====
print("\n" + "="*80)
print("Key Zone Strategy - PyBroker Backtest Results")
print("="*80)
print(f"\nTotal Return: {result.metrics.total_return_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2f}%")
print(f"Total Trades: {result.metrics.total_trades}")
print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
```

---

## 🧪 测试计划

### Phase 1: 预计算测试

```python
def test_indicator_computation():
    """测试indicator预计算"""

    # 1. 加载January 2024数据
    bars, footprints = load_cme_data('GC', '2024-01-02', '2024-01-31')

    # 2. 预计算
    computer = KeyZoneIndicatorComputer(config)
    enhanced = computer.compute_all(bars, footprints, 'GC')

    # 3. 验证
    assert 'support_touch' in enhanced.columns
    assert 'atr' in enhanced.columns
    assert len(enhanced) == len(bars)

    # 4. 统计检查
    print(f"Total bars: {len(enhanced)}")
    print(f"Support touches: {enhanced['support_touch'].sum()}")
    print(f"Resistance touches: {enhanced['resistance_touch'].sum()}")
    print(f"Bullish structures: {(enhanced['structure_direction']=='bullish').sum()}")
    print(f"Bearish structures: {(enhanced['structure_direction']=='bearish').sum()}")
```

### Phase 2: exec_fn单元测试

```python
def test_exec_fn_logic():
    """测试exec_fn逻辑"""

    # 模拟ctx
    mock_ctx = MockExecContext()
    mock_ctx.support_touch = [False, True, False]  # T-2 touched support
    mock_ctx.resistance_touch = [False, False, False]
    mock_ctx.structure_direction = [None, None, 'bullish']  # T现在bullish
    mock_ctx.support_1 = [np.nan, 1995.0, 1995.0]
    mock_ctx.close = [2000, 2001, 2002]
    mock_ctx.atr = [3.0, 3.0, 3.0]

    # 测试
    exec_fn = create_exec_fn(config)
    exec_fn(mock_ctx)

    # 验证
    assert mock_ctx.buy_shares == 100
    assert mock_ctx.stop_loss > 0
```

### Phase 3: 完整回测对比

```python
def test_vs_standalone():
    """对比PyBroker回测和standalone回测"""

    # 1. 运行standalone
    standalone_result = run_standalone_backtest('2024-01-01', '2024-01-31')

    # 2. 运行PyBroker
    pybroker_result = run_pybroker_backtest('2024-01-01', '2024-01-31')

    # 3. 对比关键指标
    assert abs(standalone_result.total_return - pybroker_result.total_return_pct) < 0.5  # ±0.5%容差
    assert abs(standalone_result.total_trades - pybroker_result.total_trades) < 5  # ±5笔容差

    print("✓ PyBroker结果与standalone一致")
```

---

## 📦 实施步骤

### Step 1: 基础设施（2小时）

- [ ] 创建`KeyZoneIndicatorComputer`类
- [ ] 实现`_extract_*`系列方法
- [ ] 单元测试预计算逻辑

### Step 2: exec_fn实现（2小时）

- [ ] 实现`create_exec_fn()`
- [ ] 实现`get_last_valid_touch()`
- [ ] 实现`enter_long()`和`enter_short()`
- [ ] 单元测试exec_fn逻辑

### Step 3: 数据Pipeline（1小时）

- [ ] 实现`load_cme_data()`辅助函数
- [ ] 实现enhanced_bars → PyBroker DataSource转换
- [ ] 测试数据加载

### Step 4: 完整回测（1小时）

- [ ] 运行January 2024回测
- [ ] 对比standalone结果
- [ ] 调试差异

### Step 5: 文档和示例（1小时）

- [ ] 编写usage示例
- [ ] 添加docstrings
- [ ] 创建Jupyter notebook示例

**总预估时间**: 7小时

---

## ⚠️ 关键注意事项

### 1. 数据对齐

```python
# 确保dates对齐
assert len(bars) == len(footprints)
assert all(bars['date_time'].values == [fp.index[0][0] for fp in footprints])
```

### 2. NaN处理

```python
# 检查NaN before使用
if pd.isna(ctx.support_1[-1]):
    return  # 不进场
```

### 3. PyBroker的stop是正数

```python
# 错误
ctx.stop_loss = entry - stop_price  # 可能是负数

# 正确
ctx.stop_loss = abs(entry - stop_price)  # 保证正数
```

### 4. 预计算性能

```python
# 对于长时间范围，考虑并行化
from multiprocessing import Pool

def compute_chunk(chunk_data):
    # 计算一个chunk
    pass

with Pool(4) as pool:
    results = pool.map(compute_chunk, chunks)
```

---

## 🎯 成功标准

回测被认为成功当：

1. ✅ **预计算完成**：无错误，所有9列正确生成
2. ✅ **PyBroker运行**：无crash，完成整个回测
3. ✅ **结果一致性**：与standalone回测差异<2%
4. ✅ **Trade数量接近**：±10笔容差
5. ✅ **性能可接受**：<5分钟完成January 2024

---

## 📚 参考资料

- PyBroker文档: https://www.pybroker.com
- PyBroker源码: `/Users/alvinma/Desktop/work/pybroker/src/pybroker/`
- Standalone backtest: `backtest_2024q1.py`
- Backtest report: `BACKTEST_REPORT_2024_JAN.md`

---

## ✅ 批准签字

**方案确认**: 用户已确认所有设计决策
**状态**: Ready to implement
**下一步**: 开始Step 1实施

---

**End of Final Plan**
