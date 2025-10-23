# CME Bars Loader - Technical Documentation

**Version 2.0** - Based on mlfinlab API

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Architecture](#core-architecture)
5. [API Reference](#api-reference)
6. [Data Preprocessor](#data-preprocessor)
7. [Usage Examples](#usage-examples)
8. [Caching Strategy](#caching-strategy)
9. [Technical Details](#technical-details)

---

## Overview

CME Bars Loader 是一个高效的金融数据加载库，基于 mlfinlab API，支持多种 bar 类型的聚合和缓存。

### Key Features

- **基于 mlfinlab**：100% 委托聚合逻辑给 mlfinlab，确保正确性
- **多种 Bar 类型**：支持 TIME, VOLUME, TICK, DOLLAR bars
- **两级缓存**：tick cache + result cache，提升性能
- **按天管理**：自动处理单天数据，避免跨天状态污染
- **时间戳修复**：自动修复 TIME bars 的 Unix timestamp
- **预聚合**：避免重复 (时间,价格) 索引问题

---

## Installation

### 开发安装

```bash
# 激活 conda 环境 (macOS)
source /opt/homebrew/Caskroom/miniconda/base/bin/activate cs

# 开发模式安装
pip install -e .

# 验证安装
python -c "from cme_tick_loader import CMEBarsLoader; print('OK')"
```

### 数据路径配置

**自动检测**:
```python
loader = CMEBarsLoader()  # 自动检测 base_path
```

**手动指定**:
```python
loader = CMEBarsLoader(base_path='/your/data/path')
```

**数据文件结构**:
```
base_path/
├── GC_1/
│   ├── GC_1_footprint_20210104.csv
│   └── ...
└── .cache/
    ├── ticks/          # Tick cache
    └── results/        # Result cache
```

---

## Quick Start

```python
from cme_tick_loader import CMEBarsLoader

# 初始化
loader = CMEBarsLoader()

# 加载 5分钟 TIME bars
result = loader.load_bars(
    symbol='GC',
    date='20210104',
    resolution='MIN',
    num_units=5
)

# 访问数据
bars = result['bars']           # OHLC + volume + 统计信息
footprint = result['footprint'] # MultiIndex(bar_timestamp, price)

print(f"Loaded {len(bars)} bars")
```

---

## Core Architecture

### 数据流

```
CME CSV (按天)
    ↓
TickLoader.load_ticks()
    ↓
转换为 mlfinlab 格式 + 预聚合
    ↓
mlfinlab API (batch_size=50000000)
    ↓
{'bars': DataFrame, 'footprint': DataFrame}
    ↓
修复 TIME bars timestamp
    ↓
返回结果
```

### 关键设计

| 决策点 | 设计 | 原因 |
|--------|------|------|
| resolution | 'MIN', 'H', 'VOLUME', etc. | 直接判断 bar type |
| batch_size | 50000000 | 确保单天一次处理 |
| 预聚合 | 按 (date_time, price) | 避免重复索引 |
| 缓存键 | {symbol}_{date}_{resolution}_{num_units} | 可读性 |

---

## API Reference

### CMEBarsLoader

#### `load_bars(symbol, date, resolution='MIN', num_units=5, use_cache=True, refresh_cache=False, verbose=False, timezone_naive=True, set_index=True)`

加载单日 bars 和 footprint（总是同时返回）。

**参数**:
- `symbol`: 'GC', 'ES', etc.
- `date`: 'YYYYMMDD'
- `resolution`:
  - TIME bars: `'D'`, `'H'`, `'MIN'`, `'S'`
  - 其他: `'VOLUME'`, `'TICK'`, `'DOLLAR'`
- `num_units`:
  - TIME bars: 时间单位数量
  - 其他: threshold
- `use_cache`: 是否使用缓存 (default: True)
- `refresh_cache`: 是否刷新缓存 (default: False)
- `verbose`: 是否打印mlfinlab日志 (default: False)
- `timezone_naive`: 是否移除timezone信息 (default: True)
  - `True`: datetime为naive（无timezone），适合大多数pandas操作
  - `False`: datetime为aware（UTC timezone），保留时区信息
- `set_index`: 是否将date_time设为DataFrame的index (default: True)
  - `True`: `bars.index = bars.date_time.values`（兼容mlfinlab），保留date_time列
  - `False`: 使用默认整数索引（RangeIndex）

**返回**:
```python
{
    'bars': DataFrame,      # OHLCV + 统计信息（详见下方列结构）
    'footprint': DataFrame  # MultiIndex(bar_timestamp, price)
                           # 列：bid_vol, ask_vol, total_vol, delta,
                           #     is_open, is_high, is_low, is_close
}
```

**bars DataFrame 列结构**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `date_time` | datetime64[ns] | Bar时间戳（TIME bars对齐到边界，其他类型为最后一个tick时间） |
| `tick_num` | int | Bar形成时的累积tick计数器 |
| `open_time_ms` | int | **第一个tick的Unix毫秒时间戳**（精确微观结构分析） |
| `close_time_ms` | int | **最后一个tick的Unix毫秒时间戳**（精确微观结构分析） |
| `open` | float | 开盘价（第一个tick价格） |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价（最后一个tick价格） |
| `volume` | int | 成交量 |
| `cum_buy_volume` | int | 累积买方主动成交量 |
| `cum_ticks` | int | Bar内tick数量 |
| `cum_dollar_value` | float | 累积成交金额 |

**重要**：`open_time_ms` 和 `close_time_ms` 是mlfinlab v1.6+新增的字段，用于解决TIME bars的精度问题：

- **TIME bars**：`date_time` 被对齐到时间边界（如10:00:00.000），但实际第一个tick可能在09:59:58.123，最后一个tick在10:00:00.456
- **其他bars**：`date_time` 等于最后一个tick的时间戳，但仍可通过 `open_time_ms` 获取第一个tick的精确时间

**使用场景**：
```python
# 计算bar的实际时长（微秒精度）
bars['duration_ms'] = bars['close_time_ms'] - bars['open_time_ms']

# 转换为datetime对象
bars['open_time'] = pd.to_datetime(bars['open_time_ms'], unit='ms')
bars['close_time'] = pd.to_datetime(bars['close_time_ms'], unit='ms')

# 对于TIME bars，检查对齐偏差
bars['alignment_offset_ms'] = (
    pd.to_datetime(bars['date_time']).astype('int64') // 10**6
    - bars['close_time_ms']
)
```

注意：API设计为总是返回完整的bars和footprint数据，用户可以选择性使用footprint。

#### `load_date_range(symbol, start_date, end_date, **kwargs)`

加载日期范围，将多天数据合并返回。

**参数**:
- `symbol`: 'GC', 'ES', etc.
- `start_date`: 'YYYYMMDD'
- `end_date`: 'YYYYMMDD'
- `**kwargs`: 传递给 `load_bars()` 的其他参数
  - `resolution`, `num_units`, `use_cache`, `refresh_cache`, `verbose`
  - `timezone_naive`, `set_index`

**注意**:
- 多日bars合并时使用 `ignore_index=True` 重建整数索引
- 如果 `set_index=True`，会在合并后重新设置datetime index
- footprint保持MultiIndex结构，包含所有日期

#### `clear_cache(level='all')`

清除缓存 ('tick' | 'result' | 'all')。

#### `get_cache_info()`

获取缓存信息。

#### `get_ticksize(symbol)`

获取指定交易品种的ticksize。

**参数**:
- `symbol`: 'GC', 'ES', etc.

**返回**:
- `float`: ticksize值

**内置ticksize**:
- GC (Gold): 0.1
- ES (E-mini S&P 500): 0.25
- NQ (E-mini NASDAQ): 0.25
- CL (Crude Oil): 0.01
- ZN (10-Year Treasury): 1/64 (0.015625)
- ZB (30-Year Treasury): 1/32 (0.03125)
- 未知符号: 0.01 (default)

**使用场景**:
```python
# 配合FootprintVisualizer使用
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
ticksize = loader.get_ticksize('GC')  # 0.1

viz = FootprintVisualizer()
fig = viz.plot_footprint(result['footprint'], ticksize=ticksize)
```

---

## StandardBarData - 标准 Bar 数据格式

**StandardBarData** 是所有下游系统（key_zone_detector, trading_system）的统一输入格式。

### 核心设计

```
CMEBarsLoader → StandardBarData.from_result() → key_zone_detector / trading_system
```

**关键保证**:
- `bars.index` == `footprint.index.levels[0]` （严格对齐）
- 避免重复计算：OHLC 由 mlfinlab 聚合，只计算 VWAP/POC/VAH/VAL
- 自动验证：去重、排序、timezone-naive

### 数据结构

```python
@dataclass
class StandardBarData:
    bars: pd.DataFrame          # OHLCV + VWAP + POC/VAH/VAL
    footprint_df: pd.DataFrame  # 原始 footprint
```

**增强列**:
- `vwap`: 成交量加权平均价
- `poc`: Point of Control（成交量最密集价格）
- `vah`: Value Area High（70%成交量上界）
- `val`: Value Area Low（70%成交量下界）

### 使用方法

```python
from cme_tick_loader import CMEBarsLoader, StandardBarData

# 加载并增强
result = loader.load_bars('GC', '20210104', 'MIN', 5)
bar_data = StandardBarData.from_result(result)

# 访问数据
print(bar_data.bars[['vwap', 'poc', 'vah', 'val']].head())
```

### Volume Profile 指标

- **POC**: 成交量最大的价格点（市场公允价值）
- **VAH/VAL**: 70%成交量区间（正常波动范围）
  - 价格突破 VAH → 强势
  - 价格跌破 VAL → 弱势

---

## Usage Examples

### Example 1: 多种 Bar 类型

```python
loader = CMEBarsLoader()

# TIME bars
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
result = loader.load_bars('GC', '20210104', resolution='H', num_units=1)

# VOLUME bars (每 10000 成交量)
result = loader.load_bars('GC', '20210104', resolution='VOLUME', num_units=10000)

# TICK bars (每 1000 ticks)
result = loader.load_bars('GC', '20210104', resolution='TICK', num_units=1000)

# DOLLAR bars
result = loader.load_bars('GC', '20210104', resolution='DOLLAR', num_units=1000000)
```

### Example 2: Footprint 分析

```python
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
footprint = result['footprint']

# 获取第一个 bar
first_bar_time = result['bars']['date_time'].iloc[0]
bar = footprint.loc[first_bar_time]

# POC (Point of Control)
poc = bar['total_vol'].idxmax()

# OHLC 价格
open_price = bar[bar['is_open']].index[0] if bar['is_open'].any() else None
high_price = bar[bar['is_high']].index[0] if bar['is_high'].any() else None
```

### Example 3: 日期范围

```python
result = loader.load_date_range(
    'GC', '20210104', '20210108',
    resolution='MIN', num_units=5
)
```

### Example 4: Datetime 格式控制

```python
loader = CMEBarsLoader()

# 默认配置：timezone-naive + datetime index
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
bars = result['bars']
print(bars['date_time'].iloc[0])  # 2021-01-04 00:05:00 (无timezone)
print(bars.index[0])               # 2021-01-04 00:05:00 (DatetimeIndex)

# UTC timezone + datetime index
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                          timezone_naive=False, set_index=True)
bars = result['bars']
print(bars['date_time'].iloc[0])  # 2021-01-04 00:05:00+00:00 (UTC)
print(bars.index[0])               # 2021-01-04 00:05:00 (DatetimeIndex)

# Timezone-naive + 整数索引
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                          timezone_naive=True, set_index=False)
bars = result['bars']
print(bars['date_time'].iloc[0])  # 2021-01-04 00:05:00 (无timezone)
print(bars.index[0])               # 0 (RangeIndex)

# 使用场景：兼容mlfinlab格式
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                          timezone_naive=True, set_index=True)
bars = result['bars']
# bars.index 已设置为 datetime，可以直接用于时间序列分析
returns = bars['close'].pct_change()  # 自动使用 datetime index
```

### Example 5: 精确时间戳分析 (open_time_ms / close_time_ms)

```python
loader = CMEBarsLoader()

# 加载TIME bars
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
bars = result['bars']

# 1. 计算每个bar的实际时长
bars['duration_ms'] = bars['close_time_ms'] - bars['open_time_ms']
bars['duration_sec'] = bars['duration_ms'] / 1000

print(f"平均bar时长: {bars['duration_sec'].mean():.2f} 秒")
print(f"5分钟TIME bars理论时长: 300 秒")
print(f"实际偏差: {bars['duration_sec'].mean() - 300:.2f} 秒")

# 2. 转换为精确的datetime对象
bars['open_time'] = pd.to_datetime(bars['open_time_ms'], unit='ms')
bars['close_time'] = pd.to_datetime(bars['close_time_ms'], unit='ms')

# 3. 分析TIME bars的对齐偏差
# date_time是对齐的边界时间，close_time是实际最后一个tick时间
bars['close_offset_ms'] = (
    pd.to_datetime(bars['date_time']).astype('int64') // 10**6
    - bars['close_time_ms']
)

print("\n前3个bars的时间对齐分析:")
print(bars[['date_time', 'open_time', 'close_time', 'duration_sec', 'close_offset_ms']].head(3))

# 输出示例:
#          date_time                open_time               close_time  duration_sec  close_offset_ms
# 0  2021-01-04 00:05:00  2021-01-04 00:00:00.003  2021-01-04 00:04:59.966       299.963           34
# 1  2021-01-04 00:10:00  2021-01-04 00:05:00.012  2021-01-04 00:09:59.988       299.976           12
# 2  2021-01-04 00:15:00  2021-01-04 00:10:00.005  2021-01-04 00:14:59.995       299.990            5

# 4. 微观结构分析：检测bar形成的速度
bars['ticks_per_second'] = bars['cum_ticks'] / bars['duration_sec']

# 找到高频交易活跃的bar
high_freq_bars = bars[bars['ticks_per_second'] > bars['ticks_per_second'].quantile(0.9)]
print(f"\n高频交易活跃的bars数量: {len(high_freq_bars)}")
print(f"平均tick频率: {bars['ticks_per_second'].mean():.2f} ticks/秒")

# 5. 使用场景：检测数据质量
# 如果duration_ms过小，可能是数据缺失
suspicious_bars = bars[bars['duration_ms'] < 1000]  # 少于1秒
if len(suspicious_bars) > 0:
    print(f"\n警告：发现 {len(suspicious_bars)} 个异常短的bars（<1秒）")
```

**为什么需要 open_time_ms 和 close_time_ms？**

TIME bars有一个重要的特性：`date_time` 被强制对齐到时间边界。例如：
- 5分钟TIME bars：date_time总是 00:05:00, 00:10:00, 00:15:00...
- 但实际交易可能在 00:04:59.5 就停止了（周末前）
- 或在 00:00:00.5 才开始（周一开盘）

使用 `open_time_ms` 和 `close_time_ms` 可以：
1. **精确计算bar时长**：真实的市场活跃时间
2. **检测数据质量**：发现缺失或异常的数据段
3. **微观结构研究**：分析tick到达频率、订单流速度
4. **回测精度**：使用真实的交易时间而非对齐时间

---

## Caching Strategy

### 两级缓存

**Level 1: Tick Cache**
```
.cache/ticks/GC_20210104.pkl
```

**Level 2: Result Cache**
```
.cache/results/GC_20210104_MIN_5.pkl
.cache/results/GC_20210104_VOLUME_10000.pkl
```

### 缓存逻辑

```python
# 默认：优先使用缓存
load_bars(use_cache=True, refresh_cache=False)

# 强制刷新 result cache
load_bars(use_cache=True, refresh_cache=True)

# 不使用缓存
load_bars(use_cache=False)
```

### 缓存版本兼容性

**重要**：mlfinlab v1.6+ 新增了 `open_time_ms` 和 `close_time_ms` 字段。如果你的result cache是在旧版本生成的，将**不包含这两个字段**。

**检测方法**：
```python
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5)
bars = result['bars']

# 检查是否包含新字段
if 'open_time_ms' not in bars.columns:
    print("⚠️  检测到旧版本缓存，缺少 open_time_ms 和 close_time_ms 字段")
    print("   请清除result cache: loader.clear_cache('result')")
```

**解决方法**：
```python
# 方法1: 清除所有result cache（推荐）
loader.clear_cache('result')

# 方法2: 强制刷新特定bar的cache
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                          refresh_cache=True)

# 方法3: 临时禁用cache
result = loader.load_bars('GC', '20210104', resolution='MIN', num_units=5,
                          use_cache=False)
```

**注意**：
- Tick cache不受影响（只存储原始tick数据）
- 只需清除一次result cache，后续会自动使用新格式
- 新格式的result cache与旧版本**不兼容**

---

## Technical Details

### 时间戳处理

- **TIME bars**: mlfinlab 转为 Unix 秒 → 自动修复为 datetime
- **其他 bars**: mlfinlab 保持 datetime → 不需要修复

### 预聚合

**问题**: CME 数据可能有同一毫秒同一价格的多条记录

**解决**: 按 (date_time, price) 分组累加
```python
df.groupby(['date_time', 'price']).agg({'bid_qty': 'sum', 'ask_qty': 'sum'})
```

### batch_size

- **值**: 50000000
- **原因**: 远超单天数据量，确保 mlfinlab 一次性处理，避免跨批状态污染

### Datetime 格式处理

**设计原则**:
- 缓存层保持 UTC-aware datetime（原始格式）
- 用户层可选择 timezone-naive（大多数pandas操作更友好）
- index设置使用 `.values` 避免pandas自动对齐

**实现细节**:
```python
# 1. Timezone 转换
if timezone_naive:
    bars['date_time'] = bars['date_time'].dt.tz_localize(None)
    footprint.index = footprint.index.set_levels(
        footprint.index.levels[0].tz_localize(None), level=0
    )

# 2. Index 设置（关键：使用.values）
if set_index:
    bars.index = bars.date_time.values  # 避免pandas自动对齐
    # 保留date_time列
```

**缓存策略**:
- `timezone_naive` 和 `set_index` 不影响缓存键
- 从缓存加载后仍会应用格式转换
- 多日合并：先 `ignore_index=True`，再重设 datetime index

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
mlfinlab>=1.0.0
plotly>=5.0.0  # optional
```

---

**Version**: 2.3
**Last Updated**: 2025-10-23

## Changelog

**v2.3** (2025-10-23) - 添加数据预处理器
- 🟢 新增 `DataPreprocessor` 类，支持从 footprint 生成增强数据
- 🟢 新增 `StandardBarData` dataclass，包含 VWAP 和 Volume Profile 指标
- 🟢 新增 `get_detection_data()` 工具函数，用于提取检测窗口
- 🟢 支持 POC/VAH/VAL 计算（Volume Profile 核心指标）
- 📚 完整的文档和使用示例

**v2.2** (2025-10-21) - mlfinlab v1.6+ 兼容
- 新增 `open_time_ms` 和 `close_time_ms` 字段支持
