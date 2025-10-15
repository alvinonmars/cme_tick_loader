# CME Bars Loader - Technical Documentation

**Version 2.0** - Based on mlfinlab API

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Architecture](#core-architecture)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Caching Strategy](#caching-strategy)
8. [Technical Details](#technical-details)

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
    'bars': DataFrame,      # 列：date_time, open, high, low, close, volume...
    'footprint': DataFrame  # MultiIndex(bar_timestamp, price)
                           # 列：bid_vol, ask_vol, total_vol, delta,
                           #     is_open, is_high, is_low, is_close
}
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

**Version**: 2.1
**Last Updated**: 2025-10-16
