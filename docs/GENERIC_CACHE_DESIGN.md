# GenericTimeSeriesCache - 设计文档

**用途**: 替代 remote_indicator_service 的 DataCacheManager
**版本**: 1.2
**日期**: 2025-10-23

---

## 设计目标

纯内存时序数据缓存，数据源可插拔，支持实时更新。

**为什么设计这个**：
- remote_indicator_service 的 DataCacheManager 耦合 ZMQ/ATAS
- 需要通用的内存缓存层，可用于多个工程
- CMEBarsLoader 已有磁盘缓存，这个只管内存层

**不是什么**：
- ❌ 不是持久化层（重启丢失）
- ❌ 不是 CMEBarsLoader 的用户 API（是底层组件）
- ❌ 不是分布式缓存

---

## 架构

```
GenericTimeSeriesCache
│
├── _cache: Dict[CacheKey, CachedData]
│       CacheKey = (symbol, resolution, num_units)
│       CachedData = (bars: DataFrame, footprint: DataFrame)
│
└── _loader: callable(symbol, resolution, num_units, start, end)
                  ↓
          CMEBarsLoader / DatabaseLoader / Custom
```

**两级 failback**：
1. 检查内存缓存是否覆盖 [start, end]
2. 调用 loader，合并到缓存，返回

---

## 快速开始

### 基本用法

```python
from cme_tick_loader import GenericTimeSeriesCache, CMEBarsLoader
import pandas as pd

# 1. 创建 loader function（包装 CMEBarsLoader）
loader = CMEBarsLoader()

def load_func(symbol, resolution, num_units, start_time, end_time):
    start_date = start_time.strftime('%Y%m%d')
    end_date = end_time.strftime('%Y%m%d')

    result = loader.load_date_range(
        symbol, start_date, end_date,
        resolution=resolution, num_units=num_units,
        timezone_naive=True,  # 必须
        set_index=True
    )

    return result['bars'], result['footprint']

# 2. 创建缓存（单线程）
cache = GenericTimeSeriesCache(load_func)

# 3. 查询（自动 failback）
bars, footprint = cache.get(
    'GC', 'MIN', 5,
    pd.Timestamp('2021-01-04 00:00:00'),
    pd.Timestamp('2021-01-04 23:59:59')
)

# 4. 实时追加
cache.append('GC', 'MIN', 5, new_bar_df, new_footprint_df)

# 5. 清理
cache.clear_key('GC', 'MIN', 5)  # 删除单个 key
cache.get_stats()  # 查看内存使用
```

### 多线程用法

```python
# 启用线程安全（多线程并发写入）
cache = GenericTimeSeriesCache(load_func, thread_safe=True)

# 线程A：实时推送
def realtime_thread():
    while True:
        bar = receive_from_terminal()
        cache.append('GC', 'MIN', 5, bar, footprint)

# 线程B：补历史数据
def backfill_thread():
    if missing_data:
        historical = fetch_from_network()
        cache.append('GC', 'MIN', 5, historical, footprint)

# 自动处理并发和乱序时间戳
threading.Thread(target=realtime_thread).start()
threading.Thread(target=backfill_thread).start()
```

---

## 核心 API

完整文档见源码 `generic_cache.py`（518 行，含完整 docstring）。

主要方法：
- `get(symbol, resolution, num_units, start, end)` - 两级 failback 查询
- `append(...)` - 实时更新（自动去重）
- `merge(...)` - 批量合并（可选 keep_first/keep_last）
- `delete(...)` - 删除整个 key 或时间范围
- `clear_key(symbol, resolution, num_units)` - 结构化清除（推荐）
- `clear(pattern)` - 字符串模式清除（有限制）
- `get_stats()` - 内存统计
- `has_data(...)` - 检查覆盖范围

---

## 关键设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| **缓存层级** | 纯内存 | 磁盘由 CMEBarsLoader 或数据库负责 |
| **CacheKey** | `(symbol, resolution, num_units)` | 与 mlfinlab API 一致 |
| **时间戳** | timezone-naive | 与 CMEBarsLoader 兼容（设计妥协） |
| **数据源** | callable（duck typing） | 不强制 Protocol，保持简单 |
| **线程安全** | 可选（thread_safe=True） | RLock 全局锁，性能损失 ~10% |
| **去重策略** | keep='last' | 实时更新场景，新数据覆盖旧数据 |
| **乱序数据** | 自动排序 | sort_index() 保证时序正确 |

---

## 已知限制

### 1. 不持久化
- 重启丢失所有缓存
- 需要外部持久化层（数据库）

### 2. 线程安全（可选）
- 默认不支持（thread_safe=False）
- 多线程场景：设置 `thread_safe=True`
- 性能损失：~10%（RLock 开销）
- 自动处理乱序时间戳（sort_index）

### 3. 无内存限制
- 缓存会无限增长
- 需要手动 `clear_key()` 或定期清理
- 使用 `get_stats()` 监控内存使用

### 4. timezone-naive
- 设计妥协，为了与 CMEBarsLoader 兼容
- 未来可能改为 timezone-aware
- 修复方法：`_normalize_timestamp()` 一处修改即可

### 5. 覆盖检查不验证连续性
- 只检查 `cache_start <= start && cache_end >= end`
- 不检查中间是否有 gap
- 适用于金融数据（市场休市时本来就有 gap）

---

## FAQ

**Q1: 为什么不内置在 CMEBarsLoader？**
A: 职责分离。CMEBarsLoader 已有 tick cache + result cache，再加一层会有三层缓存。GenericTimeSeriesCache 是通用组件，可用于其他工程。

**Q2: 如何用于非 CMEBarsLoader 的数据源？**
A: 提供任意 callable 即可：
```python
def db_loader(symbol, resolution, num_units, start, end):
    return load_from_database(...)

cache = GenericTimeSeriesCache(db_loader)
```

**Q3: 如何监控内存使用？**
A: `cache.get_stats()` 返回每个 key 的 bars_count、memory_usage_mb 等。

**Q4: `use_cache=False` 有什么用？**
A: 强制重新加载，绕过缓存。用于调试或数据更新场景。

**Q5: 为什么有 `clear()` 和 `clear_key()` 两个方法？**
A: `clear()` 是字符串模式匹配，有限制（如 symbol 不能包含下划线）。`clear_key()` 是结构化清除，更可靠，推荐使用。

---

## Changelog

**v1.2** (2025-10-23) - 添加线程安全支持
- 🟢 新增 `thread_safe=True` 参数（RLock 全局锁）
- 🟢 支持多线程并发写入（实时推送 + 补历史）
- 🟢 自动处理乱序时间戳（sort_index）
- 🟢 性能损失：~10%（测试验证）
- 📊 代码行数：518 → 560

**v1.1** (2025-10-23) - 修复关键 bug 并完善
- 🔴 修复 `use_cache=False` 逻辑错误（返回旧缓存 → 返回新数据）
- 🔴 修复 `_normalize_timestamp()` timezone 处理（`tz_localize(None)` → `replace(tzinfo=None)`）
- 🟡 重构 `append()` 调用 `merge()`，消除代码重复
- 🟡 新增 `clear_key()` 方法（结构化缓存清除）
- 🟡 新增 `_validate_loader_output()` 数据结构校验
- 🟡 新增输入参数验证（num_units、start/end、strategy）
- ✅ 所有修复已通过测试验证
- 📊 代码行数：436 → 518

**v1.0** (2025-10-22) - 初始版本
- 纯内存缓存，数据源可插拔
- 支持两级 failback、实时 append、范围查询
- ~430 行代码
