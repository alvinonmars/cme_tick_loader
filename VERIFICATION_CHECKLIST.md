# FootprintVisualizer 渲染验证清单

## 生成的文件

- `output/footprint_local.html` - 本地缩放footprint图表（15 bars，222价格层级）
- `output/footprint_global.html` - 全局缩放footprint图表
- `output/detailed_verification_local.html` - 详细验证图表（20 bars）

## 关键验证点

### 1. 坐标系统 ✓
- [ ] **高价格在上方，低价格在下方**
- [ ] 价格标签从高到低正确排列
- [ ] Y轴反转设置生效（autorange='reversed'）

### 2. OHLC蜡烛图（第1列）✓
- [ ] **涨势蜡烛为绿色**（close >= open）
- [ ] **跌势蜡烛为红色**（close < open）
- [ ] 实体高度反映开盘价和收盘价差距
- [ ] 上下影线正确显示最高价和最低价
- [ ] 列宽比例为总宽度的 8.3%（2/24）

### 3. Delta条形图（第2列）✓
- [ ] **正Delta向右延伸（绿色）**
- [ ] **负Delta向左延伸（红色）**
- [ ] 条形从列中心开始延伸
- [ ] 长度按delta绝对值比例缩放
- [ ] 列宽比例为总宽度的 50%（12/24）

### 4. Volume条形图（第3列）✓
- [ ] **蓝色条形**（#4A90E2）
- [ ] 从列左侧开始向右延伸
- [ ] 长度按volume比例缩放
- [ ] 列宽比例为总宽度的 41.7%（10/24）

### 5. 缩放模式对比
- [ ] **Local scaling**: 每个bar的delta/volume独立缩放到最大值
- [ ] **Global scaling**: 所有bars的delta/volume统一缩放到全局最大值
- [ ] Local模式应该显示更多细节
- [ ] Global模式应该显示相对大小关系

### 6. 价格精度与Ticksize ✓
- [ ] 价格标签对齐到GC的0.1 ticksize
- [ ] 价格层级间距均匀
- [ ] 没有重复或错位的价格标签

### 7. 专业外观 ✓
- [ ] 白色背景
- [ ] 适当的网格线（浅灰色）
- [ ] 清晰的边框和分隔
- [ ] 三列布局清晰可见

## 技术验证数据

```
数据范围: 2021-01-04 最后15个bars
价格层级: 222个
图形元素: 574个shapes, 784个annotations
列宽比例: 2:12:10 = 8.3%:50.0%:41.7%
坐标系统: Y轴反转 ✓
颜色方案: ATAS标准 ✓
```

## 验证方法

1. **在浏览器中打开HTML文件**
2. **使用plotly的交互功能**：
   - 缩放特定区域查看细节
   - 悬停查看具体数值
   - 比较local vs global scaling差异

3. **重点检查的区域**：
   - 第一个和最后一个bar的渲染
   - 高volume价格层级的显示
   - 大正/负delta的条形显示
   - 价格轴标签的准确性

## 与参考代码的一致性

- ✓ 坐标系统：matplotlib图像坐标（高价格上方）
- ✓ 三列布局：Body:Delta:Volume = 2:12:10
- ✓ Delta渲染：中心向两边延伸
- ✓ 颜色方案：ATAS专业标准
- ✓ 价格映射：高效算法（numpy）
- ✓ 性能优化：max_bars限制