# Research Notebooks

This directory contains Jupyter notebooks for research and analysis using CME Bars Loader.

## Notebooks

### 2025-10-15_cme_bars_analysis.ipynb

**Comprehensive CME Bars Loader Analysis**

A complete research notebook demonstrating the full capabilities of CME Bars Loader API.

**Contents**:
1. **API Overview** - Quick start and basic usage
2. **Single Day Data** - Loading and exploring data for one day
3. **Bar Types Comparison** - Compare TIME, VOLUME, TICK, DOLLAR bars
4. **Footprint Analysis** - Deep dive into footprint data (POC, delta, volume profile)
5. **Statistical Properties** - Return distributions, volatility, autocorrelation
6. **Visualization** - Professional footprint charts
7. **Multi-Day Analysis** - Cross-day patterns
8. **Cache Performance** - Performance optimization
9. **Advanced Analysis** - Market microstructure insights
10. **Summary** - Key insights and next steps

**Key Features**:
- ✅ Complete API usage examples
- ✅ 4 bar types comparison (TIME, VOLUME, TICK, DOLLAR)
- ✅ Deep footprint analysis (POC, delta, aggressiveness)
- ✅ Statistical analysis (distributions, volatility, ACF)
- ✅ Interactive Plotly visualizations
- ✅ Market microstructure insights
- ✅ Performance benchmarking

**Requirements**:
```bash
# Make sure you have installed cme_tick_loader in development mode
pip install -e .

# Launch Jupyter
jupyter notebook research/2025-10-15_cme_bars_analysis.ipynb
```

**Data Requirements**:
- Gold (GC) data for 2021-01-04 to 2021-01-06
- Or modify the `SYMBOL` and `DATE` variables to use your own data

## Usage Guide

### Running a Notebook

```bash
# 1. Activate conda environment
source /opt/homebrew/Caskroom/miniconda/base/bin/activate cs

# 2. Navigate to project root
cd /path/to/cme_tick_loader

# 3. Launch Jupyter
jupyter notebook

# 4. Open the notebook in browser
# Navigate to research/2025-10-15_cme_bars_analysis.ipynb
```

### Quick Test

To quickly test if everything is set up correctly:

```python
from cme_tick_loader import CMEBarsLoader

# Initialize
loader = CMEBarsLoader()

# Load sample data
result = loader.load_bars(
    symbol='GC',
    date='20210104',
    resolution='MIN',
    num_units=5
)

print(f"Loaded {len(result['bars'])} bars")
# Should output: Loaded 275 bars (or similar)
```

## Research Topics

The notebooks in this directory explore:

### 1. Bar Sampling Methods
- **TIME bars**: Fixed time intervals
- **VOLUME bars**: Information-driven sampling
- **TICK bars**: Trade count based
- **DOLLAR bars**: Value-weighted sampling

### 2. Footprint Analysis
- **POC (Point of Control)**: Highest volume price level
- **Delta**: Bid/ask imbalance
- **Cumulative Delta**: Market sentiment tracking
- **Volume Profile**: Price distribution of volume

### 3. Statistical Properties
- **Return Distributions**: Skewness, kurtosis, fat tails
- **Volatility Estimators**: Close-to-close, Parkinson, Garman-Klass
- **Autocorrelation**: IID testing
- **Volume-Volatility Relationship**: Market dynamics

### 4. Market Microstructure
- **Trade Aggressiveness**: Imbalance ratio
- **Price Impact**: Volume effect on future returns
- **Liquidity Analysis**: Bid-ask spread patterns
- **Information Content**: Signal extraction from footprint

## Best Practices

### Data Loading
```python
# Always use cache for repeated analysis
result = loader.load_bars(
    symbol='GC',
    date='20210104',
    resolution='MIN',
    num_units=5,
    use_cache=True,  # ← Important!
    enable_footprint=True
)
```

### Memory Management
```python
# For multi-day analysis, disable footprint if not needed
result = loader.load_date_range(
    symbol='GC',
    start_date='20210104',
    end_date='20210131',
    resolution='MIN',
    num_units=5,
    enable_footprint=False  # ← Saves memory
)
```

### Visualization
```python
# Limit number of bars for footprint visualization
from cme_tick_loader import FootprintVisualizer

viz = FootprintVisualizer()

# Use last N bars only
timestamps = footprint.index.get_level_values(0).unique()[-20:]
subset = footprint.loc[timestamps]

fig = viz.plot_footprint(subset, ticksize=0.1)
fig.show()
```

## Tips for Research

1. **Start Small**: Test with single day before running multi-day analysis
2. **Use Cache**: Always enable cache for iterative research
3. **Clear Cache**: Use `loader.clear_cache('result')` when parameters change
4. **Subset Data**: For visualization, use last N bars to avoid performance issues
5. **Save Results**: Export key dataframes to CSV for later reference

## Example Workflow

```python
# 1. Initialize
from cme_tick_loader import CMEBarsLoader
loader = CMEBarsLoader()

# 2. Load data
result = loader.load_bars('GC', '20210104', 'MIN', 5)
bars = result['bars']
footprint = result['footprint']

# 3. Quick analysis
bars['log_ret'] = np.log(bars['close']).diff()
print(f"Mean return: {bars['log_ret'].mean():.6f}")
print(f"Volatility: {bars['log_ret'].std():.6f}")

# 4. Footprint analysis
first_bar_time = bars['date_time'].iloc[0]
bar_fp = footprint.loc[first_bar_time]
poc = bar_fp['total_vol'].idxmax()
print(f"POC: {poc}")

# 5. Visualize
from cme_tick_loader import FootprintVisualizer
viz = FootprintVisualizer()
fig = viz.plot_footprint(footprint.loc[bars['date_time'].iloc[:20]])
fig.show()
```

## Contributing

To add a new notebook:

1. Use date prefix: `YYYY-MM-DD_topic_name.ipynb`
2. Include markdown documentation in cells
3. Add entry to this README
4. Test full execution before committing

## Resources

- [CME Bars Loader Documentation](../CLAUDE.md)
- [mlfinlab Documentation](https://mlfinlab.readthedocs.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Last Updated**: 2025-10-15
**Maintainer**: CME Tick Loader Team
