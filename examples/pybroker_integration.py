"""
PyBroker + CME Tick Loader Integration Example

This example demonstrates how to integrate CME Tick Loader with PyBroker
to backtest futures trading strategies using order flow data.

Author: Integration Example
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Any, FrozenSet

# PyBroker imports
from pybroker.data import DataSource
from pybroker.common import DataCol


class CMEDataSource(DataSource):
    """
    Custom DataSource that fetches CME futures data using CMEBarsLoader.

    Features:
    - Supports multiple bar types (TIME, VOLUME, TICK, DOLLAR)
    - Provides standard OHLCV data for PyBroker
    - Caches footprint data for use in indicators

    Example:
        >>> from cme_tick_loader import CMEBarsLoader
        >>> data_source = CMEDataSource(
        ...     base_path='/path/to/cme/data',
        ...     resolution='MIN',
        ...     num_units=5
        ... )
        >>> # Use with PyBroker Strategy
        >>> strategy = Strategy(
        ...     data_source,
        ...     start_date='2021-01-04',
        ...     end_date='2021-01-06'
        ... )
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        resolution: str = 'MIN',
        num_units: int = 5,
        cache_footprint: bool = True
    ):
        """
        Initialize CME DataSource.

        Args:
            base_path: Path to CME data directory (auto-detects if None)
            resolution: Bar type ('MIN', 'H', 'VOLUME', 'TICK', 'DOLLAR')
            num_units: Bar size (e.g., 5 for 5-minute bars)
            cache_footprint: Whether to cache footprint data for indicators
        """
        super().__init__()

        # Lazy import to avoid dependency issues
        try:
            from cme_tick_loader import CMEBarsLoader
        except ImportError:
            raise ImportError(
                "cme_tick_loader not found. Install it first:\n"
                "pip install -e /path/to/cme_tick_loader"
            )

        self.loader = CMEBarsLoader(base_path=base_path)
        self.resolution = resolution
        self.num_units = num_units
        self.cache_footprint = cache_footprint

        # Cache for footprint data (accessible to indicators)
        self._footprint_cache = {}

        # Register custom columns for footprint data
        self._scope.register_custom_cols(
            'delta',           # Net delta (ask - bid)
            'total_volume',    # Total volume per bar
            'poc_price',       # Point of Control price
            'poc_volume',      # Volume at POC
            'imbalance_ratio'  # Delta / total_volume
        )

    def _fetch_data(
        self,
        symbols: FrozenSet[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: Optional[str],
        adjust: Optional[Any],
    ) -> pd.DataFrame:
        """
        Fetch CME data using CMEBarsLoader.

        This method is called by PyBroker's query() method.

        Args:
            symbols: CME symbols (e.g., 'GC' for Gold, 'ES' for E-mini S&P)
            start_date: Start date
            end_date: End date
            timeframe: Ignored (use self.resolution instead)
            adjust: Ignored (CME data is unadjusted)

        Returns:
            DataFrame with columns: symbol, date, open, high, low, close, volume,
                                   delta, total_volume, poc_price, poc_volume, imbalance_ratio
        """
        all_bars = []

        # Convert date range to list of dates
        date_range = pd.date_range(start_date, end_date, freq='D')

        for symbol in symbols:
            for date in date_range:
                date_str = date.strftime('%Y%m%d')

                try:
                    # Load bars and footprint
                    result = self.loader.load_bars(
                        symbol=symbol,
                        date=date_str,
                        resolution=self.resolution,
                        num_units=self.num_units,
                        use_cache=True,
                        verbose=False
                    )

                    bars = result['bars']
                    footprint = result['footprint']

                    # Calculate footprint metrics for each bar
                    if self.cache_footprint:
                        bars = self._add_footprint_metrics(bars, footprint)

                        # Store footprint for indicator access
                        for timestamp in footprint.index.get_level_values(0).unique():
                            cache_key = (symbol, timestamp)
                            self._footprint_cache[cache_key] = footprint.loc[timestamp]

                    # Convert to PyBroker format
                    bars_pybroker = self._convert_to_pybroker_format(bars, symbol)
                    all_bars.append(bars_pybroker)

                except FileNotFoundError:
                    # Skip missing dates (weekends, holidays)
                    continue
                except Exception as e:
                    self._logger.error(f"Error loading {symbol} {date_str}: {e}")
                    continue

        if not all_bars:
            return pd.DataFrame(columns=[
                DataCol.SYMBOL.value,
                DataCol.DATE.value,
                DataCol.OPEN.value,
                DataCol.HIGH.value,
                DataCol.LOW.value,
                DataCol.CLOSE.value,
                DataCol.VOLUME.value,
                'delta',
                'total_volume',
                'poc_price',
                'poc_volume',
                'imbalance_ratio'
            ])

        return pd.concat(all_bars, ignore_index=True)

    def _add_footprint_metrics(self, bars: pd.DataFrame, footprint: pd.DataFrame) -> pd.DataFrame:
        """
        Add footprint-derived metrics to bars DataFrame.

        Args:
            bars: OHLCV bars DataFrame
            footprint: Footprint DataFrame (MultiIndex: bar_timestamp, price)

        Returns:
            bars with added columns: delta, total_volume, poc_price, poc_volume, imbalance_ratio
        """
        metrics = []

        for timestamp in footprint.index.get_level_values(0).unique():
            bar_fp = footprint.loc[timestamp]

            # Calculate metrics
            total_vol = bar_fp['total_vol'].sum()
            net_delta = bar_fp['delta'].sum()
            poc_price = bar_fp['total_vol'].idxmax() if len(bar_fp) > 0 else np.nan
            poc_volume = bar_fp.loc[poc_price, 'total_vol'] if not pd.isna(poc_price) else 0
            imbalance = abs(net_delta) / total_vol if total_vol > 0 else 0

            metrics.append({
                'date_time': timestamp,
                'delta': net_delta,
                'total_volume': total_vol,
                'poc_price': poc_price,
                'poc_volume': poc_volume,
                'imbalance_ratio': imbalance
            })

        metrics_df = pd.DataFrame(metrics)

        # Merge with bars
        bars = bars.merge(metrics_df, on='date_time', how='left')

        return bars

    def _convert_to_pybroker_format(self, bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Convert CME bars to PyBroker format.

        Args:
            bars: CME bars DataFrame
            symbol: Symbol string

        Returns:
            DataFrame with PyBroker required columns
        """
        result = pd.DataFrame()

        # Required columns
        result[DataCol.SYMBOL.value] = [symbol] * len(bars)
        result[DataCol.DATE.value] = pd.to_datetime(bars['date_time'].values)
        result[DataCol.OPEN.value] = bars['open'].values
        result[DataCol.HIGH.value] = bars['high'].values
        result[DataCol.LOW.value] = bars['low'].values
        result[DataCol.CLOSE.value] = bars['close'].values
        result[DataCol.VOLUME.value] = bars['volume'].values

        # Optional footprint columns
        if 'delta' in bars.columns:
            result['delta'] = bars['delta'].values
            result['total_volume'] = bars['total_volume'].values
            result['poc_price'] = bars['poc_price'].values
            result['poc_volume'] = bars['poc_volume'].values
            result['imbalance_ratio'] = bars['imbalance_ratio'].values

        return result

    def get_footprint(self, symbol: str, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Retrieve cached footprint data for a specific bar.

        This can be used in custom indicators to access full footprint data.

        Args:
            symbol: Trading symbol
            timestamp: Bar timestamp

        Returns:
            Footprint DataFrame or None if not found
        """
        cache_key = (symbol, timestamp)
        return self._footprint_cache.get(cache_key)


# ============================================================================
# Example: Define Footprint-Based Indicators
# ============================================================================

def delta_indicator(bar_data, lookback=20):
    """
    Calculate cumulative delta indicator.

    Args:
        bar_data: BarData from PyBroker with 'delta' column
        lookback: Lookback period for cumulative sum

    Returns:
        Series of cumulative delta values
    """
    import pandas as pd

    # Access custom delta column
    delta = pd.Series(bar_data.delta)

    # Calculate rolling cumulative delta
    cum_delta = delta.rolling(window=lookback).sum()

    return cum_delta


def imbalance_indicator(bar_data, threshold=0.3):
    """
    Detect significant order flow imbalances.

    Args:
        bar_data: BarData from PyBroker
        threshold: Imbalance ratio threshold (0-1)

    Returns:
        Series: 1 for buy imbalance, -1 for sell imbalance, 0 for balanced
    """
    import pandas as pd

    delta = pd.Series(bar_data.delta)
    total_vol = pd.Series(bar_data.total_volume)

    # Calculate imbalance ratio
    imbalance_ratio = (delta / total_vol).fillna(0)

    # Classify imbalance
    signal = pd.Series(0, index=imbalance_ratio.index)
    signal[imbalance_ratio > threshold] = 1   # Buy pressure
    signal[imbalance_ratio < -threshold] = -1  # Sell pressure

    return signal


# ============================================================================
# Example: Trading Strategy Using Order Flow
# ============================================================================

def example_orderflow_strategy():
    """
    Example strategy using order flow (delta) to generate signals.
    """
    from pybroker import Strategy, indicator
    from pybroker.context import ExecContext

    # Define indicators
    @indicator('cum_delta', 'delta')
    def cumulative_delta(bar_data):
        return delta_indicator(bar_data, lookback=20)

    @indicator('imbalance', ['delta', 'total_volume'])
    def imbalance_signal(bar_data):
        return imbalance_indicator(bar_data, threshold=0.3)

    # Define execution function
    def exec_fn(ctx: ExecContext):
        """
        Trade based on cumulative delta and imbalance.

        Strategy Logic:
        - Buy when cumulative delta is positive and strong buy imbalance detected
        - Sell when cumulative delta is negative and strong sell imbalance detected
        - Use 2% stop loss
        """
        # Access indicators
        cum_delta = ctx.indicator('cum_delta')
        imbalance = ctx.indicator('imbalance')

        if len(cum_delta) < 2:
            return

        current_delta = cum_delta[-1]
        current_imbalance = imbalance[-1]

        # Entry logic
        if current_delta > 0 and current_imbalance == 1:
            # Positive cumulative delta + buy imbalance
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 2

        elif current_delta < 0 and current_imbalance == -1:
            # Negative cumulative delta + sell imbalance
            if ctx.long_pos():
                ctx.sell_shares = ctx.long_pos().shares

    # Create data source
    data_source = CMEDataSource(
        resolution='MIN',
        num_units=5,  # 5-minute bars
        cache_footprint=True
    )

    # Create strategy
    strategy = Strategy(
        data_source,
        start_date='2021-01-04',
        end_date='2021-01-06'
    )

    # Add execution
    strategy.add_execution(
        exec_fn,
        symbols=['GC'],  # Gold futures
        indicators=[cumulative_delta, imbalance_signal]
    )

    # Run backtest
    result = strategy.backtest(warmup=20)

    # Print results
    print("\n" + "="*80)
    print("Order Flow Strategy Results")
    print("="*80)
    print(f"\nTotal Return: {result.metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {result.metrics.sharpe:.2f}")
    print(f"Max Drawdown: {result.metrics.max_drawdown:.2f}%")
    print(f"Win Rate: {result.metrics.win_rate:.2f}%")
    print(f"Total Trades: {result.metrics.total_trades}")

    return result


# ============================================================================
# Example: Machine Learning with Footprint Features
# ============================================================================

def example_ml_orderflow_strategy():
    """
    Example ML strategy using footprint data as features.
    """
    from pybroker import Strategy, model, indicator
    from pybroker.context import ExecContext
    from sklearn.ensemble import RandomForestClassifier

    # Define indicators as features
    @indicator('cum_delta', 'delta')
    def cumulative_delta(bar_data):
        return delta_indicator(bar_data, lookback=20)

    @indicator('imbalance_ratio', ['delta', 'total_volume'])
    def imbalance_ratio_ind(bar_data):
        import pandas as pd
        delta = pd.Series(bar_data.delta)
        total_vol = pd.Series(bar_data.total_volume)
        return (delta / total_vol).fillna(0)

    @indicator('poc_distance', ['close', 'poc_price'])
    def poc_distance_ind(bar_data):
        """Distance from close to POC"""
        import pandas as pd
        close = pd.Series(bar_data.close)
        poc = pd.Series(bar_data.poc_price)
        return ((close - poc) / close * 100).fillna(0)

    # Define model training function
    def train_orderflow_model(symbol, train_data, test_data):
        """
        Train RandomForest classifier using footprint features.

        Features:
        - cum_delta: Cumulative delta
        - imbalance_ratio: Order flow imbalance
        - poc_distance: Distance from close to POC

        Target:
        - 1 if next bar closes higher, 0 otherwise
        """
        # Prepare features
        features = ['cum_delta', 'imbalance_ratio', 'poc_distance']
        X_train = train_data[features].fillna(0)
        X_test = test_data[features].fillna(0)

        # Prepare target (next bar direction)
        y_train = (train_data['close'].shift(-1) > train_data['close']).astype(int)
        y_train = y_train.fillna(0)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        return model

    # Define model
    orderflow_model = model(
        'orderflow_rf',
        train_orderflow_model,
        indicators=[cumulative_delta, imbalance_ratio_ind, poc_distance_ind]
    )

    # Define execution function
    def exec_ml_fn(ctx: ExecContext):
        """Trade based on ML predictions."""
        # Get prediction
        pred = ctx.preds('orderflow_rf')

        if len(pred) < 1:
            return

        # Prediction threshold
        if pred[-1] > 0.6:  # Strong buy signal
            if not ctx.long_pos():
                ctx.buy_shares = 100
                ctx.stop_loss_pct = 2

        elif pred[-1] < 0.4:  # Strong sell signal
            if ctx.long_pos():
                ctx.sell_shares = ctx.long_pos().shares

    # Create strategy (simplified for demo)
    print("\n" + "="*80)
    print("ML Order Flow Strategy (Example)")
    print("="*80)
    print("\nThis example demonstrates how to use footprint features in ML models.")
    print("For full implementation, uncomment the strategy code below.\n")

    # Uncomment to run:
    # data_source = CMEDataSource(resolution='MIN', num_units=5)
    # strategy = Strategy(data_source, '2021-01-04', '2021-02-01')
    # strategy.add_execution(exec_ml_fn, ['GC'], models=orderflow_model)
    # result = strategy.walkforward(windows=3, train_size=0.7)
    # return result


if __name__ == '__main__':
    print("="*80)
    print("PyBroker + CME Tick Loader Integration Examples")
    print("="*80)

    print("\nAvailable examples:")
    print("1. example_orderflow_strategy() - Order flow based strategy")
    print("2. example_ml_orderflow_strategy() - ML with footprint features")

    print("\nTo run an example:")
    print(">>> from examples.pybroker_integration import example_orderflow_strategy")
    print(">>> result = example_orderflow_strategy()")
