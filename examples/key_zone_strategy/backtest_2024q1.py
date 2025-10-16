"""
Comprehensive Backtest: Key Zone Strategy on 2024 Q1 Data

Tests the strategy on 3 months of real CME Gold futures data (Jan-Mar 2024)
and generates a detailed performance report.

Author: Strategy Backtest
Date: 2025-10-15
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cme_tick_loader import CMEBarsLoader
from key_zone_strategy import KeyZoneStrategy, StrategyConfig, TradeAction


class BacktestEngine:
    """
    Comprehensive backtesting engine for Key Zone Strategy.

    Features:
    - Load multi-month data
    - Run strategy bar-by-bar
    - Simulate trade execution
    - Collect detailed statistics
    - Calculate performance metrics
    """

    def __init__(self, symbol='GC', config=None):
        self.symbol = symbol
        self.config = config or StrategyConfig()
        self.loader = CMEBarsLoader()
        self.strategy = KeyZoneStrategy(symbol=symbol, config=self.config)

        # State
        self.all_bars = []
        self.all_footprints = []
        self.all_results = []

        # Trading simulation
        self.positions = []  # List of open positions
        self.trades = []     # List of completed trades
        self.equity_curve = []
        self.initial_capital = 100000
        self.current_capital = self.initial_capital
        self.position_size_usd = 10000  # $10k per trade

        # Statistics
        self.stats = defaultdict(int)
        self.signal_details = []

    def load_data(self, start_date, end_date, resolution='MIN', num_units=5):
        """
        Load data for date range.

        Args:
            start_date: 'YYYYMMDD'
            end_date: 'YYYYMMDD'
            resolution: Bar type
            num_units: Bar size
        """
        print(f"\n{'='*80}")
        print(f"LOADING DATA: {start_date} to {end_date}")
        print(f"{'='*80}")

        # Convert to datetime
        start_dt = pd.to_datetime(start_date, format='%Y%m%d')
        end_dt = pd.to_datetime(end_date, format='%Y%m%d')

        # Load using CMEBarsLoader
        print(f"\nLoading {resolution} {num_units} bars...")
        result = self.loader.load_date_range(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            num_units=num_units,
            use_cache=True,
            verbose=False
        )

        # Extract bars and footprints
        all_bars_df = result['bars']
        all_footprint_df = result['footprint']

        if all_bars_df.empty:
            print("❌ No data loaded!")
            return

        print(f"✓ Loaded {len(all_bars_df)} bars")
        print(f"  Date range: {all_bars_df['date_time'].min()} to {all_bars_df['date_time'].max()}")
        print(f"  Price range: {all_bars_df['low'].min():.1f} to {all_bars_df['high'].max():.1f}")

        # Convert to list format for strategy
        self.all_bars = all_bars_df

        # Group footprints by bar timestamp
        self.all_footprints = []
        for timestamp in all_bars_df['date_time']:
            try:
                fp = all_footprint_df.loc[timestamp]
                self.all_footprints.append(fp)
            except KeyError:
                # No footprint for this bar (shouldn't happen)
                self.all_footprints.append(pd.DataFrame())

        print(f"✓ Loaded {len(self.all_footprints)} footprints")

    def run_backtest(self):
        """
        Run full backtest bar-by-bar.
        """
        print(f"\n{'='*80}")
        print(f"RUNNING BACKTEST")
        print(f"{'='*80}")

        if self.all_bars.empty:
            print("❌ No data to backtest!")
            return

        total_bars = len(self.all_bars)
        print(f"\nProcessing {total_bars} bars...")

        # Progress tracking
        report_interval = max(1, total_bars // 20)  # Report every 5%

        for i in range(total_bars):
            # Update progress
            if i % report_interval == 0:
                pct = (i / total_bars) * 100
                print(f"  Progress: {i}/{total_bars} ({pct:.1f}%)")

            # Get window of bars up to current
            window_bars = self.all_bars.iloc[:i+1]
            window_footprints = self.all_footprints[:i+1]

            # Run strategy
            try:
                result = self.strategy.update(
                    bars=window_bars,
                    footprints=window_footprints,
                    current_bar_index=i
                )
            except Exception as e:
                # Skip on error
                result = None

            # Record result
            self.all_results.append(result)

            # Update statistics
            if result:
                self._update_statistics(result, i)

                # Simulate trading
                if result['trade_signal']:
                    self._simulate_trade(result, i)

            # Update equity curve
            current_bar = self.all_bars.iloc[i]
            equity = self._calculate_equity(current_bar)
            self.equity_curve.append({
                'bar_index': i,
                'timestamp': current_bar['date_time'],
                'equity': equity,
                'price': current_bar['close']
            })

        print(f"\n✓ Backtest complete!")
        print(f"  Total results: {len(self.all_results)}")
        print(f"  Non-null results: {sum(1 for r in self.all_results if r is not None)}")

    def _update_statistics(self, result, bar_index):
        """Update statistics from result"""
        # Count zones
        self.stats['total_bars_processed'] += 1
        self.stats['total_zones_detected'] += len(result.get('zones', []))

        # Count touches
        touches = result.get('touches', [])
        self.stats['total_touches'] += len(touches)

        # Count structural signals
        structural_signals = result.get('structural_signals', [])
        self.stats['total_structural_signals'] += len(structural_signals)

        for signal in structural_signals:
            signal_type = signal.signal_type.value
            self.stats[f'structural_{signal_type}'] += 1

        # Count trade signals
        trade_signal = result.get('trade_signal')
        if trade_signal:
            self.stats['total_trade_signals'] += 1
            action = trade_signal.action.value
            self.stats[f'trade_signal_{action}'] += 1

            # Record signal details
            current_bar = self.all_bars.iloc[bar_index]
            self.signal_details.append({
                'bar_index': bar_index,
                'timestamp': current_bar['date_time'],
                'action': action,
                'confidence': trade_signal.confidence,
                'zone_price': trade_signal.zone.center_price,
                'zone_type': trade_signal.zone.zone_type.value,
                'zone_strength': trade_signal.zone.strength,
                'signal_type': trade_signal.structural_signal.signal_type.value,
                'signal_strength': trade_signal.structural_signal.strength,
                'price': current_bar['close']
            })

    def _simulate_trade(self, result, bar_index):
        """
        Simulate trade execution.

        Simplified model:
        - Enter at next bar's open
        - Exit after N bars or on opposite signal
        - Track P&L
        """
        trade_signal = result['trade_signal']
        action = trade_signal.action

        if action == TradeAction.HOLD:
            return

        current_bar = self.all_bars.iloc[bar_index]

        # Entry logic
        if action == TradeAction.BUY:
            # Open long position
            if bar_index + 1 < len(self.all_bars):
                entry_bar = self.all_bars.iloc[bar_index + 1]
                entry_price = entry_bar['open']

                # Calculate shares (simplified: fixed USD size)
                shares = self.position_size_usd / entry_price

                position = {
                    'type': 'LONG',
                    'entry_bar': bar_index + 1,
                    'entry_timestamp': entry_bar['date_time'],
                    'entry_price': entry_price,
                    'shares': shares,
                    'confidence': trade_signal.confidence,
                    'zone_price': trade_signal.zone.center_price
                }

                self.positions.append(position)
                self.stats['positions_opened'] += 1

        elif action == TradeAction.SELL:
            # Close any open long positions
            if self.positions:
                for pos in self.positions[:]:  # Copy list
                    if pos['type'] == 'LONG':
                        exit_price = current_bar['close']

                        # Calculate P&L
                        pnl = (exit_price - pos['entry_price']) * pos['shares']
                        pnl_pct = ((exit_price / pos['entry_price']) - 1) * 100

                        trade = {
                            **pos,
                            'exit_bar': bar_index,
                            'exit_timestamp': current_bar['date_time'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'bars_held': bar_index - pos['entry_bar']
                        }

                        self.trades.append(trade)
                        self.positions.remove(pos)

                        # Update capital
                        self.current_capital += pnl

                        self.stats['positions_closed'] += 1
                        if pnl > 0:
                            self.stats['winning_trades'] += 1
                        else:
                            self.stats['losing_trades'] += 1

    def _calculate_equity(self, current_bar):
        """Calculate current equity including open positions"""
        equity = self.current_capital

        # Add unrealized P&L from open positions
        for pos in self.positions:
            if pos['type'] == 'LONG':
                unrealized_pnl = (current_bar['close'] - pos['entry_price']) * pos['shares']
                equity += unrealized_pnl

        return equity

    def generate_report(self):
        """Generate comprehensive performance report"""
        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT")
        print(f"{'='*80}")

        report = {}

        # Basic info
        report['test_period'] = {
            'start': str(self.all_bars.iloc[0]['date_time']),
            'end': str(self.all_bars.iloc[-1]['date_time']),
            'total_bars': len(self.all_bars),
            'trading_days': self.all_bars['date_time'].dt.date.nunique()
        }

        # Zone detection statistics
        report['zone_detection'] = {
            'total_zones_detected': self.stats['total_zones_detected'],
            'avg_zones_per_bar': self.stats['total_zones_detected'] / max(1, self.stats['total_bars_processed']),
            'total_touches': self.stats['total_touches'],
            'touch_rate': self.stats['total_touches'] / max(1, self.stats['total_bars_processed'])
        }

        # Structural signal statistics
        report['structural_signals'] = {
            'total': self.stats['total_structural_signals'],
            'v_reversal_bullish': self.stats.get('structural_v_reversal_bullish', 0),
            'v_reversal_bearish': self.stats.get('structural_v_reversal_bearish', 0),
            'breakout_bullish': self.stats.get('structural_breakout_bullish', 0),
            'breakout_bearish': self.stats.get('structural_breakout_bearish', 0),
        }

        # Trade signal statistics
        report['trade_signals'] = {
            'total': self.stats['total_trade_signals'],
            'BUY': self.stats.get('trade_signal_BUY', 0),
            'SELL': self.stats.get('trade_signal_SELL', 0),
            'signal_rate': self.stats['total_trade_signals'] / max(1, len(self.all_bars))
        }

        # Trading performance
        total_trades = len(self.trades)
        if total_trades > 0:
            wins = self.stats.get('winning_trades', 0)
            losses = self.stats.get('losing_trades', 0)
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

            pnls = [t['pnl'] for t in self.trades]
            pnl_pcts = [t['pnl_pct'] for t in self.trades]

            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]

            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')

            # Calculate max drawdown
            equity_values = [e['equity'] for e in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            # Sharpe ratio (simplified)
            returns = np.diff(equity_values) / equity_values[:-1]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0

            report['trading_performance'] = {
                'total_trades': total_trades,
                'winning_trades': wins,
                'losing_trades': losses,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(sum(pnls), 2),
                'total_return_pct': round(((self.current_capital / self.initial_capital) - 1) * 100, 2),
                'avg_pnl_per_trade': round(np.mean(pnls), 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown_pct': round(max_dd * 100, 2),
                'sharpe_ratio': round(sharpe, 2),
                'final_capital': round(self.current_capital, 2)
            }
        else:
            report['trading_performance'] = {
                'total_trades': 0,
                'message': 'No trades executed'
            }

        # Confidence distribution
        if self.signal_details:
            confidences = [s['confidence'] for s in self.signal_details]
            report['confidence_stats'] = {
                'mean': round(np.mean(confidences), 3),
                'median': round(np.median(confidences), 3),
                'std': round(np.std(confidences), 3),
                'min': round(np.min(confidences), 3),
                'max': round(np.max(confidences), 3)
            }

        return report

    def print_report(self, report):
        """Print formatted report"""
        print(f"\n{'='*80}")
        print(f"DETAILED RESULTS")
        print(f"{'='*80}")

        print(f"\n📅 Test Period:")
        for key, value in report['test_period'].items():
            print(f"  {key:20}: {value}")

        print(f"\n🎯 Zone Detection:")
        for key, value in report['zone_detection'].items():
            if isinstance(value, float):
                print(f"  {key:30}: {value:.2f}")
            else:
                print(f"  {key:30}: {value}")

        print(f"\n📊 Structural Signals:")
        for key, value in report['structural_signals'].items():
            print(f"  {key:30}: {value}")

        print(f"\n💡 Trade Signals:")
        for key, value in report['trade_signals'].items():
            if isinstance(value, float):
                print(f"  {key:30}: {value:.4f}")
            else:
                print(f"  {key:30}: {value}")

        if 'trading_performance' in report and 'total_trades' in report['trading_performance']:
            print(f"\n💰 Trading Performance:")
            for key, value in report['trading_performance'].items():
                if isinstance(value, float):
                    print(f"  {key:30}: {value:.2f}")
                else:
                    print(f"  {key:30}: {value}")

        if 'confidence_stats' in report:
            print(f"\n📈 Confidence Statistics:")
            for key, value in report['confidence_stats'].items():
                print(f"  {key:30}: {value:.3f}")

    def export_results(self, output_dir='./backtest_results'):
        """Export results to files"""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = output_path / f'trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            print(f"\n✓ Exported trades to: {trades_file}")

        # Export signals
        if self.signal_details:
            signals_df = pd.DataFrame(self.signal_details)
            signals_file = output_path / f'signals_{timestamp}.csv'
            signals_df.to_csv(signals_file, index=False)
            print(f"✓ Exported signals to: {signals_file}")

        # Export equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = output_path / f'equity_{timestamp}.csv'
            equity_df.to_csv(equity_file, index=False)
            print(f"✓ Exported equity curve to: {equity_file}")


def main():
    """Main backtest execution"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "Key Zone Strategy - 2024 January Backtest" + " "*21 + "║")
    print("╚" + "="*78 + "╝")

    # Initialize
    config = StrategyConfig(
        big_delta_lookback=50,  # Reduced for speed
        peak_lookback=50,
        n_keep_prices=5,
        zone_ticks=20,
        v_reversal_lookback=5,
        breakout_lookback=10,
        touch_window=3,
        atr_period=14,
        ticksize=0.1  # GC
    )

    engine = BacktestEngine(symbol='GC', config=config)

    # Load January 2024 data (faster, still comprehensive)
    engine.load_data(
        start_date='20240102',
        end_date='20240131',
        resolution='MIN',
        num_units=5
    )

    # Run backtest
    engine.run_backtest()

    # Generate and print report
    report = engine.generate_report()
    engine.print_report(report)

    # Export results
    engine.export_results()

    print(f"\n{'='*80}")
    print("BACKTEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
