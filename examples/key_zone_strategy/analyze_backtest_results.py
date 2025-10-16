"""
Deep Analysis of Backtest Results

Generates comprehensive analysis and insights from the backtest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def analyze_trades(trades_file):
    """Analyze trade details"""
    trades = pd.DataFrame() if not Path(trades_file).exists() else pd.read_csv(trades_file)

    if trades.empty:
        return {}

    # Convert timestamps
    trades['entry_timestamp'] = pd.to_datetime(trades['entry_timestamp'])
    trades['exit_timestamp'] = pd.to_datetime(trades['exit_timestamp'])

    # Time analysis
    trades['entry_hour'] = trades['entry_timestamp'].dt.hour
    trades['exit_hour'] = trades['exit_timestamp'].dt.hour
    trades['holding_hours'] = trades['bars_held'] * 5 / 60  # 5-min bars to hours

    # Performance by confidence
    conf_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    trades['conf_bin'] = pd.cut(trades['confidence'], bins=conf_bins)

    perf_by_conf = trades.groupby('conf_bin').agg({
        'pnl': ['mean', 'std', 'count'],
        'pnl_pct': ['mean', 'std']
    }).round(2)

    # Performance by zone strength
    perf_by_zone = trades.groupby('zone_price').agg({
        'pnl': ['mean', 'count'],
        'pnl_pct': 'mean'
    }).sort_values(('pnl', 'count'), ascending=False).head(10)

    # Time distribution
    perf_by_hour = trades.groupby('entry_hour').agg({
        'pnl': ['mean', 'count'],
        'pnl_pct': 'mean'
    }).round(2)

    # Holding period analysis
    hold_bins = [0, 1, 3, 6, 12, 24, 100]
    trades['hold_bin'] = pd.cut(trades['holding_hours'], bins=hold_bins)
    perf_by_hold = trades.groupby('hold_bin').agg({
        'pnl': ['mean', 'count'],
        'pnl_pct': 'mean'
    }).round(2)

    # Win/Loss streaks
    trades['is_win'] = trades['pnl'] > 0
    streaks = []
    current_streak = 1
    for i in range(1, len(trades)):
        if trades.iloc[i]['is_win'] == trades.iloc[i-1]['is_win']:
            current_streak += 1
        else:
            streaks.append(current_streak)
            current_streak = 1
    streaks.append(current_streak)

    return {
        'total_trades': len(trades),
        'perf_by_confidence': perf_by_conf,
        'perf_by_zone': perf_by_zone,
        'perf_by_hour': perf_by_hour,
        'perf_by_holding': perf_by_hold,
        'max_win_streak': max([s for s, w in zip(streaks, trades['is_win'][::max(1, len(streaks)//len(trades))]) if w], default=0),
        'max_loss_streak': max([s for s, w in zip(streaks, trades['is_win'][::max(1, len(streaks)//len(trades))]) if not w], default=0),
        'avg_holding_hours': trades['holding_hours'].mean(),
        'best_trade': trades.nlargest(1, 'pnl')[['entry_timestamp', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']].to_dict('records')[0],
        'worst_trade': trades.nsmallest(1, 'pnl')[['entry_timestamp', 'entry_price', 'exit_price', 'pnl', 'pnl_pct']].to_dict('records')[0],
    }


def analyze_signals(signals_file):
    """Analyze signal characteristics"""
    signals = pd.DataFrame() if not Path(signals_file).exists() else pd.read_csv(signals_file)

    if signals.empty:
        return {}

    signals['timestamp'] = pd.to_datetime(signals['timestamp'])
    signals['hour'] = signals['timestamp'].dt.hour
    signals['day_of_week'] = signals['timestamp'].dt.dayofweek

    # Signal distribution by action
    action_dist = signals['action'].value_counts()

    # Signal distribution by hour
    signals_by_hour = signals.groupby('hour').size()

    # Signal distribution by day
    signals_by_day = signals.groupby('day_of_week').size()

    # Confidence distribution by signal type
    conf_by_signal = signals.groupby('signal_type')['confidence'].agg(['mean', 'std', 'count']).round(3)

    # Zone type distribution
    zone_dist = signals['zone_type'].value_counts()

    return {
        'total_signals': len(signals),
        'action_distribution': action_dist.to_dict(),
        'signals_by_hour': signals_by_hour.to_dict(),
        'signals_by_day': signals_by_day.to_dict(),
        'confidence_by_signal_type': conf_by_signal,
        'zone_type_distribution': zone_dist.to_dict(),
        'avg_confidence': signals['confidence'].mean(),
        'high_confidence_signals': (signals['confidence'] > 0.5).sum(),
    }


def analyze_equity(equity_file):
    """Analyze equity curve"""
    equity = pd.DataFrame() if not Path(equity_file).exists() else pd.read_csv(equity_file)

    if equity.empty:
        return {}

    equity['timestamp'] = pd.to_datetime(equity['timestamp'])

    # Calculate drawdowns
    equity['peak'] = equity['equity'].cummax()
    equity['drawdown'] = (equity['peak'] - equity['equity']) / equity['peak']

    # Daily returns
    equity['date'] = equity['timestamp'].dt.date
    daily_equity = equity.groupby('date')['equity'].last()
    daily_returns = daily_equity.pct_change().dropna()

    # Volatility
    annual_vol = daily_returns.std() * np.sqrt(252)

    # Underwater periods
    equity['underwater'] = equity['drawdown'] > 0

    return {
        'max_drawdown': equity['drawdown'].max(),
        'avg_drawdown': equity[equity['underwater']]['drawdown'].mean() if equity['underwater'].any() else 0,
        'annual_volatility': annual_vol,
        'daily_returns_mean': daily_returns.mean(),
        'daily_returns_std': daily_returns.std(),
        'best_day': daily_returns.max() if len(daily_returns) > 0 else 0,
        'worst_day': daily_returns.min() if len(daily_returns) > 0 else 0,
        'underwater_days': equity['underwater'].sum() / len(equity),
    }


def generate_comprehensive_report():
    """Generate comprehensive analysis report"""

    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST ANALYSIS")
    print("2024 January - Key Zone Strategy")
    print("="*80)

    # Load data
    result_dir = Path('./backtest_results')
    files = sorted(result_dir.glob('*.csv'))

    if not files:
        print("\n❌ No backtest results found!")
        return

    # Get latest files
    trades_file = sorted(result_dir.glob('trades_*.csv'))[-1]
    signals_file = sorted(result_dir.glob('signals_*.csv'))[-1]
    equity_file = sorted(result_dir.glob('equity_*.csv'))[-1]

    print(f"\nAnalyzing files:")
    print(f"  Trades:  {trades_file.name}")
    print(f"  Signals: {signals_file.name}")
    print(f"  Equity:  {equity_file.name}")

    # Analyze
    trades_analysis = analyze_trades(trades_file)
    signals_analysis = analyze_signals(signals_file)
    equity_analysis = analyze_equity(equity_file)

    # Print reports
    print("\n" + "="*80)
    print("TRADE ANALYSIS")
    print("="*80)

    if trades_analysis:
        print(f"\nTotal Trades: {trades_analysis['total_trades']}")
        print(f"Average Holding Period: {trades_analysis['avg_holding_hours']:.2f} hours")

        print(f"\n📈 Best Trade:")
        best = trades_analysis['best_trade']
        print(f"  Date: {best['entry_timestamp']}")
        print(f"  Entry: ${best['entry_price']:.2f}")
        print(f"  Exit:  ${best['exit_price']:.2f}")
        print(f"  P&L:   ${best['pnl']:.2f} ({best['pnl_pct']:.2f}%)")

        print(f"\n📉 Worst Trade:")
        worst = trades_analysis['worst_trade']
        print(f"  Date: {worst['entry_timestamp']}")
        print(f"  Entry: ${worst['entry_price']:.2f}")
        print(f"  Exit:  ${worst['exit_price']:.2f}")
        print(f"  P&L:   ${worst['pnl']:.2f} ({worst['pnl_pct']:.2f}%)")

        print(f"\n🎯 Performance by Confidence Level:")
        print(trades_analysis['perf_by_confidence'])

        print(f"\n⏰ Performance by Entry Hour (Top 10):")
        top_hours = trades_analysis['perf_by_hour'].nlargest(10, ('pnl', 'mean'))
        print(top_hours)

        print(f"\n⏱️  Performance by Holding Period:")
        print(trades_analysis['perf_by_holding'])

    print("\n" + "="*80)
    print("SIGNAL ANALYSIS")
    print("="*80)

    if signals_analysis:
        print(f"\nTotal Signals: {signals_analysis['total_signals']}")
        print(f"Average Confidence: {signals_analysis['avg_confidence']:.3f}")
        print(f"High Confidence Signals (>0.5): {signals_analysis['high_confidence_signals']}")

        print(f"\n📊 Action Distribution:")
        for action, count in signals_analysis['action_distribution'].items():
            print(f"  {action:10}: {count:4} ({count/signals_analysis['total_signals']*100:.1f}%)")

        print(f"\n🏷️  Zone Type Distribution:")
        for zone_type, count in signals_analysis['zone_type_distribution'].items():
            print(f"  {zone_type:15}: {count:4} ({count/signals_analysis['total_signals']*100:.1f}%)")

        print(f"\n📈 Confidence by Signal Type:")
        print(signals_analysis['confidence_by_signal_type'])

        print(f"\n⏰ Signal Distribution by Hour:")
        print("Hour | Signals")
        print("-" * 20)
        for hour, count in sorted(signals_analysis['signals_by_hour'].items()):
            print(f"{hour:4} | {'█' * (count // 10)}{count}")

    print("\n" + "="*80)
    print("EQUITY CURVE ANALYSIS")
    print("="*80)

    if equity_analysis:
        print(f"\n📉 Drawdown Analysis:")
        print(f"  Max Drawdown:     {equity_analysis['max_drawdown']*100:.2f}%")
        print(f"  Avg Drawdown:     {equity_analysis['avg_drawdown']*100:.2f}%")
        print(f"  Underwater Time:  {equity_analysis['underwater_days']*100:.2f}%")

        print(f"\n📊 Volatility:")
        print(f"  Annual Volatility: {equity_analysis['annual_volatility']*100:.2f}%")
        print(f"  Daily Std Dev:     {equity_analysis['daily_returns_std']*100:.2f}%")

        print(f"\n📈 Daily Returns:")
        print(f"  Mean:  {equity_analysis['daily_returns_mean']*100:.4f}%")
        print(f"  Best:  {equity_analysis['best_day']*100:.2f}%")
        print(f"  Worst: {equity_analysis['worst_day']*100:.2f}%")

    print("\n" + "="*80)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("="*80)

    if trades_analysis and signals_analysis:
        win_rate = (trades_analysis['total_trades'] - trades_analysis.get('losing_trades', 0)) / trades_analysis['total_trades'] if trades_analysis['total_trades'] > 0 else 0

        print(f"\n✅ Strengths:")
        if signals_analysis['total_signals'] > 500:
            print(f"  • High signal frequency ({signals_analysis['total_signals']} signals)")
        if signals_analysis['high_confidence_signals'] > 100:
            print(f"  • Good number of high-confidence signals ({signals_analysis['high_confidence_signals']})")

        print(f"\n⚠️  Areas for Improvement:")
        if win_rate < 0.5:
            print(f"  • Win rate below 50% ({win_rate*100:.1f}%) - consider tightening entry criteria")
        if equity_analysis.get('max_drawdown', 0) > 0.05:
            print(f"  • Max drawdown {equity_analysis['max_drawdown']*100:.1f}% - implement better stop loss")

        print(f"\n💡 Recommendations:")
        print(f"  1. Focus on high-confidence signals (>0.5) - these may perform better")
        print(f"  2. Optimize holding period - current avg is {trades_analysis.get('avg_holding_hours', 0):.1f} hours")
        print(f"  3. Consider time-of-day filters based on performance by hour")
        print(f"  4. Implement dynamic position sizing based on confidence levels")
        print(f"  5. Add stop-loss and take-profit levels to improve risk/reward")

    print("\n" + "="*80)


if __name__ == '__main__':
    generate_comprehensive_report()
