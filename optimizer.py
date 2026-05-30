from datetime import datetime
from backtest import run_backtest, calculate_metrics, print_metrics, download_data, SP500_TICKER, VIX_TICKER
import warnings
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description='Optimize S&P 500 return threshold and VIX scale for backtesting strategy across multiple tickers.')

parser.add_argument('--cache', action=argparse.BooleanOptionalAction, default=True, help='Whether to use caching for downloaded data (default: True)')

args = parser.parse_args()

if args.cache:
    CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_FILE = os.path.join(CACHE_DIR, 'data_cache.pkl')

VIX_SCALES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'COIN', 'BRK-B', 'SOXS', 'JNJ']
# TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B', 'JPM', 'JNJ', 'SPHY', 'SCYB']
TICKERS = [
    # Mega Cap Tech / High Beta
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    'AMD', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'INTC', 'QCOM',
    'MU', 'SMCI', 'PLTR', 'SNOW', 'NET', 'CRWD',

    # Semiconductor / AI Supply Chain
    'ASML', 'ARM', 'TSM', 'AMAT', 'LRCX', 'KLAC', 'MRVL',

    # Financials
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'SCHW',
    'AXP', 'BLK', 'KKR',

    # Consumer / Cyclical
    'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS',
    'BKNG', 'RCL', 'CCL', 'ABNB', 'UBER',

    # Industrials / Economic Sensitivity
    'CAT', 'DE', 'GE', 'BA', 'RTX', 'ETN', 'PH',
    'UPS', 'FDX', 'UNP',

    # Energy
    'XOM', 'CVX', 'SLB', 'HAL', 'OXY',

    # Healthcare / Defensive Growth
    'JNJ', 'LLY', 'UNH', 'PFE', 'ABBV', 'MRK',

    # High Yield / Credit / Bond Correlation
    'SPHY', 'SCYB', 'HYG', 'JNK', 'USHY', 'ANGL',

    # Leveraged / Market Amplifiers
    'TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'SPXL', 'SPXS',

    # Broad ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',

    # Volatility / Risk-On-Risk-Off
    'ARKK', 'BITO', 'MSTR', 'COIN'
]

if __name__ == "__main__":
    # overall collectors across tickers for top-performers
    overall_stats = {
        'total_return': [],  # tuples of (ticker, value, threshold)
        'cagr': [],
        'ann_vol': [],
        'sharpe': [],
        'max_dd': []
    }

    start = '2008-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    plots_root = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_root, exist_ok=True)

    # helper to plot and save per-ticker
    def save_plot(x, y, xlabel, ylabel, title, path, highlight_x=None, highlight_y=None, highlight_info=None):
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if highlight_x is not None and highlight_y is not None:
            plt.scatter([highlight_x], [highlight_y], color='red', zorder=5)
            ann_text = f"thr: {highlight_x:.4f}"
            if highlight_info is not None:
                ann_text += "\n" + highlight_info
            plt.annotate(ann_text, xy=(highlight_x, highlight_y), xytext=(5,5), textcoords='offset points')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    def save_combined_plot(thresholds, returns, cagr_vals, vol_vals, sharpe_vals, dd_vals, path,
                           h_return=None, h_cagr=None, h_vol=None, h_sharpe=None, h_dd=None):
        fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

        axes[0].plot(thresholds, returns, marker='o')
        axes[0].set_ylabel('Total return')
        if h_return is not None:
            axes[0].axvline(h_return, color='red', linestyle='--')

        axes[1].plot(thresholds, cagr_vals, marker='o')
        axes[1].set_ylabel('CAGR')
        if h_cagr is not None:
            axes[1].axvline(h_cagr, color='red', linestyle='--')

        axes[2].plot(thresholds, vol_vals, marker='o')
        axes[2].set_ylabel('Annual Vol')
        if h_vol is not None:
            axes[2].axvline(h_vol, color='red', linestyle='--')

        axes[3].plot(thresholds, sharpe_vals, marker='o')
        axes[3].set_ylabel('Sharpe')
        if h_sharpe is not None:
            axes[3].axvline(h_sharpe, color='red', linestyle='--')

        axes[4].plot(thresholds, dd_vals, marker='o')
        axes[4].set_ylabel('Max Drawdown')
        axes[4].set_xlabel('Threshold')
        if h_dd is not None:
            axes[4].axvline(h_dd, color='red', linestyle='--')

        fig.suptitle('Metrics vs Threshold')
        for ax in axes:
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(path)
        plt.close()
        print(f"Saved combined plot: {path}")

    def save_heatmap(x_vals, y_vals, matrix, xlabel, ylabel, title, path, fmt='.2f'):
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(matrix, aspect='auto', origin='lower', cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(title)

        x_step = max(1, len(x_vals) // 12)
        x_ticks = list(range(0, len(x_vals), x_step))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x_vals[i]:.4f}" for i in x_ticks], rotation=45, ha='right')
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{y:.2f}" for y in y_vals])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        finite_vals = [
            (row_idx, col_idx, val)
            for row_idx, row in enumerate(matrix)
            for col_idx, val in enumerate(row)
            if not math.isnan(val)
        ]
        if finite_vals:
            best_row, best_col, best_val = max(finite_vals, key=lambda item: item[2])
            ax.scatter([best_col], [best_row], marker='x', color='red', s=80)
            ax.annotate(
                format(best_val, fmt),
                xy=(best_col, best_row),
                xytext=(6, 6),
                textcoords='offset points',
                color='white',
                weight='bold',
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved heatmap: {path}")

    # per-ticker sweep
    # Attempt to load a single cache file that contains previously downloaded data.
    cache = {}
    if args.cache:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"Loaded cache from {CACHE_FILE}")
            except Exception as e:
                print(f"Failed to load cache: {e}")

    for ticker in TICKERS:
        print(f"\nOptimizing ticker: {ticker}")
        ticker_dir = os.path.join(plots_root, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        highest_return = float('-inf')
        highest_return_threshold = 0.0
        highest_return_trades = 0
        highest_return_pct_gain = float('nan')
        highest_cagr = float('-inf')
        highest_cagr_threshold = 0.0
        highest_cagr_trades = 0
        highest_cagr_pct_gain = float('nan')
        lowest_annual_vol = float('inf')
        lowest_annual_vol_threshold = 0.0
        lowest_annual_vol_trades = 0
        lowest_annual_vol_pct_gain = float('nan')
        highest_sharpe = float('-inf')
        highest_sharpe_threshold = 0.0
        highest_sharpe_trades = 0
        highest_sharpe_pct_gain = float('nan')
        lowest_dd = float('inf')
        lowest_dd_threshold = 0.0
        lowest_dd_trades = 0
        lowest_dd_pct_gain = float('nan')

        thresholds_list = []
        total_returns = []
        cagr_list = []
        ann_vol_list = []
        sharpe_list = []
        max_dd_list = []

        def _get_data_arg():
            cache_key = (start, end, (ticker,))
            range_global_key = (start, end, 'global')
            if cache_key in cache:
                return cache[cache_key]
            if range_global_key in cache:
                return cache[range_global_key]
            return cache.get('global_union')

        def _ensure_vix_cached():
            cache_key = (start, end, (ticker,))
            data_arg = _get_data_arg()
            if data_arg is not None and VIX_TICKER in data_arg:
                return data_arg

            full_data = download_data([ticker], start, end, include_vix=True)
            cache[cache_key] = full_data
            union = cache.get('global_union', {})
            for k, df_k in full_data.items():
                union[k] = df_k
            cache['global_union'] = union
            if args.cache:
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(cache, f)
            return full_data

        for i in range(-20, 100):
            threshold = 0.001 * i

            df = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Prepare data argument: try per-ticker cache first, then a range-keyed global cache,
                # then fall back to a merged union cache that collects all previously downloaded tickers.
                data_arg = None
                cache_key = (start, end, (ticker,))
                range_global_key = (start, end, 'global')
                # prefer exact per-ticker cached dataset
                if cache_key in cache:
                    data_arg = cache[cache_key]
                # then prefer a global dataset that matches the same date range
                elif range_global_key in cache:
                    data_arg = cache[range_global_key]
                # finally fall back to a union/global cache containing merged tickers (any date range)
                else:
                    data_arg = cache.get('global_union')

                final, df, trade_info = run_backtest(start, end, return_threshold=threshold, tickers=[ticker], data=data_arg)

                # after first successful run with downloaded data, persist to cache if empty
                if (cache_key not in cache) and df is not None:
                    # store the data dict returned by run_backtest's internal download step by invoking
                    # download once separately to capture all tickers + SP500. This keeps cache simple.
                    try:
                        # on first use, create a global cache with the ticker data, S&P, and VIX
                        full_data = download_data([ticker], start, end, include_vix=True)
                        # save the per-ticker, range-keyed cache
                        cache[cache_key] = full_data
                        # also save a range-keyed global snapshot for this date-range
                        cache[range_global_key] = full_data
                        # merge into a persistent union/global cache which aggregates all tickers
                        union = cache.get('global_union', {})
                        for k, df_k in full_data.items():
                            union[k] = df_k
                        cache['global_union'] = union
                        if args.cache:
                            with open(CACHE_FILE, 'wb') as f:
                                pickle.dump(cache, f)
                            print(f"Saved data to cache: {CACHE_FILE}")
                    except Exception as e:
                        print(f"Failed to build/save cache: {e}")

            total_return, cagr, ann_vol, sharpe, max_dd, num_trades, pct_gain, pct_loss = calculate_metrics(df, trade_info)

            if total_return > highest_return:
                highest_return = total_return
                highest_return_threshold = threshold
                highest_return_trades = num_trades
                highest_return_pct_gain = pct_gain
            if cagr > highest_cagr:
                highest_cagr = cagr
                highest_cagr_threshold = threshold
                highest_cagr_trades = num_trades
                highest_cagr_pct_gain = pct_gain
            if ann_vol < lowest_annual_vol:
                lowest_annual_vol = ann_vol
                lowest_annual_vol_threshold = threshold
                lowest_annual_vol_trades = num_trades
                lowest_annual_vol_pct_gain = pct_gain
            if sharpe > highest_sharpe:
                highest_sharpe = sharpe
                highest_sharpe_threshold = threshold
                highest_sharpe_trades = num_trades
                highest_sharpe_pct_gain = pct_gain
            if abs(max_dd) < abs(lowest_dd):
                lowest_dd = max_dd
                lowest_dd_threshold = threshold
                lowest_dd_trades = num_trades
                lowest_dd_pct_gain = pct_gain

            thresholds_list.append(threshold)
            total_returns.append(total_return)
            cagr_list.append(cagr)
            ann_vol_list.append(ann_vol)
            sharpe_list.append(sharpe)
            max_dd_list.append(max_dd)

        vix_total_return_matrix = []
        vix_sharpe_matrix = []
        vix_dd_matrix = []
        vix_trades_matrix = []
        best_vix_sharpe = float('-inf')
        best_vix_threshold = 0.0
        best_vix_scale = 0.0
        best_vix_trades = 0
        vix_data_arg = _ensure_vix_cached()

        for vix_scale in VIX_SCALES:
            row_returns = []
            row_sharpe = []
            row_dd = []
            row_trades = []

            for threshold in thresholds_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    final, df, trade_info = run_backtest(
                        start,
                        end,
                        return_threshold=threshold,
                        tickers=[ticker],
                        data=vix_data_arg,
                        vix_threshold_scale=vix_scale,
                    )

                total_return, cagr, ann_vol, sharpe, max_dd, num_trades, pct_gain, pct_loss = calculate_metrics(df, trade_info)
                row_returns.append(total_return)
                row_sharpe.append(sharpe)
                row_dd.append(max_dd)
                row_trades.append(num_trades)

                if not math.isnan(sharpe) and sharpe > best_vix_sharpe:
                    best_vix_sharpe = sharpe
                    best_vix_threshold = threshold
                    best_vix_scale = vix_scale
                    best_vix_trades = num_trades

            vix_total_return_matrix.append(row_returns)
            vix_sharpe_matrix.append(row_sharpe)
            vix_dd_matrix.append(row_dd)
            vix_trades_matrix.append(row_trades)

        # print per-ticker bests
        print(f"Highest total return for {ticker}: {highest_return:.2%} at threshold {highest_return_threshold:.4f}")
        print(f"Highest CAGR for {ticker}: {highest_cagr:.2%} at threshold {highest_cagr_threshold:.4f}")
        print(f"Lowest annual volatility for {ticker}: {lowest_annual_vol:.2%} at threshold {lowest_annual_vol_threshold:.4f}")
        print(f"Highest Sharpe for {ticker}: {highest_sharpe:.2f} at threshold {highest_sharpe_threshold:.4f}")
        print(f"Lowest Max Drawdown for {ticker}: {lowest_dd:.2%} at threshold {lowest_dd_threshold:.4f}")
        print(f"Highest VIX-scaled Sharpe for {ticker}: {best_vix_sharpe:.2f} at threshold {best_vix_threshold:.4f}, VIX scale {best_vix_scale:.2f}, trades {best_vix_trades}")

        # prepare highlight info strings that include number of trades and percent gains
        def _info(trades, pct):
            if isinstance(pct, float) and math.isnan(pct):
                pct_label = 'N/A'
            else:
                pct_label = f"{pct:.1f}%"
            return f"trades: {trades}\n%gain: {pct_label}"

        hr_info = _info(highest_return_trades, highest_return_pct_gain)
        hc_info = _info(highest_cagr_trades, highest_cagr_pct_gain)
        lav_info = _info(lowest_annual_vol_trades, lowest_annual_vol_pct_gain)
        hs_info = _info(highest_sharpe_trades, highest_sharpe_pct_gain)
        ld_info = _info(lowest_dd_trades, lowest_dd_pct_gain)

        # save per-ticker plots with extra info
        save_plot(thresholds_list, total_returns, 'Threshold', 'Total return', f'{ticker} Total return vs Threshold', os.path.join(ticker_dir, 'total_return_vs_threshold.png'), highlight_x=highest_return_threshold, highlight_y=highest_return, highlight_info=hr_info)
        save_plot(thresholds_list, cagr_list, 'Threshold', 'CAGR', f'{ticker} CAGR vs Threshold', os.path.join(ticker_dir, 'cagr_vs_threshold.png'), highlight_x=highest_cagr_threshold, highlight_y=highest_cagr, highlight_info=hc_info)
        save_plot(thresholds_list, ann_vol_list, 'Threshold', 'Annual Volatility', f'{ticker} Annual Volatility vs Threshold', os.path.join(ticker_dir, 'ann_vol_vs_threshold.png'), highlight_x=lowest_annual_vol_threshold, highlight_y=lowest_annual_vol, highlight_info=lav_info)
        save_plot(thresholds_list, sharpe_list, 'Threshold', 'Sharpe Ratio', f'{ticker} Sharpe Ratio vs Threshold', os.path.join(ticker_dir, 'sharpe_vs_threshold.png'), highlight_x=highest_sharpe_threshold, highlight_y=highest_sharpe, highlight_info=hs_info)
        save_plot(thresholds_list, max_dd_list, 'Threshold', 'Max Drawdown', f'{ticker} Max Drawdown vs Threshold', os.path.join(ticker_dir, 'max_dd_vs_threshold.png'), highlight_x=lowest_dd_threshold, highlight_y=lowest_dd, highlight_info=ld_info)

        save_combined_plot(thresholds_list, total_returns, cagr_list, ann_vol_list, sharpe_list, max_dd_list, os.path.join(ticker_dir, 'combined_metrics_vs_threshold.png'),
                           h_return=highest_return_threshold, h_cagr=highest_cagr_threshold, h_vol=lowest_annual_vol_threshold, h_sharpe=highest_sharpe_threshold, h_dd=lowest_dd_threshold)

        save_heatmap(thresholds_list, VIX_SCALES, vix_total_return_matrix, 'Base Threshold', 'VIX Scale', f'{ticker} Total Return: Threshold x VIX Scale', os.path.join(ticker_dir, 'vix_grid_total_return.png'), fmt='.2%')
        save_heatmap(thresholds_list, VIX_SCALES, vix_sharpe_matrix, 'Base Threshold', 'VIX Scale', f'{ticker} Sharpe: Threshold x VIX Scale', os.path.join(ticker_dir, 'vix_grid_sharpe.png'), fmt='.2f')
        save_heatmap(thresholds_list, VIX_SCALES, vix_dd_matrix, 'Base Threshold', 'VIX Scale', f'{ticker} Max Drawdown: Threshold x VIX Scale', os.path.join(ticker_dir, 'vix_grid_max_dd.png'), fmt='.2%')
        save_heatmap(thresholds_list, VIX_SCALES, vix_trades_matrix, 'Base Threshold', 'VIX Scale', f'{ticker} Trades: Threshold x VIX Scale', os.path.join(ticker_dir, 'vix_grid_trades.png'), fmt='.0f')

        # collect for overall top performers
        overall_stats['total_return'].append((ticker, highest_return, highest_return_threshold, highest_return_trades, highest_return_pct_gain))
        overall_stats['cagr'].append((ticker, highest_cagr, highest_cagr_threshold, highest_cagr_trades, highest_cagr_pct_gain))
        overall_stats['ann_vol'].append((ticker, lowest_annual_vol, lowest_annual_vol_threshold, lowest_annual_vol_trades, lowest_annual_vol_pct_gain))
        overall_stats['sharpe'].append((ticker, highest_sharpe, highest_sharpe_threshold, highest_sharpe_trades, highest_sharpe_pct_gain))
        overall_stats['max_dd'].append((ticker, lowest_dd, lowest_dd_threshold, lowest_dd_trades, lowest_dd_pct_gain))

    # create top-performers plots
    top_dir = os.path.join(plots_root, 'top_performers')
    os.makedirs(top_dir, exist_ok=True)

    def save_top_bars(stat_key, better='higher'):
        items = overall_stats[stat_key]
        if better == 'higher':
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        else:
            items_sorted = sorted(items, key=lambda x: x[1])

        top5 = items_sorted[:5]
        # items are (ticker, value, threshold, num_trades, pct_gain)
        tickers_ = [t for t, v, thr, tr, pct in top5]
        values_ = [v for t, v, thr, tr, pct in top5]
        thresholds_ = [thr for t, v, thr, tr, pct in top5]
        trades_ = [tr for t, v, thr, tr, pct in top5]
        pct_gains_ = [pct for t, v, thr, tr, pct in top5]

        plt.figure(figsize=(8,4))
        bars = plt.bar(tickers_, values_)
        plt.title(f'Top 5 tickers by {stat_key}')
        plt.ylabel(stat_key)
        # label each bar with threshold, trades, and percent gain
        for rect, thr, tr, pct in zip(bars, thresholds_, trades_, pct_gains_):
            height = rect.get_height()
            pct_label = f"{pct:.1f}%" if not (isinstance(pct, float) and math.isnan(pct)) else 'N/A'
            plt.annotate(f'{thr:.4f}\ntrades:{tr}\n%gain:{pct_label}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0,3), textcoords='offset points', ha='center', va='bottom')

        path = os.path.join(top_dir, f'top5_{stat_key}.png')
        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig(path)
        plt.close()
        print(f"Saved top performers plot: {path}")

    save_top_bars('total_return', better='higher')
    save_top_bars('cagr', better='higher')
    save_top_bars('ann_vol', better='lower')
    save_top_bars('sharpe', better='higher')
    save_top_bars('max_dd', better='higher') # max_dd is negative so higher is better
