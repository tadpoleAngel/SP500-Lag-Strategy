from datetime import datetime
from backtest import run_backtest, calculate_metrics, print_metrics
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    highest_return = float('-inf')
    highest_return_df = None
    highest_return_threshold = 0.0
    highest_cagr = float('-inf')
    highest_cagr_df = None
    highest_cagr_threshold = 0.0
    lowest_annual_vol = float('inf')
    lowest_annual_vol_df = None
    lowest_annual_vol_threshold = 0.0
    highest_sharpe = float('-inf')
    highest_sharpe_df = None
    highest_sharpe_threshold = 0.0
    lowest_dd = float('inf')
    lowest_dd_df = None
    lowest_dd_threshold = 0.0

    # prepare lists to collect metrics for plotting
    thresholds_list = []
    total_returns = []
    cagr_list = []
    ann_vol_list = []
    sharpe_list = []
    max_dd_list = []

    for i in range(100):
        start = '2008-01-01'
        end = datetime.today().strftime('%Y-%m-%d')
        threshold = 0.0005 * i + 0.02

        df = None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final, df = run_backtest(start, end, return_threshold=threshold)
        
        total_return, cagr, ann_vol, sharpe, max_dd = calculate_metrics(df)

        if total_return > highest_return:
            highest_return = total_return
            highest_return_df = df
            highest_return_threshold = threshold
        if cagr > highest_cagr:
            highest_cagr = cagr
            highest_cagr_df = df
            highest_cagr_threshold = threshold
        if ann_vol < lowest_annual_vol:
            lowest_annual_vol = ann_vol
            lowest_annual_vol_df = df
            lowest_annual_vol_threshold = threshold
        if sharpe > highest_sharpe:
            highest_sharpe = sharpe
            highest_sharpe_df = df
            highest_sharpe_threshold = threshold
        if abs(max_dd) < abs(lowest_dd):
            lowest_dd = max_dd
            lowest_dd_df = df
            lowest_dd_threshold = threshold

        # collect metrics for plotting
        thresholds_list.append(threshold)
        total_returns.append(total_return)
        cagr_list.append(cagr)
        ann_vol_list.append(ann_vol)
        sharpe_list.append(sharpe)
        max_dd_list.append(max_dd)
    
    print(f"Highest total return: {highest_return:.2%} at threshold {highest_return_threshold:.4f}")
    print_metrics(*calculate_metrics(highest_return_df))
    print()
    print(f"Highest CAGR: {highest_cagr:.2%} at threshold {highest_cagr_threshold:.4f}")
    print_metrics(*calculate_metrics(highest_cagr_df))
    print()
    print(f"Lowest annual volatility: {lowest_annual_vol:.2%} at threshold {lowest_annual_vol_threshold:.4f}")
    print_metrics(*calculate_metrics(lowest_annual_vol_df))
    print()
    print(f"Highest Sharpe ratio: {highest_sharpe:.2f} at threshold {highest_sharpe_threshold:.4f}")
    print_metrics(*calculate_metrics(highest_sharpe_df))
    print()
    print(f"Lowest maximum drawdown: {lowest_dd:.2%} at threshold {lowest_dd_threshold:.4f}")
    print_metrics(*calculate_metrics(lowest_dd_df))

    # ensure plots directory exists
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # helper to plot and save
    def save_plot(x, y, xlabel, ylabel, title, filename, highlight_x=None, highlight_y=None):
        plt.figure()
        plt.plot(x, y, marker='o', linestyle='-')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if highlight_x is not None and highlight_y is not None:
            plt.scatter([highlight_x], [highlight_y], color='red', zorder=5)
            plt.annotate(f"best: {highlight_x:.4f}", xy=(highlight_x, highlight_y), xytext=(5,5), textcoords='offset points')
        path = os.path.join(plots_dir, filename)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Saved plot: {path}")

    # create plots for each metric
    save_plot(thresholds_list, total_returns, 'Threshold', 'Total return', 'Total return vs Threshold', 'total_return_vs_threshold.png', highlight_x=highest_return_threshold, highlight_y=highest_return)
    save_plot(thresholds_list, cagr_list, 'Threshold', 'CAGR', 'CAGR vs Threshold', 'cagr_vs_threshold.png', highlight_x=highest_cagr_threshold, highlight_y=highest_cagr)
    save_plot(thresholds_list, ann_vol_list, 'Threshold', 'Annual Volatility', 'Annual Volatility vs Threshold', 'ann_vol_vs_threshold.png', highlight_x=lowest_annual_vol_threshold, highlight_y=lowest_annual_vol)
    save_plot(thresholds_list, sharpe_list, 'Threshold', 'Sharpe Ratio', 'Sharpe Ratio vs Threshold', 'sharpe_vs_threshold.png', highlight_x=highest_sharpe_threshold, highlight_y=highest_sharpe)
    save_plot(thresholds_list, max_dd_list, 'Threshold', 'Max Drawdown', 'Max Drawdown vs Threshold', 'max_dd_vs_threshold.png', highlight_x=lowest_dd_threshold, highlight_y=lowest_dd)

    # save a combined figure with stacked subplots
    def save_combined_plot(thresholds, returns, cagr_vals, vol_vals, sharpe_vals, dd_vals, filename):
        fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

        axes[0].plot(thresholds, returns, marker='o')
        axes[0].set_ylabel('Total return')
        axes[0].axvline(highest_return_threshold, color='red', linestyle='--')

        axes[1].plot(thresholds, cagr_vals, marker='o')
        axes[1].set_ylabel('CAGR')
        axes[1].axvline(highest_cagr_threshold, color='red', linestyle='--')

        axes[2].plot(thresholds, vol_vals, marker='o')
        axes[2].set_ylabel('Annual Vol')
        axes[2].axvline(lowest_annual_vol_threshold, color='red', linestyle='--')

        axes[3].plot(thresholds, sharpe_vals, marker='o')
        axes[3].set_ylabel('Sharpe')
        axes[3].axvline(highest_sharpe_threshold, color='red', linestyle='--')

        axes[4].plot(thresholds, dd_vals, marker='o')
        axes[4].set_ylabel('Max Drawdown')
        axes[4].set_xlabel('Threshold')
        axes[4].axvline(lowest_dd_threshold, color='red', linestyle='--')

        fig.suptitle('Metrics vs Threshold')
        for ax in axes:
            ax.grid(True)

        path = os.path.join(plots_dir, filename)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(path)
        plt.close()
        print(f"Saved combined plot: {path}")

    save_combined_plot(thresholds_list, total_returns, cagr_list, ann_vol_list, sharpe_list, max_dd_list, 'combined_metrics_vs_threshold.png')