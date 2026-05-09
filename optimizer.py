from datetime import datetime
from backtest import run_backtest, calculate_metrics, print_metrics
import warnings

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

    for i in range(100):
        start = '2024-05-09'
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