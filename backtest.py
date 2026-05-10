import warnings
import yfinance as yf
import pandas as pd
from datetime import datetime
from math import floor
from typing import Dict, Tuple

from trade import compute_desired_from_prices, compute_signal_from_closes

# Backtest configuration - can be adjusted when running
TICKERS = ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
SP500_TICKER = "^GSPC"
INITIAL_CAPITAL = 100000.0
CAPITAL_FRACTION = 0.9
RETURN_THRESHOLD = 0.05


def download_data(tickers, start: str, end: str) -> Dict[str, pd.DataFrame]:
    all_tickers = tickers + [SP500_TICKER]
    df = yf.download(all_tickers, start=start, end=end, group_by='ticker', progress=False)

    result = {}
    for t in all_tickers:
        # yfinance returns multi-level columns when multiple tickers are requested
        if isinstance(df.columns, pd.MultiIndex) and t in df.columns.get_level_values(0):
            result[t] = df[t].copy()
        else:
            result[t] = df.copy()

    return result


def compute_signal(sp_df: pd.DataFrame, idx: int, threshold: float) -> int:
    # wrapper that pulls the two closes and calls the canonical helper
    if idx < 2:
        return 0

    yesterday_close = sp_df['Close'].iat[idx - 1]
    prev_close = sp_df['Close'].iat[idx - 2]
    if prev_close == 0 or pd.isna(prev_close) or pd.isna(yesterday_close):
        return 0

    return compute_signal_from_closes(prev_close, yesterday_close, threshold)


def get_exec_price(df: pd.DataFrame, date) -> float | None:
    if df is None or date not in df.index:
        return None
    if 'Open' in df.columns and not pd.isna(df.at[date, 'Open']):
        return df.at[date, 'Open']
    return df.at[date, 'Close']


def compute_desired_for_date(data: Dict[str, pd.DataFrame], date, signal: int, per_ticker: float) -> Tuple[Dict[str, int], Dict[str, float]]:
    desired = {}
    exec_prices = {}
    for t in TICKERS:
        df = data.get(t)
        exec_price = get_exec_price(df, date)
        exec_prices[t] = exec_price

        if not exec_price or pd.isna(exec_price) or exec_price <= 0:
            desired_qty = 0
        else:
            qty = int(per_ticker // exec_price)
            desired_qty = qty if signal == 1 else -qty

        desired[t] = desired_qty

    return desired, exec_prices


def execute_trades(positions: Dict[str, int], desired: Dict[str, int], exec_prices: Dict[str, float]) -> Tuple[float, list]:
    trades = []
    cash_delta = 0.0
    for t in TICKERS:
        curr = positions.get(t, 0)
        diff = desired[t] - curr
        price = exec_prices.get(t)
        if diff == 0 or price is None or pd.isna(price):
            continue
        cost = diff * price
        cash_delta -= cost
        positions[t] = curr + diff
        trades.append({'symbol': t, 'qty': diff, 'price': price})
    return cash_delta, trades


def compute_portfolio_value(positions: Dict[str, int], cash: float, data: Dict[str, pd.DataFrame], date) -> float:
    pv = cash
    for t in TICKERS:
        df = data.get(t)
        if df is None or date not in df.index:
            price = None
        else:
            price = df.at[date, 'Close'] if 'Close' in df.columns else None
        if price is not None and not pd.isna(price):
            pv += positions.get(t, 0) * price
    return pv


def run_backtest(start: str, end: str, return_threshold: float = RETURN_THRESHOLD, initial_capital: float = INITIAL_CAPITAL) -> Tuple[float, pd.DataFrame]:
    data = download_data(TICKERS, start, end)
    sp = download_data([SP500_TICKER], start, end)[SP500_TICKER]

    dates = sp.index
    positions = dict.fromkeys(TICKERS, 0)
    cash = initial_capital
    history = []

    for i, date in enumerate(dates):
        signal = compute_signal(sp, i, return_threshold)

        if signal == 0:
            pv = compute_portfolio_value(positions, cash, data, date)
            history.append({'date': date, 'signal': 0, 'cash': cash, 'pv': pv, 'positions': positions.copy(), 'trades': []})
            continue

        total_alloc = cash * CAPITAL_FRACTION
        per_ticker = total_alloc / len(TICKERS)

        # compute execution prices and desired using canonical helper
        _, exec_prices = compute_desired_for_date(data, date, signal, per_ticker)

        # use helper from trade.py for desired qty calculation
        desired = compute_desired_from_prices(per_ticker, signal, exec_prices)

        cash_delta, trades = execute_trades(positions, desired, exec_prices)
        cash += cash_delta

        pv = compute_portfolio_value(positions, cash, data, date)
        history.append({'date': date, 'signal': signal, 'cash': cash, 'pv': pv, 'positions': positions.copy(), 'trades': trades})

    hist_df = pd.DataFrame(history).set_index('date')
    final_value = hist_df['pv'].iloc[-1] if not hist_df.empty else initial_capital
    return final_value, hist_df

def calculate_metrics(df: pd.DataFrame):
    # compute simple performance metrics
    df = df.dropna()
    if not df.empty:
        returns = df['pv'].pct_change().dropna()
        total_return = df['pv'].iloc[-1] / df['pv'].iloc[0] - 1
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else 1
        cagr = float('nan')

        warnings.filterwarnings("error", message="invalid value encountered in scalar power", category=RuntimeWarning)
        try:
            cagr = (1 + total_return) ** (1 / years) - 1
        except RuntimeWarning:
            print("Hey man, your cagr went negative you're losing a ton of money")
        finally:
            warnings.resetwarnings()

        ann_vol = returns.std() * (252 ** 0.5)
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else float('nan')
        # max drawdown
        cum = df['pv']
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()
        return total_return, cagr, ann_vol, sharpe, max_dd

def print_metrics(total_return, cagr, ann_vol, sharpe, max_dd):
    print(f'Total return: {total_return:.2%}')
    print(f'CAGR: {cagr:.2%}')
    print(f'Annualized vol: {ann_vol:.2%}')
    print(f'Sharpe (rf=0): {sharpe:.2f}')
    print(f'Max drawdown: {max_dd:.2%}')


if __name__ == '__main__':
    start = '2024-05-09'
    end = datetime.today().strftime('%Y-%m-%d')
    final, df = run_backtest(start, end)
    print(f'Final portfolio value: ${final:,.2f}')  
    print_metrics(*calculate_metrics(df))
    print(df[['signal', 'cash', 'pv']].tail(10))

    
