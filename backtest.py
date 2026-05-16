import warnings
import yfinance as yf
import pandas as pd
from datetime import datetime
from math import floor
from typing import Dict, Tuple

from trade import compute_desired_from_prices, compute_signal_from_closes

# Backtest configuration - can be adjusted when running
# TICKERS = ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
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


def compute_desired_for_date(data: Dict[str, pd.DataFrame], date, signal: int, per_ticker: float, tickers: list[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    exec_prices: Dict[str, float] = {}
    for t in tickers:
        exec_prices[t] = get_exec_price(data.get(t), date)

    # Reuse canonical helper from trade.py to compute integer desired quantities
    desired = compute_desired_from_prices(per_ticker, signal, exec_prices)
    return desired, exec_prices


def execute_trades(positions: Dict[str, int], desired: Dict[str, int], exec_prices: Dict[str, float], tickers: list[str]) -> Tuple[float, list]:
    trades = []
    cash_delta = 0.0
    for t in tickers:
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


def compute_portfolio_value(positions: Dict[str, int], cash: float, data: Dict[str, pd.DataFrame], tickers: list[str], date) -> float:
    pv = cash
    for t in tickers:
        df = data.get(t)
        if df is None or date not in df.index:
            price = None
        else:
            price = df.at[date, 'Close'] if 'Close' in df.columns else None
        if price is not None and not pd.isna(price):
            pv += positions.get(t, 0) * price
    return pv


def run_backtest(start: str, end: str, return_threshold: float = RETURN_THRESHOLD, tickers: list[str] = None, data: Dict[str, pd.DataFrame] = None, initial_capital: float = INITIAL_CAPITAL) -> Tuple[float, pd.DataFrame, list]:
    if tickers is None:
        tickers = ["SPHY", "SCYB"]

    if data is None:
        data = download_data(tickers, start, end)
    # reuse SP500 data from the provided `data` dict when available to avoid re-downloading
    if SP500_TICKER in data and data.get(SP500_TICKER) is not None:
        sp = data[SP500_TICKER]
    else:
        sp = download_data([SP500_TICKER], start, end)[SP500_TICKER]

    dates = sp.index
    positions = dict.fromkeys(tickers, 0)
    cash = initial_capital
    history = []

    # ledger for open lots per ticker: list of tuples (qty_signed, entry_price, entry_date)
    open_lots = {t: [] for t in tickers}
    # collect closed trade events as detailed dicts per closed chunk
    # each closed trade: {symbol, qty, entry_price, entry_date, exit_price, exit_date, pnl}
    closed_trade_events = []
    # count of executed trade instructions placed (each execution line is one "trade placed")
    trades_placed_count = 0

    for i, date in enumerate(dates):
        signal = compute_signal(sp, i, return_threshold)

        if signal == 0:
            # On neutral signal, close/rebalance to zero desired quantities so positions are liquidated
            exec_prices = {t: get_exec_price(data.get(t), date) for t in tickers}
            desired = {t: 0 for t in tickers}

            cash_delta, trades = execute_trades(positions, desired, exec_prices, tickers)
            trades_placed_count += len(trades)

            # update ledger & compute closed trade P&L (FIFO) - reuse same logic as below
            for tr in trades:
                sym = tr['symbol']
                qty = tr['qty']  # signed: positive = buy, negative = sell
                price = tr['price']
                lots = open_lots.get(sym, [])
                qty_remaining = qty

                if qty > 0:
                    j = 0
                    while qty_remaining > 0 and j < len(lots):
                        lot_qty, lot_price, lot_date = lots[j]
                        if lot_qty < 0:
                            close_qty = min(qty_remaining, -lot_qty)
                            pnl = close_qty * (lot_price - price)
                            closed_trade_events.append({'symbol': sym, 'qty': close_qty, 'entry_price': lot_price, 'entry_date': lot_date, 'exit_price': price, 'exit_date': date, 'pnl': pnl})
                            new_lot_qty = lot_qty + close_qty
                            if new_lot_qty == 0:
                                lots.pop(j)
                            else:
                                lots[j] = (new_lot_qty, lot_price, lot_date)
                                j += 1
                            qty_remaining -= close_qty
                        else:
                            j += 1
                    if qty_remaining > 0:
                        lots.append((qty_remaining, price, date))

                elif qty < 0:
                    sell_qty = -qty
                    j = 0
                    while sell_qty > 0 and j < len(lots):
                        lot_qty, lot_price, lot_date = lots[j]
                        if lot_qty > 0:
                            close_qty = min(sell_qty, lot_qty)
                            pnl = close_qty * (price - lot_price)
                            closed_trade_events.append({'symbol': sym, 'qty': close_qty, 'entry_price': lot_price, 'entry_date': lot_date, 'exit_price': price, 'exit_date': date, 'pnl': pnl})
                            new_lot_qty = lot_qty - close_qty
                            if new_lot_qty == 0:
                                lots.pop(j)
                            else:
                                lots[j] = (new_lot_qty, lot_price, lot_date)
                                j += 1
                            sell_qty -= close_qty
                        else:
                            j += 1
                    if sell_qty > 0:
                        lots.append((-sell_qty, price, date))

                open_lots[sym] = lots

            cash += cash_delta

            pv = compute_portfolio_value(positions, cash, data, tickers, date)
            history.append({'date': date, 'signal': 0, 'cash': cash, 'pv': pv, 'positions': positions.copy(), 'trades': trades})
            continue

        total_alloc = cash * CAPITAL_FRACTION
        per_ticker = total_alloc / len(tickers)

        # compute execution prices and desired using canonical helper
        _, exec_prices = compute_desired_for_date(data, date, signal, per_ticker, tickers)

        # use helper from trade.py for desired qty calculation
        desired = compute_desired_from_prices(per_ticker, signal, exec_prices)

        cash_delta, trades = execute_trades(positions, desired, exec_prices, tickers)
        trades_placed_count += len(trades)

        # update ledger & compute closed trade P&L (FIFO)
        for tr in trades:
            sym = tr['symbol']
            qty = tr['qty']  # signed: positive = buy, negative = sell
            price = tr['price']
            # convenience
            lots = open_lots.get(sym, [])
            qty_remaining = qty

            # If buying, first attempt to close existing short lots (negative lots)
            if qty > 0:
                j = 0
                while qty_remaining > 0 and j < len(lots):
                    lot_qty, lot_price, lot_date = lots[j]
                    if lot_qty < 0:  # short lot, can be closed by buying
                        close_qty = min(qty_remaining, -lot_qty)
                        pnl = close_qty * (lot_price - price)  # short entry - buy price
                        closed_trade_events.append({
                            'symbol': sym,
                            'qty': close_qty,
                            'entry_price': lot_price,
                            'entry_date': lot_date,
                            'exit_price': price,
                            'exit_date': date,
                            'pnl': pnl,
                        })
                        # adjust the lot
                        new_lot_qty = lot_qty + close_qty  # less negative
                        if new_lot_qty == 0:
                            lots.pop(j)
                        else:
                            lots[j] = (new_lot_qty, lot_price, lot_date)
                            j += 1
                        qty_remaining -= close_qty
                    else:
                        j += 1
                # any remaining becomes a new long lot
                if qty_remaining > 0:
                    lots.append((qty_remaining, price, date))

            # If selling, first attempt to close existing long lots (positive lots)
            elif qty < 0:
                sell_qty = -qty
                j = 0
                while sell_qty > 0 and j < len(lots):
                    lot_qty, lot_price, lot_date = lots[j]
                    if lot_qty > 0:  # long lot, can be closed by selling
                        close_qty = min(sell_qty, lot_qty)
                        pnl = close_qty * (price - lot_price)  # sell price - long entry
                        closed_trade_events.append({
                            'symbol': sym,
                            'qty': close_qty,
                            'entry_price': lot_price,
                            'entry_date': lot_date,
                            'exit_price': price,
                            'exit_date': date,
                            'pnl': pnl,
                        })
                        new_lot_qty = lot_qty - close_qty
                        if new_lot_qty == 0:
                            lots.pop(j)
                        else:
                            lots[j] = (new_lot_qty, lot_price, lot_date)
                            j += 1
                        sell_qty -= close_qty
                    else:
                        j += 1
                # any remaining sold quantity opens a short lot
                if sell_qty > 0:
                    lots.append((-sell_qty, price, date))

            open_lots[sym] = lots

        cash += cash_delta

        pv = compute_portfolio_value(positions, cash, data, tickers, date)
        history.append({'date': date, 'signal': signal, 'cash': cash, 'pv': pv, 'positions': positions.copy(), 'trades': trades})

    hist_df = pd.DataFrame(history).set_index('date')
    final_value = hist_df['pv'].iloc[-1] if not hist_df.empty else initial_capital

    # compute realized P&L (sum of closed events) and unrealized P&L from open lots
    realized_total = sum(t['pnl'] for t in closed_trade_events)
    unrealized_total = 0.0
    last_date = dates[-1] if len(dates) > 0 else None
    if last_date is not None:
        for sym, lots in open_lots.items():
            df_sym = data.get(sym)
            if df_sym is None or last_date not in df_sym.index:
                continue
            last_price = df_sym.at[last_date, 'Close'] if 'Close' in df_sym.columns else None
            if last_price is None or pd.isna(last_price):
                continue
            for qty, entry_price, entry_date in lots:
                unrealized_total += qty * (last_price - entry_price)

    return final_value, hist_df, {
        'closed_trades': closed_trade_events,
        'placed_count': trades_placed_count,
        'realized_pnl': realized_total,
        'unrealized_pnl': unrealized_total,
        'open_lots': open_lots,
    }


def calculate_metrics(df: pd.DataFrame, trade_info: dict = None):
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

        # trade-level stats from trade_info (if available)
        num_trades_placed = 0
        pct_gain = float('nan')
        pct_loss = float('nan')

        if trade_info is not None:
            closed = trade_info.get('closed_trades', [])
            num_trades_placed = trade_info.get('placed_count', 0)
            closed_count = len(closed)
            if closed_count > 0:
                gains = sum(1 for t in closed if t.get('pnl', 0) > 0)
                losses = sum(1 for t in closed if t.get('pnl', 0) < 0)
                pct_gain = gains / closed_count * 100.0
                pct_loss = losses / closed_count * 100.0

        return total_return, cagr, ann_vol, sharpe, max_dd, num_trades_placed, pct_gain, pct_loss

def print_metrics(total_return, cagr, ann_vol, sharpe, max_dd, num_trades=0, pct_gain=float('nan'), pct_loss=float('nan')):
    print(f'Total return: {total_return:.2%}')
    print(f'CAGR: {cagr:.2%}')
    print(f'Annualized vol: {ann_vol:.2%}')
    print(f'Sharpe (rf=0): {sharpe:.2f}')
    print(f'Max drawdown: {max_dd:.2%}')
    print(f'Number of trades placed: {num_trades}')
    if not pd.isna(pct_gain):
        print(f'Percent of closed trades that gained: {pct_gain:.2f}%')
    if not pd.isna(pct_loss):
        print(f'Percent of closed trades that lost: {pct_loss:.2f}%')


if __name__ == '__main__':
    start = '2025-12-01'
    # end = datetime.today().strftime('%Y-%m-%d')
    end = '2026-02-26'
    final, df, trade_info = run_backtest(start, end, tickers=['SQQQ', 'SOXS', 'SPXS'], return_threshold=0.0) #0.0255
    print(f'Final portfolio value: ${final:,.2f}')  
    print(f"Realized P&L: {trade_info.get('realized_pnl', 0):,.2f}")
    print(f"Unrealized P&L: {trade_info.get('unrealized_pnl', 0):,.2f}")
    # closed = trade_info.get('closed_trades', [])
    # if closed:
    #     print('Closed trades (sample up to 10):')
    #     for t in closed[:10]:
    #         print(f"{t['exit_date'].date()} {t['symbol']} qty={t['qty']} entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} pnl={t['pnl']:.2f}")
    print_metrics(*calculate_metrics(df, trade_info))
    final, df, trade_info = run_backtest(start, end, tickers=['CAT', 'OXY'], return_threshold=0.0) #0.0520
    print(f'Final portfolio value: ${final:,.2f}')  
    print(f"Realized P&L: {trade_info.get('realized_pnl', 0):,.2f}")
    print(f"Unrealized P&L: {trade_info.get('unrealized_pnl', 0):,.2f}")
    # closed = trade_info.get('closed_trades', [])
    # if closed:
    #     print('Closed trades (sample up to 10):')
    #     for t in closed[:10]:
    #         print(f"{t['exit_date'].date()} {t['symbol']} qty={t['qty']} entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} pnl={t['pnl']:.2f}")
    print_metrics(*calculate_metrics(df, trade_info))
    # print(df[['signal', 'cash', 'pv']].tail(10))


