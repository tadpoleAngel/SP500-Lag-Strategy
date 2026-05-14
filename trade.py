import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv
import math

# ================= CONFIG =================
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

TICKERS = ["SPHY", "SCYB"]
SP500_TICKER = "^GSPC"
CAPITAL_FRACTION = 0.9  # use 90% of buying power
RETURN_THRESHOLD = 0.015  # decimal representation of percent
TIMEZONE = "US/Eastern"

# Alpaca client is created lazily to avoid side-effects at import time
_api = None

def get_api():
    """Return a cached Alpaca REST client or None if credentials are not configured.

    This avoids creating the client on import so modules that only need the
    pure helpers can import `trade` without requiring Alpaca credentials.
    """
    global _api
    if _api is not None:
        return _api

    if not API_KEY or not API_SECRET:
        # Credentials not present; caller should handle None
        return None

    try:
        _api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    except Exception:
        _api = None
    return _api

# ================= DATA MODULE =================
def get_sp500_signal() -> int:
    """Returns +1 (bullish), -1 (bearish), or 0 (neutral) based on yesterday's S&P move and RETURN_THRESHOLD."""
    data = yf.download(SP500_TICKER, period="3d", interval="1d")

    if len(data) < 2:
        raise ValueError("Not enough data to compute S&P 500 signal")

    yesterday = data.iloc[-2]["Close"]
    prev_day = data.iloc[-3]["Close"]

    return compute_signal_from_closes(prev_day, yesterday, RETURN_THRESHOLD)


def compute_signal_from_closes(prev_close: float, yesterday_close: float, threshold: float) -> int:
    """Pure helper to compute signal from two consecutive close prices and a threshold.

    Returns +1, -1, or 0.
    """
    if prev_close == 0 or prev_close is None or yesterday_close is None:
        return 0

    ret = (yesterday_close - prev_close) / prev_close

    if ret >= threshold:
        return 1
    elif ret <= -threshold:
        return -1
    else:
        return 0

# ================= POSITION MODULE =================
def get_current_positions() -> dict[str, float]:
    """Returns dict of current positions."""
    positions = {}
    api = get_api()
    if api is None:
        # Running in backtest/local mode - no live API available
        return positions

    try:
        api_response = api.list_positions()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return positions
    
    for pos in api_response:
        positions[pos.symbol] = float(pos.qty)
    return positions


def compute_desired_positions(signal: int) -> dict[str, int]:
    """Compute desired signed quantities for each ticker based on signal.

    Returns a mapping of symbol -> desired_qty (int). Positive = long, negative = short.
    """
    allocation = allocate_capital()
    desired: dict[str, int] = {}

    for ticker in TICKERS:
        try:
            price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            price = None

        if not price or price <= 0:
            qty = 0
        else:
            qty = int(allocation / price)

        desired_qty = 0
        if signal == 1:
            desired_qty = qty
        elif signal == -1:
            desired_qty = -qty

        desired[ticker] = desired_qty

    return desired


def compute_desired_from_prices(per_ticker_allocation: float, signal: int, price_map: dict[str, float]) -> dict[str, int]:
    """Pure helper: compute desired signed integer quantities for each ticker given per-ticker dollar allocation.

    - per_ticker_allocation: dollars to allocate per ticker
    - signal: +1 long, -1 short, 0 neutral
    - price_map: mapping ticker -> execution price (float)

    Returns dict ticker -> desired_qty (int)
    """

    desired: dict[str, int] = {}

    for ticker, price in price_map.items():
        # treat None / NaN / non-positive as unavailable -> zero allocation
        if price is None:
            qty = 0
        else:
            try:
                p = float(price)
            except Exception:
                qty = 0
            else:
                if math.isnan(p) or p <= 0:
                    qty = 0
                else:
                    qty = int(per_ticker_allocation // p)

        if signal == 1:
            desired_qty = qty
        elif signal == -1:
            desired_qty = -qty
        else:
            desired_qty = 0

        desired[ticker] = desired_qty

    return desired


def close_unwanted_positions(desired_positions: dict[str, int], current_positions: dict[str, float]) -> None:
    """Close positions that are held but not desired anymore."""
    for symbol, curr_qty in current_positions.items():
        if symbol not in desired_positions:
            qty_to_close = abs(int(curr_qty))
            if qty_to_close == 0:
                continue
            side = 'sell' if curr_qty > 0 else 'buy'
            try:
                api = get_api()
                if api is None:
                    print(f"Skipping close of {symbol}: no API configured")
                    continue
                api.submit_order(
                    symbol=symbol,
                    qty=qty_to_close,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                print(f"Closed out {symbol}: {side} {qty_to_close}")
            except Exception as e:
                print(f"Error closing {symbol}: {e}")


def adjust_desired_positions(desired_positions: dict[str, int], current_positions: dict[str, float]) -> None:
    """For tickers in desired_positions, trade only the difference (desired - current)."""
    for symbol, desired_qty in desired_positions.items():
        curr_qty = int(current_positions.get(symbol, 0))
        diff = int(desired_qty) - curr_qty

        if diff == 0:
            continue

        side = 'buy' if diff > 0 else 'sell'
        trade_qty = abs(int(diff))

        if trade_qty == 0:
            continue

        try:
            api = get_api()
            if api is None:
                print(f"Skipping submit {side} {trade_qty} for {symbol}: no API configured")
                continue
            api.submit_order(
                symbol=symbol,
                qty=trade_qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f"Submitted {side} {trade_qty} for {symbol} (desired {desired_qty}, current {curr_qty})")
        except Exception as e:
            print(f"Error rebalancing {symbol}: {e}")


def rebalance_positions(desired_positions: dict[str, int], current_positions: dict[str, float]) -> None:
    """Wrapper that closes unwanted positions then adjusts desired ones."""
    close_unwanted_positions(desired_positions, current_positions)
    adjust_desired_positions(desired_positions, current_positions)

# ================= EXECUTION MODULE =================
def close_all_positions() -> None:
    """Liquidate all positions."""
    api = get_api()
    if api is None:
        print("No API configured - skipping close_all_positions")
        return

    for pos in api.list_positions():
        side = 'sell' if float(pos.qty) > 0 else 'buy'
        api.submit_order(
            symbol=pos.symbol,
            qty=abs(int(float(pos.qty))),
            side=side,
            type='market',
            time_in_force='day'
        )

def allocate_capital() -> float:
    """Returns per-ticker dollar allocation."""
    api = get_api()
    if api is None:
        raise RuntimeError("Alpaca API not configured - cannot allocate capital")

    account = api.get_account()
    buying_power = float(account.buying_power)
    total_alloc = buying_power * CAPITAL_FRACTION
    return total_alloc / len(TICKERS)

def open_positions(signal: int) -> None:
    """Open positions for the given signal (+1 long, -1 short).

    Note: kept for backward compatibility but not used by the rebalance runner.
    """
    allocation = allocate_capital()

    api = get_api()
    if api is None:
        print("No API configured - skipping open_positions")
        return

    for ticker in TICKERS:
        price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        qty = int(allocation / price)

        if qty == 0:
            continue

        side = "buy" if signal == 1 else "sell"

        api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )

# ================= STRATEGY RUNNER =================
def run_strategy() -> None:
    print(f"[{datetime.now()}] Running strategy...")

    signal = get_sp500_signal()
    if signal == 1:
        print("Signal: LONG")
    elif signal == -1:
        print("Signal: SHORT")
    else:
        print(f"Signal: NEUTRAL (change within +/-{RETURN_THRESHOLD:.3f}) - no trades placed")
        return

    current_positions = get_current_positions()

    # Compute desired positions and rebalance only the differences to avoid wash trades.
    desired_positions = compute_desired_positions(signal)
    print(f"Desired positions: {desired_positions}")
    rebalance_positions(desired_positions, current_positions)

    print("Done.")

# ================= MAIN =================
if __name__ == "__main__":
    run_strategy()