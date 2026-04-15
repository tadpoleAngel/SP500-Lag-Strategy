from dis import Positions
import alpaca_trade_api as tradeapi
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv

# ================= CONFIG =================
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

TICKERS = ["SPHY", "SCYB"]
SP500_TICKER = "^GSPC"
CAPITAL_FRACTION = 0.9  # use 90% of buying power
TIMEZONE = "US/Eastern"

# ================= API INIT =================
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# ================= DATA MODULE =================
def get_sp500_signal() -> int:
    """Returns +1 (bullish) or -1 (bearish) based on yesterday's S&P move."""
    data = yf.download(SP500_TICKER, period="3d", interval="1d")

    if len(data) < 2:
        raise Exception("Not enough data")

    yesterday = data.iloc[-2]
    prev_day = data.iloc[-3]

    ret = (yesterday["Close"][SP500_TICKER] - prev_day["Close"][SP500_TICKER]) / prev_day["Close"][SP500_TICKER]

    return 1 if ret > 0 else -1

# ================= POSITION MODULE =================
def get_current_positions() -> dict[str, float]:
    """Returns dict of current positions."""
    positions = {}
    api_response = {}

    try:
        api_response = api.list_positions()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return positions
    
    for pos in api_response:
        positions[pos.symbol] = float(pos.qty)
    return positions

# ================= EXECUTION MODULE =================
def close_all_positions() -> None:
    """Liquidate all positions."""
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
    account = api.get_account()
    buying_power = float(account.buying_power)
    total_alloc = buying_power * CAPITAL_FRACTION
    return total_alloc / len(TICKERS)

def open_positions(signal: int) -> None:
    """
    signal = +1 (long) or -1 (short)
    """
    allocation = allocate_capital()

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
    print(f"Signal: {'LONG' if signal == 1 else 'SHORT'}")

    current_positions = get_current_positions()

    # Determine if we need to flip
    if current_positions:
        print("Closing existing positions...")
        close_all_positions()

    print("Opening new positions...")
    open_positions(signal)

    print("Done.")

# ================= MAIN =================
if __name__ == "__main__":
    run_strategy()