import time
from datetime import datetime, timedelta
import pytz
from trade import run_strategy

# ================= CONFIG =================
TIMEZONE = pytz.timezone("US/Eastern")
RUN_HOUR = 9
RUN_MINUTE = 25

# ================= TIME LOGIC =================
def get_next_run_time() -> datetime:
    """Compute next run time (today or tomorrow at target time)."""
    now = datetime.now(TIMEZONE)

    target = now.replace(
        hour=RUN_HOUR,
        minute=RUN_MINUTE,
        second=0,
        microsecond=0
    )

    if now >= target:
        target += timedelta(days=1)

    return target

def format_timedelta(td: timedelta) -> str:
    """Format timedelta nicely."""
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ================= MAIN LOOP =================
def main():
    print("Starting runner...")

    while True:
        next_run = get_next_run_time()
        print(f"\nNext run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Countdown loop
        while True:
            now = datetime.now(TIMEZONE)
            remaining = next_run - now

            if remaining.total_seconds() <= 0:
                break

            print(f"\rTime until next run: {format_timedelta(remaining)}", end="")
            time.sleep(1)

        print("\nExecuting strategy...")
        try:
            run_strategy()
        except Exception as e:
            print(f"Error during strategy execution: {e}")

        print("Run complete. Scheduling next execution...")

        # Small buffer to avoid double-triggering
        time.sleep(5)

# ================= ENTRY =================
if __name__ == "__main__":
    main()