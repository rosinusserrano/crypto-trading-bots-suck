"Fetch Bybit Historical data"

from datetime import datetime
from time import sleep
import pandas as pd
from pybit.unified_trading import HTTP


def get_all_symbols(bybit_session: HTTP):
    """Get all symbols available on Bybit"""
    all_symbols = [
        l["symbol"]
        for l in bybit_session.get_tickers(category="linear")["result"]["list"]
    ]
    return all_symbols


def get_date_of_first_entry(market_symbol: str, bybit_session: HTTP):
    """Get the datetime of the first entry of the market_symbol"""
    print("Getting datetime of first entry for", market_symbol)

    d = bybit_session.get_kline(
        symbol=market_symbol,
        category="linear",
        interval="1",
        limit=10,
        start=datetime.fromisoformat("2000-01-01 00:00:00").timestamp() * 1000)

    try:
        start_time = datetime.fromtimestamp(
            int(d["result"]["list"][-1][0]) / 1000)
        print(f"Datetime of first entry {start_time.isoformat(' ')}")
        return start_time
    except IndexError:
        print(f"Data not found for {market_symbol}")
        return None


def get_data(market_symbol, interval, bybit_session: HTTP):
    """Get data for a market_symbol with interval"""
    print("Retrieving", market_symbol, "with interval", interval)

    start_time = get_date_of_first_entry(market_symbol, bybit_session)

    if start_time is None:
        return

    total_time = datetime.now() - start_time

    alldata = []
    current_time = start_time
    while current_time < datetime.now():
        remaining_time = datetime.now() - current_time

        print(f"{((1 - remaining_time.days / total_time.days) * 100):.2f}%",
              current_time.isoformat(sep=" "))

        d = bybit_session.get_kline(symbol=market_symbol,
                                    interval=interval,
                                    category="linear",
                                    limit=1000,
                                    start=current_time.timestamp() * 1000)

        alldata.extend(
            reversed([
                [datetime.fromtimestamp(int(l[0]) / 1000).isoformat(sep=" ")] +
                l[1:] for l in d["result"]["list"]
            ]))

        last_datetime = datetime.fromisoformat(alldata[-1][0])
        if current_time == last_datetime:
            break
        current_time = last_datetime

        sleep(1)

    df = pd.DataFrame(alldata,
                      columns=[
                          "timestamp", "open", "high", "low", "close",
                          "volume", "turnover"
                      ])

    print(f"Last timestamp: {df.iloc[[-1]]['timestamp']}")

    df.to_csv(f"data/{market_symbol}-{interval}.csv", index=False)
