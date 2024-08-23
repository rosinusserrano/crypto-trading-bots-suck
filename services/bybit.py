from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import os

load_dotenv()

def get_bybit_session():
    """Get a bybit session"""
    session = HTTP(
        testnet=False,
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
    )

    return session