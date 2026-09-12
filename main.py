from fastapi import FastAPI
from datetime import datetime
import requests

app = FastAPI(title="NIFTY AI Backend")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "NIFTY AI Backend",
        "time": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/nifty")
def nifty():

    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"

    try:
        response = requests.get(
            url,
            params={
                "range": "1d",
                "interval": "1m"
            },
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=10
        )

        if response.status_code != 200:
            return {
                "symbol": "NIFTY 50",
                "price": None,
                "status": "data_error",
                "http_status": response.status_code
            }

        data = response.json()

        result = data["chart"]["result"][0]
        meta = result["meta"]

        price = meta.get("regularMarketPrice")
        previous_close = meta.get("previousClose")

        change = None
        change_percent = None

        if price is not None and previous_close is not None:
            change = price - previous_close
            change_percent = (change / previous_close) * 100

        return {
            "symbol": "NIFTY 50",
            "price": price,
            "previous_close": previous_close,
            "change": round(change, 2) if change is not None else None,
            "change_percent": round(change_percent, 2) if change_percent is not None else None,
            "market_status": "live" if price is not None else "unavailable",
            "signal": "NEUTRAL",
            "trend": "UNKNOWN",
            "time": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
