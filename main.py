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
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        info = data["metadata"]

        return {
            "symbol": "NIFTY 50",
            "price": info.get("last"),
            "change": info.get("change"),
            "change_percent": info.get("percentChange"),
            "market_status": "live",
            "time": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
