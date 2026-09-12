from fastapi import FastAPI
from datetime import datetime

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
    return {
        "symbol": "NIFTY 50",
        "price": None,
        "change": None,
        "change_percent": None,
        "market_status": "waiting_for_live_data",
        "signal": "NEUTRAL",
        "trend": "UNKNOWN",
        "time": datetime.now().isoformat()
    }            "price": info.get("last"),
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
