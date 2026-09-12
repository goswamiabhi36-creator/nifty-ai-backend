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
        "price": 25000,
        "status": "demo_data"
    }
