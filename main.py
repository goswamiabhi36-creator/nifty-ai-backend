from fastapi import FastAPI
from datetime import datetime
import requests

app = FastAPI(title="NIFTY AI Backend")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "NIFTY AI Backend",
        "version": "V2",
        "time": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "V2"
    }


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def moving_average(prices, period):
    if len(prices) < period:
        return None

    return sum(prices[-period:]) / period


@app.get("/nifty")
def nifty():

    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"

    try:
        response = requests.get(
            url,
            params={
                "range": "1d",
                "interval": "5m"
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

        timestamps = result.get("timestamp", [])
        quote = result["indicators"]["quote"][0]

        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        prices = [
            float(price)
            for price in closes
            if price is not None
        ]

        if not prices:
            return {
                "symbol": "NIFTY 50",
                "price": None,
                "status": "no_price_data"
            }

        price = prices[-1]
        previous_close = meta.get("previousClose")

        change = None
        change_percent = None

        if previous_close is not None:
            change = price - previous_close
            change_percent = (change / previous_close) * 100

        # Moving averages
        ma_5 = moving_average(prices, 5)
        ma_10 = moving_average(prices, 10)
        ma_20 = moving_average(prices, 20)

        # RSI
        rsi = calculate_rsi(prices, 14)

        # Momentum
        momentum = None

        if len(prices) >= 10:
            momentum = price - prices[-10]

        # Trend analysis
        trend = "SIDEWAYS"

        if ma_5 and ma_10 and ma_20:

            if price > ma_5 > ma_10 > ma_20:
                trend = "STRONG_BULLISH"

            elif price > ma_10 and ma_10 > ma_20:
                trend = "BULLISH"

            elif price < ma_5 < ma_10 < ma_20:
                trend = "STRONG_BEARISH"

            elif price < ma_10 and ma_10 < ma_20:
                trend = "BEARISH"

        # Signal engine
        bullish_points = 0
        bearish_points = 0

        if trend in ["BULLISH", "STRONG_BULLISH"]:
            bullish_points += 2

        if trend in ["BEARISH", "STRONG_BEARISH"]:
            bearish_points += 2

        if rsi is not None:

            if 50 < rsi < 70:
                bullish_points += 1

            elif rsi >= 70:
                bearish_points += 1

            elif 30 < rsi < 50:
                bearish_points += 1

            elif rsi <= 30:
                bullish_points += 1

        if momentum is not None:

            if momentum > 0:
                bullish_points += 1

            elif momentum < 0:
                bearish_points += 1

        # Final signal
        if bullish_points >= 3 and bullish_points > bearish_points:
            signal = "BUY"

        elif bearish_points >= 3 and bearish_points > bullish_points:
            signal = "SELL"

        else:
            signal = "NEUTRAL"

        # Confidence
        total_points = bullish_points + bearish_points

        if total_points == 0:
            confidence = 0
        else:
            confidence = round(
                max(bullish_points, bearish_points)
                / total_points * 100,
                1
            )

        # Volume
        recent_volumes = [
            float(v)
            for v in volumes[-10:]
            if v is not None
        ]

        average_volume = None

        if recent_volumes:
            average_volume = sum(recent_volumes) / len(recent_volumes)

        latest_volume = (
            recent_volumes[-1]
            if recent_volumes
            else None
        )

        volume_status = "UNKNOWN"

        if latest_volume is not None and average_volume is not None:

            if latest_volume > average_volume * 1.2:
                volume_status = "HIGH"

            elif latest_volume < average_volume * 0.8:
                volume_status = "LOW"

            else:
                volume_status = "NORMAL"

        return {
            "symbol": "NIFTY 50",

            "price": round(price, 2),

            "previous_close": previous_close,

            "change": round(change, 2)
            if change is not None else None,

            "change_percent": round(change_percent, 2)
            if change_percent is not None else None,

            "trend": trend,

            "signal": signal,

            "confidence": confidence,

            "rsi_14": round(rsi, 2)
            if rsi is not None else None,

            "ma_5": round(ma_5, 2)
            if ma_5 is not None else None,

            "ma_10": round(ma_10, 2)
            if ma_10 is not None else None,

            "ma_20": round(ma_20, 2)
            if ma_20 is not None else None,

            "momentum": round(momentum, 2)
            if momentum is not None else None,

            "volume_status": volume_status,

            "market_status": "live",

            "analysis_version": "V2",

            "time": datetime.now().isoformat()
        }

    except Exception as e:

        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
