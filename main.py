from fastapi import FastAPI
from datetime import datetime
import requests

app = FastAPI(title="NIFTY AI Backend")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "NIFTY AI Backend",
        "version": "V2.1",
        "time": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "V2.1"
    }


def sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices, period):
    if len(prices) < period:
        return None

    ema = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_macd(prices):
    if len(prices) < 26:
        return None, None, None

    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)

    if ema12 is None or ema26 is None:
        return None, None, None

    macd = ema12 - ema26

    # Approximate signal line using recent MACD calculations
    macd_values = []

    for i in range(26, len(prices) + 1):
        subset = prices[:i]
        e12 = calculate_ema(subset, 12)
        e26 = calculate_ema(subset, 26)

        if e12 is not None and e26 is not None:
            macd_values.append(e12 - e26)

    if len(macd_values) >= 9:
        signal = calculate_ema(macd_values, 9)
    else:
        signal = None

    histogram = (
        macd - signal
        if signal is not None
        else None
    )

    return macd, signal, histogram


def calculate_vwap(prices, volumes):
    if not prices or not volumes:
        return None

    total_pv = 0
    total_volume = 0

    for price, volume in zip(prices, volumes):
        if price is not None and volume is not None:
            total_pv += price * volume
            total_volume += volume

    if total_volume == 0:
        return None

    return total_pv / total_volume


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

        quote = result["indicators"]["quote"][0]

        closes = quote.get("close", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        volumes = quote.get("volume", [])

        prices = [
            float(x)
            for x in closes
            if x is not None
        ]

        high_prices = [
            float(x)
            for x in highs
            if x is not None
        ]

        low_prices = [
            float(x)
            for x in lows
            if x is not None
        ]

        clean_volumes = [
            float(x)
            for x in volumes
            if x is not None
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

            change_percent = (
                change / previous_close
            ) * 100

        # Moving averages
        ma5 = sma(prices, 5)
        ma10 = sma(prices, 10)
        ma20 = sma(prices, 20)

        # RSI
        rsi = calculate_rsi(prices, 14)

        # MACD
        macd, macd_signal, macd_histogram = calculate_macd(prices)

        # VWAP
        vwap = calculate_vwap(
            prices,
            clean_volumes
        )

        # Momentum
        momentum = None

        if len(prices) >= 10:
            momentum = price - prices[-10]

        # Support / Resistance
        support = None
        resistance = None

        if len(low_prices) >= 10:
            support = min(low_prices[-10:])

        if len(high_prices) >= 10:
            resistance = max(high_prices[-10:])

        # Trend
        trend = "SIDEWAYS"

        if ma5 and ma10 and ma20:

            if price > ma5 > ma10 > ma20:
                trend = "STRONG_BULLISH"

            elif price > ma10 and ma10 > ma20:
                trend = "BULLISH"

            elif price < ma5 < ma10 < ma20:
                trend = "STRONG_BEARISH"

            elif price < ma10 and ma10 < ma20:
                trend = "BEARISH"

        bullish = 0
        bearish = 0

        reasons = []

        # Trend points
        if trend == "STRONG_BULLISH":
            bullish += 3
            reasons.append("Strong bullish trend")

        elif trend == "BULLISH":
            bullish += 2
            reasons.append("Bullish trend")

        elif trend == "STRONG_BEARISH":
            bearish += 3
            reasons.append("Strong bearish trend")

        elif trend == "BEARISH":
            bearish += 2
            reasons.append("Bearish trend")

        # RSI
        if rsi is not None:

            if rsi >= 70:
                bearish += 1
                reasons.append("RSI overbought")

            elif rsi <= 30:
                bullish += 1
                reasons.append("RSI oversold")

            elif rsi > 50:
                bullish += 1
                reasons.append("RSI bullish")

            else:
                bearish += 1
                reasons.append("RSI bearish")

        # Momentum
        if momentum is not None:

            if momentum > 0:
                bullish += 1
                reasons.append("Positive momentum")

            elif momentum < 0:
                bearish += 1
                reasons.append("Negative momentum")

        # VWAP
        if vwap is not None:

            if price > vwap:
                bullish += 1
                reasons.append("Price above VWAP")

            elif price < vwap:
                bearish += 1
                reasons.append("Price below VWAP")

        # MACD
        if macd is not None and macd_signal is not None:

            if macd > macd_signal:
                bullish += 2
                reasons.append("MACD bullish")

            else:
                bearish += 2
                reasons.append("MACD bearish")

        # Volume
        volume_status = "UNKNOWN"

        if len(clean_volumes) >= 10:

            avg_volume = (
                sum(clean_volumes[-10:])
                / len(clean_volumes[-10:])
            )

            latest_volume = clean_volumes[-1]

            if latest_volume > avg_volume * 1.2:
                volume_status = "HIGH"

                if price > previous_close:
                    bullish += 1
                    reasons.append("High bullish volume")

                else:
                    bearish += 1
                    reasons.append("High bearish volume")

            elif latest_volume < avg_volume * 0.8:
                volume_status = "LOW"

            else:
                volume_status = "NORMAL"

        # Final signal
        difference = bullish - bearish

        if difference >= 3:
            signal = "BUY"

        elif difference <= -3:
            signal = "SELL"

        else:
            signal = "NEUTRAL"

        # Realistic confidence score
        total = bullish + bearish

        if total > 0:

            confidence = (
                abs(difference) / total
            ) * 100

        else:
            confidence = 0

        # Cap confidence to avoid fake certainty
        confidence = min(confidence, 90)

        # Signal strength
        if confidence >= 70:
            signal_strength = "STRONG"

        elif confidence >= 50:
            signal_strength = "MODERATE"

        else:
            signal_strength = "WEAK"

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

            "signal_strength": signal_strength,

            "confidence": round(confidence, 1),

            "rsi_14": round(rsi, 2)
            if rsi is not None else None,

            "ma_5": round(ma5, 2)
            if ma5 is not None else None,

            "ma_10": round(ma10, 2)
            if ma10 is not None else None,

            "ma_20": round(ma20, 2)
            if ma20 is not None else None,

            "vwap": round(vwap, 2)
            if vwap is not None else None,

            "macd": round(macd, 4)
            if macd is not None else None,

            "macd_signal": round(macd_signal, 4)
            if macd_signal is not None else None,

            "macd_histogram": round(macd_histogram, 4)
            if macd_histogram is not None else None,

            "momentum": round(momentum, 2)
            if momentum is not None else None,

            "support": round(support, 2)
            if support is not None else None,

            "resistance": round(resistance, 2)
            if resistance is not None else None,

            "volume_status": volume_status,

            "bullish_points": bullish,

            "bearish_points": bearish,

            "reasons": reasons,

            "market_status": "live",

            "analysis_version": "V2.1",

            "time": datetime.now().isoformat()
        }

    except Exception as e:

        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
