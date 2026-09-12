from fastapi import FastAPI
from datetime import datetime
import requests

app = FastAPI(title="NIFTY AI Backend")


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "NIFTY AI Backend",
        "version": "V2.2",
        "time": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "V2.2"
    }


def sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def ema(prices, period):
    if len(prices) < period:
        return None

    value = sum(prices[:period]) / period
    multiplier = 2 / (period + 1)

    for price in prices[period:]:
        value = (price - value) * multiplier + value

    return value


def rsi(prices, period=14):
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


def macd(prices):
    if len(prices) < 35:
        return None, None, None

    macd_values = []

    for i in range(26, len(prices) + 1):
        subset = prices[:i]

        e12 = ema(subset, 12)
        e26 = ema(subset, 26)

        if e12 is not None and e26 is not None:
            macd_values.append(e12 - e26)

    if len(macd_values) < 9:
        return None, None, None

    macd_value = macd_values[-1]
    signal_value = ema(macd_values, 9)

    if signal_value is None:
        return None, None, None

    histogram = macd_value - signal_value

    return macd_value, signal_value, histogram


def vwap(prices, volumes):
    if not prices or not volumes:
        return None

    total_volume = 0
    total_value = 0

    for price, volume in zip(prices, volumes):

        if price is None or volume is None:
            continue

        if volume <= 0:
            continue

        total_value += price * volume
        total_volume += volume

    if total_volume == 0:
        return None

    return total_value / total_volume


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

        # -------------------------
        # INDICATORS
        # -------------------------

        ma5 = sma(prices, 5)
        ma10 = sma(prices, 10)
        ma20 = sma(prices, 20)

        rsi_value = rsi(prices, 14)

        macd_value, macd_signal, macd_hist = macd(prices)

        vwap_value = vwap(
            prices,
            clean_volumes
        )

        momentum = None

        if len(prices) >= 10:
            momentum = price - prices[-10]

        # -------------------------
        # SUPPORT / RESISTANCE
        # -------------------------

        support = None
        resistance = None

        if len(low_prices) >= 10:
            support = min(low_prices[-10:])

        if len(high_prices) >= 10:
            resistance = max(high_prices[-10:])

        # -------------------------
        # TREND
        # -------------------------

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

        # -------------------------
        # SCORING
        # -------------------------

        bullish_score = 0
        bearish_score = 0

        reasons = []
        warnings = []

        # Trend weight
        if trend == "STRONG_BULLISH":
            bullish_score += 3
            reasons.append("Strong bullish trend")

        elif trend == "BULLISH":
            bullish_score += 2
            reasons.append("Bullish trend")

        elif trend == "STRONG_BEARISH":
            bearish_score += 3
            reasons.append("Strong bearish trend")

        elif trend == "BEARISH":
            bearish_score += 2
            reasons.append("Bearish trend")

        # RSI
        if rsi_value is not None:

            if rsi_value >= 70:
                bearish_score += 1
                reasons.append("RSI overbought")

            elif rsi_value <= 30:
                bullish_score += 1
                reasons.append("RSI oversold")

            elif rsi_value >= 55:
                bullish_score += 1
                reasons.append("RSI bullish")

            elif rsi_value <= 45:
                bearish_score += 1
                reasons.append("RSI bearish")

            else:
                warnings.append("RSI neutral")

        # Momentum
        if momentum is not None:

            if momentum > 0:
                bullish_score += 2
                reasons.append("Positive momentum")

            elif momentum < 0:
                bearish_score += 2
                reasons.append("Negative momentum")

        # MACD
        if macd_value is not None and macd_signal is not None:

            if macd_value > macd_signal:

                bullish_score += 2

                reasons.append("MACD bullish")

            else:

                bearish_score += 2

                reasons.append("MACD bearish")

        else:

            warnings.append("MACD unavailable")

        # VWAP
        if vwap_value is not None:

            if price > vwap_value:

                bullish_score += 1

                reasons.append("Price above VWAP")

            elif price < vwap_value:

                bearish_score += 1

                reasons.append("Price below VWAP")

        else:

            warnings.append(
                "VWAP unavailable for NIFTY index"
            )

        # -------------------------
        # SUPPORT / RESISTANCE
        # -------------------------

        support_distance = None
        resistance_distance = None

        if support is not None:

            support_distance = (
                (price - support)
                / price
            ) * 100

        if resistance is not None:

            resistance_distance = (
                (resistance - price)
                / price
            ) * 100

        # Near support
        if (
            support_distance is not None
            and support_distance <= 0.15
        ):

            bullish_score += 1

            reasons.append(
                "Price near support"
            )

        # Near resistance
        if (
            resistance_distance is not None
            and resistance_distance <= 0.15
        ):

            bearish_score += 1

            reasons.append(
                "Price near resistance"
            )

        # -------------------------
        # VOLUME
        # -------------------------

        volume_status = "UNKNOWN"

        if len(clean_volumes) >= 10:

            recent = clean_volumes[-10:]

            average_volume = (
                sum(recent)
                / len(recent)
            )

            latest_volume = recent[-1]

            if latest_volume > average_volume * 1.2:

                volume_status = "HIGH"

            elif latest_volume < average_volume * 0.8:

                volume_status = "LOW"

            else:

                volume_status = "NORMAL"

        # -------------------------
        # SIGNAL
        # -------------------------

        difference = (
            bullish_score
            - bearish_score
        )

        total_score = (
            bullish_score
            + bearish_score
        )

        if difference >= 5:

            signal = "STRONG BUY"

        elif difference >= 3:

            signal = "BUY"

        elif difference <= -5:

            signal = "STRONG SELL"

        elif difference <= -3:

            signal = "SELL"

        else:

            signal = "NEUTRAL"

        # -------------------------
        # CONFIDENCE
        # -------------------------

        if total_score > 0:

            raw_confidence = (
                abs(difference)
                / total_score
            ) * 100

        else:

            raw_confidence = 0

        # Do not allow false certainty
        confidence = min(
            max(raw_confidence, 20),
            85
        )

        # Conflicting indicators reduce confidence
        if (
            bullish_score > 0
            and bearish_score > 0
        ):

            confidence -= 10

        confidence = max(
            confidence,
            20
        )

        # -------------------------
        # SIGNAL STRENGTH
        # -------------------------

        if confidence >= 70:

            signal_strength = "STRONG"

        elif confidence >= 50:

            signal_strength = "MODERATE"

        else:

            signal_strength = "WEAK"

        # -------------------------
        # ENTRY / AVOID ZONE
        # -------------------------

        entry_zone = None
        avoid_zone = None

        if signal in [
            "BUY",
            "STRONG BUY"
        ]:

            if support is not None:

                entry_zone = (
                    round(support, 2),
                    round(price, 2)
                )

            if resistance is not None:

                avoid_zone = (
                    round(resistance, 2),
                    round(resistance * 1.002, 2)
                )

        elif signal in [
            "SELL",
            "STRONG SELL"
        ]:

            if resistance is not None:

                entry_zone = (
                    round(price, 2),
                    round(resistance, 2)
                )

            if support is not None:

                avoid_zone = (
                    round(support * 0.998, 2),
                    round(support, 2)
                )

        return {

            "symbol": "NIFTY 50",

            "price": round(price, 2),

            "previous_close": previous_close,

            "change": round(change, 2)
            if change is not None
            else None,

            "change_percent": round(
                change_percent,
                2
            )
            if change_percent is not None
            else None,

            "trend": trend,

            "signal": signal,

            "signal_strength": signal_strength,

            "confidence": round(
                confidence,
                1
            ),

            "rsi_14": round(
                rsi_value,
                2
            )
            if rsi_value is not None
            else None,

            "ma_5": round(
                ma5,
                2
            )
            if ma5 is not None
            else None,

            "ma_10": round(
                ma10,
                2
            )
            if ma10 is not None
            else None,

            "ma_20": round(
                ma20,
                2
            )
            if ma20 is not None
            else None,

            "vwap": round(
                vwap_value,
                2
            )
            if vwap_value is not None
            else None,

            "macd": round(
                macd_value,
                4
            )
            if macd_value is not None
            else None,

            "macd_signal": round(
                macd_signal,
                4
            )
            if macd_signal is not None
            else None,

            "macd_histogram": round(
                macd_hist,
                4
            )
            if macd_hist is not None
            else None,

            "momentum": round(
                momentum,
                2
            )
            if momentum is not None
            else None,

            "support": round(
                support,
                2
            )
            if support is not None
            else None,

            "resistance": round(
                resistance,
                2
            )
            if resistance is not None
            else None,

            "support_distance_percent":
                round(
                    support_distance,
                    3
                )
                if support_distance is not None
                else None,

            "resistance_distance_percent":
                round(
                    resistance_distance,
                    3
                )
                if resistance_distance is not None
                else None,

            "volume_status":
                volume_status,

            "bullish_score":
                bullish_score,

            "bearish_score":
                bearish_score,

            "reasons":
                reasons,

            "warnings":
                warnings,

            "entry_zone":
                entry_zone,

            "avoid_zone":
                avoid_zone,

            "market_status":
                "live",

            "analysis_version":
                "V2.2",

            "time":
                datetime.now().isoformat()
        }

    except Exception as e:

        return {
            "symbol": "NIFTY 50",
            "price": None,
            "status": "data_error",
            "error": str(e)
        }
